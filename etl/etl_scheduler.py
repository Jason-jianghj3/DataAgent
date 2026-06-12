"""ETL调度器 - 每小时自动执行数据仓库ETL管道"""
import threading
from datetime import datetime
from utils.logger import logger

_etl_lock = threading.Lock()
_last_status = {
    "last_run": None,
    "last_success": None,
    "next_run": None,
    "tasks": {},
    "running": False
}


def _run_etl():
    """执行ETL管道（带锁保护）"""
    if not _etl_lock.acquire(blocking=False):
        logger.warning("[ETL Scheduler] ETL正在运行中，跳过本次调度")
        return
    try:
        _last_status["running"] = True
        _last_status["last_run"] = datetime.now().isoformat()
        logger.info("[ETL Scheduler] 开始执行ETL管道...")

        from etl.etl_pipeline import ETLPipeline
        pipeline = ETLPipeline()
        results = pipeline.run_all()

        _last_status["tasks"] = results
        _last_status["last_success"] = datetime.now().isoformat()
        all_ok = all(r.get("success") for r in results.values())
        logger.info(f"[ETL Scheduler] ETL完成, 全部成功: {all_ok}")
    except Exception as e:
        logger.error(f"[ETL Scheduler] ETL执行失败: {e}")
        _last_status["last_success"] = None
    finally:
        _last_status["running"] = False
        _etl_lock.release()


def start_scheduler(app=None):
    """启动ETL调度器"""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        # 每小时第5分钟执行
        scheduler.add_job(_run_etl, 'cron', minute=5, id='etl_hourly')
        scheduler.start()
        logger.info("[ETL Scheduler] 调度器已启动, 每小时第5分钟执行ETL")

        # 计算下次运行时间
        from apscheduler.triggers.cron import CronTrigger
        trigger = CronTrigger(minute=5)
        next_time = trigger.get_next_fire_time(None, datetime.now())
        _last_status["next_run"] = next_time.isoformat() if next_time else None

        # 注册Flask蓝图
        if app:
            _register_blueprint(app)

        # 应用关闭时停止调度器
        import atexit
        atexit.register(lambda: scheduler.shutdown(wait=False))

    except ImportError:
        logger.warning("[ETL Scheduler] apscheduler未安装, ETL调度器未启动。安装: pip install apscheduler")
    except Exception as e:
        logger.error(f"[ETL Scheduler] 启动失败: {e}")


def _register_blueprint(app):
    """注册ETL管理API蓝图"""
    from flask import Blueprint, jsonify
    etl_bp = Blueprint('etl', __name__, url_prefix='/etl')

    @etl_bp.route('/status', methods=['GET'])
    def etl_status():
        return jsonify(_last_status)

    @etl_bp.route('/trigger', methods=['POST'])
    def etl_trigger():
        if _last_status.get("running"):
            return jsonify({"error": "ETL正在运行中"}), 409
        import threading
        t = threading.Thread(target=_run_etl, daemon=True)
        t.start()
        return jsonify({"message": "ETL已触发", "status": "running"})

    app.register_blueprint(etl_bp)
    logger.info("[ETL Scheduler] ETL管理API已注册 (/etl/status, /etl/trigger)")


def get_etl_status():
    """获取ETL状态（供其他模块调用）"""
    return _last_status.copy()
