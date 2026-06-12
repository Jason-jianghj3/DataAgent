"""
通用报表智能解读与口语播报 - Web API 服务 v2.0

支持任意类型报表的智能摘要生成和语音播报
v2.1: 新增流式输出(SSE)、向量语义匹配、对话记忆
"""
import json
import uuid
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils.logger import logger
from utils.serialization import safe_json_dumps

# 创建Flask应用
app = Flask(__name__)

# 禁用JSON排序，保持字段原始顺序
app.config['JSON_SORT_KEYS'] = False


# ==================== FMCS 辅助函数 ====================


def _parse_fmcs_time_range(query: str, start_date: str, end_date: str, days: int, now) -> tuple:
    """
    解析FMCS查询的时间范围 - 支持自然语言时间表达
    """
    from datetime import timedelta
    import re

    fmt = "%Y-%m-%d %H:%M:%S"

    # 如果已经指定了完整的时间范围，直接使用
    if start_date and end_date:
        return start_date, end_date

    query_lower = query.lower() if query else ""

    # === 具体日期范围：X到Y / X至Y ===
    range_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s*[到至~]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})', query)
    if range_match:
        sd = range_match.group(1).replace('/', '-')
        ed = range_match.group(2).replace('/', '-')
        return f"{sd} 00:00:00", f"{ed} 23:59:59"

    # === 具体单日期：2024年6月15日 / 2024-06-15 ===
    single_date_match = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?', query)
    if single_date_match:
        y, m, d = int(single_date_match.group(1)), int(single_date_match.group(2)), int(single_date_match.group(3))
        target = now.replace(year=y, month=m, day=d, hour=0, minute=0, second=0)
        return target.strftime(fmt), target.replace(hour=23, minute=59, second=59).strftime(fmt)

    date_only_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', query)
    if date_only_match and not range_match:
        d = date_only_match.group(1).replace('/', '-')
        return f"{d} 00:00:00", f"{d} 23:59:59"

    # === 相对日期关键词 ===
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if '前天' in query or '前日' in query:
        target = today - timedelta(days=2)
        return target.strftime(fmt), target.replace(hour=23, minute=59, second=59).strftime(fmt)

    if '昨天' in query or '昨日' in query:
        target = today - timedelta(days=1)
        return target.strftime(fmt), target.replace(hour=23, minute=59, second=59).strftime(fmt)

    if '今天' in query or '今日' in query or '当天' in query:
        return today.strftime(fmt), now.strftime(fmt)

    if '上周' in query or '上星期' in query:
        days_since_monday = now.weekday()
        last_monday = today - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6)
        return last_monday.strftime(fmt), last_sunday.replace(hour=23, minute=59, second=59).strftime(fmt)

    if '本周' in query or '这周' in query or '这星期' in query:
        days_since_monday = now.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        return this_monday.strftime(fmt), now.strftime(fmt)

    if '上月' in query or '上个月' in query:
        first_of_this = today.replace(day=1)
        last_of_prev = first_of_this - timedelta(days=1)
        first_of_prev = last_of_prev.replace(day=1)
        return first_of_prev.strftime(fmt), last_of_prev.replace(hour=23, minute=59, second=59).strftime(fmt)

    if '本月' in query or '这个月' in query:
        first_of_month = today.replace(day=1)
        return first_of_month.strftime(fmt), now.strftime(fmt)

    # === N天前 ===
    n_days_ago = re.search(r'(\d+)\s*天前', query)
    if n_days_ago:
        n = int(n_days_ago.group(1))
        target = today - timedelta(days=n)
        return target.strftime(fmt), target.replace(hour=23, minute=59, second=59).strftime(fmt)

    # === 最近N天 / 近N天 ===
    recent_days = re.search(r'(?:最近|近)\s*(\d+)\s*天', query)
    if recent_days:
        n = int(recent_days.group(1))
        start = now - timedelta(days=n)
        return start.strftime(fmt), now.strftime(fmt)

    # === 默认：最近N天 ===
    if not start_date and not end_date:
        start = now - timedelta(days=days)
        return start.strftime(fmt), now.strftime(fmt)

    # 补全缺失的参数
    if not end_date:
        end_date = now.strftime(fmt)
    if not start_date:
        start_date = (now - timedelta(days=days)).strftime(fmt)

    return start_date, end_date


# ==================== FMCS设备数据查询接口 ====================


@app.route("/api/fmcs/query", methods=["POST", "GET"])
def fmcs_query():
    """
    FMCS/SCADA 多场景智能查询接口

    支持分析类型：
      - raw: 原始时序数据查看
      - threshold: 阈值超标分析（超标时段、持续时长）
      - comparison: 多设备对比
      - trend: 趋势分析（变化率、异常点）
      - aggregation: 聚合统计（按小时/天 avg/max/min）

    调用方式:
    POST /api/fmcs/query  Body: {
        "query": "纯化间温度超过25度持续多久",
        "days": 3,
        "analysis_type": "threshold",   // 可选，默认自动推断
        "threshold": 25,                // threshold分析时必传
        "threshold_operator": ">",      // 可选，默认">"
        "agg_interval": "hour"          // aggregation分析时可选，"hour"|"day"
    }
    """
    try:
        if request.method == "GET":
            query = (request.args.get("q") or request.args.get("query") or "").strip()
            days = int(request.args.get("days", "1"))
            tagname = (request.args.get("tagname") or "").strip()
            start_date = (request.args.get("start_date") or "").strip()
            end_date = (request.args.get("end_date") or "").strip()
            analysis_type = (request.args.get("analysis_type") or "").strip()
            threshold = request.args.get("threshold", type=float)
            threshold_operator = (request.args.get("threshold_operator") or ">").strip()
            agg_interval = (request.args.get("agg_interval") or "hour").strip()
        else:
            data = request.get_json(silent=True) or {}
            query = (data.get("query") or "").strip()
            days = int(data.get("days", 1))
            tagname = (data.get("tagname") or "").strip()
            start_date = (data.get("start_date") or "").strip()
            end_date = (data.get("end_date") or "").strip()
            analysis_type = (data.get("analysis_type") or "").strip()
            threshold = data.get("threshold")
            threshold_operator = (data.get("threshold_operator") or ">").strip()
            agg_interval = (data.get("agg_interval") or "hour").strip()

        if not query and not tagname:
            return jsonify({"success": False, "error": "请提供查询内容(query)或设备名(tagname)"}), 400

        from utils.fmcs_registry import get_fmcs_registry
        from utils.db_config import HISTORIAN_CONFIG
        from services.scada_analyzer import SCADAAnalyzer
        from datetime import datetime, timedelta

        registry = get_fmcs_registry()
        analyzer = SCADAAnalyzer(HISTORIAN_CONFIG)

        # 解析查询意图
        parsed = analyzer.parse_scada_query(query)

        # 如果前端指定了 analysis_type，覆盖自动推断的结果
        if analysis_type:
            parsed["analysis_type"] = analysis_type
        if threshold is not None:
            parsed["threshold"] = threshold
            parsed["analysis_type"] = "threshold"
        if threshold_operator:
            parsed["threshold_operator"] = threshold_operator

        # 匹配设备
        tagnames = parsed.get("tagnames", [])
        device_info = parsed.get("device_info", {})

        if tagname and not tagnames:
            device = registry.get_device(tagname)
            if device:
                tagnames = [tagname]
                measure_info = registry.get_measure_type_info(device.get('measure_type', 'PV'))
                device_info[tagname] = {
                    "tagname": tagname,
                    "cn_desc": device.get('cn_desc', tagname),
                    "room_code": device.get('room_code', ''),
                    "room_name": device.get('room_name', ''),
                    "measure_type": device.get('measure_type', 'PV'),
                    "measure_label": measure_info.get('label', ''),
                    "unit": measure_info.get('unit', ''),
                    "color": measure_info.get('color', '#4fc3f7'),
                }

        if not tagnames:
            return jsonify({
                "success": False,
                "error": f"未找到匹配的设备数据点: {query}",
                "suggestion": "试试输入房间号(如A1S115)或设备描述(如纯化间温度)"
            })

        # 计算时间范围
        now = datetime.now()
        time_range = parsed.get("time_range", {})
        if not start_date:
            start_date = time_range.get("start_date", (now - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00"))
        if not end_date:
            end_date = time_range.get("end_date", now.strftime("%Y-%m-%d %H:%M:%S"))

        logger.info(f"[FMCS] 查询: '{query}', 分析类型: {parsed['analysis_type']}, "
                     f"设备: {tagnames}, 时间: {start_date} ~ {end_date}")

        # 查询Historian时序数据
        if not HISTORIAN_CONFIG:
            return jsonify({"success": False, "error": "Historian数据库未配置"}), 500

        resolution = parsed.get("resolution", 60000)
        multi_data = analyzer.fetch_timeseries(tagnames, start_date, end_date, resolution)

        if not multi_data or all(len(v) == 0 for v in multi_data.values()):
            return jsonify({
                "success": True,
                "query": query,
                "analysis_type": parsed["analysis_type"],
                "matched_devices": [{"tagname": tn, **device_info.get(tn, {})} for tn in tagnames],
                "start_date": start_date,
                "end_date": end_date,
                "data": {},
                "stats": {},
                "analysis_result": {},
                "chart_config": {},
                "measure_info": device_info.get(tagnames[0], {}) if tagnames else {},
            })

        # 根据分析类型执行不同分析
        atype = parsed["analysis_type"]
        analysis_result = {}
        chart_config = {}
        stats = {}

        if atype == "threshold":
            # 阈值超标分析
            thresh_val = parsed.get("threshold", 25.0)
            thresh_op = parsed.get("threshold_operator", ">")
            for tn in tagnames:
                data_points = multi_data.get(tn, [])
                result = analyzer.analyze_threshold(data_points, thresh_val, thresh_op)
                analysis_result[tn] = result
            chart_config = analyzer.build_chart_config(
                multi_data, device_info, analysis_type="threshold",
                threshold_info={"threshold": thresh_val, "operator": thresh_op}
            )
            # 统计汇总
            all_periods = []
            for tn, r in analysis_result.items():
                all_periods.extend(r.get("periods", []))
            total_duration = sum(p["duration_min"] for p in all_periods)
            max_duration = max((p["duration_min"] for p in all_periods), default=0)
            stats = {
                "exceed_count": len(all_periods),
                "total_duration_min": round(total_duration, 1),
                "max_duration_min": round(max_duration, 1),
                "threshold": thresh_val,
                "threshold_operator": thresh_op,
            }

        elif atype == "comparison":
            # 多设备对比
            analysis_result = analyzer.analyze_comparison(multi_data, device_info)
            chart_config = analyzer.build_chart_config(multi_data, device_info, analysis_type="comparison")
            stats = analysis_result.get("devices", {})

        elif atype == "trend":
            # 趋势分析
            for tn in tagnames:
                data_points = multi_data.get(tn, [])
                result = analyzer.analyze_trend(data_points)
                analysis_result[tn] = result
            chart_config = analyzer.build_chart_config(multi_data, device_info, analysis_type="trend")

        elif atype == "aggregation":
            # 聚合统计
            for tn in tagnames:
                data_points = multi_data.get(tn, [])
                result = analyzer.analyze_aggregation(data_points, interval=agg_interval)
                analysis_result[tn] = result
            chart_config = analyzer.build_chart_config(
                multi_data, device_info, analysis_type="aggregation",
                aggregation_data=analysis_result
            )

        elif atype == "anomaly":
            # 异常检测
            anomaly_data = {}
            for tn in tagnames:
                data_points = multi_data.get(tn, [])
                result = analyzer.analyze_anomaly(data_points)
                analysis_result[tn] = result
                anomaly_data[tn] = result
            chart_config = analyzer.build_chart_config(
                multi_data, device_info, analysis_type="anomaly",
                anomaly_data=anomaly_data
            )

        else:
            # raw: 原始时序查看
            for tn in tagnames:
                values = [p["value"] for p in multi_data.get(tn, [])]
                if values:
                    stats[tn] = {
                        "current": round(values[-1], 2),
                        "max": round(max(values), 2),
                        "min": round(min(values), 2),
                        "avg": round(sum(values) / len(values), 2),
                        "count": len(values),
                    }
            chart_config = analyzer.build_chart_config(multi_data, device_info, analysis_type="raw")

        # 构建返回结果
        best_tagname = tagnames[0]
        best_device_info = device_info.get(best_tagname, {})

        return jsonify({
            "success": True,
            "query": query,
            "analysis_type": atype,
            "matched_devices": [{"tagname": tn, **device_info.get(tn, {})} for tn in tagnames],
            "start_date": start_date,
            "end_date": end_date,
            "data": multi_data,
            "stats": stats,
            "analysis_result": analysis_result,
            "chart_config": chart_config,
            "measure_info": best_device_info,
            "threshold": parsed.get("threshold"),
            "threshold_operator": parsed.get("threshold_operator", ">"),
            "agg_interval": agg_interval,
        })

    except Exception as e:
        logger.error(f"[FMCS] 查询异常: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== NL2API 服务 v1.0 ====================

try:
    from solutions.core.nl2api_service import create_nl2api_blueprint
    nl2api_bp = create_nl2api_blueprint()
    app.register_blueprint(nl2api_bp)
    logger.info("[NL2API] v1.0 服务已注册 | 端点: /api/nl2api/*")
except Exception as e:
    logger.warning(f"[NL2API] 注册失败: {e}")


# ==================== 流式对话 API (v2.1 新增) ====================

from services.llm_service import LLMService, SYSTEM_PROMPT
from services.conversation_manager import get_conversation_manager

_llm_service = None
_vector_store = None


def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        try:
            from services.vector_store import ReportVectorStore
            _vector_store = ReportVectorStore()
            config_path = os.path.join(os.path.dirname(__file__), "report_config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            reports = config_data.get("reports", {})
            _vector_store.build_index(reports)
            logger.info(f"[VectorStore] 向量索引构建完成，共{_vector_store.report_entries}个数据集")
        except Exception as e:
            logger.warning(f"[VectorStore] 向量索引构建失败: {e}，将降级为关键词匹配")
            _vector_store = None
    return _vector_store


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """
    流式对话接口 (SSE)

    支持流式输出、对话记忆、向量语义匹配

    调用方式:
    POST /api/chat/stream
    Body: {
        "query": "CI部门工单情况",
        "session_id": "可选，不传则自动生成"
    }

    返回: Server-Sent Events 流
    """
    params = request.get_json(silent=True) or {}
    query = (params.get("query") or "").strip()
    session_id = params.get("session_id", "")

    if not query:
        return jsonify({"code": 400, "msg": "query不能为空"}), 400

    conv_mgr = get_conversation_manager()
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    session = conv_mgr.get_or_create(session_id)

    resolved_query = session.resolve_reference(query)

    vs = get_vector_store()
    matched_datasets = []
    if vs:
        try:
            matched_datasets = vs.search(resolved_query, top_k=3)
        except Exception as e:
            logger.warning(f"[Chat Stream] 向量检索失败: {e}")

    def generate():
        full_text = ""
        try:
            yield f"data: {safe_json_dumps({'type': 'session', 'session_id': session_id}, ensure_ascii=False)}\n\n"

            if matched_datasets:
                dataset_info = matched_datasets[0]
                yield f"data: {safe_json_dumps({'type': 'dataset', 'data': dataset_info}, ensure_ascii=False)}\n\n"

            messages = session.build_context_messages(
                user_query=resolved_query,
                system_prompt=SYSTEM_PROMPT,
            )

            llm = get_llm_service()
            for chunk in llm.chat_stream(messages):
                full_text += chunk
                yield f"data: {safe_json_dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"

            session.add_exchange(resolved_query, full_text)
            if matched_datasets:
                ds = matched_datasets[0]
                session.update_business_context(
                    report=ds.get("report_name", ""),
                    dataset=ds.get("dataset_name", ""),
                    params={},
                    result_summary=full_text[:200],
                )

            yield f"data: {safe_json_dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"[Chat Stream] 流式生成异常: {e}")
            yield f"data: {safe_json_dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ==================== Agent 查询 API (v2.1 新增) ====================


@app.route("/api/agent/query", methods=["POST"])
def agent_query():
    """
    Agent智能查询接口（SSE流式）

    支持多步推理：LLM自主决定执行什么SQL，前一轮结果作为后一轮输入

    POST /api/agent/query
    Body: {"query": "4月哪天处理的工单最多？其中哪个部门处理的最多？"}
    """
    params = request.get_json(silent=True) or {}
    query = (params.get("query") or "").strip()
    context = params.get("context", {})

    if not query:
        return jsonify({"code": 400, "msg": "query不能为空"}), 400

    try:
        from solutions.core.agent_query_engine import get_agent_engine
        engine = get_agent_engine()
    except Exception as e:
        logger.error(f"[Agent] 引擎初始化失败: {e}")
        return jsonify({"code": 503, "msg": f"Agent引擎不可用: {e}"}), 503

    def generate():
        try:
            for event in engine.query(query, context=context):
                yield f"data: {safe_json_dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[Agent] 查询异常: {e}")
            yield f"data: {safe_json_dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ==================== 静态文件服务 ====================


@app.route("/", methods=["GET"])
def serve_index():
    """前端主页 - NL2API智能对话界面(统一入口)"""
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), 'chat.html')


@app.route("/chat", methods=["GET"])
def serve_chat():
    """NL2API 智能对话界面"""
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), 'chat.html')


# ==================== 启动入口 ====================

if __name__ == "__main__":
    print("""
====================================================
   通用报表智能解读与口语播报 v3.0

   支持: 销售 | 生产 | 质量 | 财务 | 库存
   接口: http://localhost:5000/api/agent/query
====================================================
    """)

    # 启动ETL调度器（每小时自动刷新数仓宽表）
    try:
        from etl.etl_scheduler import start_scheduler
        start_scheduler(app)
        logger.info("ETL调度器已启动")
    except Exception as e:
        logger.warning(f"ETL调度器启动失败（不影响主服务）: {e}")

    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=False
    )
