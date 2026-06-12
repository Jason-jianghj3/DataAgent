"""
统一数据库执行器

将分散在 agent_query_engine / nl2api_service 中的 pymssql 连接和查询逻辑
统一到此处，消除重复代码，统一类型转换和连接管理。
"""
import os
import time
import re
import threading
from utils.logger import logger
from utils.serialization import convert_rows_types


# 默认最大返回行数
MAX_ROWS = 200

# 连接池：按 connection_name 缓存连接，线程安全
_connection_pool: dict = {}
_pool_lock = threading.Lock()
_pool_max_idle = 300  # 连接最大空闲秒数


def _is_mysql_connection(connection_name: str) -> bool:
    """判断连接是否为MySQL"""
    if connection_name == 'DW':
        return os.getenv('DB_DW_TYPE', 'mysql').lower() == 'mysql'
    return False


def _get_connection_config(connection_name: str) -> dict:
    """获取数据库连接配置，优先从 db_config 模块读取，降级到环境变量"""
    from utils.db_config import get_db_config
    cfg = get_db_config(connection_name)
    if cfg:
        return cfg.to_pymssql_kwargs()

    # 降级：从环境变量读取
    prefix = f"DB_{connection_name.upper()}_"
    return {
        'host': os.getenv(f"{prefix}HOST", 'localhost'),
        'port': int(os.getenv(f"{prefix}PORT", '1433')),
        'user': os.getenv(f"{prefix}USER", 'readonly'),
        'password': os.getenv(f"{prefix}PASSWORD", ''),
        'database': os.getenv(f"{prefix}DATABASE", connection_name.upper()),
    }


def _get_pooled_connection(connection_name: str, charset: str = 'utf8',
                           timeout: int = 60, login_timeout: int = 15):
    """从连接池获取 pymssql 连接，如果池中连接已失效则重建"""
    import pymssql

    pool_key = f"mssql_{connection_name}"
    with _pool_lock:
        if pool_key in _connection_pool:
            cached = _connection_pool[pool_key]
            conn = cached['conn']
            created_at = cached['created_at']
            # 检查连接是否过期或已关闭
            if time.time() - created_at < _pool_max_idle:
                try:
                    # 简单心跳检测
                    cursor = conn.cursor()
                    cursor.execute('SELECT 1')
                    cursor.fetchone()
                    logger.debug(f"[DBExecutor] 复用连接池连接: {connection_name}")
                    return conn
                except Exception:
                    logger.debug(f"[DBExecutor] 连接池连接已失效，重建: {connection_name}")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    del _connection_pool[pool_key]
            else:
                try:
                    conn.close()
                except Exception:
                    pass
                del _connection_pool[pool_key]

    # 新建连接
    cfg = _get_connection_config(connection_name)
    conn = pymssql.connect(
        server=cfg.get('server') or cfg.get('host', 'localhost'),
        port=cfg.get('port', 1433),
        user=cfg.get('user', ''),
        password=cfg.get('password', ''),
        database=cfg.get('database', ''),
        charset=charset,
        timeout=timeout,
        login_timeout=login_timeout,
    )
    with _pool_lock:
        _connection_pool[pool_key] = {'conn': conn, 'created_at': time.time()}
    logger.debug(f"[DBExecutor] 新建连接池连接: {connection_name}")
    return conn


def _get_pooled_mysql_connection(connection_name: str, timeout: int = 60):
    """从连接池获取 pymysql 连接"""
    import pymysql

    pool_key = f"mysql_{connection_name}"
    with _pool_lock:
        if pool_key in _connection_pool:
            cached = _connection_pool[pool_key]
            conn = cached['conn']
            created_at = cached['created_at']
            if time.time() - created_at < _pool_max_idle:
                try:
                    conn.ping(reconnect=True)
                    logger.debug(f"[DBExecutor] 复用MySQL连接池连接: {connection_name}")
                    return conn
                except Exception:
                    logger.debug(f"[DBExecutor] MySQL连接池连接已失效，重建: {connection_name}")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    del _connection_pool[pool_key]
            else:
                try:
                    conn.close()
                except Exception:
                    pass
                del _connection_pool[pool_key]

    cfg = _get_connection_config(connection_name)
    conn = pymysql.connect(
        host=cfg.get('server') or cfg.get('host', 'localhost'),
        port=cfg.get('port', 3306),
        user=cfg.get('user', ''),
        password=cfg.get('password', ''),
        database=cfg.get('database', ''),
        charset='utf8mb4',
        connect_timeout=timeout,
        read_timeout=timeout,
    )
    with _pool_lock:
        _connection_pool[pool_key] = {'conn': conn, 'created_at': time.time()}
    logger.debug(f"[DBExecutor] 新建MySQL连接池连接: {connection_name}")
    return conn


def close_all_connections():
    """关闭连接池中所有连接（用于应用关闭时清理）"""
    with _pool_lock:
        for key, cached in _connection_pool.items():
            try:
                cached['conn'].close()
            except Exception:
                pass
        _connection_pool.clear()
    logger.info("[DBExecutor] 连接池已清空")


def execute_query(sql: str, connection_name: str = 'EAM',
                  max_rows: int = MAX_ROWS, timeout: int = 60,
                  charset: str = 'utf8') -> dict:
    """
    执行 SQL 查询并返回结果字典。

    返回格式:
        成功: {"success": True, "data": [...], "total_rows": int, "truncated": bool, "elapsed": float}
        失败: {"success": False, "error": str}
    """
    try:
        if _is_mysql_connection(connection_name):
            return _execute_mysql_query(sql, connection_name, max_rows, timeout)

        conn = _get_pooled_connection(connection_name, charset=charset, timeout=timeout)

        t_start = time.time()
        cursor = conn.cursor(as_dict=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.time() - t_start

        # 类型转换
        truncated = rows[:max_rows]
        convert_rows_types(truncated)

        logger.info(
            f"[DBExecutor] SQL执行成功 | 连接={connection_name} | "
            f"行数={len(rows)} | 耗时={elapsed:.2f}s")

        return {
            "success": True,
            "data": truncated,
            "total_rows": len(rows),
            "truncated": len(rows) > max_rows,
            "elapsed": round(elapsed, 2)
        }

    except Exception as e:
        logger.error(f"[DBExecutor] SQL执行失败: {e}")
        return {"success": False, "error": f"执行失败: {str(e)[:200]}"}


def _execute_mysql_query(sql: str, connection_name: str = 'DW',
                         max_rows: int = MAX_ROWS, timeout: int = 60) -> dict:
    """执行 MySQL 查询（用于DW数仓库）"""
    try:
        import pymysql
        conn = _get_pooled_mysql_connection(connection_name, timeout=timeout)

        t_start = time.time()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.time() - t_start

        # 类型转换
        truncated = rows[:max_rows]
        convert_rows_types(truncated)

        logger.info(
            f"[DBExecutor] MySQL查询成功 | 连接={connection_name} | "
            f"行数={len(rows)} | 耗时={elapsed:.2f}s")

        return {
            "success": True,
            "data": truncated,
            "total_rows": len(rows),
            "truncated": len(rows) > max_rows,
            "elapsed": round(elapsed, 2)
        }

    except Exception as e:
        logger.error(f"[DBExecutor] MySQL查询失败: {e}")
        return {"success": False, "error": f"执行失败: {str(e)[:200]}"}


def execute_query_raw(sql: str, connection_name: str = 'EAM',
                      timeout: int = 60) -> list:
    """
    执行 SQL 查询，直接返回行列表（用于不需要统计信息的场景）。

    返回: List[Dict] 或空列表
    """
    try:
        if _is_mysql_connection(connection_name):
            return _execute_mysql_query_raw(sql, connection_name, timeout)

        conn = _get_pooled_connection(connection_name, charset='UTF-8',
                                       timeout=timeout, login_timeout=15)

        cursor = conn.cursor(as_dict=True)
        cursor.execute(sql)
        rows = cursor.fetchall()

        # 类型转换
        convert_rows_types(rows)

        return rows

    except Exception as e:
        logger.error(f"[DBExecutor] SQL执行失败: {e}")
        return []


def _execute_mysql_query_raw(sql: str, connection_name: str = 'DW',
                             timeout: int = 60) -> list:
    """执行 MySQL 查询，直接返回行列表（用于不需要统计信息的场景）"""
    try:
        import pymysql
        conn = _get_pooled_mysql_connection(connection_name, timeout=timeout)

        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(sql)
        rows = cursor.fetchall()

        # 类型转换
        convert_rows_types(rows)

        return rows

    except Exception as e:
        logger.error(f"[DBExecutor] MySQL查询失败: {e}")
        return []


def validate_sql_safety(sql: str) -> tuple:
    """
    SQL 安全校验（轻量级，用于 db_executor 层面的基本防护）。
    返回 (is_safe, error_message)
    """
    # 移除注释
    sql_cleaned = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql_cleaned = re.sub(r'/\*.*?\*/', '', sql_cleaned, flags=re.DOTALL)
    sql_stripped = sql_cleaned.strip().upper()

    # 只允许 SELECT / WITH
    if not any(sql_stripped.startswith(s) for s in ('SELECT', 'WITH')):
        return False, "安全限制: 只允许SELECT/CTE查询"

    # 禁止关键字
    forbidden = ['INSERT ', 'UPDATE ', 'DELETE ', 'DROP ', 'CREATE ',
                 'ALTER ', 'TRUNCATE ', 'EXEC ', 'EXECUTE ']
    for kw in forbidden:
        if kw in sql_stripped:
            return False, f"安全限制: 不允许{kw.strip()}操作"

    return True, None
