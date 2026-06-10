"""
统一数据库执行器

将分散在 agent_query_engine / nl2api_service 中的 pymssql 连接和查询逻辑
统一到此处，消除重复代码，统一类型转换和连接管理。
"""
import os
import time
import re
from utils.logger import logger
from utils.serialization import convert_rows_types


# 默认最大返回行数
MAX_ROWS = 200


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
        import pymssql

        cfg = _get_connection_config(connection_name)

        conn = pymssql.connect(
            server=cfg.get('server') or cfg.get('host', 'localhost'),
            port=cfg.get('port', 1433),
            user=cfg.get('user', ''),
            password=cfg.get('password', ''),
            database=cfg.get('database', ''),
            charset=charset,
            timeout=timeout
        )

        t_start = time.time()
        cursor = conn.cursor(as_dict=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.time() - t_start
        conn.close()

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


def execute_query_raw(sql: str, connection_name: str = 'EAM',
                      timeout: int = 60) -> list:
    """
    执行 SQL 查询，直接返回行列表（用于不需要统计信息的场景）。

    返回: List[Dict] 或空列表
    """
    try:
        import pymssql

        cfg = _get_connection_config(connection_name)

        conn = pymssql.connect(
            server=cfg.get('server') or cfg.get('host', 'localhost'),
            port=cfg.get('port', 1433),
            user=cfg.get('user', ''),
            password=cfg.get('password', ''),
            database=cfg.get('database', ''),
            charset='UTF-8',
            timeout=timeout,
            login_timeout=15
        )

        cursor = conn.cursor(as_dict=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        # 类型转换
        convert_rows_types(rows)

        return rows

    except Exception as e:
        logger.error(f"[DBExecutor] SQL执行失败: {e}")
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
