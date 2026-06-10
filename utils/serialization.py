"""
公共序列化工具模块

统一处理 Decimal / datetime / date 等 pymssql 返回的特殊类型，
确保所有 json.dumps / jsonify 调用都能安全序列化。
"""
import json
from decimal import Decimal
from datetime import datetime, date


class SafeJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 Decimal / datetime / date 类型"""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        return super().default(obj)


def safe_json_dumps(obj, ensure_ascii=False, **kwargs):
    """安全的 json.dumps，自动处理 Decimal/datetime/date"""
    return json.dumps(obj, ensure_ascii=ensure_ascii, cls=SafeJSONEncoder, **kwargs)


def safe_serialize(obj):
    """
    递归安全的 JSON 序列化，将特殊类型转为 JSON 兼容类型。
    用于 jsonify 前的数据预处理，或需要逐字段控制的场景。
    """
    if obj is None:
        return None
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, date):
        return obj.strftime("%Y-%m-%d")
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    # 兜底：尝试序列化，失败则转字符串
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def convert_row_types(row: dict) -> dict:
    """
    转换单行 pymssql 查询结果中的特殊类型（原地修改）。
    Decimal → float, datetime → str, date → str
    """
    for key, value in row.items():
        if isinstance(value, Decimal):
            row[key] = float(value)
        elif isinstance(value, datetime):
            row[key] = value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(value, date):
            row[key] = value.strftime("%Y-%m-%d")
    return row


def convert_rows_types(rows: list) -> list:
    """批量转换 pymssql 查询结果行中的特殊类型"""
    for row in rows:
        convert_row_types(row)
    return rows
