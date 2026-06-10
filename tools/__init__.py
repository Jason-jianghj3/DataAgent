"""
LangChain BaseTool 工具定义 + OutputParser

将 agent_query_engine 中的三个工具（query_by_template / execute_sql / query_scada）
重构为 langchain-core BaseTool 子类，统一工具接口和参数校验。
同时提供 JsonOutputParser + 健壮 fallback 的统一 JSON 解析器。
"""
import json
import re
from typing import Optional, Type, Dict, List, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import JsonOutputParser


# ============================================================
#  Input Schemas
# ============================================================

class QueryByTemplateInput(BaseModel):
    """模板查询工具的输入参数"""
    template_name: str = Field(
        description="报表模板名称，如：工单统计报表、部门工单对比报表"
    )
    start_time: Optional[str] = Field(
        default=None,
        description="开始时间，格式 YYYY-MM-DD"
    )
    end_time: Optional[str] = Field(
        default=None,
        description="结束时间，格式 YYYY-MM-DD"
    )
    dept: Optional[str] = Field(
        default=None,
        description="部门代码，如 CI/DS/EHS/FF/FM/LG/OM/PD/PM/QA/QC/TM/VM"
    )
    flow_type: Optional[str] = Field(
        default=None,
        description="工单流程类型"
    )
    flow_category: Optional[str] = Field(
        default=None,
        description="工单流程分类"
    )


class ExecuteSQLInput(BaseModel):
    """SQL 执行工具的输入参数"""
    sql: str = Field(
        description="要执行的 SQL 查询语句"
    )
    connection_name: str = Field(
        default="EAM",
        description="数据库连接名，可选: EAM / WMS_PROD"
    )


class QuerySCADAInput(BaseModel):
    """SCADA 查询工具的输入参数"""
    devices: str = Field(
        description="设备名称，多个设备用逗号分隔"
    )
    time_range: Optional[str] = Field(
        default="最近1天",
        description="时间范围，如: 最近1天、最近7天、本周、上月"
    )
    analysis_type: Optional[str] = Field(
        default="raw",
        description="分析类型: raw(原始数据) / threshold(阈值分析) / comparison(对比分析) / trend(趋势分析)"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="阈值（用于 threshold 分析）"
    )
    threshold_operator: Optional[str] = Field(
        default=">",
        description="阈值比较运算符: > / < / >= / <="
    )


# ============================================================
#  Tool Definitions
# ============================================================

class QueryByTemplateTool(BaseTool):
    """使用帆软报表预定义 SQL 模板查询数据"""
    name: str = "query_by_template"
    description: str = (
        "使用预定义的报表SQL模板查询数据。"
        "当用户查询涉及工单统计、部门对比等常见报表场景时使用。"
        "需要指定template_name，可选指定时间范围和部门。"
    )
    args_schema: Type[BaseModel] = QueryByTemplateInput

    # 运行时注入的引擎引用
    engine: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("同步执行不支持，请使用 _arun 或直接调用引擎方法")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError("异步执行暂不支持")


class ExecuteSQLTool(BaseTool):
    """自由编写 SQL 查询数据库"""
    name: str = "execute_sql"
    description: str = (
        "直接执行SQL查询语句获取数据。"
        "当预定义模板无法满足需求时，可自由编写SQL查询。"
        "需要指定sql和connection_name(EAM/WMS_PROD)。"
        "仅允许SELECT查询，禁止DML/DDL操作。"
    )
    args_schema: Type[BaseModel] = ExecuteSQLInput

    engine: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("同步执行不支持，请使用 _arun 或直接调用引擎方法")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError("异步执行暂不支持")


class QuerySCADATool(BaseTool):
    """查询 SCADA/Historian 实时监控数据"""
    name: str = "query_scada"
    description: str = (
        "查询SCADA/Historian实时监控数据。"
        "支持温度、湿度、压差、压力、流量等设备数据点的时序查询。"
        "支持阈值分析、对比分析、趋势分析等多种分析模式。"
    )
    args_schema: Type[BaseModel] = QuerySCADAInput

    engine: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("同步执行不支持，请使用 _arun 或直接调用引擎方法")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError("异步执行暂不支持")


# ============================================================
#  工具注册表
# ============================================================

def get_tool_definitions() -> List[Dict]:
    """
    获取 OpenAI Function Calling 格式的工具定义列表。
    兼容现有 agent_query_engine 的 TOOL_DEFINITIONS 格式。
    """
    tools = [QueryByTemplateTool(), ExecuteSQLTool(), QuerySCADATool()]
    return [tool.to_function_schema() for tool in tools]


def get_tools_dict() -> Dict[str, BaseTool]:
    """获取工具名到工具实例的映射"""
    tools = [QueryByTemplateTool(), ExecuteSQLTool(), QuerySCADATool()]
    return {tool.name: tool for tool in tools}


# ============================================================
#  OutputParser - 统一 JSON 解析
# ============================================================

# 全局 JsonOutputParser 实例（无需指定 pydantic_model，用于通用 JSON 解析）
_json_parser = JsonOutputParser()


def get_format_instructions() -> str:
    """获取 JSON 输出格式指导，可插入 LLM prompt 中"""
    return _json_parser.get_format_instructions()


def parse_json_response(text: str, pydantic_model: Type[BaseModel] = None) -> Optional[Dict]:
    """
    统一的 LLM JSON 响应解析器。

    解析策略（按优先级）：
    1. LangChain JsonOutputParser（标准解析）
    2. Markdown 代码块提取（中文 LLM 常见）
    3. 括号计数提取（处理嵌套 JSON）
    4. 直接解析整个文本

    Args:
        text: LLM 返回的文本
        pydantic_model: 可选的 Pydantic 模型，用于结构化解析和校验

    Returns:
        解析后的 dict，失败返回 None
    """
    if not text:
        return None

    # 策略1: LangChain JsonOutputParser
    try:
        parser = JsonOutputParser(pydantic_object=pydantic_model) if pydantic_model else _json_parser
        result = parser.parse(text)
        if result is not None:
            return result
    except Exception:
        pass

    # 策略2: 提取 markdown 代码块中的 JSON（中文 LLM 常见）
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 策略3: 括号计数提取第一个 {...}（处理嵌套）
    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
        if end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx])
            except (json.JSONDecodeError, ValueError):
                pass

    # 策略4: 直接解析整个文本
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    return None
