"""
LLM数据总结服务模块（通用版）
调用大语言模型API将任意类型报表数据生成口语化摘要
支持：销售报表、生产报表、财务报表、质量报表、库存报表等任意类型
支持流式输出、对话上下文
"""
import json
import time
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LLMConfig
from utils.logger import logger


@dataclass
class SummaryResult:
    """总结结果封装"""
    success: bool = False
    summary_text: str = ""
    raw_response: str = ""
    error: str = ""
    token_usage: dict = None
    source: str = "llm"  # "llm" | "template" — 标注数据来源
    detected_type: str = ""  # LLM自动识别的报表类型（如"销售月报"、"生产日报"）


# ============================================================
#  核心Prompt：通用报表智能解读引擎
# ============================================================
SYSTEM_PROMPT = """你是一位资深的商业数据分析师，擅长解读各类企业业务报表。你的任务是将任意类型的业务报表数据，转换成一段清晰、简洁的口语化汇报。

## 核心原则
**精准 > 完美**。你的输出是给人听的语音播报，听众需要的是数据事实，不是官场话术。

## 绝对禁止（违反任何一条即为不合格输出）
- ❌ 禁止使用："各位领导"、"我来为大家汇报"、"下面我来说一下"、"总的来说"、"总而言之"等开场白/过渡词/结束语
- ❌ 禁止使用："建议相关部门"、"建议进一步调查"、"建议采取措施"等空泛建议（除非数据中明确提到了具体措施）
- ❌ 禁止编造或推断数据中没有的数字、百分比、统计量（如"占总数的96.8%"这种原始数据里没有的计算结果）
- ❌ 禁止使用编号列表格式（1. 2. 3.）— 用段落自然衔接即可
- ❌ 禁止出现"以下是摘要"、"根据数据分析"、"从数据来看"等元叙述

## 输出要求
1. **字数**：100-250字之间，精炼为上
2. **格式**：纯文本段落，不要用markdown、列表或表格（因为要转语音朗读）
3. **语气**：直接、干练，像同事之间快速同步信息，不要像领导讲话
4. **结构**：
   - 开头：一句话概括整体状况（直接说结论，不要寒暄）
   - 中间：按重要性列出关键数据和异常点（只报原始数据中有的数字和事实）
   - 结尾：一句收束即可，不要长篇总结和建议

## 数据解读规则
- **只报事实，不发表观点** — 原始数据说了什么你就报什么
- 同比/环比变化超过10%的 → 明确指出涨跌幅度
- 接近目标线或警戒线的 → 提示需关注
- 表现突出的 → 给予正面确认
- 所有正常无异常 → 一句话带过，不需要展开
- **数字必须来自原始数据，禁止自行计算百分比或汇总数**

## 示例输出风格（这就是你要模仿的标准）

### 仓储/WMS类：
"当前仓库物资物料区库存总量约3200件，与ERP账面数量基本一致，差异在允许范围内。成品区实际库存略低于系统记录，差异约15件需核实。呆滞物料方面，B、C类物资合计占比18%，其中包材类呆滞量最高达450件，超过180天未出库，建议优先处理。近效期物料共3批涉及两个SKU，最近一批距效期不足90天，需加快周转。"

### 质量检验类：
"本周验收记录显示物料质量整体良好，大部分物资均通过验收。需注意三点：高效过滤器连续多日批量接收共25批，需关注库存情况；液氮在4月7日和8日出现两个不同批次的合格记录，请确认是否有误；部分物资验收备注为空，建议检查是否遗漏。质量状况稳定，细节方面仍需留意。"

### 销售类：
"3月份全国销售额完成1250万，达成季度目标104%，同比增长8%。华东区最佳贡献420万超目标15%。西南区订单量较上月下滑12%，主要受两个大客户延期影响。新品X系列首月出货200万，一季度开局顺利。"

### 生产类：
"本周计划产量5000批实际完成4850批，达成率97%。A线良率98.2%保持稳定；B线周三设备停机导致缺口200批已恢复。关键包材供应商交货延迟一天，采购跟进中。生产运行基本可控。"
"""


class LLMService:
    """
    通用大语言模型服务
    
    支持多种OpenAI兼容API接口，
    可处理任意类型的报表数据生成口语化摘要。
    支持主备降级：主模型失败时自动切换到备用模型。
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client: Optional[OpenAI] = None
        self._fallback_client: Optional[OpenAI] = None
        self._using_fallback = False
        self._init_client()

    def _init_client(self):
        if not self.config.api_key:
            logger.warning("未配置LLM API Key，LLM功能不可用")
            return

        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=120,
            )
            logger.info(f"LLM客户端初始化成功 (model={self.config.model}, base={self.config.api_base})")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")

        if self.config.fallback_api_key and self.config.fallback_api_base:
            try:
                self._fallback_client = OpenAI(
                    api_key=self.config.fallback_api_key,
                    base_url=self.config.fallback_api_base,
                    timeout=60,
                )
                logger.info(f"LLM降级客户端初始化成功 (model={self.config.fallback_model})")
            except Exception as e:
                logger.warning(f"LLM降级客户端初始化失败: {e}")

    def _get_active_client(self) -> Optional[OpenAI]:
        if self._using_fallback and self._fallback_client:
            if self._should_try_primary() and self.client:
                return self.client
            return self._fallback_client
        return self.client

    def _get_active_model(self) -> str:
        if self._using_fallback and self.config.fallback_model:
            return self.config.fallback_model
        return self.config.model

    def _switch_to_fallback(self):
        if self._fallback_client and not self._using_fallback:
            logger.warning(f"LLM主模型不可用，切换到降级模型: {self.config.fallback_model}")
            self._using_fallback = True
            self._fallback_since = time.time()

    def _should_try_primary(self) -> bool:
        if not self._using_fallback:
            return True
        if not hasattr(self, '_fallback_since'):
            return False
        if time.time() - self._fallback_since > 300:
            logger.info("[LLM] 降级已超过5分钟，尝试切回主模型")
            self._using_fallback = False
            del self._fallback_since
            return True
        return False

    def chat(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        tools: List[Dict] = None,
        tool_choice: str = "auto",
    ) -> Dict:
        """
        通用Chat接口（支持Function Calling）
        
        Args:
            messages: 对话消息列表 [{"role": "system/user/assistant", "content": "..."}]
            model: 模型名称（默认使用配置中的模型）
            temperature: 温度参数
            max_tokens: 最大token数
            tools: OpenAI Function Calling工具定义列表
            tool_choice: 工具选择策略 ("auto"/"none"/"required"/{"type":"function","function":{"name":"xxx"}})
        
        Returns:
            Dict: {
                "content": str or None,
                "tool_calls": List[Dict] or None,
                "tool_call_id": str or None,
                "usage": Dict,
                "model": str,
                "raw_response": object
            }
        """
        if not self.client and not self._fallback_client:
            return {"error": "LLM客户端未初始化", "content": None}
        
        active_client = self._get_active_client()
        if not active_client:
            return {"error": "LLM客户端未初始化", "content": None}

        kwargs = {
            "model": model or self._get_active_model(),
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        
        try:
            response = active_client.chat.completions.create(**kwargs)
            
            choice = response.choices[0]
            message = choice.message
            
            result = {
                "content": message.content,
                "tool_calls": None,
                "tool_call_id": None,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "model": response.model,
                "finish_reason": choice.finish_reason,
                "raw_response": response,
            }
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function_name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                        "raw_arguments": tc.function.arguments,
                    }
                    for tc in message.tool_calls
                ]
            
            return result
        
        except Exception as e:
            logger.error(f"LLM chat调用失败(model={kwargs['model']}): {e}")
            if not self._using_fallback and self._fallback_client:
                self._switch_to_fallback()
                return self.chat(messages, model, temperature, max_tokens, tools, tool_choice)
            return {"error": str(e), "content": None}

    def generate_summary(
        self,
        report_data: str,
        report_type: str = "",           # 报表类型，如"销售月报"、"生产日报"；为空则让LLM自行判断
        report_title: str = "",           # 报表标题，帮助LLM理解上下文
        context: str = "",                # 补充上下文信息（如"这是Q1的数据"、"重点关注华北区"等）
        shift: str = "",                  # 兼容旧参数（班次信息），优先使用 context
        extra_context: str = "",          # 兼容旧参数
    ) -> SummaryResult:
        """
        生成报表数据的口语化摘要（通用版）
        
        Args:
            report_data: 报表数据文本（由数据源格式化后传入）
            report_type: 报表类型提示（可选，不传则LLM自动识别）
            report_title: 报表标题（可选，辅助理解）
            context: 补充上下文（如时间范围、关注重点等）
            shift: 兼容旧参数（班次）
            extra_context: 兼容旧参数
            
        Returns:
            SummaryResult 包含生成的摘要文本
        """
        # 兼容旧接口
        effective_context = context or extra_context
        if shift and not effective_context:
            effective_context = f"当前{shift}数据"
        if not report_type and shift:
            report_type = f"{shift}工作汇报"

        if not self.client:
            return self._generate_fallback_summary(report_data, report_type, effective_context)

        user_message = self._build_user_prompt(report_data, report_type, report_title, effective_context)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            summary_text = response.choices[0].message.content.strip()
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            logger.info(f"LLM摘要生成成功，token使用: {token_usage}")
            return SummaryResult(
                success=True,
                summary_text=summary_text,
                raw_response=summary_text,
                token_usage=token_usage,
                detected_type=report_type or "auto"
            )

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return self._generate_fallback_summary(report_data, report_type, effective_context, error=str(e))

    def generate_summary_stream(
        self,
        report_data: str,
        report_type: str = "",
        report_title: str = "",
        context: str = "",
        messages: List[Dict] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成报表数据的口语化摘要
        
        Args:
            report_data: 报表数据文本
            report_type: 报表类型提示
            report_title: 报表标题
            context: 补充上下文
            messages: 带上下文的对话历史（由ConversationManager构建），优先使用
            
        Yields:
            str: 逐个文本片段
        """
        if not self.client and not self._fallback_client:
            fallback = self._generate_fallback_summary(report_data, report_type, context)
            yield fallback.summary_text
            return

        active_client = self._get_active_client()
        if not active_client:
            fallback = self._generate_fallback_summary(report_data, report_type, context)
            yield fallback.summary_text
            return

        if messages:
            full_messages = messages
        else:
            user_message = self._build_user_prompt(report_data, report_type, report_title, context)
            full_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]

        try:
            stream = active_client.chat.completions.create(
                model=self._get_active_model(),
                messages=full_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM流式调用失败: {e}")
            if not self._using_fallback and self._fallback_client:
                self._switch_to_fallback()
                active_client = self._get_active_client()
                try:
                    stream = active_client.chat.completions.create(
                        model=self._get_active_model(),
                        messages=full_messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e2:
                    logger.error(f"LLM降级流式调用也失败: {e2}")
            fallback = self._generate_fallback_summary(report_data, report_type, context, error=str(e))
            yield fallback.summary_text

    def chat_stream(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> Generator[str, None, None]:
        """
        通用流式Chat接口
        
        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            
        Yields:
            str: 逐个文本片段
        """
        if not self.client and not self._fallback_client:
            yield "[错误] LLM客户端未初始化"
            return

        active_client = self._get_active_client()
        if not active_client:
            yield "[错误] LLM客户端未初始化"
            return

        kwargs = {
            "model": model or self._get_active_model(),
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": True,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            stream = active_client.chat.completions.create(**kwargs)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM chat_stream调用失败: {e}")
            if not self._using_fallback and self._fallback_client:
                self._switch_to_fallback()
                active_client = self._get_active_client()
                kwargs["model"] = self._get_active_model()
                try:
                    stream = active_client.chat.completions.create(**kwargs)
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e2:
                    logger.error(f"LLM降级chat_stream也失败: {e2}")
            yield f"[错误] {str(e)}"

    def _build_user_prompt(self, data_text: str, report_type: str, report_title: str, context: str) -> str:
        """构建用户消息prompt"""
        parts = []
        
        if report_type:
            parts.append(f"## 报表类型\n{report_type}")
        if report_title:
            parts.append(f"## 报表标题\n{report_title}")
        if context:
            parts.append(f"## 背景信息\n{context}")
            
        parts.append(f"## 原始数据\n{data_text}")
        parts.append("\n请根据以上数据生成一段清晰的口语化汇报：")

        return "\n\n".join(parts)

    def _generate_fallback_summary(
        self,
        data_text: str,
        report_type: str = "",
        context: str = "",
        error: str = ""
    ) -> SummaryResult:
        """
        降级方案：当LLM API不可用时，使用通用规则模板生成基础摘要
        
        适用于任意类型报表的关键词检测+模板填充方案
        """
        logger.info("使用降级模板生成摘要")

        # 通用关键词检测（覆盖多场景）
        warning_keywords = ["下降", "↓", "上升", "↑", "接近警戒", "关注", 
                           "异常", "未达标", "超额", "缺口", "延迟", "逾期",
                           "不合格", "超出", "低于", "高于"]
        positive_keywords = ["超额完成", "↑", "达标", "优秀", "创新高", "增长"]
        negative_keywords = ["下降", "↓", "亏损", "不合格", "异常", "延误"]

        has_warning = any(kw in data_text for kw in warning_keywords)
        has_positive = any(kw in data_text for kw in positive_keywords)
        has_negative = any(kw in data_text for kw in negative_keywords)

        # 构建基础模板（根据正负信号动态调整）
        type_label = report_type or "本组数据"
        
        if has_negative and not has_positive:
            overall = "存在一些需要关注的指标"
        elif has_positive and not has_negative:
            overall = "整体表现良好"
        elif has_warning:
            overall = "基本平稳，有部分指标需关注"
        else:
            overall = "运行平稳"

        summary_parts = [f"{type_label}，{overall}。"]

        # 从数据中提取关键信息行
        lines = data_text.split("\n")[1:]  # 跳过表头
        key_points = []
        
        for line in lines[:20]:
            if any(kw in line for kw in warning_keywords + positive_keywords[:2] + negative_keywords[:2]):
                # 截取每行的核心内容（去掉多余空格）
                clean_line = line.strip()
                if clean_line and len(clean_line) > 5:
                    # 尝试提取有意义的片段（不超过60字）
                    snippet = clean_line[:80]
                    if "|" in snippet:
                        cells = [c.strip() for c in snippet.split("|")]
                        if len(cells) >= 3:
                            # 取前几个非空字段作为要点
                            meaningful = [c for c in cells[:6] if c]
                            point = "，".join(meaningful[:4])
                            if point and len(point) > 4:
                                key_points.append(point)

        if key_points:
            summary_parts.append("重点关注：" + "；".join(key_points[:5]) + "。")
        else:
            summary_parts.append("各主要指标均在正常范围内。")

        result = "".join(summary_parts)

        return SummaryResult(
            success=True,
            summary_text=result,
            source="template",
            detected_type=report_type or "unknown",
            error=error
        )


def get_llm_service() -> LLMService:
    """获取LLM服务实例"""
    return LLMService()


if __name__ == "__main__":
    # 测试：通用LLM服务测试
    from services.fine_report_client import FineReportClient
    
    client = FineReportClient()
    
    # 测试不同类型报表
    test_cases = [
        ("检验数据", {"shift": "夜班"}),
        ("销售数据", {"report_category": "sales"}),
    ]
    
    for name, params in test_cases:
        data_result = client.fetch_report_data(params=params)
        llm = LLMService()
        summary = llm.generate_summary(
            data_result.raw_text, 
            report_type=name,
            context=params.get("shift", "")
        )
        print(f"\n=== {name} 摘要 ===")
        print(summary.summary_text)
        print(f"(来源: {summary.source}, 类型: {summary.detected_type})")
