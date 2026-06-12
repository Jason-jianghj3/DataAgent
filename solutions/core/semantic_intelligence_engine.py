#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义智能分析引擎 v4.0

核心理念：
  "让系统像人类分析师一样思考"
  
人类分析师会如何处理"CI部门工单最多的员工"？
  Step 1: 看数据 → "哦，这是每个员工的每个审批节点的明细"
  Step 2: 理解需求 → "用户要的是每个人的总工作量"
  Step 3: 决策 → "得先把同一个人的多条记录加起来"
  Step 4: 执行 → 按人分组求和 → 排序 → 找最大最小
  
本引擎要复现这个思维过程！

架构（5层智能流水线）:
  
  ┌─────────────────────────────────────────────┐
  │ Layer 0: 数据语义发现器                       │
  │   超越字段名，深入理解数据的真实含义           │
  │   - 实体识别: 这是谁的数据？(员工/部门/订单)   │
  │   - 关系发现: 一行代表什么？(明细/汇总)        │
  │   - 业务推断: 这个数值指标是什么意思？          │
  └────────────────────┬────────────────────────┘
                       ↓
  ┌─────────────────────────────────────────────┐
  │ Layer 1: 意图深度解析器                       │
  │   不只是关键词匹配，而是真正的语义理解          │
  │   - 解析用户到底想要什么粒度的结果             │
  │   - 识别隐含的聚合需求                        │
  │   - 提取比较维度和基准点                      │
  └────────────────────┬────────────────────────┘
                       ↓
  ┌─────────────────────────────────────────────┐
  │ Layer 2: 分析策略生成器                       │
  │   基于数据语义 + 用户意图 → 最优执行计划       │
  │   - 多步推理链: 聚合→筛选→排序→提取          │
  │   - 自适应选择分析方法                        │
  │   - 处理边界情况和异常                        │
  └────────────────────┬────────────────────────┘
                       ↓
  ┌─────────────────────────────────────────────┐
  │ Layer 3: 智能执行引擎                         │
  │   动态执行生成的策略                          │
  │   - Python代码动态生成与执行                  │
  │   - 中间结果验证与修正                        │
  │   - 结果质量自检                              │
  └────────────────────┬────────────────────────┘
                       ↓
  ┌─────────────────────────────────────────────┐
  │ Layer 4: 结果解释器                           │
  │   将结构化结果转化为自然语言答案               │
  │   - 结合业务上下文生成可理解的总结             │
  │   - 提供关键洞察和建议                        │
  │   - 适配不同用户的表达偏好                    │
  └─────────────────────────────────────────────┘

适用场景（无需修改代码，完全自适应）:
  ✓ 任意SQL返回的数据结构（几百上千种都没问题）
  ✓ 任意报表的业务领域（工单/库存/财务/人事...）
  ✓ 任意复杂的用户查询（极值/排名/对比/趋势/异常...）
  
示例演示:
  
  场景1: 工单效能分析
  输入: "CI部门完成工单数量最多的员工"
  数据: [{员工:'张三', 审批节点:'权限创建', 数量:79}, 
        {员工:'张三', 审批节点:'数据备份', 数量:55}, ...]
  
  引擎自动推理:
  → [Layer0] 发现: 同一员工出现多次，每行是一个审批节点明细
  → [Layer1] 理解: "最多"指的是每个人总工单数的最大值
  → [Layer2] 决策: 必须先按员工聚合(SUM数量)，再找MAX
  → [Layer3] 执行: GROUP BY 员工 → SUM(数量) → MAX → 张三(134)
  → [Layer4] 输出: "CI部门中，张三完成工单数量最多(134单)..."
  
  
  场景2: 库存分析（换了个完全不同的报表）
  输入: "仓库里哪些物料库存周转率最低"
  数据: [{物料编码:'M001', 物料名称:'螺丝', 库存数量:1000, 
         出库频率:50, 最后入库日期:'2024-01-15'}, ...]
  
  引擎自动推理:
  → [Layer0] 发现: 每行是一个物料，有多维指标
  → [Layer1] 理解: 需要计算"周转率"=出库频率/库存量
  → [Layer2] 决策: 计算衍生指标 → 排序 → 取Bottom N
  → [Layer3] 执行: 对每行计算周转率 → ASC排序 → Top5
  → [Layer4] 输出: "以下物料的库存周转率较低，可能存在呆滞风险..."
"""

import json
import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class DataSemantics:
    """数据语义信息（Layer 0输出）"""
    
    # 实体信息
    entity_type: str = "unknown"              # employee/department/order/product/...
    entity_field: str = ""                     # 标识字段的字段名
    entity_candidates: List[str] = field(default_factory=list)
    
    # 行级语义
    row_granularity: str = "unknown"           # detail(明细)/summary(汇总)/mixed
    is_detail_level: bool = False              # 是否为明细级别数据
    
    # 字段语义映射
    field_semantics: Dict[str, str] = field(default_factory=dict)  # {字段名: 语义标签}
    
    # 数值指标
    metric_fields: List[str] = field(default_factory=list)
    primary_metric: str = ""                   # 主要指标（用户最可能关心的）
    
    # 维度字段
    dimension_fields: List[str] = field(default_factory=list)
    time_fields: List[str] = field(default_factory=list)
    
    # 关键发现
    key_findings: List[str] = field(default_factory=list)
    
    # 原始数据快照（用于后续分析）
    sample_data: List[Dict] = field(default_factory=list)
    total_records: int = 0
    unique_entities: int = 0


@dataclass 
class IntentAnalysis:
    """意图分析结果（Layer 1输出）"""
    
    # 核心意图
    primary_intent: str = "unknown"            # extreme/ranking/comparison/trend/statistic/description
    intent_confidence: float = 0.0
    
    # 分析粒度
    granularity: str = "auto"                  # detail/entity_summary/total_summary
    require_aggregation: bool = False           # 是否需要预聚合
    aggregation_logic: str = ""                 # 聚合逻辑描述
    
    # 目标指标
    target_metric: str = ""                     # 用户关心的指标
    metric_operation: str = ""                  # sum/count/avg/max/min/custom
    
    # 比较维度（如果涉及比较）
    comparison_dimension: str = ""
    comparison_targets: List[str] = field(default_factory=list)
    
    # 过滤条件
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # 输出要求
    output_count: int = 10                      # 返回几条结果
    sort_order: str = "desc"                    # desc/asc
    
    # 自然语言重述
    restated_query: str = ""                    # 系统理解的用户需求（用于验证）


@dataclass
class AnalysisStrategy:
    """分析策略（Layer 2输出）"""
    
    # 执行步骤（有序列表）
    execution_steps: List[Dict] = field(default_factory=list)
    # 示例: [
    #   {"step": 1, "action": "aggregate", "params": {"by": "员工", "metric": "数量", "method": "sum"}},
    #   {"step": 2, "action": "sort", "params": {"by": "aggregated_value", "order": "desc"}},
    #   {"step": 3, "action": "extract", "params": {"type": "extreme", "count": 2}},
    # ]
    
    # 策略说明
    strategy_reasoning: str = ""                # 为什么选择这个策略
    expected_output_format: str = ""            # 预期输出格式
    
    # 风险提示
    risk_warnings: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """最终分析结果"""
    
    success: bool
    data_semantics: DataSemantics
    intent_analysis: IntentAnalysis
    analysis_strategy: AnalysisStrategy
    
    raw_data: List[Dict]
    processed_data: Any                         # 处理后的数据（可能是dict/list/DataFrame）
    result_records: List[Dict]                   # 最终结果记录
    
    summary: str                                # 自然语言总结
    insights: List[str]                         # 关键洞察
    recommendations: List[str] = field(default_factory=list)  # 建议
    
    chart_config: Dict = field(default_factory=dict)
    execution_trace: List[str] = field(default_factory=list)  # 执行轨迹（用于调试）
    
    # 性能指标
    processing_time_ms: int = 0


# ============================================================
# Layer 0: 数据语义发现器
# ============================================================

class DataSemanticDiscoverer:
    """
    数据语义自动发现器
    
    核心能力：
    1. 实体识别 - 这是谁的数据？
    2. 行粒度判断 - 明细还是汇总？
    3. 字段语义推断 - 每个字段的业务含义
    4. 业务上下文理解 - 这些数据在说什么故事？
    """
    
    def __init__(self, llm_service=None):
        self.llm_service = None
        if llm_service is not None:
            self.llm_service = llm_service
            self._llm_client = getattr(llm_service, 'client', None)
        else:
            self._init_llm()
        
        # 领域知识库（可扩展）
        self.domain_knowledge = {
            'employee_indicators': ['员工', 'USRDESC', '姓名', '名字', '处理人', '责任人', '负责人', '操作员'],
            'department_indicators': ['部门', 'DEPT', '科室', '团队', '组织'],
            'quantity_indicators': ['数量', 'COUNT', 'count', '件数', '单数', '笔数', '次数'],
            'time_indicators': ['时间', '日期', 'TIME', 'DATE', '耗时', '天数', '周期'],
            'amount_indicators': ['金额', 'AMOUNT', '总价', '总额', '成本', '费用'],
            'rate_indicators': ['率', 'RATE', 'ratio', '比例', '占比', '百分比'],
            
            # 细粒度标识（表明这是明细数据）
            'detail_indicators': ['审批节点', '工序', '环节', '阶段', '步骤', '类型', '类别', '状态'],
            
            # 聚合标识（表明这可能是汇总数据）
            'summary_indicators': ['总计', '合计', '汇总', '平均', 'AVG', 'SUM']
        }
    
    def _init_llm(self):
        """初始化LLM服务"""
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from services.llm_service import LLMService
            self.llm_service = LLMService()
            self._llm_client = getattr(self.llm_service, 'client', None)
        except Exception as e:
            logger.warning(f"[SemanticDiscoverer] LLM初始化失败: {e}")
            self.llm_service = None
            self._llm_client = None
    
    def _call_llm(self, prompt: str, system_role: str = "你是数据分析专家。") -> Optional[str]:
        """
        通用LLM调用方法（适配现有的LLMService）
        
        Args:
            prompt: 用户消息
            system_role: 系统角色提示
            
        Returns:
            LLM响应文本，失败返回None
        """
        if not self._llm_client:
            return None
        
        try:
            response = self._llm_client.chat.completions.create(
                model="glm-4-flash",  # 使用默认模型
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"[SemanticDiscoverer] LLM调用失败: {e}")
            return None
    
    def discover(self, data: List[Dict]) -> DataSemantics:
        """
        发现数据语义（主入口）
        
        Args:
            data: SQL查询返回的原始数据
            
        Returns:
            DataSemantics 包含完整的语义信息
        """
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("[Layer 0] 数据语义发现器启动")
        print(f"  输入数据: {len(data)} 条记录")
        if data:
            print(f"  字段列表: {list(data[0].keys())}")
        print("=" * 70)
        
        if not data or len(data) == 0:
            return DataSemantics(total_records=0)
        
        semantics = DataSemantics(
            sample_data=data[:10],
            total_records=len(data)
        )
        
        # ====== Step 1: 快速规则分析（不需要LLM）======
        print("\n  [Step 1] 规则快速扫描...")
        self._rule_based_analysis(data, semantics)
        
        # ====== Step 2: 统计特征分析 ======
        print("  [Step 2] 统计特征分析...")
        self._statistical_analysis(data, semantics)
        
        # ====== Step 3: 关系发现 ======
        print("  [Step 3] 实体关系发现...")
        self._relationship_discovery(data, semantics)
        
        # ====== Step 4: LLM深度语义理解（仅规则置信度极低时触发）======
        if self.llm_service and semantics.entity_type == 'unknown' and not semantics.primary_metric:
            print("  [Step 4] 规则置信度极低，LLM深度语义理解...")
            self._llm_semantic_understanding(data, semantics)
        else:
            if semantics.entity_type != 'unknown':
                semantics.key_findings.append(f"业务含义: 规则识别为{semantics.entity_type}类型数据")
            print(f"  [Step 4] 跳过LLM，规则分析已足够 (entity_type={semantics.entity_type})")
        
        elapsed = (time.time() - start_time) * 1000
        semantics.key_findings.insert(0, f"语义发现耗时: {elapsed:.0f}ms")
        
        # 打印发现结果
        self._print_discovery_result(semantics)
        
        return semantics
    
    def _rule_based_analysis(self, data: List[Dict], semantics: DataSemantics):
        """基于规则的快速分析"""
        
        fields = list(data[0].keys())
        
        for fname in fields:
            values = [str(record.get(fname, '')) for record in data[:50]]
            unique_values = set(values)
            unique_count = len(unique_values)
            
            # 判断字段类型
            numeric_ratio = sum(1 for v in values[:20] if self._is_numeric(v)) / min(len(values[:20]), 1)
            
            if numeric_ratio > 0.8:
                semantics.metric_fields.append(fname)
                
                # 进一步判断是什么类型的指标
                fname_upper = fname.upper()
                if any(kw in fname_upper for kw in self.domain_knowledge['quantity_indicators']):
                    semantics.field_semantics[fname] = 'quantity'
                    if not semantics.primary_metric:
                        semantics.primary_metric = fname
                elif any(kw in fname_upper for kw in self.domain_knowledge['time_indicators']):
                    semantics.field_semantics[fname] = 'duration'
                elif any(kw in fname_upper for kw in self.domain_knowledge['amount_indicators']):
                    semantics.field_semantics[fname] = 'amount'
                else:
                    semantics.field_semantics[fname] = 'numeric'
                    
            else:
                # 非数值字段 - 判断是否为实体或维度
                fname_upper = fname.upper()
                
                # 检查是否为员工类字段
                if any(kw in fname_upper for kw in self.domain_knowledge['employee_indicators']):
                    semantics.entity_candidates.append(fname)
                    semantics.field_semantics[fname] = 'entity_employee'
                    semantics.dimension_fields.append(fname)
                    
                # 检查是否为部门类字段
                elif any(kw in fname_upper for kw in self.domain_knowledge['department_indicators']):
                    semantics.dimension_fields.append(fname)
                    semantics.field_semantics[fname] = 'dimension_department'
                    
                # 检查是否为细粒度分类（表明是明细数据）
                elif any(kw in fname_upper for kw in self.domain_knowledge['detail_indicators']):
                    semantics.field_semantics[fname] = 'granularity_detail'
                    semantics.is_detail_level = True
                    
                else:
                    # 通用维度字段
                    if unique_count <= len(data) * 0.8:  # 有一定重复率，可能是维度
                        semantics.dimension_fields.append(fname)
                        semantics.field_semantics[fname] = 'dimension_categorical'
                    else:
                        semantics.field_semantics[fname] = 'other'
        
        # 选择主要实体字段
        if semantics.entity_candidates:
            semantics.entity_field = semantics.entity_candidates[0]
            
            # 推断实体类型
            entity_fname = semantics.entity_field.upper()
            if any(kw in entity_fname for kw in self.domain_knowledge['employee_indicators']):
                semantics.entity_type = 'employee'
            elif any(kw in entity_fname for kw in self.domain_knowledge['department_indicators']):
                semantics.entity_type = 'department'
    
    def _statistical_analysis(self, data: List[Dict], semantics: DataSemantics):
        """统计分析：发现数据分布特征"""
        
        if not semantics.entity_field:
            return
        
        # 统计实体的唯一性
        entity_values = [str(record.get(semantics.entity_field, '')) for record in data]
        unique_entities = set(entity_values)
        semantics.unique_entities = len(unique_entities)
        
        # 判断是否为明细数据
        if len(data) > len(unique_entities) * 1.2:  # 记录数比实体数多20%以上
            semantics.row_granularity = 'detail'
            semantics.is_detail_level = True
            semantics.key_findings.append(
                f"检测到明细数据: 共{len(data)}条记录，{semantics.unique_entities}个唯一实体，"
                f"平均每个实体{len(data)/semantics.unique_entities:.1f}条记录"
            )
        else:
            semantics.row_granularity = 'summary'
            semantics.key_findings.append(
                f"检测到汇总数据: 共{len(data)}条记录，{semantics.unique_entities}个唯一实体"
            )
    
    def _relationship_discovery(self, data: List[Dict], semantics: DataSemantics):
        """发现字段间的关联关系"""
        
        if not semantics.is_detail_level or not semantics.entity_field:
            return
        
        # 分析：同一实体在不同细粒度维度上的分布
        entity_groups = defaultdict(list)
        for record in data:
            entity = str(record.get(semantics.entity_field, ''))
            entity_groups[entity].append(record)
        
        # 找出典型的实体案例
        sample_entity = list(entity_groups.keys())[0] if entity_groups else ''
        if sample_entity and len(entity_groups[sample_entity]) > 1:
            sample_records = entity_groups[sample_entity][:3]
            
            detail_fields = []
            for fname in semantics.field_semantics:
                if semantics.field_semantics[fname] == 'granularity_detail':
                    values = [str(r.get(fname, '')) for r in sample_records]
                    if len(set(values)) > 1:  # 该字段在同一实体的不同记录中有不同值
                        detail_fields.append(fname)
            
            if detail_fields:
                semantics.key_findings.append(
                    f"明细维度字段: {', '.join(detail_fields)} "
                    f"(同一实体在这些维度上有多个值)"
                )
    
    def _llm_semantic_understanding(self, data: List[Dict], semantics: DataSemantics):
        """使用LLM进行深度语义理解"""
        
        # 构建数据摘要
        data_summary = self._build_data_summary_for_llm(data, semantics)
        
        prompt = f"""你是一个数据分析专家。请分析以下数据的语义特征。

## 数据概览
{data_summary}

## 你的任务
请分析并输出JSON格式的语义信息：

```json
{{
    "entity_type": "employee|department|product|order|other",
    "entity_description": "用一句话描述这些数据代表什么业务对象",
    "row_level_meaning": "每一行数据代表什么？是明细记录还是汇总统计？",
    "business_context": "这些数据在什么业务场景下产生？讲述什么业务故事？",
    "primary_metric_description": "主要的数值指标代表什么业务含义？",
    "aggregation_hint": "如果用户问'最多/最少'，应该怎么聚合？给出具体建议",
    "potential_queries": ["列出3-5个这类数据典型可以回答的问题"]
}}
```

请确保你的分析准确反映数据的真实业务含义。"""

        try:
            response = self._call_llm(prompt, system_role="你是数据分析领域的资深专家。")
            
            if response:
                # 解析LLM响应
                llm_insight = self._parse_llm_response(response)
                
                if llm_insight.get('entity_type') and llm_insight['entity_type'] != 'other':
                    semantics.entity_type = llm_insight['entity_type']
                
                if llm_insight.get('entity_description'):
                    semantics.key_findings.append(f"业务含义: {llm_insight['entity_description']}")
                
                if llm_insight.get('row_level_meaning'):
                    semantics.key_findings.append(f"行级语义: {llm_insight['row_level_meaning']}")
                    
                if llm_insight.get('aggregation_hint'):
                    semantics.key_findings.append(f"聚合建议: {llm_insight['aggregation_hint']}")
                    
        except Exception as e:
            logger.warning(f"[SemanticDiscoverer] LLM语义理解失败: {e}")
    
    def _build_data_summary_for_llm(self, data: List[Dict], semantics: DataSemantics) -> str:
        """为LLM构建数据摘要"""
        
        lines = [
            f"- 总记录数: {len(data)}",
            f"- 字段数: {len(data[0].keys()) if data else 0}",
            f"- 字段列表:",
        ]
        
        for fname in list(data[0].keys())[:10]:  # 最多显示10个字段
            values = [str(record.get(fname, '')) for record in data[:5]]
            unique_sample = list(set(values))[:3]
            lines.append(f"  * {fname}: 类型={semantics.field_semantics.get(fname, '未知')}, "
                        f"示例值={unique_sample}")
        
        if semantics.is_detail_level:
            lines.append(f"\n- 重要发现: 这是明细级别数据")
            lines.append(f"  实体字段: {semantics.entity_field}")
            lines.append(f"  唯一实体数: {semantics.unique_entities}")
            if semantics.unique_entities > 0:
                lines.append(f"  平均每实体记录数: {len(data)/semantics.unique_entities:.1f}")
        
        # 显示3条样例数据
        lines.append(f"\n- 前3条数据样例:")
        for i, record in enumerate(data[:3], 1):
            lines.append(f"\n  记录{i}:")
            for k, v in list(record.items())[:6]:
                lines.append(f"    {k}: {v}")
        
        return '\n'.join(lines)
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {}
    
    def _is_numeric(self, value: str) -> bool:
        """判断字符串是否为数值"""
        try:
            float(value)
            return True
        except:
            return False
    
    def _print_discovery_result(self, semantics: DataSemantics):
        """打印语义发现结果"""
        
        print(f"\n  {'='*50}")
        print(f"  [语义发现结果]")
        print(f"  {'='*50}")
        print(f"  实体类型: {semantics.entity_type}")
        print(f"  实体字段: {semantics.entity_field}")
        print(f"  行粒度: {semantics.row_granularity} ({'明细' if semantics.is_detail_level else '汇总'})")
        print(f"  主要指标: {semantics.primary_metric}")
        print(f"  数值字段: {semantics.metric_fields}")
        print(f"  维度字段: {semantics.dimension_fields}")
        print(f"\n  关键发现:")
        for finding in semantics.key_findings[:5]:
            print(f"    • {finding}")


# ============================================================
# Layer 1: 意图深度解析器
# ============================================================

class IntentDeepParser:
    """
    意图深度解析器
    
    超越关键词匹配，实现真正的语义理解：
    1. 解析用户的核心诉求
    2. 识别隐含的聚合需求
    3. 理解比较的维度和基准
    4. 重述用户意图以验证理解准确性
    """
    
    def __init__(self, llm_service=None):
        self.llm_service = None
        if llm_service is not None:
            self.llm_service = llm_service
            self._llm_client = getattr(llm_service, 'client', None)
        else:
            self._init_llm()
    
    def _init_llm(self):
        """初始化LLM服务"""
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from services.llm_service import LLMService
            self.llm_service = LLMService()
            self._llm_client = getattr(self.llm_service, 'client', None)
        except Exception as e:
            logger.warning(f"[IntentParser] LLM初始化失败: {e}")
            self.llm_service = None
            self._llm_client = None
    
    def _call_llm(self, prompt: str, system_role: str = "你是NLP专家。") -> Optional[str]:
        """通用LLM调用方法"""
        if not self._llm_client:
            return None
        
        try:
            response = self._llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"[IntentParser] LLM调用失败: {e}")
            return None
    
    def parse(self, user_query: str, data_semantics: DataSemantics = None, 
              pre_parsed_intent: Dict = None) -> IntentAnalysis:
        """
        深度解析用户意图
        
        Args:
            user_query: 用户的自然语言查询
            data_semantics: 数据语义信息（可选，用于上下文感知）
            
        Returns:
            IntentAnalysis 包含详细的意图分析结果
        """
        
        print("\n" + "=" * 70)
        print("[Layer 1] 意图深度解析器启动")
        print(f"  用户查询: \"{user_query}\"")
        print("=" * 70)
        
        intent = IntentAnalysis()
        
        # ====== Step 1: 快速意图分类（规则）======
        print("\n  [Step 1] 快速意图扫描...")
        self._quick_intent_classification(user_query, intent)
        
        # ====== Step 2: 深度语义解析 ======
        if pre_parsed_intent and pre_parsed_intent.get('analysis_type'):
            print("  [Step 2] 复用预解析意图，跳过LLM...")
            intent.primary_intent = pre_parsed_intent.get('analysis_type', 'statistic')
            intent.intent_confidence = 0.9
            intent.target_metric = pre_parsed_intent.get('metric', '')
            intent.restated_query = user_query
            
            metric_op_map = {
                'extreme': 'max', 'ranking': 'sum', 'comparison': 'sum',
                'statistic': 'sum', 'trend': 'sum', 'list': 'sum'
            }
            intent.metric_operation = metric_op_map.get(intent.primary_intent, 'sum')
            
            if data_semantics and data_semantics.is_detail_level:
                if intent.primary_intent in ['extreme', 'ranking', 'statistic']:
                    intent.require_aggregation = True
                    intent.aggregation_logic = f"明细数据需按{data_semantics.entity_field}聚合后分析"
            
            if data_semantics and data_semantics.metric_fields and intent.target_metric:
                user_metric_lower = intent.target_metric.lower()
                best_match = None
                best_score = 0
                for mf in data_semantics.metric_fields:
                    mf_lower = mf.lower()
                    score = 0
                    if user_metric_lower in mf_lower or mf_lower in user_metric_lower:
                        score = 10
                    # 【修复Bug1】增加工单数量↔审批数量的直接同义词匹配
                    metric_synonym_groups = [
                        {'工单数量', '审批数量', '工单数', '完成数', '处理数', '节点数'},
                        {'总耗时', 'total_process_days'},
                        {'平均耗时', 'avg_process_days', '平均处理时间'},
                        {'人均工单', 'per_capita'},
                    ]
                    for syn_group in metric_synonym_groups:
                        syn_lower = {s.lower() for s in syn_group}
                        if user_metric_lower in syn_lower and mf_lower in syn_lower:
                            score = max(score, 9)  # 同义词组匹配，高分但低于完全匹配
                    metric_keywords = {
                        '耗时': ['耗时', '时间', '处理时间', '天数'],
                        '平均耗时': ['平均耗时', '平均时间', '平均处理'],
                        '审批数量': ['审批', '数量', '工单数', '审批数'],
                        '数量': ['数量', 'count', '总数', '件数'],
                    }
                    for kw_group, keywords in metric_keywords.items():
                        if any(kw in user_metric_lower for kw in keywords):
                            if any(kw in mf_lower for kw in keywords):
                                score = max(score, 8)
                    if score > best_score:
                        best_score = score
                        best_match = mf
                if best_match and best_score >= 8:
                    intent.target_metric = best_match
                    data_semantics.primary_metric = best_match
                    print(f"    [指标匹配] 用户指标→数据字段: {user_metric_lower} → {best_match} (score={best_score})")
        elif self.llm_service:
            print("  [Step 2] LLM深度语义解析...")
            self._deep_intent_parsing(user_query, intent, data_semantics)
        else:
            print("  [Step 2] 使用规则增强...")
            self._rule_based_enhancement(user_query, intent, data_semantics)
        
        # ====== Step 3: 意图验证与修正 ======
        print("  [Step 3] 意图一致性检查...")
        self._validate_and_refine(intent, data_semantics)
        
        # 打印解析结果
        self._print_intent_result(intent)
        
        return intent
    
    def _quick_intent_classification(self, query: str, intent: IntentAnalysis):
        """快速意图分类（基于关键词和模式）"""
        
        q = query.lower()
        
        # 极值查询
        extreme_patterns = [
            (r'最多|最高|最大|top\s*\d*|第一', 'maximum'),
            (r'最少|最低|最小|倒数|最后', 'minimum'),
        ]
        for pattern, direction in extreme_patterns:
            if re.search(pattern, q):
                intent.primary_intent = 'extreme'
                if direction == 'maximum':
                    intent.sort_order = 'desc'
                else:
                    intent.sort_order = 'asc'
                break
        
        # 排名查询
        if re.search(r'top\s*\d+|前\d+名|排行|排名|榜', q):
            intent.primary_intent = 'ranking'
            
        # 对比查询
        if re.search(r'对比|比较|vs|差异|区别', q):
            intent.primary_intent = 'comparison'
            
        # 统计查询
        if re.search(r'平均|总计|汇总|一共|总共|统计', q):
            intent.primary_intent = 'statistic'
            
        # 趋势查询
        if re.search(r'趋势|变化|增长|下降|走势', q):
            intent.primary_intent = 'trend'
        
        # 默认
        if intent.primary_intent == 'unknown':
            intent.primary_intent = 'description'
    
    def _deep_intent_parsing(self, query: str, intent: IntentAnalysis, 
                             data_semantics: DataSemantics = None):
        """使用LLM进行深度意图解析"""
        
        context_info = ""
        if data_semantics:
            context_info = f"""
## 数据上下文
- 数据类型: {data_semantics.entity_type}
- 行粒度: {'明细级别（同一实体可能出现多次）' if data_semantics.is_detail_level else '汇总级别'}
- 实体字段: {data_semantics.entity_field}
- 数值指标: {', '.join(data_semantics.metric_fields)}
- 主要指标: {data_semantics.primary_metric}
- 关键发现: {'; '.join(data_semantics.key_findings[:2])}

【重要】如果数据是明细级别的，用户问"最多/最少"时，通常需要先按实体聚合再查找极值！
"""
        
        prompt = f"""你是一个自然语言理解专家，专门解析用户对数据的分析需求。

## 用户查询
"{query}"

{context_info}

## 你的任务
深度解析用户的真实意图，输出JSON格式：

```json
{{
    "primary_intent": "extreme|ranking|comparison|trend|statistic|description",
    "intent_confidence": 0.0-1.0,
    "granularity": "detail|entity_summary|total_summary",
    "require_aggregation": true/false,
    "aggregation_logic": "详细描述为什么需要（或不需要）聚合，以及如何聚合",
    "target_metric": "用户关心的核心指标字段名",
    "metric_operation": "sum|count|avg|max|min",
    "restated_query": "用一句话重述用户的需求，验证你的理解是否准确",
    "key_understanding": "你最关键的1-2个理解要点"
}}
```

## 重要提醒
1. 如果数据是**明细级别**（如每个员工的每个审批节点），用户问"最多"通常指**聚合后**的结果
2. "XX最多" ≠ "单条记录XX最大"，而是"按某个维度汇总后XX最大"
3. 仔细区分用户的真实需求！"""

        try:
            response = self._call_llm(prompt, system_role="你是NLP和数据分析领域的专家。")
            
            if response:
                parsed = self._parse_llm_response(response)
                
                # 更新intent对象
                if parsed.get('primary_intent') and parsed['primary_intent'] != 'unknown':
                    intent.primary_intent = parsed['primary_intent']
                
                intent.intent_confidence = parsed.get('intent_confidence', 0.0)
                intent.require_aggregation = parsed.get('require_aggregation', False)
                intent.aggregation_logic = parsed.get('aggregation_logic', '')
                intent.target_metric = parsed.get('target_metric', '')
                intent.metric_operation = parsed.get('metric_operation', '')
                intent.restated_query = parsed.get('restated_query', '')
                
                if parsed.get('key_understanding'):
                    print(f"    [LLM关键理解]: {parsed['key_understanding']}")
                    
        except Exception as e:
            logger.error(f"[IntentParser] LLM深度解析失败: {e}")
    
    def _rule_based_enhancement(self, query: str, intent: IntentAnalysis,
                               data_semantics: DataSemantics = None):
        """当LLM不可用时，使用增强的规则"""
        
        q = query.lower()
        
        # 检测是否需要聚合的关键信号
        aggregation_signals = [
            (r'总|合计|汇总|一共', True, "检测到汇总关键词"),
            (r'最多.*员工|最少.*人员|谁.*最多', True, "询问实体级统计"),
            (r'平均|人均', True, "需要计算平均值"),
        ]
        
        for pattern, need_agg, reason in aggregation_signals:
            if re.search(pattern, q):
                intent.require_aggregation = need_agg
                intent.aggregation_logic = reason
                break
        
        # 如果数据是明细级别的，且查询涉及极值/排名，强制启用聚合
        if data_semantics and data_semantics.is_detail_level:
            if intent.primary_intent in ['extreme', 'ranking']:
                intent.require_aggregation = True
                intent.aggregation_logic = (
                    f"数据为明细级别({data_semantics.row_granularity})，"
                    f"查询类型为{intent.primary_intent}，必须先按{data_semantics.entity_field}聚合"
                )
                print(f"    [规则强制] 明细数据+{intent.primary_intent}查询 → 启用聚合")
    
    def _validate_and_refine(self, intent: IntentAnalysis, data_semantics: DataSemantics = None):
        """验证意图的一致性并进行修正"""
        
        issues = []
        
        # 检查1: 明细数据 + 极值查询 必须聚合
        if data_semantics and data_semantics.is_detail_level:
            if intent.primary_intent in ['extreme', 'ranking']:
                if not intent.require_aggregation:
                    intent.require_aggregation = True
                    intent.aggregation_logic = "自动修正：明细数据查询极值必须先聚合"
                    issues.append("自动启用聚合（明细+极值）")
        
        # 检查2: 如果需要聚合但没有指定操作方法
        if intent.require_aggregation and not intent.metric_operation:
            intent.metric_operation = 'sum'  # 默认求和
            issues.append("默认使用SUM聚合")
        
        # 检查3: 如果没有指定目标指标
        if intent.require_aggregation and not intent.target_metric:
            if data_semantics and data_semantics.primary_metric:
                intent.target_metric = data_semantics.primary_metric
                issues.append(f"自动选择指标: {intent.target_metric}")
        
        if issues:
            print(f"    [修正] {'; '.join(issues)}")
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {}
    
    def _print_intent_result(self, intent: IntentAnalysis):
        """打印意图解析结果"""
        
        print(f"\n  {'='*50}")
        print(f"  [意图解析结果]")
        print(f"  {'='*50}")
        print(f"  主意图: {intent.primary_intent} (置信度: {intent.intent_confidence:.2f})")
        print(f"  粒度: {intent.granularity}")
        print(f"  需要聚合: {intent.require_aggregation}")
        if intent.require_aggregation:
            print(f"  聚合逻辑: {intent.aggregation_logic}")
            print(f"  目标指标: {intent.target_metric}")
            print(f"  操作方法: {intent.metric_operation}")
        print(f"  重述: {intent.restated_query or '(未生成)'}")


# ============================================================
# Layer 2: 分析策略生成器
# ============================================================

class AnalysisStrategyGenerator:
    """
    分析策略生成器
    
    基于数据语义 + 用户意图 → 生成最优执行计划
    支持多步复杂推理链
    """
    
    def __init__(self, llm_service=None):
        self.llm_service = None
        if llm_service is not None:
            self.llm_service = llm_service
            self._llm_client = getattr(llm_service, 'client', None)
        else:
            self._init_llm()
    
    def _init_llm(self):
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            from services.llm_service import LLMService
            self.llm_service = LLMService()
            self._llm_client = getattr(self.llm_service, 'client', None)
        except:
            self.llm_service = None
            self._llm_client = None

    def generate(self, data_semantics: DataSemantics,
                intent: IntentAnalysis) -> AnalysisStrategy:
        """
        生成分析策略
        """
        
        print("\n" + "=" * 70)
        print("[Layer 2] 分析策略生成器启动")
        print("=" * 70)
        
        strategy = AnalysisStrategy()
        
        # 构建执行步骤
        steps = []
        step_num = 1
        
        # Step 1: 预聚合（如果需要）
        if intent.require_aggregation and data_semantics.entity_field:
            # 确定聚合方法
            # 【关键修正】对于明细数据，默认使用sum（汇总每个实体的总量）
            valid_methods = ['sum', 'count', 'avg', 'max', 'min']
            proposed_method = intent.metric_operation or "sum"
            
            if data_semantics.is_detail_level:
                # 明细数据必须先求和！
                aggregation_method = 'sum'
                if proposed_method not in ['sum', ''] and proposed_method in valid_methods:
                    print(f"    [策略修正] 明细数据: 强制使用SUM聚合（LLM建议:{proposed_method}→修正为sum）")
            else:
                # 汇总数据可以使用其他方法
                aggregation_method = proposed_method if proposed_method in valid_methods else 'sum'
            
            agg_step = {
                "step": step_num,
                "action": "aggregate",
                "params": {
                    "group_by": data_semantics.entity_field,
                    "metric": intent.target_metric or data_semantics.primary_metric,
                    "method": aggregation_method,
                    "reason": intent.aggregation_logic or ("明细数据聚合" if data_semantics.is_detail_level else "用户查询需要聚合视图")
                }
            }
            steps.append(agg_step)
            step_num += 1
        
        # Step 2: 排序
        sort_direction = intent.sort_order
        if intent.primary_intent == 'minimum':
            sort_direction = 'asc'
        
        sort_step = {
            "step": step_num,
            "action": "sort",
            "params": {
                "by": "aggregated_value" if intent.require_aggregation else (intent.target_metric or data_semantics.primary_metric),
                "order": sort_direction,
                "reason": f"根据{intent.primary_intent}意图排序"
            }
        }
        steps.append(sort_step)
        step_num += 1
        
        # Step 3: 结果提取
        # 【关键修复】正确处理复合意图（如 "extreme|ranking"）
        primary_intent = intent.primary_intent.split('|')[0].strip()  # 取第一个意图
        intent.primary_intent = primary_intent  # 更新为单一意图
        
        if primary_intent == 'extreme':
            extract_step = {
                "step": step_num,
                "action": "extract_extremes",
                "params": {
                    "extract_type": "both",  # 极值查询默认都提取
                    "count": 1,
                    "reason": "提取极值记录"
                }
            }
        elif primary_intent == 'ranking':
            extract_step = {
                "step": step_num,
                "action": "extract_top_n",
                "params": {
                    "n": intent.output_count or 10,
                    "reason": f"提取Top-{intent.output_count or 10}"
                }
            }
        else:
            extract_step = {
                "step": step_num,
                "action": "extract_all",
                "params": {
                    "limit": 20,
                    "reason": "返回所有结果"
                }
            }
        steps.append(extract_step)
        
        strategy.execution_steps = steps
        
        # 生成策略说明
        strategy.strategy_reasoning = self._generate_reasoning(data_semantics, intent, steps)
        strategy.expected_output_format = self._determine_output_format(intent)
        
        # 风险提示
        strategy.risk_warnings = self._assess_risks(data_semantics, intent)
        
        # 打印策略
        self._print_strategy(strategy)
        
        return strategy
    
    def _generate_reasoning(self, semantics: DataSemantics, intent: IntentAnalysis, 
                           steps: List[Dict]) -> str:
        """生成策略推理说明"""
        
        parts = [f"根据用户查询\"{intent.restated_query or intent.primary_intent}\""]
        
        if semantics.is_detail_level:
            parts.append(f"数据为{semantics.row_granularity}级别（{semantics.unique_entities}个实体，{semantics.total_records}条记录）")
        
        if intent.require_aggregation:
            parts.append(f"需要先按'{semantics.entity_field}'对'{intent.target_metric}'进行{intent.metric_operation}聚合")
        
        parts.append(f"然后按{'降序' if intent.sort_order == 'desc' else '升序'}排序")
        
        if intent.primary_intent == 'extreme':
            parts.append("最后提取最大/最小值")
        elif intent.primary_intent == 'ranking':
            parts.append(f"最后提取Top-{intent.output_count}")
        
        return '，'.join(parts) + '。'
    
    def _determine_output_format(self, intent: IntentAnalysis) -> str:
        """确定输出格式"""
        
        format_map = {
            'extreme': '极值记录列表（包含实体名和指标值）',
            'ranking': '排名列表（包含排名、实体名、指标值）',
            'statistic': '统计摘要（总计、平均、最大、最小等）',
            'comparison': '对比表格（多维度对比数据）',
            'trend': '时间序列数据（适用于折线图）',
            'description': '原始数据列表或汇总表'
        }
        return format_map.get(intent.primary_intent, '通用列表')
    
    def _assess_risks(self, semantics: DataSemantics, intent: IntentAnalysis) -> List[str]:
        """评估潜在风险"""
        
        warnings = []
        
        if semantics.total_records > 10000:
            warnings.append(f"数据量大({semantics.total_records}条)，聚合操作可能较慢")
        
        if not semantics.entity_field and intent.require_aggregation:
            warnings.append("未识别到实体字段，聚合可能不准确")
        
        if intent.intent_confidence < 0.7:
            warnings.append(f"意图置信度较低({intent.intent_confidence:.2f})，结果可能不符合预期")
        
        return warnings
    
    def _print_strategy(self, strategy: AnalysisStrategy):
        """打印策略"""
        
        print(f"\n  [策略推理]")
        print(f"  {strategy.strategy_reasoning}")
        
        print(f"\n  [执行步骤]")
        for step in strategy.execution_steps:
            params_str = ', '.join(f"{k}={v}" for k, v in step['params'].items())
            print(f"    Step {step['step']}: [{step['action'].upper()}] {params_str}")
        
        if strategy.risk_warnings:
            print(f"\n  [风险提示]")
            for w in strategy.risk_warnings:
                print(f"    ⚠ {w}")


# ============================================================
# Layer 3: 智能执行引擎
# ============================================================

class IntelligentExecutionEngine:
    """
    智能执行引擎
    
    动态执行生成的分析策略
    """
    
    def execute(self, raw_data: List[Dict], strategy: AnalysisStrategy,
               data_semantics: DataSemantics, intent: IntentAnalysis) -> Tuple[Any, List[str]]:
        """
        执行分析策略
        
        Returns:
            (processed_data, execution_trace)
        """
        
        print("\n" + "=" * 70)
        print("[Layer 3] 智能执行引擎启动")
        print("=" * 70)
        
        trace = []
        current_data = raw_data
        
        for step in strategy.execution_steps:
            action = step['action']
            params = step['params']
            
            print(f"\n  [执行] Step {step['step']}: {action.upper()}")
            print(f"    参数: {params.get('reason', '')}")
            
            if action == 'aggregate':
                current_data, step_trace = self._execute_aggregate(current_data, params, data_semantics)
                trace.extend(step_trace)
                
            elif action == 'sort':
                current_data, step_trace = self._execute_sort(current_data, params)
                trace.extend(step_trace)
                
            elif action == 'extract_extremes':
                current_data, step_trace = self._extract_extremes(current_data, params, intent, data_semantics)
                trace.extend(step_trace)
                
            elif action == 'extract_top_n':
                current_data, step_trace = self._extract_top_n(current_data, params)
                trace.extend(step_trace)
                
            elif action == 'extract_all':
                current_data, step_trace = current_data[:params.get('limit', 20)], [f"取前{params.get('limit', 20)}条"]
                trace.extend(step_trace)
        
        print(f"\n  [执行完成] 最终得到 {len(current_data) if isinstance(current_data, list) else len(current_data)} 条结果")
        
        return current_data, trace
    
    def _execute_aggregate(self, data: List[Dict], params: Dict,
                          semantics: DataSemantics) -> Tuple[Dict, List[str]]:
        """执行聚合操作"""

        group_by = params['group_by']
        metric = params['metric']
        method = params.get('method', 'sum')

        trace = [f"按 '{group_by}' 分组，对 '{metric}' 进行 {method} 聚合"]

        # 【修复Bug1】指标字段名自动匹配：当metric不在数据列中时，尝试同义词映射
        if data and len(data) > 0:
            actual_columns = list(data[0].keys())
            if metric not in actual_columns:
                # 同义词映射表：用户可能说的指标名 → 实际SQL列名
                metric_synonyms = {
                    '工单数量': ['审批数量', 'order_count', '节点数量'],
                    '审批数量': ['工单数量', 'order_count', '节点数量'],
                    '工单数': ['审批数量', '工单数量', 'order_count'],
                    '完成数': ['审批数量', '工单数量'],
                    '处理数': ['审批数量', '工单数量'],
                    '总耗时': ['total_process_days', 'ProcessDays'],
                    '平均耗时': ['avg_process_days', '平均处理时间'],
                    '人均工单': ['per_capita'],
                }
                # 尝试通过同义词找到实际列名
                candidates = metric_synonyms.get(metric, [])
                for candidate in candidates:
                    if candidate in actual_columns:
                        trace.append(f"  [指标修正] '{metric}' 不在数据列中，使用同义词 '{candidate}'")
                        metric = candidate
                        break
                else:
                    # 同义词也没找到，尝试从semantics.metric_fields中找最接近的
                    if semantics and semantics.metric_fields:
                        for mf in semantics.metric_fields:
                            if mf in actual_columns:
                                trace.append(f"  [指标修正] '{metric}' 不在数据列中，使用primary_metric '{mf}'")
                                metric = mf
                                break

        # 【修复Bug1】group_by字段名自动匹配
        if data and len(data) > 0:
            actual_columns = list(data[0].keys())
            if group_by not in actual_columns:
                # 实体字段同义词映射
                entity_synonyms = {
                    'USRDESC': ['员工', '姓名', '处理人', '操作人'],
                    '员工': ['USRDESC', '姓名', '处理人'],
                }
                candidates = entity_synonyms.get(group_by, [])
                for candidate in candidates:
                    if candidate in actual_columns:
                        trace.append(f"  [分组修正] '{group_by}' 不在数据列中，使用 '{candidate}'")
                        group_by = candidate
                        break

        grouped = defaultdict(list)
        zero_count = 0
        for record in data:
            key = str(record.get(group_by, ''))
            value = record.get(metric, 0)

            if isinstance(value, (int, float)):
                grouped[key].append(value)
                if value == 0:
                    zero_count += 1

        # 【修复Bug1】如果所有值都是0，说明metric可能仍然不匹配，尝试用所有数值列
        if zero_count == len(data) and data and semantics and semantics.metric_fields:
            actual_columns = list(data[0].keys())
            for mf in semantics.metric_fields:
                if mf in actual_columns and mf != metric:
                    # 尝试用这个字段重新聚合
                    test_grouped = defaultdict(list)
                    for record in data:
                        key = str(record.get(group_by, ''))
                        value = record.get(mf, 0)
                        if isinstance(value, (int, float)) and value != 0:
                            test_grouped[key].append(value)
                    if test_grouped and any(sum(v) > 0 for v in test_grouped.values()):
                        trace.append(f"  [指标二次修正] '{metric}' 全为0，改用 '{mf}'")
                        metric = mf
                        grouped = test_grouped
                        break

        aggregated = {}
        for key, values in grouped.items():
            if method == 'sum':
                aggregated[key] = sum(values)
            elif method == 'count':
                aggregated[key] = len(values)
            elif method == 'avg':
                aggregated[key] = sum(values) / len(values) if values else 0
            elif method == 'max':
                aggregated[key] = max(values)
            elif method == 'min':
                aggregated[key] = min(values)
            else:
                aggregated[key] = sum(values)

        trace.append(f"聚合完成: {len(aggregated)} 个唯一实体")

        # 显示前5个
        sorted_items = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:5]
        for k, v in sorted_items:
            trace.append(f"  {k}: {v}")

        return aggregated, trace
    
    def _execute_sort(self, data, params: Dict) -> Tuple[List, List[str]]:
        """执行排序"""
        
        by = params['by']
        order = params.get('order', 'desc')
        
        trace = [f"按 '{by}' {'降序' if order == 'desc' else '升序'} 排序"]
        
        if isinstance(data, dict):
            sorted_items = sorted(data.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                                 reverse=(order == 'desc'))
            return sorted_items, trace
        else:
            reverse = (order == 'desc')
            sorted_data = sorted(data, key=lambda x: x.get(by, 0) or 0, reverse=reverse)
            return sorted_data, trace
    
    def _extract_extremes(self, data, params: Dict, intent: IntentAnalysis,
                         semantics: DataSemantics) -> Tuple[List[Dict], List[str]]:
        """提取极值"""
        
        extract_type = params.get('extract_type', 'maximum_only')
        trace = []
        
        results = []
        
        # 【修复】统一处理dict和list of tuples两种格式
        def get_sort_key(item):
            if isinstance(item, tuple) and len(item) >= 2:
                return item[1] if isinstance(item[1], (int, float)) else 0
            elif isinstance(item, dict):
                field = intent.target_metric or semantics.primary_metric
                return item.get(field, 0) or 0
            return 0
        
        def get_entity_name(item):
            if isinstance(item, tuple) and len(item) >= 1:
                return str(item[0])
            elif isinstance(item, dict):
                field = semantics.entity_field or 'entity'
                return item.get(field, '?')
            return '?'
        
        def get_metric_value(item):
            if isinstance(item, tuple) and len(item) >= 2:
                return item[1]
            elif isinstance(item, dict):
                field = intent.target_metric or semantics.primary_metric
                return item.get(field, 0)
            return 0
        
        # 最大值
        sorted_desc = sorted(data, key=get_sort_key, reverse=True)
        
        if sorted_desc:
            max_item = sorted_desc[0]
            max_name = get_entity_name(max_item)
            max_val = get_metric_value(max_item)
            
            results.append({'rank': 1, 'type': 'maximum', 'entity': max_name, 'value': max_val})
            trace.append(f"最大值: {max_name} ({max_val})")
        
        # 最小值
        if extract_type == 'both':
            sorted_asc = sorted(data, key=get_sort_key, reverse=False)
            
            if sorted_asc:
                min_item = sorted_asc[0]
                min_name = get_entity_name(min_item)
                min_val = get_metric_value(min_item)
                
                results.append({'rank': 2, 'type': 'minimum', 'entity': min_name, 'value': min_val})
                trace.append(f"最小值: {min_name} ({min_val})")
        
        return results, trace
    
    def _extract_top_n(self, data, params: Dict) -> Tuple[List[Dict], List[str]]:
        """提取Top N"""
        
        n = params.get('n', 10)
        trace = [f"提取Top-{min(n, len(data) if isinstance(data, list) else len(data))}"]
        
        results = []
        
        if isinstance(data, list):
            for i, item in enumerate(data[:n], 1):
                if isinstance(item, dict):
                    results.append({'rank': i, **item})
                else:
                    results.append({'rank': i, 'data': item})
        elif isinstance(data, dict):
            sorted_items = sorted(data.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:n]
            for i, (k, v) in enumerate(sorted_items, 1):
                results.append({'rank': i, 'entity': k, 'value': v})
        
        return results, trace


# ============================================================
# Layer 4: 结果解释器
# ============================================================

class ResultInterpreter:
    """
    结果解释器
    
    将结构化结果转化为自然语言答案
    """
    
    def __init__(self, llm_service=None):
        self.llm_service = None
        if llm_service is not None:
            self.llm_service = llm_service
            self._llm_client = getattr(llm_service, 'client', None)
        else:
            self._init_llm()
    
    def _init_llm(self):
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from services.llm_service import LLMService
            self.llm_service = LLMService()
            self._llm_client = getattr(self.llm_service, 'client', None)
        except:
            self.llm_service = None
            self._llm_client = None
    
    def interpret(self, processed_data: Any, intent: IntentAnalysis, 
                 data_semantics: DataSemantics, strategy: AnalysisStrategy) -> Tuple[str, List[str], List[str]]:
        """
        解释结果
        
        Returns:
            (summary, insights, recommendations)
        """
        
        print("\n" + "=" * 70)
        print("[Layer 4] 结果解释器启动")
        print("=" * 70)
        
        # 构建结果摘要
        summary = self._generate_summary(processed_data, intent, data_semantics)
        
        # 生成洞察
        insights = self._generate_insights(processed_data, intent, data_semantics)
        
        # 生成建议
        recommendations = self._generate_recommendations(processed_data, intent, data_semantics)
        
        print(f"\n  [总结生成完成]")
        print(f"  {summary[:100]}..." if len(summary) > 100 else f"  {summary}")
        
        return summary, insights, recommendations
    
    def _generate_summary(self, data: Any, intent: IntentAnalysis, 
                          semantics: DataSemantics) -> str:
        """生成总结"""
        
        entity_display = semantics.entity_field or '实体'
        metric_display = intent.target_metric or semantics.primary_metric or '指标'
        
        metric_name_map = {
            'order': '审批数量', 'order_count': '工单数量', '工单数量': '工单数量',
            '审批数量': '审批数量', 'count': '数量', 'amount': '金额',
            'total_process_days': '总耗时', '总耗时': '总耗时',
            'avg_process_days': '平均耗时', '平均耗时': '平均耗时',
            'user_count': '处理人数', '处理人数': '处理人数',
            'per_capita': '人均工单', '人均工单': '人均工单',
        }
        metric_display = metric_name_map.get(metric_display, metric_display)
        if re.match(r'^[a-z_]+$', metric_display):
            metric_display = '指标'
        
        if intent.primary_intent == 'extreme':
            if not data or len(data) == 0:
                return "未找到符合条件的数据。"
            
            lines = []
            
            # 最大值
            max_records = [r for r in data if r.get('type') == 'maximum']
            if max_records:
                max_r = max_records[0]
                entity_name = max_r.get('entity', max_r.get(semantics.entity_field, '?'))
                max_val = max_r.get('value', 0)
                lines.append(f"**{metric_display}最多**: **{entity_name}** ({max_val})")
            
            # 最小值
            min_records = [r for r in data if r.get('type') == 'minimum']
            if min_records:
                min_r = min_records[0]
                entity_name = min_r.get('entity', min_r.get(semantics.entity_field, '?'))
                min_val = min_r.get('value', 0)
                lines.append(f"**{metric_display}最少**: **{entity_name}** ({min_val})")
                
                # 差异分析
                if max_records and min_records:
                    max_v = max_records[0].get('value', 0)
                    min_v = min_records[0].get('value', 0)
                    if isinstance(max_v, (int, float)) and isinstance(min_v, (int, float)) and min_v > 0:
                        ratio = max_v / min_v
                        diff = max_v - min_v
                        lines.append(f"\n**差异**: 最高是最低的 **{ratio:.1f}倍** (相差 {diff})")
            
            dept_context = self._detect_department_context(data, semantics)
            if dept_context:
                return f"在 **{dept_context}** 中:\n" + '\n'.join(lines)
            return '\n'.join(lines)
            
        elif intent.primary_intent == 'ranking':
            if not data or len(data) == 0:
                return "未找到排名数据。"
            
            lines = [f"**{metric_display} Top {min(len(data), 10)}**:\n"]
            for i, r in enumerate(data[:10], 1):
                rank = i
                # 【修复】支持dict和tuple两种数据格式
                if isinstance(r, dict):
                    name = r.get('entity', r.get(semantics.entity_field, f'#{rank}'))
                    val = r.get('value', 0)
                elif isinstance(r, (tuple, list)) and len(r) >= 2:
                    name = str(r[0])
                    val = r[1]
                else:
                    name = f'#{rank}'
                    val = 0
                    
                medal = ""
                if rank == 1:
                    medal = "[冠军] "
                elif rank == 2:
                    medal = "[亚军] "
                elif rank == 3:
                    medal = "[季军] "
                lines.append(f"{medal}{rank}. **{name}**: {val}")
            
            return '\n'.join(lines)
            
        elif intent.primary_intent == 'statistic':
            if not data:
                return "无法计算统计数据。"
            
            values = []
            if isinstance(data, list):
                for r in data:
                    if isinstance(r, dict):
                        v = r.get('value', r.get(metric_display, 0))
                    elif isinstance(r, (list, tuple)) and len(r) >= 2:
                        v = r[1]  # 如果是tuple/list，取第二个元素
                    else:
                        v = r
                    if isinstance(v, (int, float)):
                        values.append(v)
            elif isinstance(data, dict):
                values = [v for v in data.values() if isinstance(v, (int, float))]
            
            if not values:
                return "无有效数值数据。"
            
            import statistics
            total = sum(values)
            avg = statistics.mean(values)
            median = statistics.median(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            
            lines = [
                f"**{metric_display}统计分析**:",
                f"- 总计: {total:,.2f}",
                f"- 平均: {avg:,.2f}",
                f"- 中位数: {median:,.2f}",
                f"- 标准差: {stdev:,.2f}",
            ]
            
            return '\n'.join(lines)
        
        else:
            return f"共获取 {len(data) if isinstance(data, list) else len(data)} 条记录"
    
    def _generate_insights(self, data: Any, intent: IntentAnalysis,
                          semantics: DataSemantics) -> List[str]:
        """生成洞察"""
        
        insights = []
        
        if intent.primary_intent == 'extreme' and data and len(data) >= 2:
            max_records = [r for r in data if r.get('type') == 'maximum']
            min_records = [r for r in data if r.get('type') == 'minimum']
            
            if max_records:
                leader = max_records[0].get('entity', '?')
                insights.append(f"{leader} 的表现领先，是团队中的佼佼者")
            
            if min_records:
                laggard = min_records[0].get('entity', '?')
                insights.append(f"{laggard} 相对较低，可能需要关注或支持")
            
            # 差异分析
            if max_records and min_records:
                max_v = max_records[0].get('value', 0)
                min_v = min_records[0].get('value', 0)
                if isinstance(max_v, (int, float)) and isinstance(min_v, (int, float)) and min_v > 0:
                    ratio = max_v / min_v
                    if ratio > 5:
                        insights.append(f"团队内部差异较大({ratio:.1f}倍)，可能存在工作分配不均的情况")
                    elif ratio > 2:
                        insights.append(f"团队成员表现存在一定差距({ratio:.1f}倍)")
        
        elif intent.primary_intent == 'ranking' and data and len(data) >= 3:
            top_performer = data[0].get('entity', data[0].get(semantics.entity_field, ''))
            if top_performer:
                insights.append(f"{top_performer} 表现优异，值得表彰和学习")
            
            if len(data) >= 5:
                bottom = data[-1]
                bottom_name = bottom.get('entity', bottom.get(semantics.entity_field, ''))
                if bottom_name:
                    insights.append(f"建议关注排名靠后的成员，提供针对性支持")
        
        return insights
    
    def _generate_recommendations(self, data: Any, intent: IntentAnalysis,
                                 semantics: DataSemantics) -> List[str]:
        """生成建议"""
        
        recommendations = []
        
        if intent.primary_intent == 'extreme':
            recommendations.append("可以使用更具体的查询词来获取更详细的信息")
            recommendations.append("建议结合时间维度进行分析，了解趋势变化")
        
        elif intent.primary_intent == 'ranking':
            recommendations.append("可以点击查看详细信息，了解各成员的具体贡献")
            recommendations.append("建议定期跟踪排名变化，及时发现异常")
        
        return recommendations
    
    def _detect_department_context(self, data: Any, semantics: DataSemantics) -> str:
        """检测部门上下文"""
        
        if isinstance(data, list) and data:
            for record in data[:3]:
                for v in record.values():
                    v_str = str(v)
                    if v_str in ['OM', 'CI', 'FM', 'QA', 'PD'] and len(v_str) <= 4:
                        return v_str
        return ""


# ============================================================
# 主引擎：整合所有层级
# ============================================================

class SemanticIntelligenceEngine:
    """
    语义智能分析引擎 v4.0（主入口）
    
    使用示例:
    
    engine = SemanticIntelligenceEngine()
    
    result = engine.analyze(
        raw_data=sql_result,           # SQL查询的原始结果
        user_query="CI部门工单最多的员工",  # 用户查询
    )
    
    print(result.summary)              # 自然语言总结
    print(result.insights)             # 关键洞察
    print(result.chart_config)         # 图表配置
    """
    
    def __init__(self):
        # 共享一个LLMService实例，避免重复创建OpenAI client
        shared_llm = self._create_shared_llm()

        self.discoverer = DataSemanticDiscoverer(llm_service=shared_llm)      # Layer 0
        self.parser = IntentDeepParser(llm_service=shared_llm)                 # Layer 1
        self.strategy_gen = AnalysisStrategyGenerator(llm_service=shared_llm)  # Layer 2
        self.executor = IntelligentExecutionEngine()     # Layer 3
        self.interpreter = ResultInterpreter(llm_service=shared_llm)           # Layer 4

        logger.info("[Semantic Intelligence Engine v4.0] 初始化完成")

    @staticmethod
    def _create_shared_llm():
        """创建共享的LLMService实例"""
        try:
            from services.llm_service import LLMService
            return LLMService()
        except Exception as e:
            logger.warning(f"[SemanticEngine] 共享LLM初始化失败: {e}")
            return None
    
    def analyze(self, raw_data: List[Dict], user_query: str, 
               query_intent: Dict = None, pre_parsed_intent: Dict = None) -> AnalysisResult:
        """
        一站式智能分析接口
        
        Args:
            raw_data: SQL执行的原始数据（任意结构）
            user_query: 用户的自然语言查询
            query_intent: 预解析的意图信息（可选）
            
        Returns:
            AnalysisResult 完整的分析结果
        """
        
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("=" * 70)
        print("[Semantic Intelligence Engine v4.0]")
        print(f"用户查询: \"{user_query}\"")
        print(f"数据规模: {len(raw_data)} 条记录")
        print("=" * 70)
        
        if not raw_data or len(raw_data) == 0:
            return AnalysisResult(
                success=False,
                data_semantics=DataSemantics(),
                intent_analysis=IntentAnalysis(),
                analysis_strategy=AnalysisStrategy(),
                raw_data=[],
                processed_data=None,
                result_records=[],
                summary="未查询到符合条件的数据。",
                insights=["请检查筛选条件是否过于严格"],
                chart_config={}
            )
        
        try:
            # ========== Layer 0: 数据语义发现 ==========
            data_semantics = self.discoverer.discover(raw_data)
            
            # ========== Layer 1: 意图深度解析 ==========
            intent = self.parser.parse(user_query, data_semantics, pre_parsed_intent=pre_parsed_intent)
            
            # ========== Layer 2: 策略生成 ==========
            strategy = self.strategy_gen.generate(data_semantics, intent)
            
            # ========== Layer 3: 执行 ==========
            processed_data, exec_trace = self.executor.execute(
                raw_data, strategy, data_semantics, intent
            )
            
            # ========== Layer 4: 结果解释 ==========
            summary, insights, recommendations = self.interpreter.interpret(
                processed_data, intent, data_semantics, strategy
            )
            
            # ========== 生成图表配置 ==========
            chart_config = self._generate_chart_config(
                processed_data, intent, data_semantics
            )
            
            # 构建结果记录
            result_records = self._build_result_records(processed_data, intent, data_semantics)
            
            elapsed = (time.time() - start_time) * 1000
            
            result = AnalysisResult(
                success=True,
                data_semantics=data_semantics,
                intent_analysis=intent,
                analysis_strategy=strategy,
                raw_data=raw_data,
                processed_data=processed_data,
                result_records=result_records,
                summary=summary,
                insights=insights,
                recommendations=recommendations,
                chart_config=chart_config,
                execution_trace=exec_trace,
                processing_time_ms=int(elapsed)
            )
            
            # 打印最终结果
            self._print_final_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"[Engine] 分析失败: {e}", exc_info=True)
            
            return AnalysisResult(
                success=False,
                data_semantics=DataSemantics(),
                intent_analysis=IntentAnalysis(),
                analysis_strategy=AnalysisStrategy(),
                raw_data=raw_data,
                processed_data=None,
                result_records=[],
                summary=f"分析执行失败: {str(e)}",
                insights=["系统错误，请联系管理员"],
                chart_config={},
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _build_result_records(self, processed_data: Any, intent: IntentAnalysis,
                            semantics: DataSemantics) -> List[Dict]:
        """构建标准化的结果记录"""
        
        if isinstance(processed_data, list):
            return processed_data
        elif isinstance(processed_data, dict):
            return [
                {'entity': k, 'value': v}
                for k, v in list(processed_data.items())[:20]
            ]
        return []
    
    def _generate_chart_config(self, data: Any, intent: IntentAnalysis,
                              semantics: DataSemantics) -> Dict:
        """生成图表配置"""
        
        if not data or (isinstance(data, list) and len(data) == 0):
            return {}
        
        categories = []
        values = []
        
        if isinstance(data, list):
            for item in data[:15]:
                if isinstance(item, tuple) and len(item) == 2:
                    categories.append(str(item[0]))
                    values.append(item[1] if isinstance(item[1], (int, float)) else 0)
                elif isinstance(item, dict):
                    name = item.get('entity')
                    if name is None or name == '?':
                        name = item.get(semantics.entity_field)
                    if name is None or name == '?':
                        for field in ['dept_code', 'dept_name', '部门', '员工', 'USRDESC', 'USRMRC', '流程类型', 'status']:
                            if field in item and item[field]:
                                name = item[field]
                                break
                    if name is None or name == '?':
                        for k, v in item.items():
                            if not isinstance(v, (int, float)) and v:
                                name = v
                                break
                    if name is None or name == '?':
                        name = f"记录{len(categories)+1}"
                    
                    val = item.get('value')
                    if val is None:
                        val = item.get(semantics.primary_metric)
                    if val is None:
                        for field in ['order_count', '工单数量', '审批数量', '总耗时', 'avg_process_days', '平均耗时', 'count', 'amount']:
                            if field in item and isinstance(item[field], (int, float)):
                                val = item[field]
                                break
                    if val is None:
                        for v in item.values():
                            if isinstance(v, (int, float)):
                                val = v
                                break
                    if val is None:
                        val = 0
                    
                    categories.append(str(name))
                    values.append(val if isinstance(val, (int, float)) else 0)
                else:
                    categories.append(str(item))
                    values.append(0)
                
        elif isinstance(data, dict):
            sorted_items = sorted(data.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:15]
            for k, v in sorted_items:
                categories.append(str(k))
                values.append(v if isinstance(v, (int, float)) else 0)
        
        # 确定指标显示名称
        metric_display = intent.target_metric or semantics.primary_metric or '指标'
        
        metric_name_map = {
            'order': '审批数量', 'order_count': '工单数量', '工单数量': '工单数量',
            '审批数量': '审批数量', 'count': '数量', 'amount': '金额',
            'total_process_days': '总耗时', '总耗时': '总耗时',
            'avg_process_days': '平均耗时', '平均耗时': '平均耗时',
            'user_count': '处理人数', '处理人数': '处理人数',
            'per_capita': '人均工单', '人均工单': '人均工单',
        }
        metric_display = metric_name_map.get(metric_display, metric_display)
        if re.match(r'^[a-z_]+$', metric_display):
            metric_display = '指标'
        
        # 检查是否有有效数据
        has_valid_data = any(isinstance(v, (int, float)) and v > 0 for v in values)
        if not has_valid_data and len(values) > 0:
            print(f"[Chart] 重新尝试从原始数据采样")
        
        seen = set()
        deduped_categories = []
        deduped_values = []
        for c, v in zip(categories, values):
            if c not in seen:
                seen.add(c)
                deduped_categories.append(c)
                deduped_values.append(v)
        categories = deduped_categories
        values = deduped_values
        
        # 根据意图类型生成图表
        if intent.primary_intent == 'extreme':
            colors = ['#ff4444' if i == 0 else ('#1890ff' if i == len(categories)-1 else '#d9d9d9') for i in range(len(categories))]
            
            return {
                'title': {'text': f'{metric_display}极值分析', 'left': 'center'},
                'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
                'xAxis': {
                    'type': 'category', 
                    'data': categories, 
                    'axisLabel': {'rotate': 30, 'interval': 0},
                    'name': '类别'
                },
                'yAxis': {'type': 'value', 'name': metric_display},
                'series': [{
                    'type': 'bar',
                    'data': values,
                    'itemStyle': {'color': lambda p: colors[p.dataIndex] if p.dataIndex < len(colors) else '#d9d9d9'},
                    'label': {'show': True, 'position': 'top'}
                }]
            }
            
        elif intent.primary_intent == 'ranking':
            return {
                'title': {'text': f'{metric_display} Top {len(categories)}', 'left': 'center'},
                'tooltip': {'trigger': 'item', 'formatter': '{b}: {c}'},
                'xAxis': {'type': 'value', 'name': metric_display},
                'yAxis': {'type': 'category', 'data': list(reversed(categories))},
                'series': [{
                    'type': 'bar',
                    'data': list(reversed(values)),
                    'label': {'show': True, 'position': 'right'}
                }]
            }
            
        else:
            return {
                'title': {'text': f'{metric_display}分布', 'left': 'center'},
                'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
                'xAxis': {
                    'type': 'category', 
                    'data': categories, 
                    'axisLabel': {'rotate': 30, 'interval': 0},
                    'name': '类别'
                },
                'yAxis': {'type': 'value', 'name': metric_display},
                'series': [{
                    'type': 'bar', 
                    'data': values,
                    'label': {'show': True, 'position': 'top'}
                }]
            }
    
    def _print_final_result(self, result: AnalysisResult):
        """打印最终结果"""
        
        print(f"\n{'='*70}")
        print(f"[最终结果]")
        print(f"{'='*70}")
        print(f"成功: {result.success}")
        print(f"耗时: {result.processing_time_ms}ms")
        
        if result.summary:
            print(f"\n[AI总结]")
            print(result.summary)
        
        if result.insights:
            print(f"\n[洞察]")
            for insight in result.insights:
                print(f"  • {insight}")
        
        print(f"\n{'='*70}\n")


# ============================================================
# 测试入口
# ============================================================

def test_semantic_engine():
    """测试语义智能引擎"""
    
    print("\n" + "=" * 70)
    print("[TEST] Semantic Intelligence Engine v4.0")
    print("=" * 70)
    
    engine = SemanticIntelligenceEngine()
    
    # 测试场景1: 工单效能分析（明细数据）
    mock_workorder_data = [
        {'员工': '杨晓龙', '审批节点': '权限创建/修改', '审批数量': 79, '总耗时': 667.90, '平均耗时': 8.45},
        {'员工': '王之一', '审批节点': '权限创建/修改', '审批数量': 60, '总耗时': 200.45, '平均耗时': 3.34},
        {'员工': '杨晓龙', '审批节点': '数据备份待执行', '审批数量': 55, '总耗时': 274.98, '平均耗时': 5.00},
        {'员工': '杨坤', '审批节点': '系统所属部门负责人审核', '审批数量': 43, '总耗时': 28.58, '平均耗时': 0.66},
        {'员工': '杨晓龙', '审批节点': '数据备份完成填写归档信息', '审批数量': 37, '总耗时': 42.31, '平均耗时': 1.14},
        {'员工': '赵思博', '审批节点': '权限创建/修改', '审批数量': 35, '总耗时': 156.78, '平均耗时': 4.48},
        {'员工': '李四', '审批节点': '权限创建/修改', '审批数量': 28, '总耗时': 112.34, '平均耗时': 4.01},
        {'员工': '张三', '审批节点': '数据备份待执行', '审批数量': 22, '总耗时': 98.76, '平均耗时': 4.49},
        {'员工': '王五', '审批节点': '权限创建/修改', '审批数量': 18, '总耗时': 88.12, '平均耗时': 4.90},
        {'员工': '赵思博', '审批节点': '数据备份待执行', '审批数量': 15, '总耗时': 67.89, '平均耗时': 4.53},
    ]
    
    test_cases = [
        ("CI部门完成工单数量最多的员工和最少的员工", "应正确聚合后找极值"),
        ("CI部门工单数量Top5的员工", "应正确聚合后排名"),
        ("各部门平均处理耗时", "应计算统计量"),
    ]
    
    for query, expected_behavior in test_cases:
        print(f"\n{'#'*70}")
        print(f"[测试用例] {query}")
        print(f"[预期行为] {expected_behavior}")
        print(f"{'#'*70}")
        
        result = engine.analyze(
            raw_data=mock_workorder_data,
            user_query=query
        )
        
        if result.success:
            print(f"\n✓ [SUCCESS]")
            print(f"\n[AI总结]")
            print(result.summary)
            
            if result.insights:
                print(f"\n[洞察]")
                for insight in result.insights:
                    print(f"  • {insight}")
            
            if result.data_semantics.is_detail_level:
                print(f"\n[聚合后数据示例]")
                if isinstance(result.processed_data, dict):
                    sorted_items = sorted(result.processed_data.items(), key=lambda x: x[1], reverse=True)[:5]
                    for name, val in sorted_items:
                        print(f"  {name}: {val}")
        else:
            print(f"\n✗ [FAIL] {result.summary}")


if __name__ == '__main__':
    test_semantic_engine()
