#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Semantic Layer - 语义层核心引擎

职责：
  1. 加载语义层配置JSON（指标、维度、实体、同义词、典型问题）
  2. 意图分类（限制输出空间到6种枚举类型）
  3. 实体识别与标准化（部门代码、时间表达式）
  4. 同义词展开
  5. 参数槽位填充（填空模式：LLM只做映射，不做自由生成）

架构设计原则：
  - 确定性优先：规则匹配 > 向量检索 > LLM
  - 输出空间受限：意图只有6种，指标从预定义列表选择
  - 可解释：每一步决策都有明确的规则来源
"""

import json
import os
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedIntent:
    """分类后的用户意图（输出空间受限）"""
    intent_type: str           # extreme | ranking | comparison | statistic | trend | list
    confidence: float = 0.0
    target_indicator: str = ""   # 从indicators列表中选择的指标ID或名称
    operation: str = ""          # max | min | count | sum | avg (用于extreme/ranking)
    dimension_primary: str = ""  # 主维度: employee | department | time | all
    dimension_values: Dict = field(default_factory=dict)  # {department: "CI", time_range: "本月"}
    entity_mentions: List[str] = field(default_factory=list)    # 提到的实体值
    time_range: Optional[Dict] = None                       # {start, end, text}
    raw_query: str = ""
    method: str = "rule"        # rule | vector | llm | hybrid


@dataclass
class FilledSlot:
    """填充后的参数槽位"""
    param_name: str            # dept, start_time, end_time, flow_type...
    value: Any
    source: str                 # semantic_rule | llm_fc | user_explicit | default
    confidence: float = 1.0
    normalized_value: Any = None


class SemanticLayer:
    """
    语义层引擎
    
    核心能力：
      1. load(config_path) → 加载语义层JSON
      2. classify_intent(query) → 返回ClassifiedIntent(6选1)
      3. resolve_entities(query) → 提取部门/时间/实体
      4. fill_slots(intent, query) → 填充参数槽位
      5. get_indicator_hint(indicator_name) → 指标元信息
    """

    DEFAULT_CONFIG_PATH = "data/semantic_layer.json"

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config: Dict = {}
        self.loaded = False

        # 索引结构（加载时构建）
        self._indicator_index: Dict[str, Dict] = {}       # name/alias → indicator def
        self._dimension_index: Dict[str, Dict] = {}       # name/alias → dimension def
        self._intent_keywords: Dict[str, List[str]] = {}   # intent_type → keywords
        self._dept_code_map: Dict[str, str] = {}          # name/full_name → code
        self._synonym_map: Dict[str, List[str]] = {}       # canonical → [aliases]
        self._reverse_synonym: Dict[str, str] = {}         # alias → canonical
        self._employee_names: set = set()

        self._load()

    def _load(self):
        """加载并索引语义层配置"""
        cfg_file = Path(self.config_path)
        if not cfg_file.exists():
            logger.warning(f"[SemanticLayer] 配置文件不存在: {self.config_path}")
            return

        try:
            with open(cfg_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            self._build_indexes()
            self.loaded = True

            ind_count = len(self._indicator_index)
            dim_count = len(self._dimension_index)
            intent_count = len(self.config.get('intent_types', {}))
            q_count = len(self.config.get('typical_questions', []))

            logger.info(f"[SemanticLayer] 加载完成 | 指标:{ind_count} 维度:{dim_count} 意图:{intent_count} 典型问题:{q_count}")

            self._load_employee_names()

        except Exception as e:
            logger.error(f"[SemanticLayer] 加载失败: {e}")

    def _build_indexes(self):
        """构建快速查找索引"""

        # ====== 指标索引 ======
        for ind in self.config.get('indicators', []):
            keys = [ind['name']] + ind.get('aliases', [])
            for key in keys:
                self._indicator_index[key.lower()] = ind

        # ====== 维度索引 ======
        for dim in self.config.get('dimensions', []):
            keys = [dim['name']] + dim.get('aliases', [])
            for key in keys:
                self._dimension_index[key.lower()] = dim

        # ====== 意图关键词索引 ======
        intent_types = self.config.get('intent_types', {})
        for itype, idef in intent_types.items():
            self._intent_keywords[itype] = idef.get('keywords', [])

        # ====== 部门代码映射 ======
        entities = self.config.get('entities', {})
        depts = entities.get('departments', {}).get('values', [])
        for d in depts:
            code = d['code']
            self._dept_code_map[code.lower()] = code
            self._dept_code_map[d['full_name'].lower()] = code
            self._dept_code_map[d['short_name'].lower()] = code
            for alias in d.get('aliases', []):
                self._dept_code_map[alias.lower()] = code

        # ====== 同义词映射 ======
        synonyms = self.config.get('synonyms', {})
        for canonical, aliases in synonyms.items():
            self._synonym_map[canonical.lower()] = aliases
            for alias in aliases:
                self._reverse_synonym[alias.lower()] = canonical

    def _load_employee_names(self):
        try:
            import pymssql
            from utils.db_config import get_db_config
            cfg = get_db_config('EAM')
            if not cfg:
                logger.warning("[SemanticLayer] EAM数据库配置未找到，跳过员工名库加载")
                return
            conn = pymssql.connect(**cfg.to_pymssql_kwargs())
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT USRDESC FROM ATUSERS WHERE USRDESC IS NOT NULL AND USRDESC != '' AND USRDESC NOT LIKE '%[a-z]%' AND USRDESC NOT LIKE '%[A-Z]%' AND LEN(USRDESC) >= 2")
            rows = cursor.fetchall()
            self._employee_names = {row[0].strip() for row in rows if row[0] and len(row[0].strip()) >= 2}
            conn.close()
            logger.info(f"[SemanticLayer] 员工名库加载完成 | 共{len(self._employee_names)}人")
        except Exception as e:
            logger.warning(f"[SemanticLayer] 员工名库加载失败(非致命): {e}")

    # ================================================================
    # 公共API
    # ================================================================

    def classify_intent(self, query: str) -> ClassifiedIntent:
        """
        意图分类（输出空间限制为6种枚举类型）
        
        使用关键词规则匹配 + 典型问题相似度，不依赖LLM
        """
        q_lower = query.lower()

        result = ClassifiedIntent(intent_type='unknown', raw_query=query)

        # Step 1: 关键词打分
        scores = {}
        for itype, keywords in self._intent_keywords.items():
            score = 0
            for kw in keywords:
                pattern = kw.replace('\\d+', r'\\d+').replace('*', '.*')
                if re.search(pattern, q_lower):
                    score += 1
                    if itype in ('extreme', 'ranking'):
                        score += 1  # 员工类查询权重更高
            if score > 0:
                scores[itype] = score

        # Step 2: 同义词扩展后二次打分
        for alias, canonical in self._reverse_synonym.items():
            if alias in q_lower:
                for itype, keywords in self._intent_keywords.items():
                    if canonical in keywords or any(c in k for k in keywords for c in [canonical]):
                        scores[itype] = scores.get(itye, 0) + 0.5

        # Step 3: 选择最高分
        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]

            result.intent_type = best_type
            result.confidence = min(best_score / 3.0, 1.0)  # 归一化
            result.method = 'rule'

            # Step 4: 推断操作类型
            if best_type == 'extreme':
                if any(kw in q_lower for kw in ['最少', '最慢', '最低', '最小', '倒数']):
                    result.operation = 'min'
                else:
                    result.operation = 'max'
            elif best_type == 'ranking':
                result.operation = 'desc'  # 默认降序

        else:
            # 无明确关键词匹配 → 默认statistic
            result.intent_type = 'statistic'
            result.confidence = 0.3
            result.method = 'default'

        # Step 5: 识别目标指标
        result.target_indicator = self._detect_target_indicator(q_lower)

        # Step 6: 识别主维度
        result.dimension_primary = self._detect_primary_dimension(q_lower)

        return result

    def _detect_target_indicator(self, query_lower: str) -> str:
        """从查询文本中检测目标指标"""
        indicator_matches = []
        for key, ind_def in self._indicator_index.items():
            if key in query_lower:
                indicator_matches.append((key, ind_def, len(key)))

        if indicator_matches:
            indicator_matches.sort(key=lambda x: x[2], reverse=True)
            return indicator_matches[0][1].get('name', '')

        fuzzy_map = {
            '处理时间': '平均耗时',
            '工单处理时间': '平均耗时',
            '平均工单处理时间': '平均耗时',
            '平均处理': '平均耗时',
            '人均效能': '人均工单',
            '效能': '人均工单',
            '工单数': '工单数量',
            '审批数': '审批数量',
        }
        for fuzzy_key, indicator_name in fuzzy_map.items():
            if fuzzy_key in query_lower:
                return indicator_name

        return ""

    def _detect_primary_dimension(self, query_lower: str) -> str:
        """检测主分析维度"""
        if any(kw in query_lower for kw in ['员工', '谁', '人员', '个人', '每人']):
            return 'employee'
        if any(kw in query_lower for kw in ['部门', '各组', '各科室', '对比']):
            return 'department'
        if any(kw in query_lower for kw in ['时间', '天', '周', '月', '趋势', '变化']):
            return 'time'
        return 'all'

    def resolve_entities(self, query: str, intent: ClassifiedIntent = None) -> Dict:
        """
        实体识别与标准化
        
        Returns:
            {
                'departments': ['CI'],           # 识别到的部门代码列表
                'time_range': {...},             # 解析后的时间范围
                'employees': [],                  # 提到的员工名（原始文本）
                'flow_types': [],               # 流程类型
                'mentioned_entities': {...},     # 所有提到的实体
            }
        """
        q_lower = query.lower()
        result = {
            'departments': [],
            'time_range': None,
            'employees': [],
            'flow_types': [],
            'mentioned_entities': {},
        }

        # ====== 部门识别 ======
        detected_depts = set()
        for text_key, code in self._dept_code_map.items():
            if text_key in q_lower:
                detected_depts.add(code.upper())
                result['mentioned_entities'][text_key] = code

        if detected_depts:
            result['departments'] = list(detected_depts)
            intent.dimension_values['department'] = ','.join(detected_depts) if intent else ''

        # ====== 时间解析（复用NL2APIService的解析器）=====
        from solutions.core.nl2api_service import NL2APIService
        temp_service = object.__new__(NL2APIService)
        time_resolved = temp_service._resolve_time_expression(query)
        if time_resolved.get('resolved'):
            result['time_range'] = time_resolved
            if intent:
                intent.time_range = {
                    'start': time_resolved.get('start_time'),
                    'end': time_resolved.get('end_time'),
                    'text': time_resolved.get('time_range_text'),
                }

        # ====== 员工名称提取 ======
        # 优先使用数据库人名库精确匹配
        if self._employee_names:
            for name in self._employee_names:
                if name in query:
                    if name not in result['employees']:
                        result['employees'].append(name)

        # 正则补充匹配（仅当人名库未匹配到时）
        if not result['employees']:
            emp_patterns = [
                r'(\w{2,4})(?:部门)?(?:完成|处理|操作|审批|经办|批了|做了|办了|提交|发起)',
                r'谁(?:的|完成的|处理的|操作的|审批的|批了)',
            ]
            non_name_words = [
                '哪个', '什么', '哪', '本月', '上月', '本周', '最近', '今年',
                '平均', '最多', '最少', '总共', '合计', '工单', '数量', '耗时',
                '个月', '部门', '效能', '指标', '排名', '对比', '趋势', '统计',
                '时间', '天数', '情况', '处理', '审批', '完成', '操作',
            ]
            for pattern in emp_patterns:
                match = re.search(pattern, query)
                if match and match.group(1):
                    emp_name = match.group(1)
                    if emp_name not in non_name_words and '工单' not in emp_name and '部门' not in emp_name:
                        if not re.match(r'^[A-Z]{2,4}$', emp_name):
                            if emp_name not in result['employees']:
                                result['employees'].append(emp_name)

        if result['employees']:
            if intent:
                intent.dimension_primary = 'employee'
                intent.dimension_values['employee'] = ','.join(result['employees'])

        return result

    def fill_slots(self, intent: ClassifiedIntent, query: str,
                   api_params: List[Dict], entities: Dict = None) -> List[FilledSlot]:
        """
        参数槽位填充（填空模式）
        
        输入:
          - intent: 分类后的意图
          - query: 原始查询
          - api_params: API定义的参数列表 [{name, type, required, enum_values}, ...]
          - entities: 已识别的实体
        
        输出:
          - List[FilledSlot]: 填充好的参数槽位
        """
        entities = entities or {}
        slots = []

        for param_def in api_params:
            pname = param_def.get('name', '')
            ptype = param_def.get('param_type', 'string')

            slot = self._fill_single_slot(
                pname=pname,
                ptype=ptype,
                intent=intent,
                query=query,
                entities=entities,
                enum_values=param_def.get('enum_values'),
            )
            if slot:
                slots.append(slot)

        return slots

    def _fill_single_slot(self, pname: str, ptype: str, intent: ClassifiedIntent,
                           query: str, entities: Dict, enum_values: List = None) -> Optional[FilledSlot]:
        """填充单个参数槽位"""
        pname_lower = pname.lower()
        q_lower = query.lower()

        value = None
        source = 'unknown'

        # ====== 部门参数 ======
        if 'dept' in pname_lower:
            depts = entities.get('departments', [])
            if depts:
                value = ','.join(depts)
                source = 'entity_detected'
            elif intent and intent.dimension_values.get('department'):
                value = intent.dimension_values['department']
                source = 'intent_dimension'

        # ====== 时间参数 ======
        elif 'time' in pname_lower or 'date' in pname_lower:
            tr = entities.get('time_range') or (intent.time_range if intent else None)
            if tr:
                if 'start' in pname_lower and tr.get('start_time'):
                    value = tr['start_time']
                    source = 'time_resolved'
                elif 'end' in pname_lower and tr.get('end_time'):
                    value = tr['end_time']
                    source = 'time_resolved'

        # ====== 流程类型 ======
        elif 'flow' in pname_lower:
            pass  # TODO: 流程类型提取

        # 如果没找到值且不是可选参数
        if value is None:
            return FilledSlot(
                param_name=pname,
                value=None,
                source='not_filled',
                confidence=0.0,
            )

        if value is not None:
            return FilledSlot(
                param_name=pname,
                value=value,
                source=source,
                confidence=0.95 if source.startswith(('entity_', 'time_', 'rule_')) else 0.8,
            )

        return None

    def get_indicator_info(self, name_or_alias: str) -> Optional[Dict]:
        """获取指标的完整元信息"""
        return self._indicator_index.get(name_or_alias.lower())

    def get_dimension_info(self, name_or_alias: str) -> Optional[Dict]:
        """获取维度的完整元信息"""
        return self._dimension_index.get(name_or_alias.lower())

    def find_similar_question(self, query: str, top_k: int = 3) -> List[Dict]:
        """查找最相似的典型问题（用于Few-Shot）"""
        questions = self.config.get('typical_questions', [])

        scored = []
        for q in questions:
            q_text = q.get('query', '')
            similarity = self._text_similarity(query.lower(), q_text.lower())
            if similarity > 0.15:
                scored.append({**q, 'similarity': similarity})

        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """简单的文本相似度（词重叠率）"""
        words_a = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', a))
        words_b = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', b))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'loaded': self.loaded,
            'config_path': self.config_path,
            'indicator_count': len(self._indicator_index),
            'dimension_count': len(self._dimension_index),
            'intent_types': list(self.config.get('intent_types', {}).keys()),
            'typical_questions': len(self.config.get('typical_questions', [])),
            'department_codes': [d['code'] for d in self.config.get('entities', {}).get('departments', {}).get('values', [])],
            'synonym_groups': len(self._synonym_map),
        }

    def validate_department(self, value: str) -> Tuple[bool, str, str]:
        """
        校验部门代码
        
        Returns: (is_valid, normalized_code, error_message)
        """
        if not value:
            return False, '', '部门不能为空'

        upper_val = value.strip().upper()
        valid_codes = {d['code'].upper() for d in
                      self.config.get('entities', {}).get('departments', {}).get('values', [])}

        if upper_val in valid_codes:
            return True, upper_val, ''

        # 模糊匹配
        mapped = self._dept_code_map.get(value.lower(), '').upper()
        if mapped and mapped in valid_codes:
            return True, mapped, f'已自动修正: "{value}" → "{mapped}"'

        return False, upper_val, f'无效的部门代码"{value}", 有效选项: {", ".join(sorted(valid_codes))}'


# 全局单例
_global_semantic_layer: Optional[SemanticLayer] = None


def get_semantic_layer(config_path=None) -> SemanticLayer:
    global _global_semantic_layer
    if _global_semantic_layer is None:
        _global_semantic_layer = SemanticLayer(config_path=config_path)
    return _global_semantic_layer


def reset_semantic_layer():
    global _global_semantic_layer
    _global_semantic_layer = None


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    print("=" * 70)
    print("[TEST] Semantic Layer 引擎测试")
    print("=" * 70)

    sl = SemanticLayer()
    stats = sl.get_stats()
    print(f"\n[统计]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 测试意图分类
    print(f"\n\n[意图分类测试]")
    test_queries = [
        "CI部门完成工单最多的员工",
        "各部门工单对比",
        "OM部门排名",
        "最近一周趋势",
        "列出HR明细",
        "本月汇总",
        "QA和PD谁多",
        "RD谁最快",
    ]

    for q in test_queries:
        intent = sl.classify_intent(q)
        entities = sl.resolve_entities(q, intent)
        similar = sl.find_similar_question(q, top_k=1)

        sim_text = similar[0]['query'] if similar else '(无)'
        sim_score = f"{similar[0]['similarity']:.2f}" if similar else '-'

        print(f"  [{intent.intent_type:10s} op={intent.operation:4s}] {q:35s}")
        print(f"    指标={intent.target_indicator or '?'} 维度={intent.dimension_primary or '?'} "
              f"部门={entities.get('departments', [])} "
              f"时间={(entities.get('time_range') or {}).get('time_range_text') or ''}")
        print(f"    相似问题: {sim_text[:30]}... ({sim_score})")

    # 测试部门校验
    print(f"\n\n[部门校验测试]")
    for val in ['ci', 'OM', 'XX', '质量部', 'qa']:
        ok, norm, msg = sl.validate_department(val)
        icon = '✅' if ok else '❌'
        print(f"  {icon} '{val}' → {norm or '(invalid)'} | {msg}")

    print(f"\n{'='*70}\n")
