#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NL2API Service v1.0 - 自然语言到API转换服务

核心理念：
  "用户说自然语言 → 系统自动选择正确的API并填充参数"

架构流程：
  
  用户: "CI部门工单数量最多的员工"
      ↓
  [Layer 1] 意图理解 (LLM)
      ↓ 理解用户想查什么、关心什么维度
  [Layer 2] API路由 (Function Calling)
      ↓ 从32个预定义API中选择最匹配的
  [Layer 3] 参数提取 (LLM)
      ↓ 从自然语言中提取 dept="CI"
  [Layer 4] 执行引擎
      ↓ 调用帆软API执行验证过的SQL
  [Layer 5] 结果处理
      ↓ 格式化返回 + 可视化配置

关键优势：
  ✅ SQL是预先验证过的，不会出错
  ✅ API有明确的入参出参定义
  ✅ LLM只需要做"理解+选择+提取"，不需要写SQL
  ✅ 新增报表只需注册新API，无需改代码
"""

import json
import os
import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class NL2APIResult:
    """NL2API查询结果"""

    success: bool

    # 意图理解
    user_intent: str = ""
    confidence: float = 0.0

    # API调用信息
    selected_api_id: str = ""
    selected_api_name: str = ""
    api_parameters: Dict[str, Any] = field(default_factory=dict)

    # 原始数据
    raw_data: List[Dict] = field(default_factory=list)
    data_count: int = 0

    # AI分析结果
    summary: str = ""
    insights: List[str] = field(default_factory=list)
    chart_config: Dict = field(default_factory=dict)

    # 元数据
    execution_time: float = 0
    llm_calls: int = 0

    # 参数校验（新增）
    validation_errors: List[Dict] = field(default_factory=list)   # 校验错误列表
    param_retry_count: int = 0                                    # 参数重试次数

    # 语义层分析结果（新增）
    _semantic_intent: Any = None                                   # ClassifiedIntent对象
    _semantic_entities: Dict = field(default_factory=dict)         # 实体识别结果

    # 错误信息（如果失败）
    error: str = ""
    error_stage: str = ""


@dataclass
class ValidationError:
    """单个参数校验错误"""
    field: str              # 参数名
    original_value: Any     # 原始值
    error_type: str         # 错误类型: invalid_enum / invalid_format / missing_required / out_of_range / type_mismatch
    message: str            # 人类可读的错误描述
    suggestion: Any = None  # 建议的正确值


class ParamValidator:
    """
    参数校验器
    
    职责：
    1. 对提取的参数进行业务规则校验
    2. 返回结构化的错误列表
    3. 提供错误提示用于LLM重试
    """

    VALID_DEPARTMENTS = {'CI', 'DS', 'EHS', 'FF', 'FM', 'LG', 'OM', 'PD', 'PM', 'QA', 'QC', 'TM', 'VM'}

    DEPT_FULL_NAMES = {
        'CI': '自控信息部',
        'DS': '原液生产部',
        'EHS': '安全环保部',
        'FF': '制剂生产部',
        'FM': '设备管理部',
        'LG': '物控部',
        'OM': '运行保障部',
        'PD': '采购部',
        'PM': '生产管理办公室',
        'QA': '质量保证部',
        'QC': '质量控制部',
        'TM': '技术部',
        'VM': '验证管理部',
    }

    def __init__(self):
        self.errors: List[ValidationError] = []

    def validate(self, params: Dict[str, Any], api_def) -> Dict:
        """
        校验所有参数
        
        Returns:
            {
                'valid': bool,
                'cleaned_params': Dict,      # 清洗后的参数
                'errors': List[ValidationError],
                'error_summary': str,          # 用于LLM重试的错误摘要
            }
        """
        self.errors = []
        cleaned = {}

        if not api_def or not api_def.parameters:
            return {'valid': True, 'cleaned_params': params, 'errors': [], 'error_summary': ''}

        for param in api_def.parameters:
            value = params.get(param.name)

            if value is None or (isinstance(value, str) and value.strip() == ''):
                if param.required:
                    self._add_error(
                        field=param.name,
                        original_value=value,
                        error_type='missing_required',
                        message=f'必填参数"{param.name}"未提供',
                        suggestion=param.example_value
                    )
                continue

            validated = self._validate_single_field(param, value)
            if validated is not None:
                cleaned[param.name] = validated
            else:
                cleaned[param.name] = value

        error_summary = self._build_error_summary(params, api_def)

        return {
            'valid': len(self.errors) == 0,
            'cleaned_params': cleaned,
            'errors': self.errors,
            'error_summary': error_summary,
        }

    def _validate_single_field(self, param, value: Any) -> Any:
        """校验单个字段，返回清洗后的值或None(表示校验失败)"""
        name_lower = param.name.lower()
        str_val = str(value).strip()

        # ====== 部门枚举校验 ======
        if 'dept' in name_lower or 'department' in name_lower:
            return self._validate_department(param.name, str_val, param.enum_values)

        # ====== 时间格式校验 ======
        if any(kw in name_lower for kw in ['time', 'date', 'start', 'end']):
            return self._validate_datetime(param.name, str_val)

        # ====== 数值类型校验 ======
        if param.param_type in ('number', 'integer'):
            return self._validate_number(param.name, str_val, param.param_type)

        # ====== 字符串通用校验 ======
        return self._validate_string(param.name, str_val, param.enum_values)

    def _validate_department(self, field_name: str, value: str, enum_values: List[str] = None) -> str:
        """部门代码校验：大小写归一化 + 枚举范围检查"""

        upper_val = value.upper().strip()

        if upper_val in self.VALID_DEPARTMENTS:
            return upper_val

        if enum_values:
            valid_set = {v.upper() for v in enum_values}
            if upper_val in valid_set:
                return upper_val

        self._add_error(
            field=field_name,
            original_value=value,
            error_type='invalid_enum',
            message=f'部门代码"{value}"无效，必须是以下之一: {", ".join(sorted(self.VALID_DEPARTMENTS))}',
            suggestion=self._guess_department(value)
        )
        return None

    def _guess_department(self, value: str) -> Optional[str]:
        """模糊匹配部门代码"""
        upper_val = value.upper().strip()

        for dept in self.VALID_DEPARTMENTS:
            if dept in upper_val or upper_val in dept:
                return dept

        full_name_map = {
            '自控': 'CI', '信息': 'CI',
            '原液': 'DS',
            '安环': 'EHS', '安全': 'EHS', '环保': 'EHS',
            '制剂': 'FF',
            '设备': 'FM',
            '物控': 'LG',
            '运行': 'OM', '保障': 'OM',
            '采购': 'PD',
            '生管': 'PM', '生产管理': 'PM',
            '质保': 'QA', '质量保证': 'QA',
            '质控': 'QC', '质量控制': 'QC',
            '技术': 'TM',
            '验证': 'VM',
        }

        for keyword, dept in full_name_map.items():
            if keyword in value:
                return dept

        return 'CI'

    def _validate_datetime(self, field_name: str, value: str) -> str:
        """时间格式校验与标准化"""

        if not value:
            return None

        from datetime import datetime

        # 尝试解析各种格式
        formats_to_try = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%Y年%m月%d日',
        ]

        for fmt in formats_to_try:
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.strftime('%Y-%m-%d %H:%M:%S') if '%H' in fmt else parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue

        # 相对时间转换
        relative_patterns = {
            '今天': lambda: datetime.now().strftime('%Y-%m-%d'),
            '今日': lambda: datetime.now().strftime('%Y-%m-%d'),
            '昨天': lambda: (datetime.now().__class__(datetime.now().year, datetime.now().month, datetime.now().day) - __import__('datetime').timedelta(days=1)).strftime('%Y-%m-%d'),
            '本月': lambda: datetime.now().replace(day=1).strftime('%Y-%m-%d'),
            '这个月': lambda: datetime.now().replace(day=1).strftime('%Y-%m-%d'),
            '上月': lambda: (datetime.now().replace(day=1) - __import__('datetime').timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d'),
        }

        for keyword, resolver in relative_patterns.items():
            if keyword in value:
                try:
                    return resolver()
                except Exception:
                    pass

        # 如果包含数字，尝试宽松匹配
        import re as _re
        date_match = _re.search(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})', value)
        if date_match:
            y, m, d = date_match.groups()
            try:
                return f"{y}-{int(m):02d}-{int(d):02d}"
            except (ValueError, TypeError):
                pass

        self._add_error(
            field=field_name,
            original_value=value,
            error_type='invalid_format',
            message=f'时间格式"{value}"无法识别，请使用 YYYY-MM-DD 或 YYYY-MM-DD HH:mm:ss 格式',
            suggestion=datetime.now().strftime('%Y-%m-%d')
        )
        return None

    def _validate_number(self, field_name: str, value: str, expected_type: str) -> Any:
        """数值类型校验"""
        try:
            if expected_type == 'integer':
                return int(float(value))
            else:
                return float(value)
        except (ValueError, TypeError):
            self._add_error(
                field=field_name,
                original_value=value,
                error_type='type_mismatch',
                message=f'"{value}"不是有效的数值',
                suggestion=None
            )
            return None

    def _validate_string(self, field_name: str, value: str, enum_values: List[str] = None) -> str:
        """字符串通用校验（含枚举约束）"""
        if not value:
            return None

        if enum_values and value not in enum_values:
            enum_str = ', '.join(enum_values[:8])
            self._add_error(
                field=field_name,
                original_value=value,
                error_type='invalid_enum',
                message=f'"{value}"不在允许的选项中，可选: {enum_str}',
                suggestion=enum_values[0] if enum_values else None
            )
            return None

        if len(value) > 200:
            self._add_error(
                field=field_name,
                original_value=value,
                error_type='out_of_range',
                message=f'参数值过长({len(value)}字符)，最大允许200字符',
                suggestion=value[:200]
            )
            return None

        return value

    def _add_error(self, **kwargs):
        self.errors.append(ValidationError(**kwargs))

    def _build_error_summary(self, original_params: Dict, api_def) -> str:
        """构建用于LLM重试的错误摘要"""
        if not self.errors:
            return ''

        parts = [f"## 参数校验发现 {len(self.errors)} 个错误\n"]

        for i, err in enumerate(self.errors, 1):
            suggestion_text = f"，建议改为: {err.suggestion}" if err.suggestion else ""
            parts.append(f"{i}. [{err.field}] {err.message}{suggestion_text}")

        parts.append("\n## 正确示例")
        if api_def and api_def.parameters:
            sample_parts = []
            for p in api_def.parameters[:5]:
                if p.enum_values:
                    sample_parts.append(f'- {p.name}: "{p.enum_values[0]}" (可选: {", ".join(p.enum_values)})')
                elif p.example_value is not None:
                    sample_parts.append(f'- {p.name}: "{p.example_value}"')
                elif 'time' in p.name.lower() or 'date' in p.name.lower():
                    sample_parts.append(f'- {p.name}: "2026-01-01" 或 "今天"/"本月"')
            if sample_parts:
                parts.extend(sample_parts)

        parts.append("\n请根据以上错误修正参数后重新输出JSON。")

        return '\n'.join(parts)


class NL2APIService:
    """
    NL2API 服务主类
    
    实现完整的"自然语言→API调用"链路
    """
    
    def __init__(self):
        # 加载API注册表
        from solutions.core.api_registry import APIRegistry
        self.registry = APIRegistry()

        # 初始化LLM服务（完整实例，支持Function Calling）
        self.llm_service = None
        self.llm_client = None
        self._init_llm()

        # 初始化示例库（向量检索Few-Shot）
        self.example_store = None
        self._init_example_store()

        # 初始化语义层（意图分类+实体识别+参数槽位填充）
        self.semantic_layer = None
        self._init_semantic_layer()

        logger.info("[NL2API Service v1.0] 初始化完成")

    def _init_semantic_layer(self):
        """初始化语义层引擎"""
        try:
            from solutions.core.semantic_layer import get_semantic_layer
            self.semantic_layer = get_semantic_layer()
            if self.semantic_layer.loaded:
                stats = self.semantic_layer.get_stats()
                logger.info(f"[NL2API] 语义层初始化完成 | 指标:{stats['indicator_count']} 维度:{stats['dimension_count']} 意图:{stats['intent_types']}")
            else:
                logger.warning("[NL2API] 语义层配置加载失败")
        except Exception as e:
            logger.warning(f"[NL2API] 语义层初始化异常: {e}")
            self.semantic_layer = None

    def _init_example_store(self):
        """初始化示例库"""
        try:
            from solutions.core.example_store import get_example_store
            self.example_store = get_example_store(registry=self.registry)
            stats = self.example_store.get_stats()
            logger.info(f"[NL2API] 示例库初始化完成 | 示例数: {stats['total']} | 检索器: {stats['retriever_stats']['built']}")
        except Exception as e:
            logger.warning(f"[NL2API] 示例库初始化失败: {e}")
            self.example_store = None

    def _init_llm(self):
        """初始化LLM客户端"""
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            from services.llm_service import LLMService
            self.llm_service = LLMService()
            self.llm_client = getattr(self.llm_service, 'client', None)

            if self.llm_client:
                logger.info("[NL2API] LLM客户端初始化成功 (支持Function Calling)")
            else:
                logger.warning("[NL2API] LLM客户端初始化失败")

        except Exception as e:
            logger.error(f"[NL2API] LLM初始化异常: {e}")
            self.llm_service = None
            self.llm_client = None
    
    def query(self, user_query: str, report_path: str = "") -> NL2APIResult:
        """
        执行NL2API查询（主入口）
        
        Args:
            user_query: 用户自然语言查询
            report_path: 报表路径（可选，用于过滤API）
            
        Returns:
            NL2APIResult 包含完整的结果信息
        """
        
        start_time = time.time()
        result = NL2APIResult(success=False)
        llm_calls = 0
        
        print("\n" + "=" * 70)
        print("=" * 70)
        print("[NL2API Service v1.0] 开始智能查询")
        print(f"  用户查询: \"{user_query}\"")
        if report_path:
            print(f"  报表路径: {report_path}")
        print("=" * 70)
        
        try:
            # ====== Layer 1: 意图理解（规则优先+LLM回退）======
            print("\n[Layer 1] 意图理解...")
            
            if self.semantic_layer and self.semantic_layer.loaded:
                sl_intent_pre = self.semantic_layer.classify_intent(user_query)
                sl_entities_pre = self.semantic_layer.resolve_entities(user_query, sl_intent_pre)
                
                if sl_intent_pre.intent_type and sl_intent_pre.intent_type != 'unknown':
                    intent_result = {
                        'intent': sl_intent_pre.operation or user_query,
                        'confidence': 0.85,
                        'entities': {
                            'department': ','.join(sl_entities_pre.get('departments', [])) if sl_entities_pre.get('departments') else '全部',
                            'metric': sl_intent_pre.target_indicator or '',
                            'time_range': sl_entities_pre.get('time_range', {}).get('time_range_text', '') if sl_entities_pre.get('time_range') else '',
                            'analysis_type': sl_intent_pre.intent_type or 'statistic'
                        },
                        'clarity': 'high',
                        'suggestions': []
                    }
                    result.user_intent = intent_result.get('intent', '')
                    result.confidence = intent_result.get('confidence', 0.0)
                    result._semantic_intent = sl_intent_pre
                    result._semantic_entities = sl_entities_pre
                    print(f"  [规则优先] 语义层直接识别意图: {sl_intent_pre.intent_type}")
                    print(f"  意图: {result.user_intent[:100]}...")
                    print(f"  置信度: {result.confidence:.2f}")
                else:
                    intent_result = self._understand_intent(user_query)
                    result.user_intent = intent_result.get('intent', '')
                    result.confidence = intent_result.get('confidence', 0.0)
                    llm_calls += 1
                    print(f"  [LLM回退] 语义层置信度不足，使用LLM")
                    print(f"  意图: {result.user_intent[:100]}...")
                    print(f"  置信度: {result.confidence:.2f}")
            else:
                intent_result = self._understand_intent(user_query)
                result.user_intent = intent_result.get('intent', '')
                result.confidence = intent_result.get('confidence', 0.0)
                llm_calls += 1
                print(f"  意图: {result.user_intent[:100]}...")
                print(f"  置信度: {result.confidence:.2f}")

            # ====== Layer 1.5: 时间表达式解析（规则优先，不依赖LLM）======
            print("\n[Layer 1.5] 时间表达式解析...")
            time_resolved = self._resolve_time_expression(user_query, intent_result)
            if time_resolved['resolved']:
                print(f"  ✓ 时间解析: {time_resolved['time_range_text']} → {time_resolved.get('start_time', '')} ~ {time_resolved.get('end_time', '')} (方法:{time_resolved['method']})")
            else:
                print(f"  - 未检测到时间表达式，将查询全部时间范围")

            # ====== Layer 1.7: 语义层实体识别（Layer 1已调用，直接复用）======
            if self.semantic_layer and self.semantic_layer.loaded:
                if not result._semantic_intent:
                    print("\n[Layer 1.7] 语义层分析...")
                    sl_intent = self.semantic_layer.classify_intent(user_query)
                    sl_entities = self.semantic_layer.resolve_entities(user_query, sl_intent)
                    result._semantic_intent = sl_intent
                    result._semantic_entities = sl_entities
                
                sl_intent = result._semantic_intent
                sl_entities = result._semantic_entities
                
                print(f"\n[Layer 1.7] 语义层结果复用:")
                print(f"  意图: {sl_intent.intent_type} | 操作: {sl_intent.operation} | "
                      f"指标: {sl_intent.target_indicator or '?'} | 主维度: {sl_intent.dimension_primary or '?'}")
                if sl_entities.get('departments'):
                    print(f"  实体-部门: {sl_entities['departments']}")
                if sl_entities.get('time_range') and sl_entities['time_range'].get('resolved'):
                    tr = sl_entities['time_range']
                    print(f"  实体-时间: {tr.get('time_range_text', '')} ({tr.get('start_time','')}~{tr.get('end_time','')})")

            # ====== Layer 2: API路由（Function Calling）======
            print("\n[Layer 2] API智能路由...")
            route_result = self._route_to_api(user_query, report_path)
            
            if not route_result.get('success'):
                result.error = route_result.get('error', '未找到合适的API')
                result.error_stage = 'route'
                return result
            
            result.selected_api_id = route_result['api_id']
            result.selected_api_name = route_result['api_name']
            llm_calls += 1
            
            print(f"  ✓ 选择API: {result.selected_api_name}")
            print(f"  API ID: {result.selected_api_id}")
            
            # ====== Layer 3: 参数提取（Function Calling增强）======
            print("\n[Layer 3] 参数智能提取...")
            param_result = self._extract_parameters(
                user_query=user_query,
                api_id=result.selected_api_id,
                intent_context=result.user_intent,
                route_result=route_result,
                time_hint=time_resolved if time_resolved.get('resolved') else None
            )
            
            result.api_parameters = param_result.get('parameters', {})
            llm_calls += 1

            # 校验错误信息（新增）
            validation_errors = param_result.get('validation_errors', [])
            retry_count = param_result.get('retry_count', 0)
            if validation_errors:
                result.validation_errors = validation_errors
                result.param_retry_count = retry_count
                print(f"  ⚠️ 参数校验: {len(validation_errors)}个警告 (重试{retry_count}次)")
                for ve in validation_errors:
                    print(f"    - [{ve['field']}] {ve['message']}")
            else:
                print(f"  ✓ 参数校验通过")

            print(f"  ✓ 提取到 {len(result.api_parameters)} 个参数:")
            for pname, pvalue in result.api_parameters.items():
                print(f"    - {pname}: {pvalue}")
            
            # ====== Layer 3.5: 语义层部门注入（如果LLM未提取到dept参数）======
            if result._semantic_entities and 'departments' in result._semantic_entities:
                semantic_depts_detected = result._semantic_entities['departments']
                if semantic_depts_detected and len(semantic_depts_detected) > 0:
                    has_dept_param = any(
                        'dept' in k.lower() or 'department' in k.lower()
                        for k in result.api_parameters.keys()
                    )
                    if not has_dept_param:
                        api_def_tmp = self.registry.get_api(result.selected_api_id)
                        dept_param_name = None
                        if api_def_tmp:
                            for p in api_def_tmp.parameters:
                                if 'dept' in p.name.lower() or 'department' in p.name.lower():
                                    dept_param_name = p.name
                                    break
                        if dept_param_name:
                            dept_value = ','.join(semantic_depts_detected)
                            result.api_parameters[dept_param_name] = dept_value
                            print(f"  [语义层注入] dept参数未提取，自动注入: {dept_param_name}={dept_value}")
                        else:
                            print(f"  [语义层注入] API无dept参数，跳过注入")
            
            # ====== Layer 4: 执行API调用（带降级机制）=====
            print("\n[Layer 4] 执行API调用...")
            # 提取语义层识别到的部门信息，用于后处理
            semantic_depts = None
            if result._semantic_entities and 'departments' in result._semantic_entities:
                semantic_depts = result._semantic_entities['departments']
                print(f"  [语义层部门] 识别到部门: {semantic_depts}")
            
            semantic_employees = None
            if result._semantic_entities and 'employees' in result._semantic_entities:
                semantic_employees = result._semantic_entities['employees']
                if semantic_employees:
                    print(f"  [语义层员工] 识别到员工: {semantic_employees}")
            
            # 【修复Bug2】确保时间参数不被防幻觉检查误删
            # 如果规则解析了时间但api_parameters中没有，强制补回
            if time_resolved.get('resolved'):
                if 'start_time' not in result.api_parameters and time_resolved.get('start_time'):
                    result.api_parameters['start_time'] = time_resolved['start_time']
                    print(f"  [时间修复] 补回start_time: {time_resolved['start_time']}")
                if 'end_time' not in result.api_parameters and time_resolved.get('end_time'):
                    result.api_parameters['end_time'] = time_resolved['end_time']
                    print(f"  [时间修复] 补回end_time: {time_resolved['end_time']}")
            
            exec_result = self._execute_api(
                api_id=result.selected_api_id,
                parameters=result.api_parameters,
                semantic_depts=semantic_depts,
                semantic_employees=semantic_employees,
                resolved_time=time_resolved if time_resolved.get('resolved') else None
            )
            
            if not exec_result.get('success'):
                error_msg = exec_result.get('error', 'API执行失败')
                
                # 【P0降级-1】503错误 → 友好提示
                if '503' in error_msg or 'memory' in error_msg.lower():
                    result.error = (
                        "⚠️ 帆软服务器当前负载较高，请稍后重试。\n"
                        "建议：等待1-2分钟后重新查询"
                    )
                    result.error_stage = 'execute'
                    result.success = False
                    return result
                
                # 【P0降级-2】SQL语法错误 → 先尝试仅保留时间参数，再尝试无参数
                if '400' in error_msg or '语法' in error_msg or '|' in error_msg:
                    print("  [P0降级] SQL语法错误，尝试宽松查询...")
                    
                    time_only_params = {k: v for k, v in result.api_parameters.items()
                                       if 'time' in k.lower() or 'start' in k.lower() or 'end' in k.lower()}
                    
                    if time_only_params:
                        retry_result = self._execute_api(
                            api_id=result.selected_api_id,
                            parameters=time_only_params,
                            semantic_depts=semantic_depts,
                            semantic_employees=semantic_employees,
                            resolved_time=time_resolved if time_resolved.get('resolved') else None
                        )
                        if retry_result.get('success') and retry_result.get('data'):
                            exec_result = retry_result
                            result.api_parameters = time_only_params
                            print(f"  [P0降级成功] 仅时间参数查询返回 {len(retry_result['data'])} 条数据")
                    
                    if exec_result is None or not exec_result.get('success') or not exec_result.get('data'):
                        retry_result = self._execute_api(
                            api_id=result.selected_api_id,
                            parameters={},
                            semantic_depts=semantic_depts,
                            semantic_employees=semantic_employees,
                            resolved_time=None
                        )
                        if retry_result.get('success') and retry_result.get('data'):
                            exec_result = retry_result
                            result.api_parameters = {}
                            print(f"  [P0降级成功] 无参数查询返回 {len(retry_result['data'])} 条数据")
                        else:
                            result.error = f"❌ 数据查询异常: {error_msg[:200]}"
                            result.error_stage = 'execute'
                            return result
            
            # 检查是否返回了数据
            result.raw_data = exec_result.get('data', [])
            result.data_count = len(result.raw_data)
            
            # 【P0降级-3】空数据 → 逐步放宽条件重试
            if result.data_count == 0 and result.api_parameters:
                print(f"  [P0警告] 返回{result.data_count}条数据，尝试宽松查询...")
                
                # 第1轮: 移除非时间参数（保留时间参数，去掉可能错误的flow_type等）
                time_only_params = {k: v for k, v in result.api_parameters.items()
                                   if 'time' in k.lower() or 'start' in k.lower() or 'end' in k.lower()}
                
                retry_success = False
                if time_only_params and time_only_params != result.api_parameters:
                    retry_result = self._execute_api(
                        api_id=result.selected_api_id,
                        parameters=time_only_params,
                        semantic_depts=semantic_depts,
                        semantic_employees=semantic_employees,
                        resolved_time=time_resolved if time_resolved.get('resolved') else None
                    )
                    if retry_result.get('success') and retry_result.get('data') and len(retry_result['data']) > 0:
                        result.raw_data = retry_result['data']
                        result.data_count = len(result.raw_data)
                        result.api_parameters = time_only_params
                        retry_success = True
                        print(f"  [P0降级成功] 仅时间参数查询返回 {result.data_count} 条数据")
                
                # 第2轮: 只保留部门参数
                if not retry_success:
                    dept_only_params = {k: v for k, v in result.api_parameters.items()
                                       if 'dept' in k.lower() or 'department' in k.lower()}
                    if dept_only_params and dept_only_params != result.api_parameters:
                        retry_result = self._execute_api(
                            api_id=result.selected_api_id,
                            parameters=dept_only_params,
                            semantic_depts=semantic_depts,
                            semantic_employees=semantic_employees,
                            resolved_time=None
                        )
                        if retry_result.get('success') and retry_result.get('data') and len(retry_result['data']) > 0:
                            result.raw_data = retry_result['data']
                            result.data_count = len(result.raw_data)
                            result.api_parameters = dept_only_params
                            retry_success = True
                            print(f"  [P0降级成功] 仅部门参数查询返回 {result.data_count} 条数据")
                
                # 第3轮: 无参数查询
                if not retry_success:
                    retry_result = self._execute_api(
                        api_id=result.selected_api_id,
                        parameters={},
                        semantic_depts=semantic_depts,
                        semantic_employees=semantic_employees,
                        resolved_time=None
                    )
                    if retry_result.get('success') and retry_result.get('data') and len(retry_result['data']) > 0:
                        result.raw_data = retry_result['data']
                        result.data_count = len(result.raw_data)
                        result.api_parameters = {}
                        print(f"  [P0降级成功] 无参数查询返回 {result.data_count} 条数据")
            
            # 最终检查
            if result.data_count == 0:
                print(f"  ⚠️ 最终仍返回{result.data_count}条数据")
                result.error = f"ℹ️ 查询完成但未找到匹配数据\n\n建议：尝试调整查询条件或扩大范围"
                result.error_stage = 'execute'
                result.success = False  # 空数据也算失败，但不是系统错误
                return result
            else:
                print(f"  ✓ 获取 {result.data_count} 条记录")
            
            # ====== Layer 5: 结果分析与格式化 ======
            print("\n[Layer 5] 结果智能分析...")
            
            # 【双重保险】确保数据按语义层识别的部门过滤
            # 但如果SQL层已经通过dept参数过滤了，跳过双重保险（避免数据无部门字段导致误过滤）
            final_data = result.raw_data
            dept_in_sql_params = any(
                'dept' in k.lower() or 'department' in k.lower()
                for k in result.api_parameters.keys()
            )
            if semantic_depts and len(semantic_depts) > 0 and not dept_in_sql_params:
                print(f"  [双重保险] 按语义层部门重新过滤: {semantic_depts}")
                api_def = self.registry.get_api(result.selected_api_id)
                final_data = self._filter_by_department(final_data, ','.join(semantic_depts), api_def)
                result.raw_data = final_data
                result.data_count = len(final_data)
                print(f"  [双重保险] 过滤后: {result.data_count} 条")
            elif dept_in_sql_params:
                print(f"  [双重保险] 跳过: dept已在SQL层处理")
            
            # 【双重保险】确保数据按语义层识别的员工过滤
            if semantic_employees and len(semantic_employees) > 0 and final_data:
                emp_fields = ['USRDESC', 'CREATEDBY', '员工', '操作人', '审批人']
                filtered_by_emp = []
                for row in final_data:
                    matched = False
                    for field in emp_fields:
                        if field in row and row[field]:
                            val = str(row[field]).strip()
                            for emp_name in semantic_employees:
                                if emp_name in val or val in emp_name:
                                    matched = True
                                    break
                        if matched:
                            break
                    if matched:
                        filtered_by_emp.append(row)
                
                if filtered_by_emp:
                    final_data = filtered_by_emp
                    result.raw_data = final_data
                    result.data_count = len(final_data)
                    print(f"  [双重保险] 按语义层员工过滤: {semantic_employees} → {result.data_count} 条")
                else:
                    print(f"  [双重保险] 员工过滤未匹配，保留全部数据")
            
            # 使用语义引擎分析数据
            if not hasattr(self, '_semantic_engine') or self._semantic_engine is None:
                from solutions.core.semantic_intelligence_engine import SemanticIntelligenceEngine
                self._semantic_engine = SemanticIntelligenceEngine()
            
            pre_parsed = None
            if result._semantic_intent:
                pre_parsed = {
                    'analysis_type': result._semantic_intent.intent_type,
                    'metric': result._semantic_intent.target_indicator or '',
                    'dimension': result._semantic_intent.dimension_primary or '',
                }
            elif intent_result:
                entities = intent_result.get('entities', {})
                pre_parsed = {
                    'analysis_type': entities.get('analysis_type', ''),
                    'metric': entities.get('metric', ''),
                }
            
            analyzed = self._semantic_engine.analyze(
                raw_data=result.raw_data,
                user_query=user_query,
                pre_parsed_intent=pre_parsed
            )
            
            if analyzed.success:
                result.summary = analyzed.summary
                result.insights = analyzed.insights
                result.chart_config = analyzed.chart_config

                # 【修复Bug1】将语义引擎处理后的聚合数据写回raw_data
                # 对于extreme/ranking等需要聚合的查询，processed_data才是正确的结果
                if analyzed.result_records and len(analyzed.result_records) > 0:
                    intent_type = pre_parsed.get('analysis_type', '') if pre_parsed else ''
                    if intent_type in ('extreme', 'ranking'):
                        result.raw_data = analyzed.result_records
                        result.data_count = len(analyzed.result_records)
                        print(f"  [聚合修复] 使用语义引擎聚合结果替换原始数据: {result.data_count} 条")

                print(f"  ✓ AI总结生成完成")
                print(f"  ✓ 图表配置已生成")
            
            # ====== 完成 ======
            result.success = True
            result.execution_time = time.time() - start_time
            result.llm_calls = llm_calls

            # 自动积累成功查询到示例库（Few-Shot自增长）
            if self.example_store and result.data_count > 0:
                try:
                    self.example_store.add_from_result(user_query, result)
                except Exception as e:
                    logger.debug(f"[Few-Shot] 示例积累异常: {e}")
            
            print(f"\n{'='*70}")
            print(f"[NL2API] 查询完成! | 耗时: {result.execution_time:.2f}s | LLM调用: {llm_calls}次")
            print(f"{'='*70}\n")
            
            return result
            
        except Exception as e:
            logger.error(f"[NL2API] 查询失败: {e}", exc_info=True)
            
            result.error = str(e)
            result.error_stage = 'unknown'
            result.execution_time = time.time() - start_time
            result.llm_calls = llm_calls
            
            return result

    def _resolve_time_expression(self, query: str, intent_result: Dict = None) -> Dict:
        """
        从查询文本中解析时间表达式，转换为标准日期 (v2.0)
        
        支持的格式:
          - 绝对时间: "2026-01-15", "2026年5月", "5月1日"
          - 相对时间: "今天", "昨天", "前天", "本月", "上月", "本周", "上周", "上上周"
          - 范围表达: "昨天到今天", "5月1日到5月20日", "前天到今天"
          - 模糊时间: "最近N天/周/月", "前N天", "近N天", "过去N天"
          - 月份/季度: "3月份", "第一季度", "上月", "去年"
          - 特殊: "月初", "月末", "年初"
        """
        import re
        from datetime import datetime, timedelta
        now = datetime.now()
        today = now.date()
        result = {'start_time': None, 'end_time': None, 'time_range_text': '', 'resolved': False, 'method': 'rule'}

        # ====== 规则0: 范围表达 "A到B" (最高优先级) ======
        range_to_match = re.search(
            r'(\d{4}年\d{1,2}月\d{1,2}[日号]?|\d{1,2}月\d{1,2}[日号]?|昨天|前天|大前天|今天|本周|上周|上上周|月初|月末|年初)'
            r'\s*[到至~\-]\s*'
            r'(\d{4}年\d{1,2}月\d{1,2}[日号]?|\d{1,2}月\d{1,2}[日号]?|昨天|前天|大前天|今天|本周|上周|上上周|月初|月末|年初|现在|今天为止|为止)',
            query
        )
        if range_to_match:
            start_expr = range_to_match.group(1)
            end_expr = range_to_match.group(2)
            start_dt = self._parse_single_time_expr(start_expr, now, today)
            end_dt = self._parse_single_time_expr(end_expr, now, today)
            if start_dt and end_dt:
                result['start_time'] = start_dt.strftime('%Y-%m-%d')
                result['end_time'] = end_dt.strftime('%Y-%m-%d')
                result['time_range_text'] = f'{start_expr}到{end_expr}'
                result['resolved'] = True
                return result

        # ====== 规则1: 绝对日期 "2026-01-15" ======
        abs_date_match = re.search(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})', query)
        if abs_date_match:
            y, m, d = int(abs_date_match.group(1)), int(abs_date_match.group(2)), int(abs_date_match.group(3))
            try:
                date_val = datetime(y, m, d)
                result['start_time'] = date_val.strftime('%Y-%m-%d')
                result['end_time'] = date_val.strftime('%Y-%m-%d')
                result['time_range_text'] = result['start_time']
                result['resolved'] = True
                return result
            except ValueError:
                pass

        # ====== 规则1.5: 年月 "2026年5月" / 月份 "3月份" / 季度 "第一季度" ======
        year_month_match = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月', query)
        if year_month_match:
            y, m = int(year_month_match.group(1)), int(year_month_match.group(2))
            try:
                start = datetime(y, m, 1)
                if m == 12:
                    end = datetime(y, 12, 31)
                else:
                    end = datetime(y, m + 1, 1) - timedelta(days=1)
                result['start_time'] = start.strftime('%Y-%m-%d')
                result['end_time'] = end.strftime('%Y-%m-%d')
                result['time_range_text'] = f'{y}年{m}月'
                result['resolved'] = True
                return result
            except ValueError:
                pass

        month_only_match = re.search(r'(\d{1,2})\s*月份?', query)
        if month_only_match:
            m = int(month_only_match.group(1))
            if 1 <= m <= 12:
                y = now.year
                if m > now.month:
                    y -= 1
                try:
                    start = datetime(y, m, 1)
                    if m == 12:
                        end = datetime(y, 12, 31)
                    else:
                        end = datetime(y, m + 1, 1) - timedelta(days=1)
                    result['start_time'] = start.strftime('%Y-%m-%d')
                    result['end_time'] = end.strftime('%Y-%m-%d')
                    result['time_range_text'] = f'{m}月份'
                    result['resolved'] = True
                    return result
                except ValueError:
                    pass

        quarter_match = re.search(r'(第|今年)?([一二三四1-4])\s*季度', query)
        if quarter_match:
            q_map = {'一': 1, '二': 2, '三': 3, '四': 4, '1': 1, '2': 2, '3': 3, '4': 4}
            q_num = q_map.get(quarter_match.group(2))
            if q_num:
                y = now.year
                start_month = (q_num - 1) * 3 + 1
                end_month = q_num * 3
                try:
                    start = datetime(y, start_month, 1)
                    end = datetime(y, end_month + 1, 1) - timedelta(days=1) if end_month < 12 else datetime(y, 12, 31)
                    result['start_time'] = start.strftime('%Y-%m-%d')
                    result['end_time'] = end.strftime('%Y-%m-%d')
                    result['time_range_text'] = f'第{q_num}季度'
                    result['resolved'] = True
                    return result
                except ValueError:
                    pass

        # ====== 规则2: 月级相对时间 ======
        month_patterns = [
            ('本月|这个月|今月', lambda: (
                now.replace(day=1).strftime('%Y-%m-%d'),
                now.strftime('%Y-%m-%d'),
                '本月'
            )),
            ('上月|上个月', lambda: (
                (now.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d'),
                (now.replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d'),
                '上月'
            )),
            ('今年', lambda: (
                now.replace(month=1, day=1).strftime('%Y-%m-%d'),
                now.strftime('%Y-%m-%d'),
                '今年'
            )),
            ('去年', lambda: (
                now.replace(year=now.year-1, month=1, day=1).strftime('%Y-%m-%d'),
                now.replace(year=now.year-1, month=12, day=31).strftime('%Y-%m-%d'),
                '去年'
            )),
        ]

        for pattern, resolver in month_patterns:
            if re.search(pattern, query):
                s, e, text = resolver()
                result['start_time'] = s
                result['end_time'] = e
                result['time_range_text'] = text
                result['resolved'] = True
                return result

        # ====== 规则3: 日级相对时间 ======
        day_patterns = [
            ('今天|今日', lambda: (now.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d'), '今天')),
            ('昨天', lambda: ((now - timedelta(days=1)).strftime('%Y-%m-%d'),) * 2 + ('昨天',)),
            ('前天', lambda: ((now - timedelta(days=2)).strftime('%Y-%m-%d'),) * 2 + ('前天',)),
            ('大前天', lambda: ((now - timedelta(days=3)).strftime('%Y-%m-%d'),) * 2 + ('大前天',)),
            ('月初|本月月初', lambda: (now.replace(day=1).strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d'), '月初')),
            ('年初|今年年初', lambda: (now.replace(month=1, day=1).strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d'), '年初')),
        ]
        for pattern, resolver in day_patterns:
            if re.search(pattern, query):
                s, e, text = resolver()
                result['start_time'] = s
                result['end_time'] = e
                result['time_range_text'] = text
                result['resolved'] = True
                return result

        # ====== 规则4: 周级相对时间 ======
        def get_week_range(week_offset=0):
            weekday = today.weekday()
            monday = today - timedelta(days=weekday + 7 * week_offset)
            sunday = monday + timedelta(days=6)
            if week_offset == 0:
                end = min(sunday, today)
                return monday.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), '本周'
            elif week_offset == 1:
                return monday.strftime('%Y-%m-%d'), sunday.strftime('%Y-%m-%d'), '上周'
            else:
                return monday.strftime('%Y-%m-%d'), sunday.strftime('%Y-%m-%d'), f'前{week_offset}周'

        if '上上周' in query or '大上周' in query:
            s, e, t = get_week_range(2)
            result.update({'start_time': s, 'end_time': e, 'time_range_text': '上上周', 'resolved': True})
            return result

        if '上周' in query:
            s, e, t = get_week_range(1)
            result.update({'start_time': s, 'end_time': e, 'time_range_text': t, 'resolved': True})
            return result

        if '本周' in query or '这周' in query:
            s, e, t = get_week_range(0)
            result.update({'start_time': s, 'end_time': e, 'time_range_text': t, 'resolved': True})
            return result

        # ====== 规则5: 最近N天/N周/N月 ======
        _cn_num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                       '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                       '两': 2, '半': 0.5}

        range_match = re.search(r'(最近|近|过去|前)(\d+)\s*(天|日|周|个月?|月)', query)
        if not range_match:
            range_match = re.search(r'(最近|近|过去|前)([一二三四五六七八九十两半]+)\s*(天|日|周|个月?|月)', query)

        if range_match:
            num_str = range_match.group(2)
            unit = range_match.group(3)

            if num_str in _cn_num_map:
                num = _cn_num_map[num_str]
            else:
                try:
                    num = int(num_str)
                except ValueError:
                    num = 1

            if unit in ('周',):
                days = num * 7
            elif unit in ('个月', '月'):
                days = int(num * 30)
            else:
                days = int(num)

            start = (now - timedelta(days=days)).strftime('%Y-%m-%d')
            result['start_time'] = start
            result['end_time'] = now.strftime('%Y-%m-%d')
            result['time_range_text'] = f'最近{num_str}{unit}'
            result['resolved'] = True
            return result

        # 特殊: "最近"/"近" 后面没有数字 → 默认7天
        if re.search(r'^最近$|^近$', query) or re.search(r'(最近|近)\s*(数据|情况|工单|统计)', query):
            start = (now - timedelta(days=7)).strftime('%Y-%m-%d')
            result['start_time'] = start
            result['end_time'] = now.strftime('%Y-%m-%d')
            result['time_range_text'] = '最近'
            result['resolved'] = True
            return result

        # ====== 规则6: 月日格式 "5月1日" ======
        md_match = re.search(r'(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]', query)
        if md_match:
            m, d = int(md_match.group(1)), int(md_match.group(2))
            try:
                y = now.year
                date_val = datetime(y, m, d)
                if date_val.date() > today:
                    y -= 1
                    date_val = datetime(y, m, d)
                result['start_time'] = date_val.strftime('%Y-%m-%d')
                result['end_time'] = date_val.strftime('%Y-%m-%d')
                result['time_range_text'] = f'{m}月{d}日'
                result['resolved'] = True
                return result
            except ValueError:
                pass

        # ====== 规则7: 从intent_result中提取（LLM已识别）======
        if intent_result:
            entities = intent_result.get('entities', {})
            time_range = entities.get('time_range', '')
            if time_range and time_range not in ('', '全部', '无'):
                result['time_range_text'] = time_range
                result['method'] = 'intent'
                sub_result = self._resolve_time_expression(time_range)
                if sub_result['resolved']:
                    result.update(sub_result)
                    result['method'] = 'intent+rule'

        return result

    def _parse_single_time_expr(self, expr: str, now: datetime, today) -> datetime:
        """解析单个时间表达式为datetime对象"""
        import re
        from datetime import datetime, timedelta
        
        expr = expr.strip()
        
        if expr in ('今天', '今日', '现在', '为止', '今天为止'):
            return now
        elif expr == '昨天':
            return now - timedelta(days=1)
        elif expr == '前天':
            return now - timedelta(days=2)
        elif expr == '大前天':
            return now - timedelta(days=3)
        elif expr == '月初':
            return now.replace(day=1)
        elif expr == '年初':
            return now.replace(month=1, day=1)
        elif expr == '月末':
            if now.month == 12:
                return now.replace(month=12, day=31)
            return datetime(now.year, now.month + 1, 1) - timedelta(days=1)
        elif expr == '本周':
            weekday = today.weekday()
            return now - timedelta(days=weekday)
        elif expr == '上周':
            weekday = today.weekday()
            return now - timedelta(days=weekday + 7)
        elif expr == '上上周':
            weekday = today.weekday()
            return now - timedelta(days=weekday + 14)
        
        ymd_match = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?', expr)
        if ymd_match:
            try:
                return datetime(int(ymd_match.group(1)), int(ymd_match.group(2)), int(ymd_match.group(3)))
            except ValueError:
                pass
        
        md_match = re.search(r'(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?', expr)
        if md_match:
            try:
                y = now.year
                dt = datetime(y, int(md_match.group(1)), int(md_match.group(2)))
                if dt.date() > today:
                    y -= 1
                    dt = datetime(y, int(md_match.group(1)), int(md_match.group(2)))
                return dt
            except ValueError:
                pass
        
        return None

    def _understand_intent(self, query: str) -> Dict:
        """
        Layer 1: 意图理解
        
        使用LLM深度理解用户的真实需求
        """
        
        prompt = f"""你是一个数据分析专家。请分析用户的查询意图。

## 用户查询
"{query}"

## 已知实体枚举（必须从以下选项中匹配，不要自创）
- 部门代码: CI, DS, EHS, FF, FM, LG, OM, PD, PM, QA, QC, TM, VM（仅此13个，大小写不敏感）
- 分析类型枚举:
  * extreme(极值): "最多/最少/最快/最慢/最大/最小/最高/最低/最好/最差"
  * ranking(排名): "排名/前N名/第几名/Top N/排行/谁第一"
  * comparison(对比): "对比/A比B/哪个更/差异/比较/vs"
  * statistic(统计): "总共/多少/平均/总计/汇总/分布"
  * trend(趋势): "走势/趋势/变化/增长/下降/近N天/近N周/近N月"
  * list(列表): "列出/所有/全部/有哪些/明细"

## Few-Shot 示例（参考这些示例的格式输出）

输入: "CI部门完成工单数量最多的员工"
→ intent: 查询CI部门中工单审批数量最多的员工
→ entities: {{department:"CI", metric:"审批数量", analysis_type:"extreme"}}

输入: "各部门工单数量对比"
→ intent: 对比各部门的工单处理数量
→ entities: {{department:"全部", metric:"工单数量", analysis_type:"comparison"}}

输入: "最近一周的工单情况"
→ intent: 统计最近一周内的工单数据概况
→ entities: {{department:"全部", time_range:"最近一周", analysis_type:"statistic"}}

输入: "OM部门的员工效能排名"
→ intent: 列出OM部门员工按效能指标排序
→ entities: {{department:"OM", metric:"人均效能", analysis_type:"ranking"}}

## 你的任务
1. 从用户查询中提取department，必须匹配上述9个部门之一（如未提及则填"全部"）
2. 识别analysis_type，必须使用上述6种枚举值之一
3. 提取metric（用户关心的核心指标词）
4. 提取time_range（如有）

## 输出JSON（严格遵守格式）
```json
{{
    "intent": "一句话描述用户想做什么",
    "confidence": 0.0-1.0,
    "entities": {{
        "department": "CI|DS|EHS|FF|FM|LG|OM|PD|PM|QA|QC|TM|VM|全部",
        "metric": "关心的指标",
        "time_range": "时间范围或空",
        "analysis_type": "extreme|ranking|comparison|statistic|trend|list"
    }},
    "clarity": "high|medium|low",
    "suggestions": ["如果意图不清晰，给出建议"]
}}
```"""

        try:
            response = self._call_llm(prompt, "你是数据分析专家，擅长理解用户需求。")
            
            if response:
                print(f"    [LLM响应] {response[:150]}...")
                parsed = self._parse_json_response(response)
                if parsed:
                    return parsed
                else:
                    print(f"    [WARN] JSON解析失败，使用原始响应")
            else:
                print(f"    [WARN] LLM返回空")
                    
        except Exception as e:
            print(f"    [ERROR] 意图理解异常: {e}")
        
        # 回退：规则解析（更健壮）
        return self._rule_based_intent(query)
    
    def _route_to_api(self, query: str, report_path: str = "") -> Dict:
        """
        Layer 2: API智能路由 (Function Calling模式)
        
        使用OpenAI Function Calling标准格式，让LLM从可用API工具中选择最合适的一个。
        
        标准流程:
          1. 构建tools = registry.get_tools_for_routing()
          2. 调用 llm_service.chat(messages, tools=tools)
          3. 从 tool_calls[0] 获取 function_name + arguments
          4. 将 function_name 映射回 api_id
        """

        # 获取可用的API列表（用于索引回退）
        available_apis = []
        if report_path:
            for api_id, api_def in self.registry.apis.items():
                if report_path in api_def.report_name or api_def.report_name in report_path:
                    available_apis.append((api_id, api_def))
        if not available_apis:
            available_apis = list(self.registry.apis.items())
        if not available_apis:
            return {'success': False, 'error': '没有可用的API'}

        # 【P0核心】预计算员工API索引（用于强制纠错）
        import re as _re
        employee_apis = []
        for i, (api_id, api_def) in enumerate(available_apis):
            sql_lower = api_def.sql_template.lower() if api_def.sql_template else ''
            gb_match = _re.search(r'group\s+by\s+(.+?)(?:order\s+by|$)', sql_lower, _re.DOTALL)
            if gb_match and 'usrdesc' in gb_match.group(1).lower():
                employee_apis.append(i)

        # 【P0核心】检测用户查询是否涉及员工维度
        query_lower = query.lower()
        ask_about_employee = any(kw in query_lower for kw in [
            '员工', '谁', '人员', '最多.*人', '最少.*人',
            'top.*人', '排名.*人', '个人', '每人'
        ])
        if not ask_about_employee:
            ask_about_employee = any(kw in query for kw in ['完成工单', '工单最多', '工单最少', '批了', '做了'])
        if not ask_about_employee and hasattr(self, 'semantic_layer') and self.semantic_layer and self.semantic_layer.loaded:
            if self.semantic_layer._employee_names:
                for name in self.semantic_layer._employee_names:
                    if name in query:
                        ask_about_employee = True
                        break

        # ========== Function Calling 路由 ==========
        try:
            # Step 1: 获取FC格式的tools定义
            tools = self.registry.get_tools_for_routing()

            # Step 1.5: 检索Few-Shot相似示例（向量检索增强）
            few_shot_context = ""
            if self.example_store:
                try:
                    similar_examples = self.example_store.search(query, top_k=3, min_score=0.2)
                    if similar_examples:
                        fs_lines = ["## 历史相似查询参考（请参考这些成功案例选择API）"]
                        for i, item in enumerate(similar_examples, 1):
                            ex = item['example']
                            params_str = json.dumps(ex.parameters, ensure_ascii=False) if ex.parameters else "{}"
                            fs_lines.append(
                                f"{i}. 用户问:\"{ex.query}\" → 选API:{ex.api_name} | 参数:{params_str} | 类型:{ex.analysis_type} (相似度{item['score']:.2f})"
                            )
                        few_shot_context = '\n'.join(fs_lines)
                        print(f"    [Few-Shot] 检索到 {len(similar_examples)} 条相似示例")
                except Exception as e:
                    logger.debug(f"[Few-Shot] 检索异常: {e}")

            # Step 2: 构建消息（精简system prompt + Few-Shot上下文）
            base_rules = """你是一个API路由专家。根据用户的自然语言查询，从提供的工具(functions)中选择最匹配的一个来调用。

## 关键规则（优先级从高到低）
0. 【最重要】优先选择API名称与用户查询关键词重叠最多的函数
1. 如果用户询问"员工/谁/人员/个人"相关的问题 → 必须选择按员工分组的API（标签含employee或描述含"按员工分组"）
2. 如果用户询问"部门对比/各部门" → 选择按部门分组的API
3. 只选择1个最匹配的function，不要编造不存在的函数名
4. 将用户查询中能确定的参数值一并填入arguments
5. 用户没有提到的参数不要填（不要猜！）"""

            system_prompt = base_rules + ("\n\n" + few_shot_context if few_shot_context else "")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请为以下查询选择最合适的API:\n\n{query}"}
            ]

            # Step 3: 调用LLM (Function Calling模式)
            print(f"    [FC路由] 发送 {len(tools)} 个工具定义给LLM...")
            fc_result = self.llm_service.chat(
                messages=messages,
                temperature=0.1,
                tools=tools,
                tool_choice="auto"
            )

            if fc_result.get('error'):
                print(f"    [ERROR] FC调用失败: {fc_result['error']}")
                raise Exception(fc_result['error'])

            # Step 4: 解析tool_calls
            tool_calls = fc_result.get('tool_calls')
            if not tool_calls or len(tool_calls) == 0:
                print(f"    [WARN] LLM未返回tool_calls，尝试从content解析...")
                content = fc_result.get('content', '')
                return self._fallback_route_from_content(content, query, available_apis)

            tc = tool_calls[0]
            func_name = tc.get('function_name', '')
            func_args = tc.get('arguments', {})

            print(f"    [FC路由结果] 函数: {func_name}")
            print(f"    [FC路由结果] 参数: {func_args}")

            # Step 5: 将函数名映射回api_id
            selected_id, selected_def = self._map_function_name_to_api(func_name, available_apis)

            if not selected_id:
                print(f"    [WARN] 无法映射函数名 '{func_name}' 到API")
                return self._fallback_keyword_search(query)

            # 【P0强制纠错】如果问员工但选了非员工API，自动纠正
            p0_corrected = False  # 标记是否被P0纠正过
            idx = next((i for i, (aid, _) in enumerate(available_apis) if aid == selected_id), -1)
            if ask_about_employee and idx not in employee_apis and employee_apis:
                print(f"    [⚠️ P0纠错] 用户问'{query}'涉及员工，但LLM选了非员工API!")
                correct_idx = employee_apis[0]
                selected_id, selected_def = available_apis[correct_idx]
                print(f"    [✓ P0纠正] 强制切换为: {selected_def.name} ⭐[按员工分组]")
                func_args = {}
                p0_corrected = True  # 标记
            
            # 【P1名称匹配纠错】检查用户查询中的关键词是否更匹配其他API名称
            # 使用中文n-gram分词(2-4字滑动窗口)而非整句匹配
            # 注意：如果已经被P0纠正过，P1就不再覆盖
            if not p0_corrected:
                def _chinese_ngrams(text, min_len=2, max_len=4):
                    """中文n-gram分词: 滑动窗口生成2-4字片段"""
                    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
                    ngrams = set()
                    for length in range(min_len, min(max_len + 1, len(chinese_chars) + 1)):
                        for i in range(len(chinese_chars) - length + 1):
                            ngrams.add(''.join(chinese_chars[i:i+length]))
                    return ngrams
                
                query_keywords = _chinese_ngrams(query)
                
                if len(query_keywords) >= 2:
                    best_name_match_score = 0
                    best_name_match_api = None
                    
                    for aid, adef in available_apis:
                        api_name_ngrams = _chinese_ngrams(adef.name)
                        overlap = len(query_keywords & api_name_ngrams)
                        
                        if overlap > best_name_match_score:
                            best_name_match_score = overlap
                            best_name_match_api = (aid, adef)
                    
                    # 如果有其他API的名称与查询关键词重叠度更高，且差距明显，则切换
                    if best_name_match_api and best_name_match_api[0] != selected_id:
                        selected_name_ngrams = _chinese_ngrams(selected_def.name)
                        current_overlap = len(query_keywords & selected_name_ngrams)
                        
                        if best_name_match_score > current_overlap + 1:
                            old_name = selected_def.name
                            selected_id, selected_def = best_name_match_api
                            print(f"    [⚠️ P1纠正] 名称匹配: '{old_name}'({current_overlap}词) → '{selected_def.name}'({best_name_match_score}词)")
                            func_args = {}  # 清除可能错误的参数

            print(f"    [✓] FC选择: {selected_def.name}")

            return {
                'success': True,
                'api_id': selected_id,
                'api_name': selected_def.name,
                'reasoning': f'Function Calling选择: {func_name}',
                'confidence': 0.9,
                'parameter_hints': func_args,
                '_fc_function_name': func_name,
                '_fc_arguments': func_args,
            }

        except Exception as e:
            logger.error(f"[Layer 2] FC路由异常: {e}")
            import traceback
            traceback.print_exc()

        # 回退0: 语义层驱动的路由（员工→员工API，部门→部门API）
        if ask_about_employee and employee_apis:
            fallback_idx = employee_apis[0]
            fallback_id, fallback_def = available_apis[fallback_idx]
            print(f"    [回退-语义层] 员工查询 → {fallback_def.name}")
            return {
                'success': True,
                'api_id': fallback_id,
                'api_name': fallback_def.name,
                'reasoning': '语义层员工识别(回退)',
                'confidence': 0.7,
                'parameter_hints': {}
            }

        # 回退1: 关键词搜索
        print(f"    [回退] 使用关键词搜索...")
        search_results = self.registry.search_apis(query, top_k=3)
        if search_results:
            best_id = search_results[0][0]
            best_def = self.registry.get_api(best_id)
            return {
                'success': True,
                'api_id': best_id,
                'api_name': best_def.name if best_def else best_id,
                'reasoning': '基于关键词匹配(回退)',
                'confidence': 0.6,
                'parameter_hints': {}
            }

        return {'success': False, 'error': '无法匹配合适的API'}

    def _map_function_name_to_api(self, func_name: str, available_apis: list) -> tuple:
        """
        将FC返回的函数名映射回api_id

        因为 _sanitize_function_name() 可能有多对一的情况，
        需要精确匹配或模糊匹配
        """

        if not func_name:
            return None, None

        # 精确匹配：遍历所有available_apis，看哪个的sanitized name等于func_name
        for api_id, api_def in available_apis:
            sanitized = self.registry._sanitize_function_name(api_id)
            if sanitized == func_name:
                return api_id, api_def

        # 模糊匹配：包含关系
        for api_id, api_def in available_apis:
            sanitized = self.registry._sanitize_function_name(api_id)
            if sanitized in func_name or func_name in sanitized:
                return api_id, api_def

        # 最后尝试：在全部registry中找
        for api_id, api_def in self.registry.apis.items():
            sanitized = self.registry._sanitize_function_name(api_id)
            if sanitized == func_name:
                return api_id, api_def

        return None, None

    def _fallback_route_from_content(self, content: str, query: str, available_apis: list) -> Dict:
        """当FC未返回tool_calls时，尝试从文本内容中解析"""
        if not content:
            return self._fallback_keyword_search(query)

        print(f"    [回退-内容解析] 尝试从LLM文本响应中提取API选择...")

        try:
            parsed = self._parse_json_response(content)
            if parsed and parsed.get('selected_index'):
                idx = parsed['selected_index'] - 1
                if 0 <= idx < len(available_apis):
                    selected_id, selected_def = available_apis[idx]
                    return {
                        'success': True,
                        'api_id': selected_id,
                        'api_name': selected_def.name,
                        'reasoning': '从文本JSON解析(回退)',
                        'confidence': 0.7,
                        'parameter_hints': {}
                    }
        except Exception as e:
            print(f"    [WARN] 内容解析失败: {e}")

        return self._fallback_keyword_search(query)

    def _fallback_keyword_search(self, query: str) -> Dict:
        """回退方案：基于关键词搜索"""
        search_results = self.registry.search_apis(query, top_k=3)
        if search_results:
            best_id = search_results[0][0]
            best_def = self.registry.get_api(best_id)
            return {
                'success': True,
                'api_id': best_id,
                'api_name': best_def.name if best_def else best_id,
                'reasoning': '基于关键词匹配(最终回退)',
                'confidence': 0.5,
                'parameter_hints': {}
            }
        return {'success': False, 'error': '无法匹配合适的API'}
    
    def _extract_parameters(self, user_query: str, api_id: str,
                           intent_context: str = "", route_result: Dict = None,
                           time_hint: Dict = None) -> Dict:
        """
        Layer 3: 参数智能提取 (Function Calling增强版)

        策略优先级:
          1. [最高优] FC路由阶段已返回的参数 (_fc_arguments)
          1.5 [最高优] 时间表达式解析结果 (time_hint) - 规则解析的时间直接注入
          2. [次优]   使用单个API的FC定义进行精确参数提取
          3. [回退]   传统prompt方式
          4. [最终]   规则匹配
        """

        api_def = self.registry.get_api(api_id)
        if not api_def or not api_def.parameters:
            return {'success': True, 'parameters': {}}

        extracted_params = {}

        # ====== 策略1.5: 注入时间表达式解析结果（规则优先，零LLM调用）======
        if time_hint and time_hint.get('resolved'):
            hint_start = time_hint.get('start_time')
            hint_end = time_hint.get('end_time')
            if hint_start or hint_end:
                for param in api_def.parameters:
                    pname_lower = param.name.lower()
                    if ('start' in pname_lower or 'begin' in pname_lower) and hint_start:
                        extracted_params[param.name] = hint_start
                        print(f"      [时间注入] {param.name}: {hint_start} (来自规则解析)")
                    elif ('end' in pname_lower or 'finish' in pname_lower) and hint_end:
                        extracted_params[param.name] = hint_end
                        print(f"      [时间注入] {param.name}: {hint_end} (来自规则解析)")

        # ====== 策略1: 复用FC路由阶段返回的参数（带防幻觉验证）======
        if route_result and route_result.get('_fc_arguments'):
            fc_args = route_result['_fc_arguments']
            if fc_args and isinstance(fc_args, dict):
                print(f"    [FC参数] 发现路由阶段返回的参数: {list(fc_args.keys())}")
                
                query_lower = user_query.lower()
                
                for param in api_def.parameters:
                    if param.name in fc_args:
                        value = fc_args[param.name]
                        
                        # 【关键】防幻觉验证: 检查用户是否真的提到了这个参数值
                        value_str = str(value).upper().strip() if value else ''
                        should_keep = True
                        
                        # 部门参数: 用户必须提到具体部门名
                        if 'dept' in param.name.lower() or 'department' in param.name.lower():
                            mentioned_depts = [d.upper() for d in [
                                'CI', 'DS', 'EHS', 'FF', 'FM', 'LG', 'OM', 'PD', 'PM', 'QA', 'QC', 'TM', 'VM',
                                'ci', 'ds', 'ehs', 'ff', 'fm', 'lg', 'om', 'pd', 'pm', 'qa', 'qc', 'tm', 'vm'
                            ] if d.lower() in query_lower]
                            if not mentioned_depts:
                                print(f"      ✗ {param.name}={value} → 丢弃! 用户未提及任何部门(不要猜!)")
                                should_keep = False
                            elif value_str not in mentioned_depts:
                                print(f"      ⚠ {param.name}={value} → 用户提到了{mentioned_depts},但FC返回了{value}")
                                should_keep = False
                        
                        # 时间参数: 用户必须提到时间相关词汇
                        elif any(kw in param.name.lower() for kw in ['time', 'start', 'end']):
                            time_keywords = [
                                '本月', '上月', '上个月', '本周', '上周', '上上周', '大上周',
                                '今年', '去年', '最近', '近', '过去', '前',
                                '今天', '昨天', '前天', '大前天', '月初', '年初', '月末',
                                '时间', '期间', '为止', '到现在', '到今天',
                                '月份', '季度', '号',
                                '2024', '2025', '2026', '2027',
                                '1月', '2月', '3月', '4月', '5月', '6月',
                                '7月', '8月', '9月', '10月', '11月', '12月',
                            ]
                            has_time_mention = any(kw in query_lower for kw in time_keywords)
                            if not has_time_mention:
                                print(f"      ✗ {param.name}={value} → 丢弃! 用户未提及时间(不要猜!)")
                                should_keep = False
                        
                        # 状态/类型参数: 用户必须提到相关关键词
                        elif any(kw in param.name.lower() for kw in ['status', 'type', 'flow_type']):
                            query_upper = user_query.upper()
                            if value_str and value_str not in query_upper:
                                print(f"      ⚠ {param.name}={value} → 需要进一步验证")
                        
                        if should_keep:
                            validated = self._validate_and_convert_param(
                                value, param.param_type, param.enum_values
                            )
                            if validated is not None:
                                if param.name not in extracted_params:
                                    extracted_params[param.name] = validated
                                    print(f"      ✓ {param.name}: {validated} (来自FC路由)")
                                else:
                                    print(f"      ⊘ {param.name}: 保留规则注入值 {extracted_params[param.name]}，跳过FC值 {validated}")

                # 如果所有必需参数都已获取，直接返回
                missing_required = [
                    p.name for p in api_def.parameters
                    if p.required and p.name not in extracted_params
                ]
                if not missing_required:
                    print(f"    [FC参数] 所有必需参数已从FC路由获取，跳过LLM提取")
                    return {
                        'success': True,
                        'parameters': extracted_params,
                        'confidence': 0.95,
                        'source': 'fc_routing',
                        'notes': f'参数来自FC路由阶段 ({len(extracted_params)}个)'
                    }

        # ====== 策略2: 单API Function Calling参数提取 ======
        missing_required = [
            p.name for p in api_def.parameters
            if p.required and p.name not in extracted_params
        ]
        if missing_required and self.llm_service:
            try:
                fc_params = self._extract_parameters_via_fc(user_query, api_def, intent_context)
                if fc_params:
                    for k, v in fc_params.items():
                        if k not in extracted_params:
                            validated = self._validate_and_convert_param(
                                v, 
                                next((p.param_type for p in api_def.parameters if p.name == k), 'string'),
                                next((p.enum_values for p in api_def.parameters if p.name == k), None)
                            )
                            if validated is not None:
                                extracted_params[k] = validated
                    
                    # 【防幻觉】策略2返回前检查
                    extracted_params = self._anti_hallucination_check(extracted_params, user_query)
                    
                    return {
                        'success': True,
                        'parameters': extracted_params,
                        'confidence': 0.9,
                        'source': 'fc_extraction',
                        'notes': f'FC提取 + 路由参数合并 ({len(extracted_params)}个)'
                    }
            except Exception as e:
                logger.warning(f"[Layer 3] FC参数提取失败，回退到prompt: {e}")

        # ====== 策略3: 传统prompt方式（保留作为兼容回退）======
        try:
            prompt_params = self._extract_parameters_via_prompt(user_query, api_def, intent_context)
            if prompt_params:
                for k, v in prompt_params.items():
                    if k not in extracted_params:
                        extracted_params[k] = v
                if extracted_params:
                    extracted_params = self._anti_hallucination_check(extracted_params, user_query)
                    return {
                        'success': True,
                        'parameters': extracted_params,
                        'confidence': 0.8,
                        'source': 'prompt_fallback',
                        'notes': f'prompt提取 + 前序合并 ({len(extracted_params)}个)'
                    }
        except Exception as e:
            logger.warning(f"[Layer 3] Prompt参数提取失败: {e}")

        # ====== 策略4: 规则匹配（最终回退）======
        rule_params = self._rule_based_param_extraction(user_query, api_def)
        rule_extracted = rule_params.get('parameters', {})
        for k, v in rule_extracted.items():
            if k not in extracted_params:
                extracted_params[k] = v

        # ====== 全局防幻觉后处理（所有策略提取完成后统一校验）======
        query_lower_for_check = user_query.lower()
        params_to_remove = []
        
        for param_name, param_value in list(extracted_params.items()):
            value_str = str(param_value).upper().strip() if param_value is not None else ''
            should_remove = False
            
            # 部门参数: 用户必须提到具体部门名
            if 'dept' in param_name.lower() or 'department' in param_name.lower():
                mentioned_depts = [d.upper() for d in [
                    'CI', 'DS', 'EHS', 'FF', 'FM', 'LG', 'OM', 'PD', 'PM', 'QA', 'QC', 'TM', 'VM'
                ] if d.lower() in query_lower_for_check]
                if not mentioned_depts:
                    print(f"    [防幻觉] ✗ {param_name}={param_value} → 丢弃! 用户未提及任何部门(不要猜!)")
                    should_remove = True
                elif isinstance(param_value, list):
                    # 如果是列表，只保留用户提到的部门
                    filtered = [d for d in param_value if str(d).upper() in mentioned_depts]
                    if filtered:
                        extracted_params[param_name] = filtered  # 保留列表格式，不管多少个
                        print(f"    [防幻觉] {param_name}: 列表→过滤为{filtered}")
                    else:
                        print(f"    [防幻觉] ✗ {param_name}={param_value} → 丢弃! 不在用户提及的部门中")
                        should_remove = True
                elif value_str not in mentioned_depts:
                    print(f"    [防幻觉] ✗ {param_name}={param_value} → 丢弃! 用户未提及此部门")
                    should_remove = True
            
            # 时间参数: 用户必须提到时间关键词
            elif any(kw in param_name.lower() for kw in ['time', 'start', 'end']):
                time_keywords = ['本月', '上月', '本周', '上周', '今年', '去年',
                                 '最近', '今天', '昨天', '时间', '期间',
                                 '2024', '2025', '2026']
                if not any(kw in query_lower_for_check for kw in time_keywords):
                    print(f"    [防幻觉] ✗ {param_name}={param_value} → 丢弃! 用户未提及时间(不要猜!)")
                    should_remove = True
            
            # flow_category/flow_type 等类型参数
            elif any(kw in param_name.lower() for kw in ['flow_type', 'flow_category', 'status', 'type']):
                if value_str and value_str not in user_query.upper() and value_str != 'ALL':
                    print(f"    [防幻觉] ⚠ {param_name}={param_value} → 可能是幻觉")
            
            if should_remove:
                params_to_remove.append(param_name)
        
        for p in params_to_remove:
            del extracted_params[p]
        
        if params_to_remove:
            print(f"    [防幻觉] 共丢弃 {len(params_to_remove)} 个幻觉参数，剩余: {list(extracted_params.keys())}")

        # ====== 参数校验 + 自动重试 ======
        validator = ParamValidator()
        validation = validator.validate(extracted_params, api_def)

        max_retries = 2
        retry_count = 0

        while not validation['valid'] and retry_count < max_retries and self.llm_service:
            retry_count += 1
            print(f"    [参数校验] 发现 {len(validation['errors'])} 个错误 (第{retry_count}次重试)")

            for err in validation['errors']:
                print(f"      ✗ [{err.field}] {err.message}")

            retry_result = self._retry_params_with_llm(
                user_query=user_query,
                api_def=api_def,
                intent_context=intent_context,
                original_params=extracted_params,
                error_summary=validation['error_summary']
            )

            if retry_result:
                extracted_params.update(retry_result)
                validator = ParamValidator()
                validation = validator.validate(extracted_params, api_def)

                if validation['valid']:
                    print(f"    [参数校验] ✓ 重试成功! 所有参数已通过校验")
                    break
            else:
                print(f"    [参数校验] 重试#{retry_count} 未返回有效参数，停止重试")
                break

        # 构建最终结果
        final_params = validation.get('cleaned_params', extracted_params)
        validation_errors = [
            {
                'field': e.field,
                'original_value': str(e.original_value)[:50],
                'error_type': e.error_type,
                'message': e.message,
                'suggestion': str(e.suggestion) if e.suggestion else None,
            }
            for e in validation.get('errors', [])
        ]

        notes_parts = [f'参数{len(final_params)}个']
        if retry_count > 0:
            notes_parts.append(f'重试{retry_count}次')
        if validation_errors and validation['valid']:
            notes_parts.append(f'{len(validation_errors)}个警告')

        return {
            'success': True,
            'parameters': final_params,
            'confidence': 0.9 if validation['valid'] else 0.6,
            'source': f'validated{"_retry"+str(retry_count) if retry_count > 0 else ""}',
            'notes': ', '.join(notes_parts),
            'validation_errors': validation_errors,
            'validation_valid': validation['valid'],
            'retry_count': retry_count,
        }

    def _retry_params_with_llm(self, user_query: str, api_def, intent_context: str,
                                 original_params: Dict, error_summary: str) -> Optional[Dict]:
        """
        参数校验失败后，让LLM带错误提示重新提取参数
        
        将校验错误作为上下文传给LLM，要求其修正参数
        """
        from solutions.core.api_registry import APIRegistry

        single_tool = {
            "type": "function",
            "function": {
                "name": self.registry._sanitize_function_name(api_def.api_id),
                "description": api_def.description + "（请根据错误提示修正参数）",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                }
            }
        }

        properties = {}
        required = []
        for param in api_def.parameters:
            prop_def = {"type": APIRegistry._map_type_to_json_schema(param.param_type)}
            if param.enum_values:
                prop_def["enum"] = param.enum_values
                prop_def["description"] = f"可选值: {', '.join(param.enum_values)}"
            elif param.description:
                prop_def["description"] = param.description
            else:
                prop_def["description"] = f"{param.name}参数"
            properties[param.name] = prop_def
            if param.required:
                required.append(param.name)

        single_tool["function"]["parameters"]["properties"] = properties
        single_tool["function"]["parameters"]["required"] = required

        retry_prompt = f"""你是一个参数修正专家。上次提取的参数校验失败了，请根据错误提示修正。

## 用户原始查询
"{user_query}"

## 上次提取的参数（有误）
{json.dumps(original_params, ensure_ascii=False, indent=2)}

{error_summary}

## 要求
1. 只修正有错误的参数，正确的保持不变
2. 部门代码必须大写，如 CI, OM, QA
3. 时间格式必须为 YYYY-MM-DD
4. 只返回修正后的参数JSON"""

        messages = [
            {"role": "system", "content": "你是参数修正专家，能根据错误提示精确修正参数值。"},
            {"role": "user", "content": retry_prompt}
        ]

        try:
            print(f"    [重试] 发送错误提示给LLM...")
            fc_result = self.llm_service.chat(
                messages=messages,
                temperature=0.05,
                tools=[single_tool],
                tool_choice={"type": "function", "function": {"name": single_tool["function"]["name"]}}
            )

            if fc_result.get('error'):
                print(f"    [重试] LLM调用失败: {fc_result['error']}")
                return None

            tool_calls = fc_result.get('tool_calls')
            if tool_calls and len(tool_calls) > 0:
                args = tool_calls[0].get('arguments', {})
                if args:
                    print(f"    [重试] LLM修正后参数: {args}")
                    return args

            content = fc_result.get('content', '')
            if content:
                parsed = self._parse_json_response(content)
                if parsed and isinstance(parsed, dict):
                    params = parsed.get('parameters', parsed)
                    if params:
                        print(f"    [重试] 从文本解析到修正参数: {params}")
                        return params

            print(f"    [重试] LLM未返回有效修正")
            return None

        except Exception as e:
            logger.warning(f"[参数重试] 异常: {e}")
            return None

    def _extract_parameters_via_fc(self, user_query: str, api_def, intent_context: str = "") -> Optional[Dict]:
        """
        使用Function Calling模式提取单个API的参数

        将单个API定义转为tools格式，让LLM只关注这个API的参数填充
        """

        # 构建单API的FC工具定义
        single_tool = {
            "type": "function",
            "function": {
                "name": self.registry._sanitize_function_name(api_def.api_id),
                "description": api_def.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                }
            }
        }

        properties = {}
        required = []
        for param in api_def.parameters:
            prop_def = {"type": self.registry._map_type_to_json_schema(param.param_type)}
            if param.enum_values:
                prop_def["enum"] = param.enum_values
                prop_def["description"] = f"可选值: {', '.join(param.enum_values)}"
            elif param.description:
                prop_def["description"] = param.description
            else:
                prop_def["description"] = f"{param.name}参数"
            properties[param.name] = prop_def
            if param.required:
                required.append(param.name)

        single_tool["function"]["parameters"]["properties"] = properties
        single_tool["function"]["parameters"]["required"] = required

        # 构建消息
        system_prompt = """你是一个参数提取专家。根据用户查询，填写函数的参数。

## 关键规则
1. 部门代码必须是: CI, DS, EHS, FF, FM, LG, OM, PD, PM, QA, QC, TM, VM 之一
2. 时间格式: YYYY-MM-DD 或 YYYY-MM-DD HH:mm:ss
3. 用户没有提到的可选参数不要填（不要猜！）
4. 只返回能从查询中明确推断出的参数值"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户查询: \"{user_query}\"\n\n意图上下文: {intent_context}\n\n请提取参数:"}
        ]

        # 调用FC
        print(f"    [FC参数提取] 发送单API工具定义...")
        fc_result = self.llm_service.chat(
            messages=messages,
            temperature=0.1,
            tools=[single_tool],
            tool_choice={"type": "function", "function": {"name": single_tool["function"]["name"]}}
        )

        if fc_result.get('error'):
            print(f"    [WARN] FC参数提取调用失败: {fc_result['error']}")
            return None

        tool_calls = fc_result.get('tool_calls')
        if not tool_calls:
            content = fc_result.get('content', '')
            print(f"    [WARN] FC未返回tool_calls, content: {content[:100]}")
            return None

        args = tool_calls[0].get('arguments', {})
        print(f"    [FC参数提取结果] {args}")
        return args if args else None

    def _extract_parameters_via_prompt(self, user_query: str, api_def, intent_context: str = "") -> Optional[Dict]:
        """传统prompt方式参数提取（兼容回退）"""

        params_info = []
        for param in api_def.parameters:
            pinfo = {
                'name': param.name,
                'type': param.param_type,
                'required': param.required,
                'description': param.description,
            }
            if param.enum_values:
                pinfo['enum_values'] = param.enum_values
            if param.example_value is not None:
                pinfo['example'] = param.example_value
            params_info.append(pinfo)

        params_json = json.dumps(params_info, ensure_ascii=False, indent=2)

        prompt = f"""你是一个参数提取专家。请从用户查询中提取API参数值。

## 用户查询
"{user_query}"

## 意图上下文
{intent_context}

## 需要提取的参数
{params_json}

## 参数值枚举约束（必须从以下选项中选择，禁止自创）
- 部门代码: CI, DS, EHS, FF, FM, LG, OM, PD, PM, QA, QC, TM, VM（仅此13个）
- 时间格式: YYYY-MM-DD 或 YYYY-MM-DD HH:mm:ss

## 规则
1. 用户没提到的可选参数不要填（不要猜！）
2. 部门必须是上述13个代码之一

## 输出JSON
```json
{{
    "success": true,
    "parameters": {{
        "参数名": "提取的值"
    }}
}}
```"""

        try:
            response = self._call_llm(prompt, "你是参数提取专家。")
            if response:
                parsed = self._parse_json_response(response)
                if parsed and parsed.get('success'):
                    validated_params = {}
                    for param in api_def.parameters:
                        value = parsed['parameters'].get(param.name)
                        if value is not None and value != '':
                            v = self._validate_and_convert_param(value, param.param_type, param.enum_values)
                            if v is not None:
                                validated_params[param.name] = v
                    return validated_params if validated_params else None
        except Exception as e:
            logger.warning(f"[Layer 3] Prompt参数提取异常: {e}")

        return None
    
    def _rule_based_intent(self, query: str) -> Dict:
        """规则回退：基于关键词的意图解析"""
        
        q_lower = query.lower()
        
        # 分析类型
        if any(kw in q_lower for kw in ['最多', '最少', '最高', '最低', '最大', '最小']):
            analysis_type = 'extreme'
        elif any(kw in q_lower for kw in ['top', '排行', '排名', '前']):
            analysis_type = 'ranking'
        elif any(kw in q_lower for kw in ['对比', '比较', '差异']):
            analysis_type = 'comparison'
        elif any(kw in q_lower for kw in ['趋势', '变化']):
            analysis_type = 'trend'
        else:
            analysis_type = 'description'
        
        # 提取实体
        entities = {}
        
        dept_match = re.search(r'([A-Z]{2,3})部门|([A-Z]{2,3})', query)
        if dept_match:
            dept = dept_match.group(1) or dept_match.group(2)
            if dept in ['OM', 'CI', 'FM', 'QA', 'PD']:
                entities['department'] = dept
        
        if any(kw in q_lower for kw in ['员工', '人员', '谁']):
            entities['entity_type'] = 'employee'
        
        # 评估清晰度
        clarity = 'high' if len(query) > 10 else ('medium' if len(query) > 5 else 'low')
        
        return {
            'intent': f'{analysis_type}: {query}',
            'confidence': 0.6,
            'entities': entities,
            'analysis_type': analysis_type,
            'clarity': clarity,
            'suggestions': []
        }
    
    def _rule_based_param_extraction(self, query: str, api_def) -> Dict:
        """规则回退：简单的正则参数提取（带防幻觉）"""
        
        parameters = {}
        q_lower = query.lower()
        
        for param in api_def.parameters:
            pname_lower = param.name.lower()
            
            # 部门参数（仅当用户明确提及时才提取，支持多部门）
            if 'dept' in pname_lower:
                # 检查用户是否真的提到了部门相关词
                dept_keywords = ['部门', 'CI', 'DS', 'EHS', 'FF', 'FM', 'LG', 'OM', 'PD', 'PM', 'QA', 'QC', 'TM', 'VM']
                has_dept_mention = any(k in q_lower or k in query.upper() for k in dept_keywords)
                
                if has_dept_mention:
                    # 提取所有出现的部门代码
                    mentioned_depts = []
                    all_depts = ['CI', 'DS', 'EHS', 'FF', 'FM', 'LG', 'OM', 'PD', 'PM', 'QA', 'QC', 'TM', 'VM']
                    for d in all_depts:
                        if d.lower() in q_lower or d in query:
                            mentioned_depts.append(d)
                    
                    if mentioned_depts:
                        if len(mentioned_depts) == 1:
                            parameters[param.name] = mentioned_depts[0]
                        else:
                            parameters[param.name] = mentioned_depts  # 多部门返回列表
            
            # 时间参数（仅当用户明确提及时才提取）
            elif 'time' in pname_lower or 'date' in pname_lower:
                time_keywords = ['本月', '上月', '本周', '上周', '今天', '昨天']
                has_time_mention = any(kw in q_lower or kw in query for kw in time_keywords)
                
                if has_time_mention:
                    if '本月' in q_lower or '这个月' in q_lower:
                        from datetime import date
                        today = date.today()
                        parameters[param.name] = today.replace(day=1).strftime('%Y-%m-%d')
                    elif '今天' in q_lower:
                        from datetime import date
                        parameters[param.name] = date.today().strftime('%Y-%m-%d')
        
        return {'success': True, 'parameters': parameters, 'confidence': 0.5}
    
    def _validate_and_convert_param(self, value: Any, expected_type: str, 
                                    enum_values: List[str] = None) -> Any:
        """验证并转换参数值"""
        
        if enum_values and value in enum_values:
            return value
        
        str_value = str(value).strip()
        
        if expected_type == 'string':
            return str_value if str_value else None
        elif expected_type == 'number':
            try:
                return float(str_value)
            except:
                return None
        elif expected_type == 'integer':
            try:
                return int(float(str_value))
            except:
                return None
        elif expected_type in ['datetime', 'date']:
            # 保持字符串格式，后续SQL会处理
            return str_value if str_value else None
        else:
            return str_value if str_value else None
    
    def _anti_hallucination_check(self, params: Dict, user_query: str) -> Dict:
        """
        全局防幻觉检查: 验证用户是否真的提到了参数值
        
        核心原则: 不要猜! 用户没提到的参数不填
        """
        if not params:
            return params
        
        query_lower = user_query.lower()
        params_to_remove = []
        
        for param_name, param_value in list(params.items()):
            value_str = str(param_value).upper().strip() if param_value is not None else ''
            should_remove = False
            
            if 'dept' in param_name.lower() or 'department' in param_name.lower():
                mentioned_depts = [d.upper() for d in [
                    'CI', 'DS', 'EHS', 'FF', 'FM', 'LG', 'OM', 'PD', 'PM', 'QA', 'QC', 'TM', 'VM'
                ] if d.lower() in query_lower]
                
                if not mentioned_depts:
                    print(f"      [防幻觉] ✗ {param_name}={param_value} → 丢弃(用户未提及部门)")
                    should_remove = True
                elif isinstance(param_value, list):
                    filtered = [d for d in param_value if str(d).upper() in mentioned_depts]
                    if filtered:
                        params[param_name] = filtered[0] if len(filtered) == 1 else filtered
                        print(f"      [防幻觉] {param_name}: 列表→{filtered}")
                    else:
                        should_remove = True
                elif value_str not in mentioned_depts and value_str != '' and value_str != 'ALL':
                    print(f"      [防幻觉] ✗ {param_name}={param_value} → 丢弃(未提及此部门)")
                    should_remove = True
            
            elif any(kw in param_name.lower() for kw in ['time', 'start', 'end']):
                time_keywords = [
                    '本月', '上月', '上个月', '本周', '上周', '上上周', '大上周',
                    '今年', '去年', '最近', '近', '过去', '前',
                    '今天', '昨天', '前天', '大前天', '月初', '年初', '月末',
                    '时间', '期间', '为止', '到现在', '到今天',
                    '月份', '季度', '号',
                    '2024', '2025', '2026', '2027',
                    '1月', '2月', '3月', '4月', '5月', '6月',
                    '7月', '8月', '9月', '10月', '11月', '12月',
                ]
                if not any(kw in query_lower for kw in time_keywords) and value_str:
                    print(f"      [防幻觉] ✗ {param_name}={param_value} → 丢弃(用户未提及时间)")
                    should_remove = True
            
            elif any(kw in param_name.lower() for kw in ['flow_type', 'flow_category', 'status']):
                if value_str and value_str not in user_query.upper() and value_str != 'ALL':
                    print(f"      [防幻觉] ⚠ {param_name}={param_value} → 可能是幻觉")
            
            if should_remove:
                params_to_remove.append(param_name)
        
        for p in params_to_remove:
            del params[p]
        
        if params_to_remove:
            print(f"      [防幻觉] 共丢弃{len(params_to_remove)}个幻觉参数")
        
        return params
    
    def _execute_api(self, api_id: str, parameters: Dict[str, Any], semantic_depts: List[str] = None, semantic_employees: List[str] = None, resolved_time: Dict = None) -> Dict:
        """
        Layer 5.0: 直连SQL Server (pymssql) + Python后处理
        
        架构流程:
          1. 从SQL模板中精确删除所有${if()}块 → 干净的标准SQL
          2. 通过pymssql直连SQL Server执行SQL（只读权限）
          3. 获取全量数据（如492条，不受帆软默认过滤限制）
          4. Python后处理：部门/时间/状态/流程类型过滤
        """
        
        api_def = self.registry.get_api(api_id)
        if not api_def:
            return {'success': False, 'error': f'API不存在: {api_id}'}
        
        sql_template = api_def.sql_template
        connection_name = api_def.connection or 'EAM'
        
        print(f"  [Layer 5.0] 直连数据库模式")
        print(f"  SQL模板长度: {len(sql_template)} 字符")
        print(f"  连接: {connection_name}")
        print(f"  参数: {list(parameters.keys()) if parameters else '无'}")
        
        # ====== 步骤1: 智能处理${if()}块（有值替换/无值删除）======
        final_sql = self._strip_fanruan_if_blocks(sql_template, parameters)
        
        print(f"  清理后SQL长度: {len(final_sql)} 字符")
        
        # ====== 步骤2: 直连SQL Server执行 ======
        try:
            data = self._execute_sql_direct(final_sql, connection_name)
            
            if data is None:
                data = []
            
            raw_count = len(data)
            print(f"  [DB返回] 原始数据量: {raw_count} 条")
            
            if raw_count == 0:
                return {'success': False, 'error': '查询返回空数据', 'sql': final_sql[:200]}
            
            # ====== 步骤3: Python后处理 ======
            # 安全策略: 始终将时间参数传入后处理（作为SQL层${if()}处理的安全网）
            # 部门参数如果已在SQL层处理则跳过Python层过滤

            sql_handled_dept = any(
                'dept' in k.lower() or 'department' in k.lower()
                for k, v in (parameters or {}).items()
                if v is not None and str(v).strip()
            )
            effective_semantic_depts = semantic_depts if not sql_handled_dept else None

            post_params = {}
            for k, v in (parameters or {}).items():
                if v is not None and str(v).strip():
                    if 'time' in k.lower():
                        post_params[k] = v
                    elif k.lower() in ('recfromstatus', 'status', 'flow_type'):
                        post_params[k] = v

            # 【修复Bug2】当resolved_time存在但parameters中无时间参数时，
            # 说明时间参数可能在SQL层被${if()}块处理时丢失了，需要补回
            if resolved_time and resolved_time.get('resolved'):
                if 'start_time' not in post_params and resolved_time.get('start_time'):
                    post_params['start_time'] = resolved_time['start_time']
                    print(f"  [时间安全网] 后处理补回start_time: {resolved_time['start_time']}")
                if 'end_time' not in post_params and resolved_time.get('end_time'):
                    post_params['end_time'] = resolved_time['end_time']
                    print(f"  [时间安全网] 后处理补回end_time: {resolved_time['end_time']}")

            if post_params or effective_semantic_depts:
                processed_data = self._post_process_data(
                    data=data,
                    parameters=post_params,
                    api_def=api_def,
                    sql_template=sql_template,
                    semantic_depts=effective_semantic_depts
                )
            else:
                processed_data = data
            
            processed_count = len(processed_data)
            print(f"  [后处理] {raw_count} → {processed_count} 条")
            
            # ====== 员工名过滤（语义层识别的员工）======
            if semantic_employees and len(semantic_employees) > 0 and processed_data:
                emp_fields = ['USRDESC', 'CREATEDBY', '员工', '操作人', '审批人']
                filtered_by_emp = []
                for row in processed_data:
                    matched = False
                    for field in emp_fields:
                        if field in row and row[field]:
                            val = str(row[field]).strip()
                            for emp_name in semantic_employees:
                                if emp_name in val or val in emp_name:
                                    matched = True
                                    break
                        if matched:
                            break
                    if matched:
                        filtered_by_emp.append(row)
                
                if filtered_by_emp:
                    processed_data = filtered_by_emp
                    processed_count = len(processed_data)
                    print(f"  [员工过滤] {semantic_employees} → {processed_count} 条")
                else:
                    print(f"  [员工过滤] 未匹配到员工 {semantic_employees}，保留全部数据")
            
            return {
                'success': True,
                'data': processed_data,
                'count': processed_count,
                'raw_count': raw_count,
                'sql_executed': f'直连SQL Server(v5.0), {len(final_sql)}字符'
            }
                
        except Exception as e:
            logger.error(f"[Layer 5.0] 执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _execute_sql_direct(self, sql: str, connection_name: str = 'EAM') -> List[Dict]:
        """
        通过pymssql直连SQL Server执行SELECT查询（只读）
        
        安全措施:
          - 只允许SELECT语句
          - 连接超时限制
          - 结果集大小限制(TOP 1000)
        """
        import pymssql
        
        # 1. 先移除SQL注释（防止注释干扰开头检查）
        sql_cleaned = sql
        # 移除单行注释 -- ...
        sql_cleaned = re.sub(r'--.*$', '', sql_cleaned, flags=re.MULTILINE)
        # 移除 /* ... */ 多行注释
        sql_cleaned = re.sub(r'/\*.*?\*/', '', sql_cleaned, flags=re.DOTALL)
        
        # 安全检查：只允许SELECT查询（包括CTE WITH ... SELECT）
        sql_stripped = sql_cleaned.strip()
        sql_upper = sql_stripped.upper()
        
        allowed_starts = ('SELECT', 'WITH')
        if not any(sql_upper.startswith(s) for s in allowed_starts):
            raise ValueError(f"安全限制: 只允许SELECT/CTE查询, 实际开头: {sql_upper[:30]}")
        
        forbidden_keywords = ['INSERT ', 'UPDATE ', 'DELETE ', 'DROP ', 'CREATE ', 
                              'ALTER ', 'TRUNCATE ', 'EXEC ', 'EXECUTE ']
        for kw in forbidden_keywords:
            if kw.upper() in sql_upper:
                raise ValueError(f"安全限制: 不允许{kw}操作")
        
        # 数据库连接配置（从环境变量或配置读取）
        db_config = self._get_db_config(connection_name)
        
        conn = pymssql.connect(
            server=db_config.get('server') or db_config.get('host', 'localhost'),
            port=db_config.get('port', 1433),
            user=db_config.get('user', ''),
            password=db_config.get('password', ''),
            database=db_config.get('database', ''),
            timeout=db_config.get('timeout', 30),
            login_timeout=db_config.get('login_timeout', 15),
            charset='UTF-8',
            as_dict=True
        )
        
        cursor = conn.cursor()
        cursor.execute(sql)
        
        rows = cursor.fetchall()
        
        conn.close()
        
        return rows
    
    def _get_db_config(self, connection_name: str) -> Dict[str, str]:
        """获取数据库连接配置（从.env环境变量读取，不硬编码）"""
        from utils.db_config import get_db_config
        cfg = get_db_config(connection_name)
        if cfg:
            return cfg.to_dict()
        
        return {
            'host': os.getenv('DB_EAM_HOST', 'localhost'),
            'user': os.getenv('DB_EAM_USER', 'readonly'),
            'password': os.getenv('DB_EAM_PASSWORD', ''),
            'database': connection_name,
            'port': int(os.getenv('DB_EAM_PORT', '1433')),
        }
    
    def _strip_fanruan_if_blocks(self, sql: str, params: Dict = None) -> str:
        """
        智能处理${if()}块 (v5.2)
        
        策略:
          - if(len(param)>0, true_part, false_part): 有值→true_part, 无值→false_part
          - if(len(param)==0, true_part, false_part): 有值→false_part, 无值→true_part
          - 支持嵌套if()条件（如时间参数的复杂嵌套结构）
        """
        
        from solutions.core.smart_sql_converter import SmartSQLConverter
        converter = SmartSQLConverter()
        
        params = params or {}
        
        blocks = converter._find_if_blocks(sql)
        
        if not blocks:
            return sql
        
        print(f"    [strip] 发现 {len(blocks)} 个 ${{if()}} 块，智能处理...")
        
        result = sql
        replaced_count = 0
        removed_count = 0
        
        for block in reversed(blocks):
            param_name = block.get('param_name')
            block_content = block.get('content', '')
            
            should_replace = (
                param_name and 
                param_name in params and 
                params[param_name] is not None and 
                str(params[param_name]).strip()
            )
            
            is_zero_condition = '==0' in block_content[:50] or '== 0' in block_content[:50]
            
            if should_replace and not is_zero_condition:
                true_part = converter._extract_true_part(block_content)
                if true_part:
                    param_value = params[param_name]
                    replacement = self._replace_param_in_part(true_part, param_name, param_value)
                    result = result[:block['start']] + replacement + result[block['end']+1:]
                    replaced_count += 1
                    print(f"      [替换] {param_name}={param_value} → {replacement.strip()}")
                else:
                    result = result[:block['start']] + '' + result[block['end']+1:]
                    removed_count += 1
            elif should_replace and is_zero_condition:
                evaluated = self._evaluate_fanruan_if(block_content, params)
                if evaluated:
                    result = result[:block['start']] + evaluated + result[block['end']+1:]
                    replaced_count += 1
                    print(f"      [替换==0] {param_name}有值 → {evaluated.strip()}")
                else:
                    result = result[:block['start']] + '' + result[block['end']+1:]
                    removed_count += 1
                    print(f"      [替换==0] {param_name}有值但评估为空，删除块")
            else:
                if is_zero_condition:
                    evaluated = self._evaluate_fanruan_if(block_content, params)
                    if evaluated:
                        result = result[:block['start']] + evaluated + result[block['end']+1:]
                        replaced_count += 1
                        print(f"      [评估==0] {param_name}无值 → {evaluated.strip()}")
                    else:
                        result = result[:block['start']] + '' + result[block['end']+1:]
                        removed_count += 1
                else:
                    result = result[:block['start']] + '' + result[block['end']+1:]
                    removed_count += 1
        
        print(f"      结果: 替换{replaced_count}个, 删除{removed_count}个")
        
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        return '\n'.join(lines)
    
    def _evaluate_fanruan_if(self, if_content: str, params: Dict) -> str:
        """
        评估帆软${if()}块，支持嵌套if()和==0条件
        
        示例:
          if(len(start_time)==0, if(len(end_time)==0,"","..."), if(len(end_time)==0,"...","..."))
          当start_time有值时，取最外层false_part，再递归评估内层if()
        """
        import re
        
        content = if_content.strip()
        if content.lower().startswith('if'):
            content = content[2:].strip()
        if not content.startswith('('):
            return ''
        content = content[1:]
        
        args = self._split_if_arguments(content)
        if len(args) < 3:
            return ''
        
        condition = args[0].strip()
        true_part = args[1].strip()
        false_part = args[2].strip()
        
        param_match = re.search(r'len\s*\(\s*(\w+)\s*\)', condition, re.IGNORECASE)
        if not param_match:
            return ''
        param_name = param_match.group(1).lower()
        
        param_value = params.get(param_name)
        has_value = param_value is not None and str(param_value).strip() != ''
        
        is_zero = '==0' in condition or '== 0' in condition
        
        if is_zero:
            selected_part = false_part if has_value else true_part
        else:
            selected_part = true_part if has_value else false_part
        
        selected_stripped = selected_part.strip()
        
        if selected_stripped.lower().startswith('if(') or selected_stripped.lower().startswith('if ('):
            return self._evaluate_fanruan_if(selected_stripped, params)
        
        cleaned = self._clean_if_value(selected_stripped, params)
        return cleaned
    
    def _split_if_arguments(self, content: str) -> list:
        """
        将if(condition, true_part, false_part)的参数部分拆分为3个参数
        
        使用括号计数和引号跟踪处理嵌套结构
        """
        args = []
        depth = 0
        in_string = False
        string_char = None
        current_arg = []
        
        i = 0
        while i < len(content):
            char = content[i]
            
            if char in ('"', "'") and (i == 0 or content[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            
            if not in_string:
                if char == '(':
                    depth += 1
                elif char == ')':
                    if depth == 0:
                        args.append(''.join(current_arg))
                        break
                    depth -= 1
                elif char == ',' and depth == 0:
                    args.append(''.join(current_arg))
                    current_arg = []
                    i += 1
                    continue
            
            current_arg.append(char)
            i += 1
        
        return args
    
    def _clean_if_value(self, value: str, params: Dict) -> str:
        """清理if()评估结果：去引号、替换参数引用、清理帆软拼接语法"""
        import re
        
        result = value.strip()
        
        for pname, pvalue in params.items():
            if pvalue is not None and str(pvalue).strip():
                sv = str(pvalue)
                result = re.sub(r'"\s*\+\s*' + re.escape(pname) + r'\s*\+\s*"', sv, result, flags=re.IGNORECASE)
                result = re.sub(r'"\s*\+\s*' + re.escape(pname.upper()) + r'\s*\+\s*"', sv, result, flags=re.IGNORECASE)
                result = re.sub(r"'\s*\+\s*" + re.escape(pname) + r"\s*\+\s*'", "'" + sv + "'", result, flags=re.IGNORECASE)
                result = re.sub(r"'\s*\+\s*" + re.escape(pname.upper()) + r"\s*\+\s*'", "'" + sv + "'", result, flags=re.IGNORECASE)
        
        result = result.replace('"', '')
        
        result = re.sub(r"""\+\s*'([^']*)'\s*\+""", r"'\1'", result)
        result = re.sub(r"""\+\s*"([^"]*)"\s*\+""", r"'\1'", result)
        result = re.sub(r"""'(\s*)"([^"]+)"(\s*)'""", r"'\2'", result)
        result = re.sub(r"""'(\s*)'([^']+)'(\s*)'""", r"'\2'", result)
        
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def _replace_param_in_part(self, part: str, param_name: str, param_value) -> str:
        """在if()的true_part/false_part中替换参数引用为实际值"""
        import re
        
        if isinstance(param_value, list):
            formatted = ','.join([f"'{v}'" for v in param_value])
        else:
            formatted = f"'{param_value}'"
        
        replacement = part
        
        sub_patterns = [
            r"SUBSTITUTE\s*\(\s*" + re.escape(param_name) + r'\s*,\s*"[^"]*"\s*,\s*"[^"]*"\s*\)',
            r"SUBSTITUTE\s*\(\s*" + re.escape(param_name) + r'\s*,\s*[\'"][^\'"]*[\'"]\s*,\s*[\'"][^\'"]*[\'"]*\s*\)',
            r"SUBSTITUTE\s*\(\s*" + re.escape(param_name) + r"[^\)]*\)",
        ]
        
        replaced_sub = False
        for sub_pattern in sub_patterns:
            match = re.search(sub_pattern, replacement, re.IGNORECASE)
            if match:
                replacement = re.sub(sub_pattern, formatted, replacement, flags=re.IGNORECASE)
                replaced_sub = True
                break
        
        if not replaced_sub:
            param_patterns = [
                r"""\+['"]?""" + re.escape(param_name) + r"""['"]?\+""",
                r"""['"]\+\s*""" + re.escape(param_name) + r"""\s*\+['"]""",
            ]
            for pp in param_patterns:
                replacement = re.sub(pp, formatted, replacement, flags=re.IGNORECASE)
        
        if param_name in replacement:
            replacement = replacement.replace(param_name, formatted)
        
        replacement = re.sub(r"""'\s*\+\s*"([^"]*)"\s*\+\s*'""", r"'\1'", replacement)
        replacement = re.sub(r"""'\s*\+\s*'([^']*)'\s*\+\s*'""", r"\1", replacement)
        replacement = re.sub(r"""\+\s*"([^"]*)"\s*\+""", r"'\1'", replacement)
        replacement = re.sub(r"""\+\s*'([^']*)'\s*\+""", r"\1", replacement)
        replacement = re.sub(r"""'(\s*)"([^"]+)"(\s*)'""", r"'\2'", replacement)
        replacement = re.sub(r"""'(\s*)'([^']+)'(\s*)'""", r"'\2'", replacement)
        
        return replacement.strip()
    
    def _post_process_data(self, data: List[Dict], parameters: Dict, 
                           api_def, sql_template: str, semantic_depts: List[str] = None) -> List[Dict]:
        """
        Python后处理器: 在内存中对数据进行过滤、聚合
        
        支持的操作:
          1. 部门过滤: dept='CI' → 只保留CI部门的数据
          2. 时间范围过滤: start_time/end_time
          3. 按员工聚合: 将同一员工的多行数据合并（求和审批数量等）
          4. 排序取Top: 按审批数量降序取前N名
        """
        
        if not data:
            return data
        if not parameters and not semantic_depts:
            return data
        
        result = data
        param_lower = {k.lower(): v for k, v in parameters.items()}
        
        # ---- 1. 部门过滤 (优先使用语义层识别结果) ----
        # 优先级：语义层识别 > API参数
        target_dept = None
        if semantic_depts and len(semantic_depts) > 0:
            # 语义层有识别到部门，优先使用
            target_dept = ','.join(semantic_depts)
            print(f"    [后处理-部门] 使用语义层识别部门: {target_dept}")
        else:
            # 没有语义层识别，使用API参数
            dept_value = param_lower.get('dept') or param_lower.get('department')
            if dept_value and str(dept_value).strip():
                target_dept = str(dept_value)
                print(f"    [后处理-部门] 使用API参数部门: {target_dept}")
        
        if target_dept:
            result = self._filter_by_department(result, target_dept, api_def)
            print(f"    [后处理-部门] 过滤后剩余{len(result)}条")
        
        # ---- 2. 时间范围过滤 ----
        start_time = param_lower.get('start_time')
        end_time = param_lower.get('end_time')
        if start_time or end_time:
            result = self._filter_by_time_range(result, start_time, end_time, sql_template)
            print(f"    [后处理-时间] 时间过滤后剩余{len(result)}条")
        
        # ---- 3. 审批节点/状态过滤 ----
        status_val = param_lower.get('recfromstatus') or param_lower.get('status')
        if status_val and str(status_val).strip():
            result = self._filter_by_field(result, 'RECFROMSTATUS', str(status_val))
            print(f"    [后处理-状态] 过滤状态={status_val}, 剩余{len(result)}条")
        
        flow_type = param_lower.get('flow_type')
        if flow_type and str(flow_type).strip():
            result = self._filter_by_flow_type(result, str(flow_type))
            print(f"    [后处理-流程类型] 过滤后剩余{len(result)}条")
        
        return result
    
    def _filter_by_department(self, data: List[Dict], dept: str, api_def) -> List[Dict]:
        """按部门代码过滤数据（支持单部门或多部门）"""
        
        # 解析部门参数：支持 'CI', 'CI,OM', ['CI', 'OM'] 格式
        dept_str = str(dept).upper().strip()
        
        # 提取部门列表
        target_depts = []
        if ',' in dept_str:
            target_depts = [d.strip() for d in dept_str.split(',') if d.strip()]
        elif isinstance(dept, (list, tuple)):
            target_depts = [str(d).upper().strip() for d in dept]
        else:
            target_depts = [dept_str]
        
        # 过滤空值
        target_depts = [d for d in target_depts if d]
        
        if not target_depts:
            return data
        
        print(f"      目标部门: {target_depts}")
        filtered = []
        
        for row in data:
            row_upper = {k.upper(): v for k, v in row.items()}
            
            # 尝试多种可能的字段名
            match = False
            
            # 方法1: 直接匹配 USRMRC / dept_code 字段
            for field in ['USRMRC', 'DEPT_CODE', 'DEPT', 'DEPARTMENT']:
                val = row_upper.get(field)
                if val is not None:
                    val_str = str(val).upper().strip()
                    # 多值检查
                    if val_str in target_depts:
                        match = True
                        break
                    # 支持单条记录是多值（如 "CI,OM"）的情况
                    row_depts = [d.strip() for d in val_str.split(',') if d.strip()]
                    if any(rd in target_depts for rd in row_depts):
                        match = True
                        break
            
            # 方法2: 通过关联的员工表字段匹配
            if not match:
                for field in ['USRDESC', 'EMPLOYEE', 'CREATEDBY', 'NAME']:
                    val = row.get(field)
                    if val and hasattr(api_def, 'name') and '员工' in api_def.name:
                        pass  # 员工维度不按此字段过滤部门
            
            if match:
                filtered.append(row)
        
        print(f"      部门过滤后: {len(filtered)}/{len(data)}")
        return filtered
    
    def _filter_by_time_range(self, data: List[Dict], start_time: str, 
                               end_time: str, sql_template: str) -> List[Dict]:
        """按时间范围过滤数据"""
        
        from datetime import datetime
        
        time_field = self._detect_time_field_from_sql(sql_template) or 'ApprovalTime'
        print(f"      时间字段检测: {time_field}")
        
        has_time_column = False
        if data:
            for field in [time_field, 'APPROVALTIME', 'LASTSAVED', 'CREATEDATE', 
                          'CREATETIME', 'APPROVAL_TIME', 'LAST_SAVED']:
                for k in data[0].keys():
                    if k.upper() == field.upper():
                        has_time_column = True
                        break
                if has_time_column:
                    break
        
        if not has_time_column:
            # 【修复Bug2】检查SQL模板是否已包含时间过滤条件
            # 如果SQL层已通过${if()}块处理了时间过滤，则信任SQL层结果
            # 如果SQL层没有时间过滤，则发出警告（数据可能未被正确过滤）
            sql_has_time_filter = False
            if sql_template:
                sql_lower = sql_template.lower()
                time_keywords_in_sql = ['approvaltime', 'lastsaved', 'createdate', 'createtime']
                time_params_in_sql = ['start_time', 'end_time']
                has_time_param_ref = any(p in sql_lower for p in time_params_in_sql)
                has_time_field_ref = any(k in sql_lower for k in time_keywords_in_sql)
                sql_has_time_filter = has_time_param_ref and has_time_field_ref

            if sql_has_time_filter:
                print(f"      [时间过滤] 数据中无时间列，但SQL层已含时间过滤条件，信任SQL层结果，保留{len(data)}条")
            else:
                print(f"      [时间过滤警告] 数据中无时间列且SQL层无时间过滤条件！数据可能未被正确过滤，保留{len(data)}条")
            return data
        
        filtered = []
        
        for row in data:
            time_val = None
            
            for field in [time_field, 'APPROVALTIME', 'LASTSAVED', 'CREATEDATE', 
                          'CREATETIME', 'APPROVAL_TIME', 'LAST_SAVED']:
                if field in row:
                    time_val = row[field]
                    break
                field_upper = field.upper()
                for k, v in row.items():
                    if k.upper() == field_upper:
                        time_val = v
                        break
                if time_val:
                    break
            
            if not time_val:
                continue
            
            try:
                if isinstance(time_val, str):
                    dt = datetime.strptime(time_val[:19], '%Y-%m-%d %H:%M:%S')
                else:
                    dt = time_val
                
                if start_time:
                    start_dt = datetime.strptime(start_time[:10], '%Y-%m-%d')
                    if dt < start_dt:
                        continue
                
                if end_time:
                    end_dt = datetime.strptime(end_time[:10], '%Y-%m-%d')
                    if dt > end_dt:
                        continue
                
                filtered.append(row)
            except (ValueError, TypeError):
                filtered.append(row)
        
        return filtered
    
    def _detect_time_field_from_sql(self, sql_template: str) -> Optional[str]:
        """从SQL模板中检测使用的时间字段名"""
        import re
        
        candidates = [
            ('da.ApprovalTime', r'da\.ApprovalTime'),
            ('af.LASTSAVED', r'af\.LASTSAVED'),
            ('ApprovalTime', r'\bApprovalTime\b'),
            ('LASTSAVED', r'\bLASTSAVED\b'),
            ('CREATEDATE', r'\bCREATEDATE\b'),
        ]
        
        for field_name, pattern in candidates:
            if re.search(pattern, sql_template, re.IGNORECASE):
                return field_name
        
        return None
    
    def _filter_by_field(self, data: List[Dict], field_name: str, value: str) -> List[Dict]:
        """按指定字段值过滤"""
        value_upper = value.upper().strip()
        return [
            row for row in data
            if any(
                str(v).upper().strip() == value_upper or 
                value_upper in str(v).upper().split(',')
                for v in [row.get(field_name), row.get(field_name.upper()), 
                          row.get(field_name.lower())]
                if v is not None
            )
        ]
    
    def _filter_by_flow_type(self, data: List[Dict], flow_type: str) -> List[Dict]:
        """按工单流程类型过滤"""
        type_values = [t.strip().upper() for t in flow_type.split(',')]
        
        return [
            row for row in data
            if any(
                any(tv in str(v).upper() for tv in type_values)
                for v in [row.get('FLODESC'), row.get('flow_type'), 
                          row.get('FLOENTITYDESC'), row.get('流程类型')]
                if v is not None
            )
        ]
    
    def _apply_parameters_to_sql(self, sql_template: str, parameters: Dict[str, Any]) -> Optional[str]:
        """将参数应用到SQL模板（增强版 - 处理所有占位符格式）"""
        
        if not sql_template:
            return None
        
        final_sql = sql_template
        
        if not parameters:
            # 如果没有参数，移除所有条件语句中的未替换占位符
            import re
            # 移除整个${if(...)...}块
            final_sql = re.sub(r'\$\{if\(.*?\}.*?\}', '', final_sql, flags=re.DOTALL)
            # 移除剩余的${param}
            final_sql = re.sub(r'\$\{\w+\}', '', final_sql)
            return final_sql
        
        for param_name, param_value in parameters.items():
            if param_value is None or param_value == '':
                continue
            
            str_value = str(param_value)
            
            # 尝试多种占位符格式（按优先级排序）
            placeholders = [
                (f"${{{param_name}}}", True),       # ${param_name}  (最常见)
                (f"${param_name}$", True),           # $param_name$
                (f"{{{param_name}}}", True),         # {param_name}
                (f"${param_name}", True),             # $param_name (无结尾$)
            ]
            
            replaced = False
            for placeholder, needs_quotes in placeholders:
                if placeholder in final_sql:
                    try:
                        # 根据参数类型决定是否加引号
                        if isinstance(param_value, (int, float)):
                            final_sql = final_sql.replace(placeholder, str(param_value))
                        else:
                            final_sql = final_sql.replace(placeholder, f"'{str_value}'")
                        replaced = True
                        print(f"    [参数替换] {placeholder} → {str_value}")
                        break
                    except Exception as e:
                        print(f"    [WARN] 替换失败 {placeholder}: {e}")
            
            if not replaced:
                print(f"    [WARN] 参数 '{param_name}' 未找到匹配的占位符")
        
        # 【关键】清理残留的未替换占位符
        import re
        
        # 统计残留占位符
        remaining = re.findall(r'\$\{[^}]+\}', final_sql)
        if remaining:
            print(f"    [WARN] 发现 {len(remaining)} 个未替换的占位符: {remaining[:3]}")
            
            # 对于可选参数，移除相关条件
            for placeholder in remaining:
                # 如果是简单变量引用，用空字符串或默认值替换
                var_name = placeholder.strip('${}')
                
                # 常见默认值
                defaults = {
                    'start_time': "''",
                    'end_time': "''", 
                    'dept': "''",
                    'flow_type': "''",
                }
                
                replacement = defaults.get(var_name, "''")
                final_sql = final_sql.replace(placeholder, replacement)
        
        # 清理多余的WHERE AND/OR
        final_sql = re.sub(r'\bWHERE\s+AND\b', 'WHERE', final_sql)
        final_sql = re.sub(r'\bWHERE\s+\)', '', final_sql)  # 空WHERE子句
        
        return final_sql
    
    def _call_fanruan_api(self, sql: str, connection: str = 'EAM') -> Optional[List[Dict]]:
        """直连数据库执行SQL（原帆软OpenAPI已替换为pymssql直连）"""

        import pymssql

        try:
            db_config = self._get_db_config(connection)

            conn = pymssql.connect(
                server=db_config.get('server') or db_config.get('host', 'localhost'),
                port=db_config.get('port', 1433),
                user=db_config.get('user', ''),
                password=db_config.get('password', ''),
                database=db_config.get('database', ''),
                timeout=db_config.get('timeout', 60),
                login_timeout=db_config.get('login_timeout', 15),
                charset='UTF-8',
                as_dict=True
            )

            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            conn.close()

            if isinstance(rows, list) and len(rows) > 0:
                print(f"  [OK] 直连查询成功! ({len(rows)}条记录)")
                return rows
            else:
                print(f"  [WARN] 返回空数据")
                return []

        except Exception as e:
            logger.error(f"[直连DB] SQL执行失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _call_llm(self, prompt: str, system_role: str = "你是AI助手。") -> Optional[str]:
        """调用LLM"""
        
        if not self.llm_client:
            return None
        
        try:
            response = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3,  # 低温度保证准确性
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"[LLM] 调用失败: {e}")
            return None
    
    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """解析LLM响应中的JSON（支持嵌套结构和markdown代码块）"""
        
        if not text:
            return None
        
        # 方法1: 尝试提取markdown代码块中的JSON
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except:
                pass
        
        # 方法2: 提取第一个{...}（使用计数器处理嵌套）
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
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
            json_str = text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"    [DEBUG] JSON解析错误: {e}")
                print(f"    [DEBUG] JSON片段: {json_str[:200]}...")
        
        # 方法3: 直接尝试解析整个文本
        try:
            return json.loads(text.strip())
        except:
            pass
        
        return None


# ============================================================
# Flask API 集成
# ============================================================

def create_nl2api_blueprint():
    """创建Flask蓝图"""
    
    from flask import Blueprint, request, jsonify
    
    nl2api_bp = Blueprint('nl2api', __name__, url_prefix='/api/nl2api')
    
    service = NL2APIService()
    
    @nl2api_bp.route('/query', methods=['POST'])
    def query():
        """NL2API查询接口"""
        
        import json as _json
        
        def safe_serialize(obj):
            """递归安全的JSON序列化"""
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, dict):
                return {k: safe_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [safe_serialize(item) for item in obj]
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif callable(obj):
                return str(obj)
            else:
                try:
                    _json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        try:
            data = request.get_json(force=True)
            user_query = data.get('query', '').strip()
            report_path = data.get('report_path', '')
            
            if not user_query:
                return jsonify({
                    'code': 400,
                    'msg': '缺少查询内容',
                    'data': {'success': False, 'error': 'query参数不能为空'}
                }), 400
            
            # 执行查询
            result = service.query(user_query, report_path)
            
            logger.info(f"[NL2API] 查询完成 | success={result.success} | api={result.selected_api_name} | params={result.api_parameters} | data_count={result.data_count} | time={result.execution_time:.1f}s | llm_calls={result.llm_calls}")
            
            if result.success:
                return jsonify({
                    'code': 200,
                    'msg': '成功',
                    'data': {
                        'success': True,
                        'intent': result.user_intent,
                        'confidence': result.confidence,
                        'selected_api': {
                            'id': result.selected_api_id,
                            'name': result.selected_api_name,
                        },
                        'parameters': safe_serialize(result.api_parameters),
                        'validation_errors': safe_serialize(result.validation_errors) if hasattr(result, 'validation_errors') else [],
                        'param_retry_count': getattr(result, 'param_retry_count', 0),
                        'raw_data': safe_serialize(result.raw_data[:100]),
                        'raw_data_count': result.data_count,
                        'summary': result.summary,
                        'insights': safe_serialize(result.insights),
                        'chart_config': safe_serialize(result.chart_config),
                        'execution_time': result.execution_time,
                        'llm_calls': result.llm_calls,
                        'semantic_entities': safe_serialize(getattr(result, '_semantic_entities', {})),
                    }
                })
            else:
                # 【P0修复】根据错误类型返回合适的HTTP状态码，不再统一返回500
                
                error_msg = result.error or '查询失败'
                error_stage = result.error_stage or 'unknown'
                
                # 判断错误类型和合适的HTTP状态码
                if '503' in error_msg or 'memory' in error_msg.lower() or '负载' in error_msg:
                    http_code = 200
                    user_msg = f"⚠️ 帆软服务器繁忙，请稍后重试\n\n{error_msg}"
                elif '400' in error_msg or '语法' in error_msg:
                    http_code = 200
                    user_msg = f"❌ 数据查询异常，已记录错误\n\n{error_msg[:200]}"
                elif '空数据' in error_msg or 'empty' in error_msg.lower():
                    http_code = 200
                    user_msg = f"ℹ️ 查询成功但未找到匹配数据\n\n建议：尝试调整查询条件"
                elif error_stage == 'route':
                    http_code = 200
                    user_msg = f"🔍 暂无匹配的API，请换个说法试试\n\n{error_msg[:150]}"
                else:
                    http_code = 200
                    user_msg = f"❌ 查询未成功: {error_msg[:200]}"
                
                return jsonify({
                    'code': http_code,
                    'msg': user_msg,
                    'data': {
                        'success': False,
                        'error': error_msg,
                        'error_stage': error_stage,
                        'intent': result.user_intent,
                        'confidence': result.confidence if hasattr(result, 'confidence') else 0,
                        'selected_api': {
                            'id': getattr(result, 'selected_api_id', None),
                            'name': getattr(result, 'selected_api_name', None),
                        } if hasattr(result, 'selected_api_id') else None,
                        'parameters': getattr(result, 'api_parameters', {}),
                        'validation_errors': getattr(result, 'validation_errors', []),
                        'param_retry_count': getattr(result, 'param_retry_count', 0),
                        'execution_time': getattr(result, 'execution_time', 0),
                        'llm_calls': getattr(result, 'llm_calls', 0),
                    }
                })
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'code': 200,
                'msg': f'查询异常: {str(e)[:200]}',
                'data': {
                    'success': False,
                    'error': str(e),
                    'error_stage': 'system'
                }
            })
    
    @nl2api_bp.route('/apis', methods=['GET'])
    def list_apis():
        """列出所有可用的API"""
        
        apis = service.registry.get_all_apis()
        
        api_list = []
        for api_def in apis:
            api_list.append({
                'id': api_def.api_id,
                'name': api_def.name,
                'category': api_def.category,
                'description': api_def.description[:100],
                'parameters': [
                    {
                        'name': p.name,
                        'type': p.param_type,
                        'required': p.required,
                        'enum': p.enum_values,
                    }
                    for p in api_def.parameters
                ],
                'tags': api_def.tags,
                'use_cases': api_def.use_cases,
            })
        
        return jsonify({
            'code': 200,
            'msg': '成功',
            'data': {
                'total': len(api_list),
                'apis': api_list
            }
        })
    
    @nl2api_bp.route('/functions', methods=['GET'])
    def get_function_definitions():
        """获取Function Calling格式的函数定义"""
        
        func_defs = service.registry.to_function_definitions()
        
        return jsonify({
            'code': 200,
            'msg': '成功',
            'data': {
                'total': len(func_defs),
                'functions': func_defs
            }
        })
    
    return nl2api_bp


# ============================================================
# 测试入口
# ============================================================

def test_nl2api_service():
    """测试NL2API服务"""
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  NL2API Service v1.0 - 自然语言到API测试  ".center(66) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    service = NL2APIService()
    
    test_cases = [
        ("CI部门完成工单数量最多的员工", "/智翔金泰设备管理平台/工单效能看板.fvs"),
        ("各部门工单处理情况", ""),
    ]
    
    for query, report in test_cases:
        print(f"\n{'─' * 70}")
        print(f"[TEST] 查询: {query}")
        if report:
            print(f"       报表: {report}")
        print(f"{'─' * 70}")
        
        result = service.query(query, report)
        
        if result.success:
            print(f"\n✓ [SUCCESS]")
            print(f"  选择API: {result.selected_api_name}")
            print(f"  参数: {result.api_parameters}")
            print(f"  数据量: {result.data_count} 条")
            print(f"  耗时: {result.execution_time:.2f}s")
            print(f"  LLM调用: {result.llm_calls} 次")
            
            if result.summary:
                print(f"\n  [AI总结]")
                print(f"  {result.summary[:200]}...")
        else:
            print(f"\n✗ [FAIL] {result.error} (阶段: {result.error_stage})")


if __name__ == '__main__':
    test_nl2api_service()
