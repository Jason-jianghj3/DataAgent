#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API Registry - API注册中心

从 report_config.json 提取所有数据集，封装为标准化的API定义。

每个API包含:
- api_id: 唯一标识
- name: 显示名称  
- description: 功能描述（用于LLM理解）
- parameters: 参数列表 [{name, type, required, description, enum_values}]
- sql_template: SQL模板（带参数占位符）
- connection: 数据库连接
- response_fields: 返回字段说明
- business_context: 业务场景说明
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class APIParameter:
    """API参数定义"""
    name: str                           # 参数名
    param_type: str = "string"          # 类型: string/number/date/enum
    required: bool = False              # 是否必填
    description: str = ""               # 参数描述
    default_value: Any = None           # 默认值
    enum_values: List[str] = field(default_factory=list)  # 可选值列表
    example_value: Any = None           # 示例值


@dataclass 
class APIDefinition:
    """标准API定义"""
    
    api_id: str                          # 唯一ID (如: workorder_employee_efficiency)
    name: str                            # 显示名称 (如: 部门人均效能)
    description: str                     # 功能描述（用于LLM匹配）
    category: str = "general"            # 分类: efficiency/inventory/quality/...
    report_name: str = ""                # 所属报表
    
    # 参数
    parameters: List[APIParameter] = field(default_factory=list)
    
    # SQL执行信息
    sql_template: str = ""               # SQL模板
    connection: str = "EAM"             # 数据库连接
    
    # 返回值
    response_fields: List[Dict] = field(default_factory=list)  # [{name, type, description}]
    sample_response: List[Dict] = field(default_factory=list)  # 示例返回数据
    
    # 元信息
    tags: List[str] = field(default_factory=list)  # 标签: [employee, department, time_range]
    use_cases: List[str] = field(default_factory=list)  # 典型使用场景
    
    # 业务上下文
    business_rules: str = ""             # 业务规则说明
    data_granularity: str = "unknown"   # 数据粒度: detail/summary/daily/monthly


class APIRegistry:
    """
    API注册中心
    
    职责：
    1. 从report_config.json加载所有数据集
    2. 解析并标准化为APIDefinition
    3. 提供API查询和匹配接口
    4. 生成Function Calling所需的function definitions
    """
    
    def __init__(self, config_path: str = 'report_config.json'):
        self.config_path = config_path
        self.apis: Dict[str, APIDefinition] = {}  # {api_id: APIDefinition}
        self._load_and_register()
        
        print(f"[APIRegistry] 初始化完成 | 已注册 {len(self.apis)} 个API")
    
    def _load_and_register(self):
        """加载配置并注册所有API"""
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            print(f"[ERROR] 配置文件不存在: {config_path}")
            return
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        reports = config.get('reports', {})
        
        if isinstance(reports, dict):
            for report_name, report_config in reports.items():
                self._register_report_apis(report_name, report_config)
        elif isinstance(reports, list):
            for report_config in reports:
                report_name = report_config.get('filename', '未知报表')
                self._register_report_apis(report_name, report_config)
    
    def _register_report_apis(self, report_name: str, report_config: Dict):
        """注册一个报表的所有数据集为API"""
        
        datasets = report_config.get('datasets', [])
        report_path = report_config.get('report_path', '')
        primary_connection = report_config.get('primary_connection', 'EAM')
        
        for ds in datasets:
            if not isinstance(ds, dict):
                continue
                
            api_def = self._create_api_from_dataset(
                dataset=ds,
                report_name=report_name,
                report_path=report_path,
                primary_connection=primary_connection
            )
            
            if api_def:
                self.apis[api_def.api_id] = api_def
    
    def _create_api_from_dataset(self, dataset: Dict, report_name: str, 
                                  report_path: str, primary_connection: str) -> Optional[APIDefinition]:
        """从数据集配置创建标准API定义"""
        
        ds_name = dataset.get('name', '未命名数据集')
        sql_template = dataset.get('sql_template', '')
        connection = dataset.get('connection', primary_connection)
        params_config = dataset.get('params', [])
        
        if not sql_template or len(sql_template) < 20:
            return None
        
        # 生成API ID（从名称生成slug）
        api_id = self._generate_api_id(ds_name, report_name)
        
        # 解析参数
        parameters = self._parse_parameters(params_config, sql_template)
        
        # 分析SQL提取返回字段
        response_fields = self._extract_response_fields(sql_template)
        
        # 生成描述（用于LLM理解）
        description = self._generate_description(ds_name, sql_template, parameters, response_fields)
        
        # 提取标签和分类
        tags, category = self._classify_api(ds_name, sql_template, parameters, response_fields)
        
        # 确定数据粒度
        granularity = self._determine_granularity(sql_template, response_fields)
        
        # 生成典型使用场景
        use_cases = self._generate_use_cases(ds_name, parameters, response_fields)
        
        return APIDefinition(
            api_id=api_id,
            name=ds_name,
            description=description,
            category=category,
            report_name=report_name,
            parameters=parameters,
            sql_template=sql_template,
            connection=connection,
            response_fields=response_fields,
            tags=tags,
            use_cases=use_cases,
            data_granularity=granularity,
            business_rules=f"来源于报表: {report_name}"
        )
    
    def _generate_api_id(self, ds_name: str, report_name: str) -> str:
        """生成唯一的API ID"""
        # 简化中文名称为英文slug
        name_part = re.sub(r'[^\w]', '_', ds_name.lower())[:30]
        report_part = re.sub(r'[^\w]', '_', report_name.lower())[:20]
        return f"{name_part}_{report_part}".strip('_').replace('__', '_')
    
    def _parse_parameters(self, params_config: List, sql_template: str) -> List[APIParameter]:
        """解析参数配置"""
        
        parameters = []
        
        for param_name in params_config:
            if not param_name or not isinstance(param_name, str):
                continue
            
            # 从SQL模板推断参数信息
            param_in_sql = f"${param_name}" in sql_template or "{" + param_name + "}" in sql_template
            
            # 根据参数名推断类型和描述
            param_type, description, enum_vals = self._infer_parameter_info(param_name, sql_template)
            
            # 判断是否必填
            required = self._is_parameter_required(param_name, sql_template)
            
            # 示例值
            example = self._get_example_value(param_name, param_type, enum_vals)
            
            parameters.append(APIParameter(
                name=param_name,
                param_type=param_type,
                required=required,
                description=description,
                enum_values=enum_vals,
                example_value=example
            ))
        
        return parameters
    
    def _infer_parameter_info(self, param_name: str, sql_template: str) -> tuple:
        """推断参数的类型、描述和可选值"""
        
        name_lower = param_name.lower()
        
        # 时间相关
        if 'time' in name_lower or 'date' in name_lower:
            if 'start' in name_lower or 'begin' in name_lower:
                return 'datetime', '开始时间（格式: YYYY-MM-DD HH:mm:ss）', []
            elif 'end' in name_lower:
                return 'datetime', '结束时间（格式: YYYY-MM-DD HH:mm:ss）', []
            else:
                return 'datetime', '时间筛选条件', []
        
        # 部门
        elif 'dept' in name_lower:
            return 'string', '部门代码（如: OM, CI, FM, QA, PD）', ['OM', 'CI', 'FM', 'QA', 'PD']
        
        # 流程类型
        elif 'flow' in name_lower or 'type' in name_lower:
            return 'string', '流程/工单类型筛选', []
        
        # 分类
        elif 'category' in name_lower:
            return 'string', '业务分类筛选', []
        
        else:
            return 'string', f'{param_name}参数', []
    
    def _is_parameter_required(self, param_name: str, sql_template: str) -> bool:
        """判断参数是否必填"""
        
        # 检查SQL中的条件逻辑
        # 如果有类似 ${if(len(param)>0,...} 的判断，说明是可选的
        optional_pattern = rf'\${{if\(len\({param_name}\)'
        if re.search(optional_pattern, sql_template):
            return False
        
        # 如果参数在WHERE子句的关键位置，可能是必填的
        # 但通常我们设计为可选，不传则查询全部
        
        return False  # 默认都是可选的
    
    def _get_example_value(self, param_name: str, param_type: str, 
                          enum_values: List[str]) -> Any:
        """获取示例值"""
        
        if enum_values:
            return enum_values[0]
        
        if param_type == 'datetime':
            return '2026-01-01'
        
        return ''
    
    def _extract_response_fields(self, sql_template: str) -> List[Dict]:
        """从SQL中提取返回字段"""
        
        fields = []
        
        # 查找SELECT子句
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_template, re.IGNORECASE | re.DOTALL)
        
        if select_match:
            select_clause = select_match.group(1)
            
            # 分割字段（处理别名）- 修复正则匹配
            field_patterns = re.findall(r'([\w.\[\]\*]+)\s+(?:AS\s+)?([\w\u4e00-\u9fff]+)', select_clause, re.IGNORECASE)
            
            for match in field_patterns:
                if len(match) >= 2:
                    expr, alias = match[0], match[1]
                    
                    # 跳过过短或无效的别名
                    if len(alias) < 1 or len(expr) < 1:
                        continue
                    
                    # 推断字段类型
                    field_type = 'string'
                    if any(kw in expr.upper() for kw in ['COUNT(', 'SUM(']):
                        field_type = 'number'
                    elif any(kw in expr.upper() for kw in ['AVG(', 'ROUND(']):
                        field_type = 'decimal'
                    elif 'DATEDIFF' in expr.upper():
                        field_type = 'decimal'
                    
                    fields.append({
                        'name': alias,
                        'expression': expr.strip(),
                        'type': field_type,
                        'description': ''
                    })
        
        return fields
    
    def _generate_description(self, ds_name: str, sql_template: str, 
                            parameters: List[APIParameter], 
                            fields: List[Dict]) -> str:
        """生成API功能描述（用于LLM理解）"""
        
        parts = [f"查询【{ds_name}】的数据"]
        
        if fields:
            field_names = [f['name'] for f in fields[:5]]
            parts.append(f"返回字段包括: {', '.join(field_names)}")
        
        if parameters:
            param_names = [p.name for p in parameters[:3]]
            if param_names:
                parts.append(f"支持按{', '.join(param_names)}等条件筛选")
        
        # 根据字段名推断用途
        field_str = ' '.join([f['name'] for f in fields]).lower()
        
        if '员工' in field_str or 'usrdesc' in field_str.lower():
            parts.append("可分析员工级效能数据")
        if '部门' in field_str or 'usrmrc' in field_str.lower():
            parts.append("支持部门维度分析")
        if '耗时' in field_str or 'process' in field_str.lower():
            parts.append("包含工单处理时长统计")
        if '数量' in field_str or 'count' in field_str.lower():
            parts.append("提供数量统计指标")
        
        return '。'.join(parts) + '。'
    
    def _classify_api(self, ds_name: str, sql_template: str, 
                    parameters: List[APIParameter], 
                    fields: List[Dict]) -> tuple:
        """分类API并打标签"""
        
        tags = []
        category = 'general'
        
        name_lower = ds_name.lower()
        field_str = ' '.join([f['name'].lower() for f in fields])
        sql_lower = sql_template.lower()
        
        # 效能类
        if any(kw in name_lower or kw in field_str for kw in ['效能', '效率', '耗时', '处理']):
            tags.append('efficiency')
            category = 'efficiency'
        
        # 工单类
        if any(kw in name_lower or kw in sql_lower for kw in ['工单', 'workflow', 'atworkflow']):
            tags.append('workorder')
            category = 'workorder'
        
        # 员工类
        if '员工' in ds_name or 'usrdesc' in field_str:
            tags.append('employee')
        
        # 部门类
        if '部门' in ds_name or 'dept' in field_str:
            tags.append('department')
        
        # 统计类
        if any(kw in name_lower for kw in ['汇总', '统计', '分布', '情况']):
            tags.append('statistics')
        
        # 明细类
        if '明细' in name_lower or ('detail' in name_lower):
            tags.append('detail')
            category = 'detail'
        
        # 极值/排名类
        if any(kw in name_lower for kw in ['top', '排行', '最多', '最少']):
            tags.append('ranking')
        
        if not tags:
            tags.append('general')
        
        return tags, category
    
    def _determine_granularity(self, sql_template: str, fields: List[Dict]) -> str:
        """确定数据粒度"""
        
        sql_lower = sql_template.lower()
        field_names = [f['name'].lower() for f in fields]
        
        # 如果GROUP BY了员工或人员字段，且SELECT中有聚合函数 → 可能是汇总
        has_group_by = 'group by' in sql_lower
        has_aggregation = any(kw in sql_lower for kw in ['count(', 'sum(', 'avg('])
        
        if has_group_by and has_aggregation:
            if any(kw in ' '.join(field_names) for kw in ['usrdesc', '员工', '姓名']):
                return 'entity_summary'  # 按实体汇总
            
        if has_aggregation:
            return 'summary'  # 总体汇总
        
        # 如果有审批节点字段，通常是明细
        if any(kw in ' '.join(field_names) for kw in ['审批节点', 'recfromstatus', 'noddesc']):
            return 'detail'  # 明细级别
        
        return 'unknown'
    
    def _generate_use_cases(self, ds_name: str, parameters: List[APIParameter],
                           fields: List[Dict]) -> List[str]:
        """生成典型使用场景"""
        
        cases = []
        
        # 基于参数生成示例查询
        dept_param = next((p for p in parameters if 'dept' in p.name.lower()), None)
        
        if dept_param and dept_param.enum_values:
            dept = dept_param.enum_values[0]
            cases.append(f"查询{dept}部门的{ds_name}")
        
        time_param = next((p for p in parameters if 'time' in p.name.lower() or 'date' in p.name.lower()), None)
        if time_param:
            cases.append(f"查看本月{ds_name}")
        
        # 基于字段生成
        if any('员工' in f.get('name', '') for f in fields):
            cases.append(f"找出{ds_name}中表现最好/最差的员工")
        
        if not cases:
            cases.append(f"查看{ds_name}的整体情况")
        
        return cases[:3]
    
    def get_all_apis(self) -> List[APIDefinition]:
        """获取所有已注册的API"""
        return list(self.apis.values())
    
    def get_api(self, api_id: str) -> Optional[APIDefinition]:
        """根据ID获取API定义"""
        return self.apis.get(api_id)
    
    def search_apis(self, query: str, top_k: int = 5) -> List[tuple]:
        """搜索匹配的API (返回 [(api_id, score), ...])"""
        
        query_lower = query.lower()
        scores = []
        
        for api_id, api_def in self.apis.items():
            score = 0
            
            # 名称匹配
            if query_lower in api_def.name.lower():
                score += 10
            
            # 描述匹配
            if query_lower in api_def.description.lower():
                score += 5
            
            # 标签匹配
            for tag in api_def.tags:
                if tag in query_lower or query_lower in tag:
                    score += 3
            
            # 使用场景匹配
            for use_case in api_def.use_cases:
                if any(qw in use_case.lower() for qw in query_lower.split()):
                    score += 2
            
            # 字段名匹配
            for field in api_def.response_fields:
                if query_lower in field['name'].lower():
                    score += 1
            
            if score > 0:
                scores.append((api_id, score, api_def))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [(item[0], item[1]) for item in scores[:top_k]]
    
    def to_function_definitions(self) -> List[Dict]:
        """
        转换为完整的 OpenAI Function Calling / Tools 标准格式
        
        符合 https://platform.openai.com/docs/guides/function-calling
        每个函数定义包含: type, function { name, description, parameters }
        
        Returns:
            List[Dict]: 符合OpenAI tools格式的函数定义列表
            [
                {
                    "type": "function",
                    "function": {
                        "name": "query_workorder_employee_efficiency",
                        "description": "...",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "dept": {"type": "string", "description": "...", "enum": [...]},
                                ...
                            },
                            "required": ["dept"],
                            "additionalProperties": False
                        }
                    }
                },
                ...
            ]
        """
        
        functions = []
        
        for api_id, api_def in self.apis.items():
            
            # 构建增强描述（包含使用场景和返回字段信息）
            enhanced_desc = self._build_fc_description(api_def)
            
            # 构建参数属性
            properties = {}
            required = []
            
            for param in api_def.parameters:
                prop_def = {
                    "type": self._map_type_to_json_schema(param.param_type),
                    "description": param.description or f"{param.name}参数",
                }
                
                # 枚举值（关键约束！）
                if param.enum_values:
                    prop_def["enum"] = param.enum_values
                
                # 示例值
                if param.example_value is not None and str(param.example_value):
                    prop_def["example"] = str(param.example_value)
                
                properties[param.name] = prop_def
                
                if param.required:
                    required.append(param.name)
            
            func_def = {
                "type": "function",
                "function": {
                    "name": self._sanitize_function_name(api_id),
                    "description": enhanced_desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    }
                }
            }
            
            functions.append(func_def)
        
        return functions
    
    def _build_fc_description(self, api_def: APIDefinition) -> str:
        """构建Function Calling专用的增强描述"""
        parts = [api_def.description]
        
        # 添加返回字段说明
        if api_def.response_fields:
            field_descs = []
            for f in api_def.response_fields[:6]:
                fd = f.get('name', '')
                ft = f.get('type', '')
                field_descs.append(f"- {fd}({ft})")
            parts.append(f"返回字段: {'; '.join(field_descs)}")
        
        # 添加典型查询场景
        if api_def.use_cases:
            parts.append(f"适用场景: {' | '.join(api_def.use_cases[:3])}")
        
        # 添加数据粒度标签
        if api_def.data_granularity != 'unknown':
            granularity_map = {
                'detail': '明细数据(每行是一条记录)',
                'summary': '汇总数据(聚合结果)',
                'entity_summary': '按实体汇总(如按员工/部门)',
            }
            parts.append(f"数据粒度: {granularity_map.get(api_def.data_granularity, api_def.data_granularity)}")
        
        # 添加业务标签
        if api_def.tags:
            parts.append(f"标签: {', '.join(api_def.tags)}")
        
        return '\n'.join(parts)
    
    def _sanitize_function_name(self, api_id: str) -> str:
        """将API ID转换为合法的函数名（符合FC规范：字母数字下划线，以字母开头）"""
        name = re.sub(r'[^a-zA-Z0-9_]', '_', api_id)
        name = re.sub(r'^[^a-zA-Z]', 'fn_', name)
        name = re.sub(r'_+', '_', name).strip('_')
        return name or 'unnamed_api'
    
    def get_tools_for_routing(self) -> Dict[str, Any]:
        """
        获取用于路由的精简工具定义（减少token消耗）
        
        只保留name/description/参数名和枚举，去掉详细字段信息
        """
        tools = []
        for api_id, api_def in self.apis.items():
            
            # 精简版描述
            desc_parts = [api_def.name]
            if api_def.tags:
                desc_parts.append(f"[{', '.join(api_def.tags)}]")
            if api_def.use_cases:
                desc_parts.append(f"例:{api_def.use_cases[0]}")
            
            properties = {}
            required = []
            for p in api_def.parameters:
                prop = {"type": self._map_type_to_json_schema(p.param_type)}
                if p.enum_values:
                    prop["enum"] = p.enum_values
                    prop["description"] = f"可选: {', '.join(p.enum_values)}"
                else:
                    prop["description"] = p.description or ""
                properties[p.name] = prop
                if p.required:
                    required.append(p.name)
            
            tools.append({
                "type": "function",
                "function": {
                    "name": self._sanitize_function_name(api_id),
                    "description": ' | '.join(desc_parts),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
            })
        
        return tools
    
    def _map_type_to_json_schema(self, param_type: str) -> str:
        """映射参数类型到JSON Schema类型"""
        
        mapping = {
            'string': 'string',
            'number': 'number',
            'integer': 'integer',
            'datetime': 'string',
            'date': 'string',
            'enum': 'string',
            'boolean': 'boolean',
        }
        
        return mapping.get(param_type, 'string')
    
    def export_registry(self, output_path: str = 'api_registry.json'):
        """导出API注册表到JSON文件"""
        
        export_data = {
            "version": "1.0",
            "generated_at": __import__('datetime').datetime.now().isoformat(),
            "total_apis": len(self.apis),
            "apis": {}
        }
        
        for api_id, api_def in self.apis.items():
            export_data["apis"][api_id] = {
                "name": api_def.name,
                "description": api_def.description,
                "category": api_def.category,
                "report": api_def.report_name,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.param_type,
                        "required": p.required,
                        "description": p.description,
                        "enum": p.enum_values,
                    }
                    for p in api_def.parameters
                ],
                "response_fields": api_def.response_fields,
                "tags": api_def.tags,
                "use_cases": api_def.use_cases,
                "granularity": api_def.data_granularity,
                "connection": api_def.connection,
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"[APIRegistry] 已导出到 {output_path} ({len(self.apis)} 个API)")
        
        return output_path


# ============================================================
# 测试入口
# ============================================================

def test_api_registry():
    """测试API注册中心"""
    
    print("\n" + "=" * 70)
    print("[TEST] API Registry - API注册中心测试")
    print("=" * 70)
    
    registry = APIRegistry()
    
    print(f"\n[统计]")
    print(f"  总API数: {len(registry.apis)}")
    
    # 显示前5个API
    print(f"\n[API列表 (前10个)]")
    for i, (api_id, api_def) in enumerate(list(registry.apis.items())[:10], 1):
        print(f"\n  {i}. {api_def.name}")
        print(f"     ID: {api_id}")
        print(f"     分类: {api_def.category}")
        print(f"     参数数: {len(api_def.parameters)}")
        print(f"     返回字段数: {len(api_def.response_fields)}")
        print(f"     标签: {api_def.tags}")
        print(f"     描述: {api_def.description[:100]}...")
        
        if api_def.parameters:
            params_str = ', '.join([f"{p.name}({p.param_type})" for p in api_def.parameters[:3]])
            print(f"     参数: {params_str}")
    
    # 测试搜索
    print(f"\n\n[搜索测试]")
    test_queries = ["员工", "CI部门", "工单数量", "效率"]
    
    for q in test_queries:
        results = registry.search_apis(q, top_k=3)
        if results:
            best = results[0]
            api_def = registry.apis.get(best[0])
            print(f"\n  查询: \"{q}\"")
            print(f"  最佳匹配: {api_def.name if api_def else best[0]} (分数: {best[1]})")
    
    # 导出Function Definitions
    print(f"\n\n[Function Calling 定义]")
    func_defs = registry.to_function_definitions()
    print(f"  生成了 {len(func_defs)} 个函数定义")
    
    if func_defs:
        print(f"\n  示例 (第一个):")
        first_func = func_defs[0]
        print(f"    名称: {first_func['name']}")
        print(f"    描述: {first_func['description'][:80]}...")
        print(f"    参数: {list(first_func['parameters']['properties'].keys())}")
    
    # 导出注册表
    print(f"\n\n[导出注册表]")
    registry.export_registry('_api_registry_export.json')


if __name__ == '__main__':
    test_api_registry()
