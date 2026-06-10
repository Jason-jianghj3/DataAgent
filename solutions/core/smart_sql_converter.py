#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
帆软SQL → 标准SQL 转换器 v3.0 (基于括号计数解析)

核心改进：
  使用基于括号计数的解析器正确处理嵌套的 ${if()} 语法
  解决v2.0中正则无法匹配嵌套引号/SUBSTITUTE函数导致40%内容丢失的问题

流程：
  1. 智能扫描提取所有 ${if(...)} 块（基于括号计数）
  2. 根据参数值进行直接替换或保留默认逻辑
  3. 清理剩余占位符和格式
"""

import re
from typing import Optional, Tuple, List, Dict, Any


class SmartSQLConverter:
    """智能SQL转换器"""
    
    def __init__(self):
        self.stats = {'total': 0, 'converted': 0, 'errors': 0}
    
    def convert(self, sql_template: str, params: Dict[str, Any] = None, 
                api_metadata: Dict = None) -> Tuple[Optional[str], List[str]]:
        """
        智能转换入口 - v3.0 基于括号计数解析
        
        Args:
            sql_template: 原始帆软SQL
            params: 已提取的参数值
            api_metadata: API元数据（可选，包含字段映射信息）
        """
        
        if not sql_template:
            return None, ["SQL为空"]

        self.stats['total'] += 1
        warnings = []
        params = params or {}
        self._current_params = params

        print(f"     [SmartConverter v3.0] 开始转换 ({len(sql_template)} 字符)")

        # ====== v3.0 核心策略: 基于括号计数的智能解析 ======
        final_sql = self._smart_convert_v3(sql_template, params)

        # 最终清理
        final_sql = self._final_cleanup(final_sql)

        # 验证
        is_valid, val_warnings = self._validate(final_sql)
        warnings.extend(val_warnings)

        if is_valid:
            self.stats['converted'] += 1
            print(f"     [SmartConverter] ✅ 转换成功! SQL长度: {len(final_sql)}")
        else:
            self.stats['errors'] += 1
            print(f"     [SmartConverter] ⚠️ 转换完成但有警告")

        return final_sql, warnings

    def _find_if_blocks(self, sql: str) -> List[Dict]:
        """
        基于括号计数提取所有 ${if(...)} 块
        
        解决正则无法匹配嵌套引号/SUBSTITUTE函数的问题
        
        Returns:
            List[Dict]: [{'start': int, 'end': int, 'content': str, 'full_match': str}, ...]
        """
        blocks = []
        i = 0
        length = len(sql)
        
        while i < length - 3:
            # 查找 ${ 开头
            if sql[i:i+2] == '${' and i + 2 < length and sql[i+2:i+4].lower() == 'if':
                # 开始括号计数
                depth = 0
                j = i + 2  # 从 'if' 之后开始
                
                while j < length:
                    char = sql[j]
                    
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                        if depth == 0:
                            # 找到匹配的 )，继续找 }
                            k = j + 1
                            while k < length:
                                if sql[k] == '}':
                                    # 找到完整的 ${if(...)} 块
                                    # i指向'$', i+2是'i'(if的起始), j指向')', k指向'}'
                                    block_content = sql[i+2:k]  # if(...) 部分 (不包含首尾括号)
                                    full_match = sql[i:k+1]
                                    
                                    blocks.append({
                                        'start': i,
                                        'end': k,
                                        'content': block_content,
                                        'full_match': full_match,
                                        'param_name': self._extract_param_from_if(block_content)
                                    })
                                    break
                                elif sql[k] == '$' and k + 1 < length and sql[k+1] == '{':
                                    # 嵌套的 ${}，跳过
                                    break
                                k += 1
                            break
                    j += 1
                
                if blocks and blocks[-1]['end'] > i:
                    i = blocks[-1]['end'] + 1
                    continue
            
            i += 1
        
        return blocks

    def _extract_param_from_if(self, if_content: str) -> Optional[str]:
        import re
        match = re.search(r'len\s*\(\s*(\w+)\s*\)', if_content, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return None

    def _is_negative_condition(self, if_content: str) -> bool:
        if_content_lower = if_content.lower()
        return '==0' in if_content_lower or '=0' in if_content_lower or '== 0' in if_content_lower

    def _smart_convert_v3(self, sql: str, params: Dict) -> str:
        blocks = self._find_if_blocks(sql)
        
        print(f"     [v3.0] 发现 {len(blocks)} 个 ${{if()}} 块")
        
        if not blocks:
            print(f"     [v3.0] 无${{if()}}块，返回原始SQL")
            return sql
        
        result = sql
        replaced_count = 0
        removed_count = 0
        
        for block in reversed(blocks):
            full_match = block['full_match']
            param_name = block.get('param_name')
            content = block['content']
            is_negative = self._is_negative_condition(content)
            
            param_value = params.get(param_name) if param_name else None
            has_value = param_value is not None and str(param_value).strip() != ''
            
            if is_negative:
                if has_value:
                    replacement = self._build_replacement_for_negative(content, param_name, param_value)
                    result = result[:block['start']] + replacement + result[block['end']+1:]
                    replaced_count += 1
                    print(f"       [v3.0替换(反向)] {param_name}={param_value}")
                else:
                    replacement = self._build_replacement_for_negative_empty(content, param_name, params)
                    result = result[:block['start']] + replacement + result[block['end']+1:]
                    if not replacement:
                        removed_count += 1
                    print(f"       [v3.0反向空值] {param_name} → {'保留默认' if replacement else '清空'}")
            else:
                if has_value:
                    replacement = self._build_replacement(content, param_name, param_value)
                    result = result[:block['start']] + replacement + result[block['end']+1:]
                    replaced_count += 1
                    print(f"       [v3.0替换] {param_name}={param_value}")
                else:
                    replacement = self._handle_empty_param(content, param_name, params)
                    result = result[:block['start']] + replacement + result[block['end']+1:]
                    if replacement == '':
                        removed_count += 1
        
        print(f"     [v3.0] 替换 {replaced_count} 个, 清空 {removed_count} 个, 结果长度: {len(result)}")
        
        return result

    def _build_replacement(self, if_content: str, param_name: str, param_value: Any) -> str:
        import re

        true_part = self._extract_true_part(if_content)

        if not true_part:
            return ''

        result = true_part

        substitute_pattern = r"SUBSTITUTE\s*\(\s*" + re.escape(param_name) + r"""\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)"""
        substitute_match = re.search(substitute_pattern, result, re.IGNORECASE)

        if substitute_match:
            separator = substitute_match.group(1)
            replacement = substitute_match.group(2)
            raw_value = str(param_value)
            substituted = raw_value.replace(separator, replacement)
            result = result[:substitute_match.start()] + substituted + result[substitute_match.end():]
        else:
            if isinstance(param_value, list):
                formatted_value = ','.join([f"'{v}'" for v in param_value])
            else:
                formatted_value = str(param_value)
            result = re.sub(r'\+"' + re.escape(param_name) + r'"\+', formatted_value, result, flags=re.IGNORECASE)
            result = re.sub(r'\+\s*' + re.escape(param_name) + r'\s*\+', formatted_value, result, flags=re.IGNORECASE)
            if re.search(r'(?<!\w)' + re.escape(param_name) + r'(?!\w)', result, re.IGNORECASE):
                result = re.sub(r'(?<!\w)' + re.escape(param_name) + r'(?!\w)', formatted_value, result, flags=re.IGNORECASE)

        result = re.sub(r"""\s*"\s*\+\s*""", '', result)
        result = re.sub(r"""\s*\+\s*"\s*""", '', result)
        result = re.sub(r"""\s*'\s*\+\s*""", '', result)
        result = re.sub(r"""\s*\+\s*'\s*""", '', result)

        result = re.sub(r"""\(\s*'([^']*)'\s*\)""", r"('\1')", result)

        return result.strip()

    def _build_replacement_for_negative(self, if_content: str, param_name: str, param_value: Any) -> str:
        import re

        if param_name and 'time' in param_name.lower():
            return self._build_time_condition(if_content, param_name, param_value, self._current_params if hasattr(self, '_current_params') else {})

        false_part = self._extract_false_part(if_content)
        if not false_part:
            return ''

        result = false_part
        if isinstance(param_value, list):
            formatted_value = ','.join([f"'{v}'" for v in param_value])
        else:
            formatted_value = str(param_value)

        result = re.sub(r"'\s*\+\s*" + re.escape(param_name) + r"\s*\+'", formatted_value, result, flags=re.IGNORECASE)
        result = re.sub(r"'\s*\+\s*" + re.escape(param_name) + r"\s*\+", f"'{formatted_value}'", result, flags=re.IGNORECASE)
        result = re.sub(r"\+\s*" + re.escape(param_name) + r"\s*\+", formatted_value, result, flags=re.IGNORECASE)

        result = re.sub(r"""\s*"\s*\+\s*""", '', result)
        result = re.sub(r"""\s*\+\s*"\s*""", '', result)
        result = re.sub(r"""\s*'\s*\+\s*""", '', result)
        result = re.sub(r"""\s*\+\s*'\s*""", '', result)

        return result.strip()

    def _build_time_condition(self, if_content: str, param_name: str, param_value: Any, params: Dict) -> str:
        import re

        time_field = 'da.ApprovalTime'
        field_match = re.search(r'(\w+\.\w+|\w+)\s*(?:>=|<=|BETWEEN)', if_content, re.IGNORECASE)
        if field_match:
            time_field = field_match.group(1)

        start_val = params.get('start_time', '')
        end_val = params.get('end_time', '')

        if isinstance(start_val, str):
            start_val = start_val.strip()
        if isinstance(end_val, str):
            end_val = end_val.strip()

        if start_val and end_val:
            return f"AND {time_field} BETWEEN '{start_val}' AND '{end_val}'"
        elif start_val:
            return f"AND {time_field} >= '{start_val}'"
        elif end_val:
            return f"AND {time_field} <= '{end_val}'"
        else:
            return ''

    def _build_replacement_for_negative_empty(self, if_content: str, param_name: str, params: Dict) -> str:
        import re

        if param_name and 'time' in param_name.lower():
            return self._build_time_condition(if_content, param_name, None, params)

        true_part = self._extract_true_part(if_content)
        if not true_part:
            return ''

        nested_if_match = re.search(r'if\s*\(', true_part, re.IGNORECASE)
        if nested_if_match:
            second_param = self._extract_param_from_if(true_part)
            if second_param and second_param in params:
                second_value = params[second_param]
                if second_value and str(second_value).strip():
                    inner_false = self._extract_false_part(true_part)
                    if inner_false:
                        result = inner_false
                        result = re.sub(r"'\s*\+\s*" + re.escape(second_param) + r"\s*\+'", str(second_value), result, flags=re.IGNORECASE)
                        result = re.sub(r"""\s*"\s*\+\s*""", '', result)
                        result = re.sub(r"""\s*\+\s*"\s*""", '', result)
                        result = re.sub(r"""\s*'\s*\+\s*""", '', result)
                        result = re.sub(r"""\s*\+\s*'\s*""", '', result)
                        return result.strip()
            return ''

        return true_part.strip()

    def _extract_false_part(self, if_content: str) -> Optional[str]:
        import re

        content = if_content
        if content.lower().startswith('if'):
            content = content[2:]

        content = content.strip()
        if not content.startswith('('):
            return None

        content = content[1:]

        depth = 0
        comma_positions = []
        i = 0
        in_string = False
        string_char = None

        while i < len(content):
            char = content[i]

            if not in_string:
                if char in ('"', "'"):
                    in_string = True
                    string_char = char
                elif char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == ',' and depth == 0:
                    comma_positions.append(i)
            else:
                if char == string_char:
                    in_string = False
                    string_char = None

            i += 1

        if len(comma_positions) < 2:
            return None

        last_comma_pos = comma_positions[-1]

        false_str = content[last_comma_pos+1:].strip()

        if false_str.startswith('"') and false_str.endswith('"'):
            false_str = false_str[1:-1]
        elif false_str.startswith("'") and false_str.endswith("'"):
            false_str = false_str[1:-1]

        return false_str

    def _extract_true_part(self, if_content: str) -> Optional[str]:
        """
        从 if(condition, true_part, false_part) 中提取 true_part
        
        帆软模板格式示例:
          1. if(len(dept)>0," and au.USRMRC IN ('"+SUBSTITUTE(dept,",","','")+"') " ,"")
          2. if(len(start_time)==0, if(len(end_time)==0,"","<=end"), ...)
        
        核心挑战: true_part中包含嵌套引号和SUBSTITUTE函数
        """
        # 去掉 'if' 前缀 (如果存在)
        content = if_content
        if content.lower().startswith('if'):
            content = content[2:]  # 跳过 'if'
        
        # 确保以 '(' 开头
        content = content.strip()
        if not content.startswith('('):
            return None
        
        # 去掉外层括号
        content = content[1:]  # 跳过 '('
        
        # 策略: 手动扫描，跟踪括号深度和字符串状态
        depth = 0
        true_part_chars = []
        i = 0
        
        # 阶段1: 跳过condition部分，找到第一个顶层逗号后的引号
        while i < len(content):
            char = content[i]
            
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                # 找到第一个顶层逗号，condition结束
                i += 1  # 跳过逗号
                break
            i += 1
        
        # 跳过逗号后的空白
        while i < len(content) and content[i] in (' ', '\t', '\n'):
            i += 1
        
        if i >= len(content):
            return None
        
        # 阶段2: 提取true_part（从引号开始到匹配的引号结束）
        if content[i] != '"':
            return None
        
        i += 1  # 跳过开始引号
        
        # 现在提取直到匹配的结束引号（处理嵌套引号）
        paren_depth = 0  # 跟踪括号深度
        while i < len(content):
            char = content[i]
            
            if char == '\\' and i + 1 < len(content):
                true_part_chars.append(char)
                true_part_chars.append(content[i+1])
                i += 2
                continue
            
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth = max(0, paren_depth - 1)
            elif char == '"':
                # 只有在括号平衡时才考虑这是结束引号
                # 如果还有未闭合的括号(如SUBSTITUTE(...))，这个引号不是结束引号
                if paren_depth == 0:
                    remaining = content[i+1:].lstrip()
                    if remaining.startswith(',') or remaining.startswith(')') or not remaining:
                        break
            
            true_part_chars.append(char)
            i += 1
        
        if true_part_chars:
            return ''.join(true_part_chars).strip()
        
        return None

    def _handle_empty_param(self, if_content: str, param_name: Optional[str], params: Dict) -> str:
        """
        处理参数为空时的逻辑
        
        对于时间参数的嵌套if: if(len(start_time)==0, if(len(end_time)==0,"默认","..."), ...)
        返回false_part或空字符串
        """
        import re
        
        # 检查是否是时间相关的特殊嵌套结构
        if param_name and 'time' in param_name.lower():
            # 时间参数的复杂嵌套: if(len(start_time)==0, if(len(end_time)==0,"","<=end"), ...)
            # 当start_time和end_time都为空时，应该返回 "" (不过滤时间)
            
            # 尝试提取最内层的默认值（当所有参数都为空时）
            default_value = self._extract_nested_default(if_content)
            if default_value is not None:
                return default_value
        
        # 默认行为：返回空字符串（移除条件）
        return ''

    def _extract_nested_default(self, if_content: str) -> Optional[str]:
        """
        提取嵌套if结构的最终默认值
        
        示例: if(len(start_time)==0, if(len(end_time)==0,"默认值","else1"), "else2")
        当 start_time="" 且 end_time="" 时，返回 "默认值"
        """
        import re
        
        # 匹配模式: if(len(param)==0, inner_if, else_part)
        pattern = r'if\s*\(\s*len\s*\(\s*(\w+)\s*\)\s*==\s*0\s*,\s*(.+?)\s*,\s*(.+?)\s*\)$'
        
        # 简化处理：查找所有的 "..." 字符串字面量
        strings = re.findall(r'"([^"]*)"', if_content)
        
        if strings:
            # 第一个字符串通常是"全空"时的默认值
            # 对于时间条件，这通常是 "" (不过滤)
            return strings[0] if strings else ''
        
        return None

    def _final_cleanup(self, sql: str) -> str:
        """最终清理"""
        
        # 【关键】处理管道符字符串连接 → CONCAT()
        pipe_count = 0
        def replace_pipe(match):
            nonlocal pipe_count
            pipe_count += 1
            return f"CONCAT({match.group(1)}, '|', {match.group(2)})"
        
        # 模式: 字段 + '|' + 字段（使用[^'\s]+确保正确匹配）
        sql = re.sub(
            r"([^'\s]+)\s*\+\s*'\|'\s*\+\s*([^'\s]+)",
            replace_pipe,
            sql
        )
        
        if pipe_count > 0:
            print(f"       修复了 {pipe_count} 个管道符连接")
        
        # 【v3.3修复】清理孤立的SQL注释行（如 --${if()} 被清空后留下的 --）
        # 这些注释行如果和后续SQL连在同一行，会把GROUP BY等变成注释
        sql = re.sub(r'^\s*--\s*$', '', sql, flags=re.MULTILINE)  # 只有--的行
        sql = re.sub(r'^\s*--\s*部门筛选\s*$', '', sql, flags=re.MULTILINE)  # -- 部门筛选
        sql = re.sub(r'^\s*--\s*时间范围筛选\s*$', '', sql, flags=re.MULTILINE)  # -- 时间范围筛选
        sql = re.sub(r'^\s*--\s*工单类型筛选\s*$', '', sql, flags=re.MULTILINE)  # -- 工单类型筛选
        sql = re.sub(r'^\s*--\s*工单子类筛选\s*$', '', sql, flags=re.MULTILINE)  # -- 工单子类筛选
        
        # 移除 WHERE AND/OR 开头的问题
        sql = re.sub(r'\bWHERE\s+(AND|OR)\b', 'WHERE', sql, flags=re.IGNORECASE)
        
        # 移除连续的 AND 或 OR
        sql = re.sub(r'\bAND\s+AND\b', 'AND', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bOR\s+OR\b', 'OR', sql, flags=re.IGNORECASE)
        
        # 移除空WHERE
        sql = re.sub(r'\bWHERE\s*;?$', ';', sql)
        sql = re.sub(r'\bWHERE\s*\)', ')', sql)
        
        # 清理多余空格
        sql = re.sub(r' {2,}', ' ', sql)
        
        # 确保以分号结尾
        sql = sql.rstrip().rstrip(';') + ';'
        
        return sql
    
    def _validate(self, sql: str) -> Tuple[bool, List[str]]:
        """基本验证"""
        
        warnings = []
        valid = True
        
        if len(sql) < 20:
            warnings.append("SQL过短")
            valid = False
        
        if not re.search(r'\b(SELECT|WITH)\b', sql, re.IGNORECASE):
            warnings.append("缺少SELECT/WITH")
            valid = False
        
        if '${' in sql:
            warnings.append("仍有${}")
            valid = False
        
        return valid, warnings
    
    def get_stats(self) -> Dict:
        return self.stats.copy()


def test_smart_converter():
    """测试智能转换器"""
    
    print("\n" + "=" * 70)
    print("[TEST] SmartSQLConverter v2.0")
    print("=" * 70)
    
    converter = SmartSQLConverter()
    
    test_sql = '''WITH Ordered AS (
    SELECT TOP 500 RECCODE, RECFLOCODE, CREATEDBY, LASTSAVED
    FROM ATWORKFLOWRECORDS
),
workflow_entity AS (
    SELECT DISTINCT da.RECCODE, da.CREATEDBY
    FROM Ordered da
    LEFT JOIN ATUSERS au ON da.CREATEDBY = au.USRCODE
    WHERE au.USRDESC IS NOT NULL
      -- 部门筛选
      ${if(len(dept)>0," AND au.USRMRC IN ('"+SUBSTITUTE(dept,",","','")+"') " ,"")}
      -- 时间范围筛选
      ${if(len(start_time)==0,"", "AND da.LASTSAVED >= '"+start_time+"' ")}
)
SELECT COUNT(DISTINCT RECCODE + '|' + RECFLOCODE) AS 工单数量
FROM workflow_entity;'''
    
    params = {'dept': 'CI', 'start_time': '2026-01-01'}
    api_meta = {
        'name': '工单数量统计',
        'field_mapping': {
            'dept': 'au.USRMRC',
            'start_time': 'da.LASTSAVED',
        }
    }
    
    print(f"\n输入长度: {len(test_sql)} 字符")
    print(f"参数: {params}\n")
    
    converted, warnings = converter.convert(test_sql, params, api_meta)
    
    if converted:
        print(f"\n✅ 成功! 输出长度: {len(converted)} 字符\n")
        
        if warnings:
            print("警告:")
            for w in warnings:
                print(f"  ⚠️  {w}\n")
        
        stats = converter.get_stats()
        print("统计:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        print("\n转换后SQL:")
        print("-" * 70)
        print(converted)
        print("-" * 70)


if __name__ == '__main__':
    test_smart_converter()
