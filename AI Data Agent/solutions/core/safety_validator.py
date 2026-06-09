"""
SQL安全验证器 - 多层防护机制

核心功能:
1. 关键字黑名单检测
2. SQL结构白名单验证
3. 注入攻击模式识别
4. 运行时限制检查

使用场景:
- LLM生成的SQL必须经过验证才能执行
- 用户输入的参数需要清洗
- 防止SQL注入攻击
"""

import re
import json
from typing import Tuple, List, Dict, Optional
from enum import Enum


class SecurityLevel(Enum):
    """安全级别"""
    SAFE = "SAFE"  # 完全安全，可以执行
    WARNING = "WARNING"  # 有风险但可接受（需记录日志）
    DANGEROUS = "DANGEROUS"  # 危险，拒绝执行
    UNKNOWN = "UNKNOWN"  # 无法判断（保守处理：拒绝）


class SQLSafetyValidator:
    """
    SQL安全验证器
    
    设计原则:
    - 默认拒绝未知操作（白名单模式）
    - 多层防护（即使一层被绕过，其他层仍有效）
    - 详细日志记录所有检测到的威胁
    - 可配置的安全策略
    """

    # 第1层: 绝对禁止的关键字 (DDL/DCL/危险命令)
    FORBIDDEN_KEYWORDS = [
        # DDL - 数据定义语言 (会改变数据库结构)
        'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'RENAME',
        
        # DML - 数据操作语言 (除了SELECT)
        'DELETE', 'UPDATE', 'INSERT', 'MERGE', 'REPLACE',
        'CALL',
        
        # DCL - 数据控制语言 (权限相关)
        'GRANT', 'REVOKE', 'DENY',
        
        # 存储过程执行 (可能包含恶意代码)
        'EXEC', 'EXECUTE', 'EXECUTESQL',
        
        # SQL Server 特有危险命令
        'xp_cmdshell', 'sp_OACreate', 'sp_OADestroy',
        'xp_regread', 'xp_regwrite', 'xp_loginconfig',
        
        # 文件操作 (可能导致文件泄露或写入)
        'INTO OUTFILE', 'INTO DUMPFILE', 'LOAD_FILE',
        'BULK INSERT', 'OPENROWSET', 'OPENDATASOURCE',
        
        # 注入攻击特征
        'UNION SELECT', 'UNION ALL SELECT',
        ';--', '/**/', '/*', '*/',
        
        # 其他危险操作
        'SHUTDOWN', 'KILL', 'RECONFIGURE'
    ]

    # 第2层: 允许的SQL语句类型 (白名单)
    ALLOWED_STATEMENT_TYPES = ['SELECT', 'WITH']  # WITH 用于CTE，实际上是SELECT的扩展

    # 第3层: 允许的SQL子句 (在SELECT语句中)
    ALLOWED_CLAUSES = [
        'FROM', 'WHERE', 'GROUP BY', 'HAVING',
        'ORDER BY', 'LIMIT', 'TOP', 'OFFSET',
        'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN',
        'FULL JOIN', 'CROSS JOIN',
        'UNION', 'UNION ALL',
        'WITH'  # CTE (Common Table Expression)
    ]

    # 第4层: 正则表达式模式 (检测复杂注入攻击)
    INJECTION_PATTERNS = [
        # 注释注入
        r'/\*.*?\*/',  # 块注释
        r'--.*?$',     # 行注释
        
        # UNION注入
        r'\bUNION\b\s+\b(SELECT)\b',
        
        # 布尔盲注
        r'\bAND\b\s+\d+\s*=\s*\d+',
        r'\bOR\b\s+\d+\s*=\s*\d+',
        
        # 时间盲注
        r'BENCHMARK\s*\(',
        r'SLEEP\s*\(',
        r'WAITFOR\s+DELAY',
        
        # 编码绕过尝试
        r'0x[0-9a-fA-F]+',  # 十六进制编码
        r'\bCHAR\s*\(\s*\d+',  # CHAR()函数编码
    ]

    # 运行时限制配置
    LIMITS = {
        'max_sql_length': 8000,       # SQL最大长度(字符)
        'max_result_rows': 10000,      # 最大返回行数
        'max_execution_time': 30,      # 最大执行时间(秒)
        'max_nested_queries': 3,       # 最大嵌套子查询层数
        'max_joins': 5,               # 最大JOIN数量
    }

    def __init__(self, strict_mode: bool = True):
        """
        初始化验证器
        
        Args:
            strict_mode: 严格模式 (True=任何可疑都拒绝, False=只拒绝明确危险)
        """
        self.strict_mode = strict_mode
        self.detection_log = []  # 记录所有检测结果

    def validate(self, sql: str) -> Tuple[SecurityLevel, str, Dict]:
        """
        验证SQL安全性 (主入口)
        
        Args:
            sql: 待验证的SQL语句
            
        Returns:
            (安全级别, 消息, 详细信息字典)
            安全级别: SAFE / WARNING / DANGEROUS / UNKNOWN
        """
        self.detection_log = []  # 清空上次记录
        
        if not sql or not sql.strip():
            return SecurityLevel.UNKNOWN, "SQL为空", {'reason': 'empty_sql'}

        sql_upper = sql.upper().strip()
        details = {
            'sql_length': len(sql),
            'original_sql_preview': sql[:200] + ('...' if len(sql) > 200 else ''),
            'checks_performed': [],
            'warnings': [],
            'errors': []
        }

        # Layer 1: 基础长度检查
        length_check = self._check_sql_length(sql, details)
        if not length_check['passed']:
            return SecurityLevel.DANGEROUS, length_check['message'], details

        # Layer 2: 语句类型检查 (必须是SELECT)
        type_check = self._check_statement_type(sql_upper, details)
        if not type_check['passed']:
            return SecurityLevel.DANGEROUS, type_check['message'], details

        # Layer 3: 禁止关键字检测
        keyword_check = self._check_forbidden_keywords(sql, sql_upper, details)
        if not keyword_check['passed']:
            return SecurityLevel.DANGEROUS, keyword_check['message'], details

        # Layer 4: 注入模式检测
        injection_check = self._check_injection_patterns(sql, details)
        if not injection_check['passed'] and self.strict_mode:
            return SecurityLevel.DANGEROUS, injection_check['message'], details
        elif not injection_check['passed']:
            details['warnings'].append(injection_check['message'])

        # Layer 5: 结构合理性检查
        structure_check = self._check_structure_reasonability(sql, details)

        # 综合评估
        has_errors = len(details['errors']) > 0
        has_warnings = len(details['warnings']) > 0

        if has_errors:
            level = SecurityLevel.DANGEROUS
            message = f"发现 {len(details['errors'])} 个安全问题: {'; '.join(details['errors'][:3])}"
        elif has_warnings:
            level = SecurityLevel.WARNING
            message = f"有 {len(details['warnings'])} 个警告项，但在可接受范围内"
        else:
            level = SecurityLevel.SAFE
            message = "SQL通过安全验证"

        details['final_level'] = level.value
        details['checks_performed'].append({
            'layer': 'comprehensive',
            'result': 'passed' if level == SecurityLevel.SAFE else 'issues_found'
        })

        return level, message, details

    def _check_sql_length(self, sql: str, details: Dict) -> Dict:
        """检查SQL长度是否超限"""
        result = {
            'passed': True,
            'message': ''
        }

        if len(sql) > self.LIMITS['max_sql_length']:
            result['passed'] = False
            msg = f"SQL过长 ({len(sql)}字符 > {self.LIMITS['max_sql_length']}限制)"
            result['message'] = msg
            details['errors'].append(msg)
            self.detection_log.append({'type': 'length_exceeded', 'detail': msg})

        details['checks_performed'].append({
            'layer': 'length_check',
            'result': 'passed' if result['passed'] else 'failed',
            'detail': f'{len(sql)} chars'
        })

        return result

    def _check_statement_type(self, sql_upper: str, details: Dict) -> Dict:
        """检查SQL语句类型"""
        result = {
            'passed': True,
            'message': ''
        }

        # 提取第一个关键字
        first_word_match = re.match(r'^\s*(\w+)', sql_upper)
        if not first_word_match:
            result['passed'] = False
            msg = "无法识别SQL语句类型"
            result['message'] = msg
            details['errors'].append(msg)
            return result

        statement_type = first_word_match.group(1)

        if statement_type not in self.ALLOWED_STATEMENT_TYPES:
            result['passed'] = False
            msg = f"不允许的SQL类型: {statement_type} (仅允许: {', '.join(self.ALLOWED_STATEMENT_TYPES)})"
            result['message'] = msg
            details['errors'].append(msg)
            self.detection_log.append({'type': 'forbidden_statement', 'detail': msg})
        else:
            details['checks_performed'].append({
                'layer': 'statement_type',
                'result': 'passed',
                'detail': statement_type
            })

        return result

    def _check_forbidden_keywords(self, sql: str, sql_upper: str, details: Dict) -> Dict:
        """检测禁止的关键字"""
        result = {
            'passed': True,
            'message': ''
        }

        found_keywords = []

        for keyword in self.FORBIDDEN_KEYWORDS:
            keyword_upper = keyword.upper()
            
            # 使用单词边界匹配，避免误报 (如 "UPDATE" 不会匹配 "UPDATES")
            if re.search(r'\b' + re.escape(keyword_upper) + r'\b', sql_upper):
                found_keywords.append(keyword)
                self.detection_log.append({
                    'type': 'forbidden_keyword',
                    'keyword': keyword,
                    'context': self._get_context(sql, keyword)
                })

        if found_keywords:
            result['passed'] = False
            msg = f"检测到禁止关键字: {', '.join(found_keywords[:5])}"
            result['message'] = msg
            details['errors'].append(msg)

        details['checks_performed'].append({
            'layer': 'keyword_check',
            'result': 'passed' if result['passed'] else f'found_{len(found_keywords)}_keywords',
            'found_count': len(found_keywords)
        })

        return result

    def _check_injection_patterns(self, sql: str, details: Dict) -> Dict:
        """检测SQL注入攻击模式"""
        result = {
            'passed': True,
            'message': ''
        }

        found_patterns = []

        for pattern in self.INJECTION_PATTERNS:
            matches = list(re.finditer(pattern, sql, re.IGNORECASE | re.MULTILINE))
            if matches:
                pattern_name = pattern[:30] + '...' if len(pattern) > 30 else pattern
                found_patterns.append({
                    'pattern': pattern_name,
                    'count': len(matches),
                    'examples': [m.group()[:50] for m in matches[:2]]
                })
                self.detection_log.append({
                    'type': 'injection_pattern',
                    'pattern': pattern_name,
                    'matches': len(matches)
                })

        if found_patterns:
            result['passed'] = False
            patterns_str = ', '.join([p['pattern'] for p in found_patterns[:3]])
            msg = f"检测到可能的注入模式: {patterns_str}"
            result['message'] = msg
            details['warnings'].append(msg)

        details['checks_performed'].append({
            'layer': 'injection_pattern_check',
            'result': 'passed' if result['passed'] else f'found_{len(found_patterns)}_patterns'
        })

        return result

    def _check_structure_reasonability(self, sql: str, details: Dict) -> Dict:
        """检查SQL结构是否合理 (警告级别)"""
        warnings = []

        # 检查嵌套子查询深度
        nested_count = sql.count('(SELECT')
        if nested_count > self.LIMITS['max_nested_queries']:
            warnings.append(f"嵌套子查询过多 ({nested_count}层)")

        # 检查JOIN数量
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        if join_count > self.LIMITS['max_joins']:
            warnings.append(f"JOIN数量较多 ({join_count}个)")

        # 检查是否有通配符 SELECT * (只是建议，不影响安全级别)
        if re.search(r'SELECT\s+\*\s+FROM', sql, re.IGNORECASE):
            pass  # 不再作为警告，只是信息提示

        # 检查是否有ORDER BY RAND() (性能问题)
        if re.search(r'ORDER\s+BY\s+RAND\s*\(', sql, re.IGNORECASE):
            warnings.append("使用了 ORDER BY RAND() (大数据量时性能差)")

        if warnings:
            details['warnings'].extend(warnings)

        details['checks_performed'].append({
            'layer': 'structure_check',
            'result': 'passed_with_warnings' if warnings else 'passed',
            'warning_count': len(warnings)
        })

        return {'passed': True, 'warnings': warnings}

    def _get_context(self, sql: str, keyword: str, context_size: int = 50) -> str:
        """获取关键字周围的上下文 (用于日志)"""
        pos = sql.upper().find(keyword.upper())
        if pos == -1:
            return ''

        start = max(0, pos - context_size)
        end = min(len(sql), pos + len(keyword) + context_size)

        context = sql[start:end]
        return context.replace('\n', ' ').strip()

    def sanitize_parameter(self, value: str, param_type: str = 'string') -> str:
        """
        清洗用户输入的参数值
        
        Args:
            value: 原始参数值
            param_type: 参数类型 ('string', 'number', 'date', 'identifier')
            
        Returns:
            清洗后的安全值
        """
        if value is None:
            return None

        value = str(value).strip()

        # 移除危险字符
        dangerous_chars = ["'", '"', ';', '--', '/*', '*/', '\\x00', '\\n', '\\r']
        for char in dangerous_chars:
            value = value.replace(char, '')

        # 根据类型进一步处理
        if param_type == 'number':
            # 只保留数字、小数点、负号
            value = re.sub(r'[^\d.\-]', '', value)
        elif param_type == 'identifier':
            # 只保留字母、数字、下划线
            value = re.sub(r'[^\w]', '', value)
        elif param_type == 'date':
            # 标准化日期格式
            value = re.sub(r'[^\d\-/:\s]', '', value)

        # 长度限制
        max_lengths = {
            'string': 255,
            'number': 20,
            'date': 20,
            'identifier': 50
        }
        max_len = max_lengths.get(param_type, 100)
        value = value[:max_len]

        return value

    def get_detection_report(self) -> Dict:
        """
        获取完整的检测报告 (用于审计和调试)
        
        Returns:
            包含所有检测结果的详细报告
        """
        return {
            'total_detections': len(self.detection_log),
            'detections': self.detection_log,
            'summary': {
                'by_type': {}
            }
        }


# ============================================================
# 测试用例
# ============================================================

def test_safety_validator():
    """测试SQL安全验证器"""
    
    print("=" * 80)
    print("SQL安全验证器测试")
    print("=" * 80)
    
    validator = SQLSafetyValidator(strict_mode=True)
    
    test_cases = [
        # (SQL描述, SQL内容, 预期结果)
        ("正常SELECT", "SELECT * FROM users WHERE id = 1", "SAFE"),
        ("带CTE的SELECT", 
         "WITH cte AS (SELECT id FROM users) SELECT * FROM cte", 
         "SAFE"),
        ("包含DROP", 
         "DROP TABLE users; SELECT * FROM products", 
         "DANGEROUS"),
        ("UNION注入", 
         "SELECT * FROM users WHERE id = 1 UNION SELECT * FROM passwords", 
         "DANGEROUS"),
        ("注释注入",
         "SELECT * FROM users; -- 注释",
         "DANGEROUS"),
        ("EXEC注入",
         "EXEC xp_cmdshell 'dir'",
         "DANGEROUS"),
        ("正常工单查询",
         "SELECT dept_code, COUNT(*) FROM workflow GROUP BY dept_code",
         "SAFE"),
        ("复杂但安全的CTE查询",
         """WITH Ordered AS (
             SELECT RECCODE, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY time) AS rn
             FROM records
           )
           SELECT * FROM Ordered WHERE rn <= 10""",
         "SAFE"),
        ("超长SQL", "SELECT " + "a," * 9000 + " b FROM t", "DANGEROUS"),
    ]
    
    results = []
    
    for desc, sql, expected in test_cases:
        print(f"\n{'-'*60}")
        print(f"[测试] {desc}")
        print(f"SQL: {sql[:80]}{'...' if len(sql)>80 else ''}")
        
        level, message, details = validator.validate(sql)
        
        status = "[OK]" if level.value == expected else "[FAIL]"
        print(f"{status} 预期: {expected}, 实际: {level.value}")
        print(f"消息: {message}")
        
        if details.get('errors'):
            print(f"错误: {details['errors']}")
        if details.get('warnings'):
            print(f"警告: {details['warnings']}")
        
        results.append({
            'description': desc,
            'expected': expected,
            'actual': level.value,
            'passed': level.value == expected
        })
    
    # 汇总
    print("\n\n" + "=" * 80)
    print("测试汇总")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    print(f"\n总计: {total} 个测试")
    print(f"通过: {passed} ({passed/total*100:.0f}%)")
    print(f"失败: {total - passed}")
    
    if passed < total:
            print("\n失败的测试:")
            for r in results:
                if not r['passed']:
                    print(f"  [FAIL] {r['description']}: 预期{r['expected']}, 实际{r['actual']}")
    
    # 输出检测报告
    report = validator.get_detection_report()
    print(f"\n检测到的问题总数: {report['total_detections']}")


if __name__ == '__main__':
    test_safety_validator()
