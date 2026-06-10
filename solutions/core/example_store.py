#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example Store - 查询示例库（向量检索Few-Shot）

核心功能：
  1. 存储 query → api_id + parameters 的成功映射示例
  2. TF-IDF 向量语义检索，找到最相似的历史案例
  3. 自动从成功的查询中积累新示例
  4. 为 Layer 2 路由层提供 Few-Shot 上下文

架构：
  ExampleStore (存储层) ← JSON持久化
      ↓
  VectorRetriever (检索层) ← TF-IDF向量化 + 余弦相似度
      ↓
  NL2APIService._route_to_api() (消费层) ← 注入few-shot到FC system prompt
"""

import json
import os
import re
import math
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class Example:
    """单条查询示例"""
    id: str = ""                              # 唯一ID
    query: str = ""                           # 用户原始查询
    normalized_query: str = ""                 # 标准化后的查询（用于匹配）
    api_id: str = ""                           # 选中的API ID
    api_name: str = ""                         # API显示名称
    parameters: Dict[str, Any] = field(default_factory=dict)  # 提取的参数
    analysis_type: str = ""                    # 分析类型: extreme/ranking/comparison/statistic/trend/list
    department: str = ""                       # 涉及部门(如有)
    tags: List[str] = field(default_factory=list)           # 业务标签
    success: bool = True                      # 是否成功返回数据
    data_count: int = 0                       # 返回数据量
    created_at: str = ""                      # 创建时间
    source: str = "manual"                    # 来源: manual/auto/feedback
    confidence: float = 0.0                   # 医信度
    usage_count: int = 0                      # 被命中次数


class VectorRetriever:
    """
    TF-IDF 向量检索器
    
    无需外部依赖，纯Python实现：
    - 中文分词（基于字符N-gram）
    - TF-IDF 加权
    - 余弦相似度排序
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}       # 词 → 文档频率
        self.doc_count: int = 0                     # 总文档数
        self.doc_vectors: Dict[str, Dict[str, float]] = {}  # doc_id → {term: tfidf}
        self.doc_norms: Dict[str, float] = {}       # doc_id → 向量模长
        self.idf_cache: Dict[str, float] = {}       # term → IDF值
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        """
        中文文本分词
        
        策略：
          1. 提取中文连续片段（2字以上）
          2. 提取英文单词
          3. 提取数字+单位组合
          4. 生成 bigram 特征
        """
        if not text:
            return []

        text_lower = text.lower().strip()
        tokens = []

        # 中文片段（2个及以上连续中文字符）
        chinese_parts = re.findall(r'[\u4e00-\u9fff]{2,}', text_lower)
        tokens.extend(chinese_parts)

        # 单个中文字符也加入（用于短查询匹配）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text_lower)
        tokens.extend(chinese_chars)

        # 英文单词
        english_words = re.findall(r'[a-zA-Z]{2,}', text_lower)
        tokens.extend(english_words)

        # 部门代码等特殊标识
        dept_codes = re.findall(r'\b(CI|DS|EHS|FF|FM|LG|OM|PD|PM|QA|QC|TM|VM)\b', text_upper := text.upper())
        tokens.extend(dept_codes)

        # 数字
        numbers = re.findall(r'\d+', text_lower)
        tokens.extend(numbers)

        # Bigram（相邻词对）
        all_tokens = chinese_parts + english_words + dept_codes
        for i in range(len(all_tokens) - 1):
            tokens.append(f"{all_tokens[i]}_{all_tokens[i+1]}")

        return tokens

    def index_documents(self, documents: List[Tuple[str, str]]):
        """
        构建索引
        
        Args:
            documents: [(doc_id, text), ...]
        """
        self.doc_count = len(documents)
        self.vocabulary = {}
        self.doc_vectors = {}

        # 第一遍：统计词频和文档频率
        doc_tf: Dict[str, Counter] = {}
        df_counter: Counter = Counter()

        for doc_id, text in documents:
            tokens = self._tokenize(text)
            tf_counter = Counter(tokens)
            doc_tf[doc_id] = tf_counter

            for term in tf_counter:
                df_counter[term] += 1

        # 计算IDF
        self.idf_cache = {}
        for term, df in df_counter.items():
            self.idf_cache[term] = math.log((self.doc_count + 1) / (df + 1)) + 1
            self.vocabulary[term] = df

        # 第二遍：计算每个文档的TF-IDF向量
        self.doc_norms = {}
        for doc_id, tf in doc_tf.items():
            vec = {}
            for term, count in tf.items():
                tf_val = 1 + math.log(count) if count > 0 else 0
                idf_val = self.idf_cache.get(term, 1.0)
                vec[term] = tf_val * idf_val

            # L2归一化
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {k: v / norm for k, v in vec.items()}
                self.doc_norms[doc_id] = norm
            else:
                self.doc_norms[doc_id] = 1.0

            self.doc_vectors[doc_id] = vec

        self._built = True
        logger.info(f"[VectorRetriever] 索引构建完成 | 文档数: {self.doc_count} | 词表大小: {len(self.vocabulary)}")

    def _query_to_vector(self, query: str) -> Dict[str, float]:
        """将查询转为TF-IDF向量"""
        tokens = self._tokenize(query)
        tf_counter = Counter(tokens)

        vec = {}
        for term, count in tf_counter.items():
            tf_val = 1 + math.log(count) if count > 0 else 0
            idf_val = self.idf_cache.get(term, 1.0)
            vec[term] = tf_val * idf_val

        # L2归一化
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}

        return vec

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.1,
               filters: Dict[str, Any] = None) -> List[Dict]:
        """
        语义相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回最相似的K条
            min_score: 最小相似度阈值
            filters: 过滤条件 {department: "CI", analysis_type: "extreme", ...}
        
        Returns:
            [{doc_id, score, ...}, ...] 按score降序
        """
        if not self._built or not self.doc_vectors:
            return []

        q_vec = self._query_to_vector(query)

        scores = []
        for doc_id, d_vec in self.doc_vectors.items():
            # 余弦相似度（向量已归一化，直接点积）
            dot_product = sum(q_vec.get(t, 0) * d_vec.get(t, 0) for t in set(q_vec) | set(d_vec))

            if dot_product >= min_score:
                result = {'doc_id': doc_id, 'score': round(dot_product, 4)}
                scores.append(result)

        # 按分数降序
        scores.sort(key=lambda x: x['score'], reverse=True)

        results = scores[:top_k]

        logger.debug(f"[VectorRetriever] 搜索: \"{query[:30]}...\" → {len(results)} 条结果 (top_k={top_k})")
        if results:
            logger.debug(f"  最佳匹配: score={results[0]['score']}, doc={results[0]['doc_id']}")

        return results

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            'doc_count': self.doc_count,
            'vocab_size': len(self.vocabulary),
            'built': self._built,
        }


class ExampleStore:
    """
    示例库 - Few-Shot查询示例存储与检索
    
    功能：
    1. 预置种子示例（覆盖核心场景）
    2. 从成功查询自动积累新示例
    3. 向量语义检索最相似示例
    4. 持久化存储（JSON文件）
    """

    DEFAULT_STORE_PATH = "data/example_store.json"

    SEED_EXAMPLES = [
        # ====== 员工极值类 (employee + extreme) ======
        {
            "query": "CI部门完成工单数量最多的员工",
            "api_name_pattern": "人均效能",
            "parameters": {"dept": "CI"},
            "analysis_type": "extreme",
            "department": "CI",
            "tags": ["employee", "extreme", "dept"],
            "key_features": ["最多", "员工", "CI"],
        },
        {
            "query": "OM部门谁处理的工单最快",
            "api_name_pattern": "人均效能",
            "parameters": {"dept": "OM"},
            "analysis_type": "extreme",
            "department": "OM",
            "tags": ["employee", "extreme", "dept"],
            "key_features": ["最快", "谁", "OM"],
        },
        {
            "query": "QA部门完成工单最少的员工是谁",
            "api_name_pattern": "人均效能",
            "parameters": {"dept": "QA"},
            "analysis_type": "extreme",
            "department": "QA",
            "tags": ["employee", "extreme", "dept"],
            "key_features": ["最少", "谁", "QA"],
        },
        {
            "query": "PD部门效率最低的员工",
            "api_name_pattern": "人均效能",
            "parameters": {"dept": "PD"},
            "analysis_type": "extreme",
            "department": "PD",
            "tags": ["employee", "extreme", "dept"],
            "key_features": ["最低", "PD"],
        },
        # ====== 员工排名类 (employee + ranking) ======
        {
            "query": "RD部门的员工效能排名",
            "api_name_pattern": "人均效能",
            "parameters": {"dept": "RD"},
            "analysis_type": "ranking",
            "department": "RD",
            "tags": ["employee", "ranking", "dept"],
            "key_features": ["排名", "效能", "RD"],
        },
        {
            "query": "QC部门谁处理工单最多，列出前5名",
            "api_name_pattern": "人均效能",
            "parameters": {"dept": "QC"},
            "analysis_type": "ranking",
            "department": "QC",
            "tags": ["employee", "ranking", "dept", "top"],
            "key_features": ["排名", "前5", "QC"],
        },
        {
            "query": "PM部门员工处理速度排行",
            "api_name_pattern": "人均效能",
            "parameters": {"dept": "PM"},
            "analysis_type": "ranking",
            "department": "PM",
            "tags": ["employee", "ranking", "dept"],
            "key_features": ["排行", "速度", "PM"],
        },
        # ====== 部门对比类 (comparison) ======
        {
            "query": "各部门工单处理情况对比",
            "api_name_pattern": "各部门",
            "parameters": {},
            "analysis_type": "comparison",
            "department": "",
            "tags": ["department", "comparison", "all"],
            "key_features": ["各部门", "对比", "情况"],
        },
        {
            "query": "QA部门和PD部门谁处理的工单多",
            "api_name_pattern": "各部门",
            "parameters": {"dept": "QA"},
            "analysis_type": "comparison",
            "department": "QA",
            "tags": ["department", "comparison", "multi_dept"],
            "key_features": ["对比", "QA", "PD", "谁多"],
        },
        {
            "query": "OM和CI哪个部门效率更高",
            "api_name_pattern": "各部门",
            "parameters": {},
            "analysis_type": "comparison",
            "department": "",
            "tags": ["department", "comparison", "multi_dept"],
            "key_features": ["对比", "OM", "CI", "效率"],
        },
        # ====== 统计汇总类 (statistic) ======
        {
            "query": "所有部门的关键指标汇总",
            "api_name_pattern": "关键指标",
            "parameters": {},
            "analysis_type": "statistic",
            "department": "",
            "tags": ["summary", "statistic", "kpi", "all"],
            "key_features": ["关键指标", "汇总", "全部"],
        },
        {
            "query": "本月HR部门的整体工单情况",
            "api_name_pattern": "关键指标",
            "parameters": {"dept": "HR"},
            "analysis_type": "statistic",
            "department": "HR",
            "tags": ["dept", "statistic", "time", "HR"],
            "key_features": ["情况", "本月", "HR"],
        },
        {
            "query": "FIN部门最近一周的数据统计",
            "api_name_pattern": "关键指标",
            "parameters": {"dept": "FIN"},
            "analysis_type": "statistic",
            "department": "FIN",
            "tags": ["dept", "statistic", "time", "FIN"],
            "key_features": ["统计", "最近一周", "FIN"],
        },
        # ====== 时间趋势类 (trend) ======
        {
            "query": "最近一周的工单处理趋势",
            "api_name_pattern": "关键指标",
            "parameters": {},
            "analysis_type": "trend",
            "department": "",
            "tags": ["time", "trend", "recent"],
            "key_features": ["最近一周", "趋势", "走势"],
        },
        {
            "query": "本月CI部门的工单变化情况",
            "api_name_pattern": "关键指标",
            "parameters": {"dept": "CI", "start_time": "本月"},
            "analysis_type": "trend",
            "department": "CI",
            "tags": ["dept", "time", "trend", "CI"],
            "key_features": ["变化", "本月", "CI"],
        },
        # ====== 明细列表类 (list/detail) ======
        {
            "query": "列出HR部门的所有工单明细",
            "api_name_pattern": "关键指标",
            "parameters": {"dept": "HR"},
            "analysis_type": "list",
            "department": "HR",
            "tags": ["dept", "detail", "list", "HR"],
            "key_features": ["列出", "明细", "HR"],
        },
        {
            "query": "OM部门最近的工单列表",
            "api_name_pattern": "关键指标",
            "parameters": {"dept": "OM"},
            "analysis_type": "list",
            "department": "OM",
            "tags": ["dept", "list", "OM"],
            "key_features": ["列表", "最近", "OM"],
        },
        # ====== 耗时分布类 ======
        {
            "query": "PM部门的人员工单耗时分布",
            "api_name_pattern": "人均效能",
            "parameters": {"dept": "PM"},
            "analysis_type": "statistic",
            "department": "PM",
            "tags": ["employee", "distribution", "time", "PM"],
            "key_features": ["耗时", "分布", "PM"],
        },
        {
            "query": "各部门工单平均处理时长",
            "api_name_pattern": "各部门",
            "parameters": {},
            "analysis_type": "statistic",
            "department": "",
            "tags": ["department", "time", "avg", "all"],
            "key_features": ["平均", "耗时", "各部门"],
        },
        # ====== 完成率/状态类 ======
        {
            "query": "CI部门工单完成率是多少",
            "api_name_pattern": "完成情况",
            "parameters": {"dept": "CI"},
            "analysis_type": "statistic",
            "department": "CI",
            "tags": ["dept", "status", "completion", "CI"],
            "key_features": ["完成率", "CI"],
        },
        {
            "query": "未完成的工单有哪些",
            "api_name_pattern": "完成情况",
            "parameters": {},
            "analysis_type": "list",
            "department": "",
            "tags": ["status", "incomplete", "list"],
            "key_features": ["未完成", "哪些"],
        },
    ]

    def __init__(self, store_path: str = None, registry=None):
        self.store_path = store_path or self.DEFAULT_STORE_PATH
        self.registry = registry
        self.examples: Dict[str, Example] = {}
        self.retriever = VectorRetriever()
        self._loaded = False

        self._ensure_data_dir()
        self._load()

    def _ensure_data_dir(self):
        p = Path(self.store_path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        """从JSON加载示例库"""
        store_file = Path(self.store_path)
        if store_file.exists():
            try:
                with open(store_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                examples_data = data.get('examples', {})
                for ex_id, ex_dict in examples_data.items():
                    self.examples[ex_id] = Example(**ex_dict)

                self._rebuild_index()
                self._loaded = True
                logger.info(f"[ExampleStore] 已加载 {len(self.examples)} 条示例 from {self.store_path}")
            except Exception as e:
                logger.warning(f"[ExampleStore] 加载失败: {e}，使用种子示例")
                self._init_seed_examples()
        else:
            self._init_seed_examples()

    def _init_seed_examples(self):
        """初始化种子示例"""
        ts = time.strftime('%Y-%m-%d %H:%M:%S')

        for i, seed in enumerate(self.SEED_EXAMPLES):
            ex_id = f"seed_{i+1:03d}"
            ex = Example(
                id=ex_id,
                query=seed['query'],
                normalized_query=self._normalize(seed['query']),
                api_id="",  # 将在bind_to_registry时填充
                api_name=seed['api_name_pattern'],
                parameters=seed.get('parameters', {}),
                analysis_type=seed.get('analysis_type', ''),
                department=seed.get('department', ''),
                tags=seed.get('tags', []),
                success=True,
                created_at=ts,
                source='seed',
                confidence=0.95,
            )
            self.examples[ex_id] = ex

        self._rebuild_index()
        self._loaded = True
        self._save()
        logger.info(f"[ExampleStore] 初始化种子示例 {len(self.examples)} 条")

    def bind_to_registry(self):
        """将示例绑定到registry中的真实API（通过名称模糊匹配）"""
        if not self.registry:
            return

        matched = 0
        for ex_id, ex in self.examples.items():
            if not ex.api_id and ex.api_name:
                best_match = None
                best_score = 0

                for api_id, api_def in self.registry.apis.items():
                    name_lower = api_def.name.lower()
                    pattern_lower = ex.api_name.lower()

                    if pattern_lower in name_lower or name_lower in pattern_lower:
                        score = len(pattern_lower)
                        if score > best_score:
                            best_score = score
                            best_match = api_id
                            ex.api_name = api_def.name

                if best_match:
                    ex.api_id = best_match
                    matched += 1

        if matched > 0:
            logger.info(f"[ExampleStore] 绑定API: {matched}/{len(self.examples)} 条示例已关联")
            self._save()

    def _normalize(self, query: str) -> str:
        """标准化查询文本（用于更好的匹配）"""
        q = query.lower().strip()
        q = re.sub(r'\s+', ' ', q)
        q = re.sub(r'[？?！!。，,]', '', q)
        return q

    def _rebuild_index(self):
        """重建向量索引"""
        docs = [(ex_id, ex.normalized_query or ex.query) for ex_id, ex in self.examples.items()]
        if docs:
            self.retriever.index_documents(docs)

    def _generate_id(self) -> str:
        return f"auto_{int(time.time()*1000)}_{len(self.examples)}"

    def add_example(self, query: str, api_id: str, api_name: str,
                   parameters: Dict = None, success: bool = True,
                   data_count: int = 0, source: str = "auto",
                   analysis_type: str = "", department: str = "",
                   tags: List[str] = None) -> Optional[Example]:
        """
        添加一条新示例
        
        Returns:
            新创建的Example对象
        """
        ex_id = self._generate_id()
        ts = time.strftime('%Y-%m-%d %H:%M:%S')

        ex = Example(
            id=ex_id,
            query=query,
            normalized_query=self._normalize(query),
            api_id=api_id,
            api_name=api_name,
            parameters=parameters or {},
            analysis_type=analysis_type,
            department=department,
            tags=tags or [],
            success=success,
            data_count=data_count,
            created_at=ts,
            source=source,
            confidence=0.8 if success else 0.3,
        )

        self.examples[ex_id] = ex
        self._rebuild_index()
        self._save()

        logger.info(f"[ExampleStore] 新增示例 #{ex_id}: \"{query[:40]}...\" → {api_name}")
        return ex

    def add_from_result(self, user_query: str, result, max_store: int = 200):
        """
        从NL2APIResult自动积累示例
        
        只记录成功的、有意义的查询
        """
        if not result.success:
            return

        if len(self.examples) >= max_store:
            return

        # 避免重复
        norm_q = self._normalize(user_query)
        for ex in self.examples.values():
            if ex.normalized_query == norm_q:
                ex.usage_count += 1
                return

        # 推断标签
        tags = []
        query_lower = user_query.lower()

        if any(kw in query_lower for kw in ['员工', '谁', '人员']):
            tags.append('employee')
        if result.selected_api_id:
            api_def = self.registry.get_api(result.selected_api_id) if self.registry else None
            if api_def:
                tags.extend(api_def.tags[:3])

        self.add_example(
            query=user_query,
            api_id=result.selected_api_id,
            api_name=result.selected_api_name,
            parameters=result.api_parameters,
            success=True,
            data_count=result.data_count,
            source='auto',
            tags=tags,
        )

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.15,
               require_success: bool = True,
               department: str = None,
               analysis_type: str = None) -> List[Dict]:
        """
        语义检索最相似的示例
        
        Returns:
            [{
                'example': Example对象,
                'score': 相似度分数,
                'matched_features': 共同特征,
            }, ...]
        """
        raw_results = self.retriever.search(
            query=query,
            top_k=top_k * 2,
            min_score=min_score
        )

        results = []
        for r in raw_results:
            ex = self.examples.get(r['doc_id'])
            if not ex:
                continue

            if require_success and not ex.success:
                continue

            if department and ex.department and ex.department != department:
                continue

            if analysis_type and ex.analysis_type and ex.analysis_type != analysis_type:
                continue

            # 计算特征重叠度
            query_tokens = set(self.retriever._tokenize(query))
            ex_tokens = set(self.retriever._tokenize(ex.query))
            overlap = query_tokens & ex_tokens

            results.append({
                'example': ex,
                'score': r['score'],
                'matched_features': list(overlap)[:10],
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def get_few_shot_context(self, query: str, top_k: int = 3,
                             format: str = "text") -> str:
        """
        获取Few-Shot上下文文本（注入到LLM prompt中）
        
        Args:
            format: "text" | "json" | "fc"
        """
        examples = self.search(query, top_k=top_k)

        if not examples:
            return ""

        if format == "text":
            lines = ["## 历史相似查询参考"]
            for i, item in enumerate(examples, 1):
                ex = item['example']
                params_str = json.dumps(ex.parameters, ensure_ascii=False) if ex.parameters else "{}"
                lines.append(f"{i}. 查询: \"{ex.query}\"")
                lines.append(f"   → 选择API: {ex.api_name} | 参数: {params_str} | 类型: {ex.analysis_type}")
                lines.append(f"   (相似度: {item['score']:.2f})")
            return '\n'.join(lines)

        elif format == "json":
            output = []
            for item in examples:
                ex = item['example']
                output.append({
                    "query": ex.query,
                    "api_name": ex.api_name,
                    "api_id": ex.api_id,
                    "parameters": ex.parameters,
                    "analysis_type": ex.analysis_type,
                    "similarity": item['score'],
                })
            return json.dumps(output, ensure_ascii=False, indent=2)

        elif format == "fc":
            output = []
            for item in examples:
                ex = item['example']
                output.append({
                    "user": ex.query,
                    "assistant": {
                        "function_call": {
                            "name": ex.api_id or ex.api_name,
                            "arguments": ex.parameters,
                        }
                    },
                    "score": item['score'],
                })
            return json.dumps(output, ensure_ascii=False, indent=2)

        return ""

    def get_stats(self) -> Dict:
        """获取示例库统计信息"""
        total = len(self.examples)
        by_source = Counter(ex.source for ex in self.examples.values())
        by_type = Counter(ex.analysis_type for ex in self.examples.values())
        by_dept = Counter(ex.department for ex in self.examples.values() if ex.department)

        return {
            'total': total,
            'by_source': dict(by_source),
            'by_analysis_type': dict(by_type),
            'by_department': dict(by_dept),
            'retriever_stats': self.retriever.get_stats(),
            'store_path': self.store_path,
        }

    def _save(self):
        """持久化到JSON"""
        try:
            data = {
                'version': '1.0',
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'examples': {ex_id: asdict(ex) for ex_id, ex in self.examples.items()},
                'stats': self.get_stats(),
            }

            with open(self.store_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"[ExampleStore] 保存失败: {e}")


# ============================================================
# 全局单例
# ============================================================
_global_store: Optional[ExampleStore] = None


def get_example_store(registry=None) -> ExampleStore:
    """获取全局ExampleStore单例"""
    global _global_store
    if _global_store is None:
        _global_store = ExampleStore(registry=registry)
        if registry:
            _global_store.bind_to_registry()
    return _global_store


def reset_example_store():
    """重置全局单例（测试用）"""
    global _global_store
    _global_store = None


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("[TEST] Example Store + Vector Retrieval 测试")
    print("=" * 70)

    store = ExampleStore(store_path="_test_example_store.json")

    stats = store.get_stats()
    print(f"\n[统计]")
    print(f"  示例总数: {stats['total']}")
    print(f"  来源分布: {stats['by_source']}")
    print(f"  分析类型: {stats['by_analysis_type']}")
    print(f"  部门分布: {stats['by_department']}")

    # 测试向量检索
    test_queries = [
        "CI部门谁完成的工单最多",
        "各部门工单对比",
        "最近一周的趋势",
        "RD部门员工排名",
        "QA和PD哪个好",
        "列出HR的明细",
        "PM部门耗时分布",
        "本月FIN的情况",
        "未完成的工单",
        "OM部门效率最低的人",
    ]

    print(f"\n\n[向量检索测试] ({len(test_queries)} 个查询)")
    print("-" * 70)

    correct_count = 0
    for q in test_queries:
        results = store.search(q, top_k=1)

        if results:
            best = results[0]
            ex = best['example']
            score = best['score']

            # 判断是否合理匹配
            q_lower = q.lower()
            is_reasonable = False

            if any(kw in q_lower for kw in ['员工', '谁', '人员']):
                is_reasonable = 'employee' in ex.tags
            elif any(kw in q_lower for kw in ['部门', '对比', '哪个']):
                is_reasonable = 'department' in ex.tags or 'comparison' in ex.tags
            elif any(kw in q_lower for kw in ['趋势', '变化', '最近']):
                is_reasonable = 'trend' in ex.tags or 'time' in ex.tags
            elif any(kw in q_lower for kw in ['列出', '明细', '哪些']):
                is_reasonable = 'list' in ex.tags or 'detail' in ex.tags
            elif any(kw in q_lower for kw in ['汇总', '指标', '情况', '统计']):
                is_reasonable = 'statistic' in ex.tags or 'summary' in ex.tags
            elif any(kw in q_lower for kw in ['排名', '排行', '前']):
                is_reasonable = 'ranking' in ex.tags
            elif any(kw in q_lower for kw in ['耗时', '分布', '时长']):
                is_reasonable = 'distribution' in ex.tags or 'time' in ex.tags
            else:
                is_reasonable = score >= 0.3

            icon = "✅" if is_reasonable else "⚠️"
            if is_reasonable:
                correct_count += 1

            print(f"\n  {icon} [{score:.3f}] \"{q}\"")
            print(f"     → 匹配: \"{ex.query}\"")
            print(f"     → API: {ex.api_name} | 类型:{ex.analysis_type} | 标签:{ex.tags}")
            if best.get('matched_features'):
                print(f"     → 共同特征: {best['matched_features'][:6]}")
        else:
            print(f"\n  ❌ 无匹配结果: \"{q}\"")

    accuracy = correct_count / len(test_queries) * 100
    print(f"\n{'='*70}")
    print(f"  检索准确率: {correct_count}/{len(test_queries)} = {accuracy:.0f}%")
    print(f"{'='*70}")

    # 测试Few-Shot上下文生成
    print(f"\n\n[Few-Shot上下文示例]")
    ctx = store.get_few_shot_context("CI部门完成最多的员工是谁", top_k=2, format="text")
    print(ctx[:500])
