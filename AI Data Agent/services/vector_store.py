"""
报表语义向量存储与检索模块
基于Embedding API构建报表数据集的向量索引，支持语义检索最相关的数据集
降级链: Embedding API → 本地TF-IDF语义匹配
"""
import json
import hashlib
import os
import re
from typing import List, Dict, Optional
from pathlib import Path
from collections import Counter

import numpy as np

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LLMConfig
from utils.logger import logger

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CACHE_DIR = Path(__file__).parent.parent / ".vector_cache"

_PROVIDER_MODELS = {
    "bigmodel.cn": ("embedding-2", 1024),
    "zhipuai": ("embedding-2", 1024),
    "openai.com": ("text-embedding-3-small", 1536),
    "api.openai.com": ("text-embedding-3-small", 1536),
    "dashscope.aliyuncs.com": ("text-embedding-v3", 1024),
}


def _detect_embedding_config(api_base: str) -> tuple:
    """根据API Base URL自动检测Embedding模型"""
    if not api_base:
        return EMBEDDING_MODEL, EMBEDDING_DIM
    api_base_lower = api_base.lower()
    for pattern, (model, dim) in _PROVIDER_MODELS.items():
        if pattern in api_base_lower:
            return model, dim
    return EMBEDDING_MODEL, EMBEDDING_DIM


_fallback_tfidf_available = True

_sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
except ImportError:
    pass


class _TFIDFSemanticMatcher:
    """
    轻量级TF-IDF语义匹配器（无需下载模型，纯Python实现）
    使用字符级n-gram + 词级分词 + 业务同义词扩展
    """

    _SYNONYM_GROUPS = [
        {"工单", "任务", "流程", "审批"},
        {"效率", "效能", "速度", "处理"},
        {"库存", "仓储", "物料", "存货"},
        {"质量", "检验", "检测", "QA", "QC"},
        {"设备", "机器", "设施", "装备"},
        {"验收", "接收", "入库", "收货"},
        {"对比", "比较", "相比", "差异"},
        {"部门", "科室", "团队", "组织"},
        {"异常", "问题", "缺陷", "不合格"},
        {"趋势", "变化", "走势", "波动"},
        {"完成", "结案", "关闭", "处理完"},
        {"维修", "保养", "维护", "检修"},
        {"采购", "购买", "进货", "下单"},
        {"生产", "制造", "加工", "产出"},
        {"近效期", "过期", "有效期", "保质期"},
        {"参观", "访问", "来访", "接待"},
        {"看板", "仪表盘", "概览", "总览"},
    ]

    def __init__(self):
        self._synonym_map = {}
        for group in self._SYNONYM_GROUPS:
            for word in group:
                self._synonym_map[word] = group

        self._vocab = {}
        self._idf = {}
        self._doc_vectors = None
        self._dim = 256

    def _tokenize(self, text: str) -> List[str]:
        """中文分词：字符bigram + 关键词提取"""
        tokens = []
        text = text.lower().strip()

        for word in re.findall(r'[a-z]{2,}|\d+|[a-z]\d+', text):
            tokens.append(word)

        for i in range(len(text) - 1):
            bigram = text[i:i + 2]
            if re.match(r'[\u4e00-\u9fff]{2}', bigram):
                tokens.append(bigram)

        for i in range(len(text) - 2):
            trigram = text[i:i + 3]
            if re.match(r'[\u4e00-\u9fff]{3}', trigram):
                tokens.append(trigram)

        expanded = set(tokens)
        for t in tokens:
            if t in self._synonym_map:
                for syn in self._synonym_map[t]:
                    if len(syn) >= 2:
                        expanded.add(syn)
        return list(expanded)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """将文本转为稀疏向量"""
        tokens = self._tokenize(text)
        counter = Counter(tokens)
        vec = np.zeros(self._dim, dtype=np.float32)

        for token, count in counter.items():
            idx = hash(token) % self._dim
            idf_weight = self._idf.get(token, 1.0)
            vec[idx] += count * idf_weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def fit(self, texts: List[str]):
        """构建词汇表和IDF"""
        N = len(texts)
        doc_freq = Counter()
        all_tokens = []

        for text in texts:
            tokens = set(self._tokenize(text))
            for t in tokens:
                doc_freq[t] += 1
            all_tokens.extend(tokens)

        self._idf = {}
        for token, df in doc_freq.items():
            self._idf[token] = np.log((N + 1) / (df + 1)) + 1

        self._doc_vectors = np.array([self._text_to_vector(t) for t in texts], dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """检索最相似的文档，返回 [(index, score), ...]"""
        if self._doc_vectors is None:
            return []

        query_vec = self._text_to_vector(query)
        similarities = self._doc_vectors @ query_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class ReportVectorStore:
    """
    报表语义向量存储
    将report_config.json中的数据集信息向量化，支持语义检索
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client: Optional[OpenAI] = None
        self._local_model = None
        self._tfidf_matcher = _TFIDFSemanticMatcher()
        self._embeddings: Optional[np.ndarray] = None
        self._documents: List[Dict] = []
        self._use_faiss = False
        self._faiss_index = None
        self._embedding_model = EMBEDDING_MODEL
        self._embedding_dim = EMBEDDING_DIM
        self._use_embedding = False
        self._init_embedding_backend()

    @property
    def report_entries(self) -> int:
        """返回已索引的数据集数量"""
        return len(self._documents)

    def _init_embedding_backend(self):
        """初始化Embedding后端：优先OpenAI API，降级到本地模型"""
        if self.config.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    timeout=30,
                )
                self._embedding_model, self._embedding_dim = _detect_embedding_config(
                    self.config.api_base or ""
                )
                logger.info(f"向量存储：Embedding后端初始化成功 (model={self._embedding_model})")
                return
            except Exception as e:
                logger.warning(f"向量存储：客户端初始化失败: {e}")

        if _sentence_transformers_available:
            try:
                self._local_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                self._embedding_dim = 384
                logger.info("向量存储：降级到sentence-transformers本地模型")
                return
            except Exception as e:
                logger.warning(f"向量存储：本地模型加载失败: {e}")

        self._embedding_model = EMBEDDING_MODEL
        self._embedding_dim = EMBEDDING_DIM
        logger.warning("向量存储：无可用Embedding后端，语义检索功能不可用")

    def build_index(self, reports: List[Dict]):
        """
        从报表数据集列表构建向量索引

        Args:
            reports: report_config.json中reports字段的值，每个元素包含datasets列表
        """
        self._documents = []
        rich_texts = []

        for report_name, report_info in reports.items():
            datasets = report_info.get("datasets", [])
            description = report_info.get("description", "")
            primary_connection = report_info.get("primary_connection", "")
            primary_table = report_info.get("primary_table", "")

            for ds in datasets:
                doc = {
                    "report_name": report_name,
                    "dataset_name": ds.get("name", ""),
                    "connection": ds.get("connection", primary_connection),
                    "table": ds.get("table", primary_table),
                    "sql_template": ds.get("sql_template", ""),
                    "params": ds.get("params", []),
                    "report_description": description,
                    "filename": report_info.get("filename", ""),
                }
                self._documents.append(doc)
                rich_texts.append(self._build_rich_text({
                    **doc,
                    "report_name": report_name,
                }))

        if not rich_texts:
            logger.warning("向量存储：无数据集可索引")
            return

        self._tfidf_matcher.fit(rich_texts)
        logger.info(f"向量存储：TF-IDF语义索引构建完成 ({len(rich_texts)}条)")

        config_hash = self._compute_config_hash(reports)
        cache_path = CACHE_DIR / f"index_{config_hash}.npz"
        meta_path = CACHE_DIR / f"meta_{config_hash}.json"

        if cache_path.exists() and meta_path.exists():
            try:
                self._embeddings = np.load(cache_path)["embeddings"]
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._documents = json.load(f)
                self._build_search_index()
                self._use_embedding = True
                logger.info(f"向量存储：从缓存加载Embedding索引 ({len(self._documents)}条)")
                return
            except Exception as e:
                logger.warning(f"向量存储：缓存加载失败，尝试在线构建: {e}")

        logger.info(f"向量存储：尝试构建Embedding索引 ({len(rich_texts)}条数据集)...")
        self._embeddings = self._get_embeddings(rich_texts)

        if self._embeddings is not None and len(self._embeddings) > 0:
            self._build_search_index()
            self._save_cache(config_hash)
            self._use_embedding = True
            logger.info(f"向量存储：Embedding索引构建完成，维度={self._embeddings.shape[1]}")
        else:
            logger.info("向量存储：Embedding不可用，使用TF-IDF语义匹配")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        语义检索最相关的Top-K数据集
        优先使用Embedding向量检索，降级到TF-IDF语义匹配
        """
        if len(self._documents) == 0:
            logger.warning("向量存储：索引未构建，无法检索")
            return []

        if self._use_embedding and self._embeddings is not None:
            return self._search_via_embedding(query, top_k)

        return self._search_via_tfidf(query, top_k)

    def _search_via_tfidf(self, query: str, top_k: int) -> List[Dict]:
        """使用TF-IDF语义匹配检索"""
        results = self._tfidf_matcher.search(query, top_k)
        output = []
        for idx, score in results:
            if idx < len(self._documents):
                result = dict(self._documents[idx])
                result["score"] = round(score, 4)
                output.append(result)
        return output

    def _search_via_embedding(self, query: str, top_k: int) -> List[Dict]:
        """使用Embedding向量检索"""
        query_embedding = self._get_embeddings([query])
        if query_embedding is None or len(query_embedding) == 0:
            return self._search_via_tfidf(query, top_k)

        query_vec = query_embedding[0]

        if self._use_faiss and self._faiss_index is not None:
            distances, indices = self._faiss_index.search(
                query_vec.reshape(1, -1).astype(np.float32), min(top_k, len(self._documents))
            )
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                result = dict(self._documents[idx])
                result["score"] = float(dist)
                results.append(result)
            return results

        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = self._embeddings / norms

        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_normed = query_vec / query_norm

        similarities = normed @ query_normed
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            result = dict(self._documents[idx])
            result["score"] = float(similarities[idx])
            results.append(result)
        return results

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        调用Embedding API获取向量，失败时降级到本地模型
        """
        if self.client:
            try:
                all_embeddings = []
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = self.client.embeddings.create(
                        model=self._embedding_model,
                        input=batch,
                    )
                    batch_embs = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embs)
                return np.array(all_embeddings, dtype=np.float32)
            except Exception as e:
                logger.error(f"向量存储：Embedding API调用失败: {e}")
                if not self._local_model and _sentence_transformers_available:
                    try:
                        import os as _os
                        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
                        self._local_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                        self._embedding_dim = 384
                        logger.info("向量存储：降级到本地模型")
                    except Exception as e2:
                        logger.warning(f"向量存储：本地模型不可用（需预下载模型）: {e2}")

        if self._local_model:
            try:
                embeddings = self._local_model.encode(texts, show_progress_bar=False)
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.error(f"向量存储：本地模型编码失败: {e}")

        logger.error("向量存储：无可用Embedding后端")
        return np.array([], dtype=np.float32)

    def _build_rich_text(self, report: Dict) -> str:
        """
        将报表数据集的多维信息合并为富文本

        Args:
            report: 包含report_name, dataset_name, connection, table, sql_template, params, report_description等字段

        Returns:
            合并后的富文本字符串
        """
        parts = []

        report_name = report.get("report_name", "")
        if report_name:
            parts.append(f"报表: {report_name}")

        dataset_name = report.get("dataset_name", "")
        if dataset_name:
            parts.append(f"数据集: {dataset_name}")

        report_desc = report.get("report_description", "")
        if report_desc:
            parts.append(f"描述: {report_desc}")

        connection = report.get("connection", "")
        if connection:
            parts.append(f"数据源: {connection}")

        table = report.get("table", "")
        if table:
            parts.append(f"数据表: {table}")

        params = report.get("params", [])
        if params:
            parts.append(f"参数: {', '.join(params)}")

        sql = report.get("sql_template", "")
        if sql:
            sql_preview = sql[:300].replace("\n", " ").strip()
            parts.append(f"SQL概要: {sql_preview}")

        return " | ".join(parts)

    def _build_search_index(self):
        """构建搜索索引（FAISS或numpy）"""
        if self._embeddings is None or len(self._embeddings) == 0:
            return

        if self._use_faiss:
            dim = self._embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normed = (self._embeddings / norms).astype(np.float32)
            self._faiss_index.add(normed)
            logger.info("向量存储：FAISS索引构建完成")

    def _compute_config_hash(self, reports: Dict) -> str:
        """计算配置的哈希值，用于缓存校验"""
        content = json.dumps(reports, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]

    def _save_cache(self, config_hash: str):
        """保存索引缓存到本地文件"""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = CACHE_DIR / f"index_{config_hash}.npz"
            meta_path = CACHE_DIR / f"meta_{config_hash}.json"

            np.savez_compressed(cache_path, embeddings=self._embeddings)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._documents, f, ensure_ascii=False, indent=2)

            logger.info(f"向量存储：索引缓存已保存 (hash={config_hash})")
        except Exception as e:
            logger.warning(f"向量存储：缓存保存失败: {e}")


def get_vector_store() -> ReportVectorStore:
    """获取向量存储实例"""
    return ReportVectorStore()


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "report_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    store = ReportVectorStore()
    store.build_index(config_data.get("reports", {}))

    test_queries = ["库存数据", "工单处理效率", "物料质量验收"]
    for q in test_queries:
        results = store.search(q, top_k=3)
        print(f"\n查询: {q}")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['report_name']} - {r['dataset_name']}")
