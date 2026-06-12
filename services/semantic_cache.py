"""
语义缓存模块 - 缓存已验证的高频查询，相似问题直接返回
"""
import json
import os
import re
import hashlib
import threading
from typing import Optional
from datetime import datetime

from utils.logger import logger


class SemanticCache:
    """语义缓存 - 缓存已验证的高频查询，相似问题直接返回"""

    MAX_CACHE_SIZE = 500
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, cache_file: str = None):
        if cache_file is None:
            cache_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'data', 'semantic_cache.json'
            )
        self._cache_file = cache_file
        self._lock = threading.Lock()
        self._cache = []
        self._load()

    def lookup(self, query: str) -> Optional[dict]:
        """查找语义相似的缓存项"""
        if not query:
            return None

        normalized = self._normalize(query)

        with self._lock:
            # 1. 精确匹配（归一化后）
            for entry in self._cache:
                if entry.get("normalized") == normalized and entry.get("verified"):
                    entry["hit_count"] = entry.get("hit_count", 0) + 1
                    self._save()
                    logger.info(f"语义缓存精确命中: {query}")
                    return {
                        "dsl": entry.get("dsl"),
                        "sql": entry.get("sql"),
                        "similarity": 1.0,
                        "source": "semantic_cache",
                    }

            # 2. 关键词相似度匹配（Jaccard相似度 > 阈值）
            best_match = None
            best_similarity = 0.0

            for entry in self._cache:
                if not entry.get("verified"):
                    continue
                sim = self._similarity(normalized, entry.get("normalized", ""))
                if sim > self.SIMILARITY_THRESHOLD and sim > best_similarity:
                    best_similarity = sim
                    best_match = entry

            if best_match is not None:
                best_match["hit_count"] = best_match.get("hit_count", 0) + 1
                self._save()
                logger.info(f"语义缓存相似命中: {query} (相似度={best_similarity:.2f})")
                return {
                    "dsl": best_match.get("dsl"),
                    "sql": best_match.get("sql"),
                    "similarity": round(best_similarity, 4),
                    "source": "semantic_cache",
                }

            return None

    def store(self, query: str, dsl: dict, sql: str, result_data: dict = None, verified: bool = False, connection_name: str = ""):
        """存储查询到缓存"""
        if not query:
            return

        normalized = self._normalize(query)

        # 计算结果的hash
        result_hash = ""
        if result_data is not None:
            try:
                result_str = json.dumps(result_data, ensure_ascii=False, sort_keys=True)
                result_hash = hashlib.md5(result_str.encode("utf-8")).hexdigest()
            except (TypeError, ValueError):
                result_hash = ""

        entry = {
            "query": query,
            "normalized": normalized,
            "dsl": dsl,
            "sql": sql,
            "connection_name": connection_name,
            "result_hash": result_hash,
            "verified": verified,
            "hit_count": 0,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with self._lock:
            # 检查是否已存在相同归一化查询，若存在则更新
            for i, existing in enumerate(self._cache):
                if existing.get("normalized") == normalized:
                    self._cache[i] = entry
                    self._save()
                    logger.info(f"语义缓存更新: {query}")
                    return

            # 超出上限时淘汰hit_count最低的
            if len(self._cache) >= self.MAX_CACHE_SIZE:
                self._cache.sort(key=lambda x: x.get("hit_count", 0), reverse=True)
                self._cache = self._cache[: self.MAX_CACHE_SIZE - 1]
                logger.info("语义缓存淘汰: 移除hit_count最低的缓存项")

            self._cache.append(entry)
            self._save()
            logger.info(f"语义缓存存储: {query} (verified={verified})")

    def mark_verified(self, query: str):
        """标记缓存项为已验证（人工确认数据准确）"""
        normalized = self._normalize(query)

        with self._lock:
            for entry in self._cache:
                if entry.get("normalized") == normalized:
                    entry["verified"] = True
                    self._save()
                    logger.info(f"语义缓存标记已验证: {query}")
                    return

        logger.warning(f"语义缓存未找到待验证项: {query}")

    def mark_incorrect(self, query: str):
        """标记缓存项为不准确（用户反馈数据错误），并删除该缓存项"""
        normalized = self._normalize(query)

        with self._lock:
            original_len = len(self._cache)
            self._cache = [
                entry for entry in self._cache
                if entry.get("normalized") != normalized
            ]
            if len(self._cache) < original_len:
                self._save()
                logger.info(f"语义缓存删除不准确项: {query}")
            else:
                logger.warning(f"语义缓存未找到待删除项: {query}")

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        with self._lock:
            total_entries = len(self._cache)
            verified_entries = sum(1 for e in self._cache if e.get("verified"))
            total_hits = sum(e.get("hit_count", 0) for e in self._cache)

            # top 10 by hit_count
            sorted_entries = sorted(
                self._cache, key=lambda x: x.get("hit_count", 0), reverse=True
            )
            top_queries = [
                {"query": e.get("query", ""), "hits": e.get("hit_count", 0)}
                for e in sorted_entries[:10]
            ]

            return {
                "total_entries": total_entries,
                "verified_entries": verified_entries,
                "total_hits": total_hits,
                "top_queries": top_queries,
            }

    def _normalize(self, query: str) -> str:
        """归一化查询文本：去除标点、空格、统一大小写"""
        if not query:
            return ""
        # 去除标点符号
        text = re.sub(r'[^\w\u4e00-\u9fff]', '', query)
        # 统一小写
        text = text.lower()
        return text

    def _similarity(self, query1: str, query2: str) -> float:
        """计算两个查询的语义相似度（基于Jaccard系数，使用字符bigram分词）"""
        if not query1 or not query2:
            return 0.0

        set1 = self._bigrams(query1)
        set2 = self._bigrams(query2)

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2
        union = set1 | set2

        return len(intersection) / len(union)

    @staticmethod
    def _bigrams(text: str) -> set:
        """字符bigram分词：例如'OM部门工单' → {'OM', 'M部', '部门', '门工', '工单'}"""
        if len(text) < 2:
            return {text} if text else set()
        return {text[i:i + 2] for i in range(len(text) - 1)}

    def _load(self):
        """从文件加载缓存"""
        if not os.path.exists(self._cache_file):
            self._cache = []
            return

        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.info(f"语义缓存加载: {len(self._cache)} 条记录")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"语义缓存加载失败，使用空缓存: {e}")
            self._cache = []

    def _save(self):
        """持久化缓存到文件"""
        try:
            cache_dir = os.path.dirname(self._cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"语义缓存保存失败: {e}")


# 全局单例
_cache_instance = None
_cache_lock = threading.Lock()


def get_semantic_cache() -> SemanticCache:
    """获取语义缓存全局单例"""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = SemanticCache()
    return _cache_instance
