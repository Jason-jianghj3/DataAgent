"""
对话记忆与上下文理解服务

管理多会话的对话历史、业务上下文和指代消解，
使LLM能够理解用户追问、省略表达等自然对话行为。

v2.0: 集成LangChain ChatMessageHistory，提供更规范的对话记忆管理
"""
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from utils.logger import logger


class ConversationSession:
    """单个会话：维护对话历史与业务上下文，集成LangChain ChatMessageHistory"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict] = []
        self.langchain_history: BaseChatMessageHistory = ChatMessageHistory()
        self.last_report: str = ""
        self.last_dataset: str = ""
        self.last_params: Dict = {}
        self.last_result_summary: str = ""
        self.user_preferences: Dict = {}
        self.created_at: datetime = datetime.now()
        self.last_active: datetime = datetime.now()

    def build_context_messages(
        self,
        user_query: str,
        system_prompt: str,
        max_turns: int = 5,
    ) -> List[Dict]:
        messages = [{"role": "system", "content": system_prompt}]

        context_parts = self._build_business_context_block()
        if context_parts:
            messages.append({
                "role": "system",
                "content": f"【对话上下文】\n{context_parts}",
            })

        recent = self.messages[-(max_turns * 2):]
        messages.extend(recent)

        resolved_query = self.resolve_reference(user_query)
        messages.append({"role": "user", "content": resolved_query})

        return messages

    def get_langchain_messages(self) -> List:
        return self.langchain_history.messages

    def add_exchange(self, user_msg: str, assistant_msg: str):
        self.messages.append({"role": "user", "content": user_msg})
        self.messages.append({"role": "assistant", "content": assistant_msg})

        self.langchain_history.add_user_message(user_msg)
        self.langchain_history.add_ai_message(assistant_msg)

        self.last_active = datetime.now()
        logger.info(
            f"[ConversationSession] 记录对话 | session={self.session_id} | "
            f"历史轮数={len(self.messages) // 2}"
        )

    def update_business_context(
        self,
        report: str = "",
        dataset: str = "",
        params: Dict = None,
        result_summary: str = "",
    ):
        if report:
            self.last_report = report
        if dataset:
            self.last_dataset = dataset
        if params:
            self.last_params = params
        if result_summary:
            self.last_result_summary = result_summary[:200]
        self.last_active = datetime.now()
        logger.info(
            f"[ConversationSession] 更新上下文 | session={self.session_id} | "
            f"report={self.last_report} dataset={self.last_dataset}"
        )

    def resolve_reference(self, query: str) -> str:
        if not self.last_report and not self.last_dataset:
            return query

        resolved = query

        compare_match = re.search(r'(?:和|跟|与|对比|比较)\s*(.+?)\s*(?:比呢|比较呢|对比呢|比一比)', query)
        if compare_match:
            target = compare_match.group(1).strip()
            supplement = self._build_supplement()
            if supplement:
                resolved = f"对比{target}的{supplement}"
                logger.info(
                    f"[ConversationSession] 指代消解[对比] | "
                    f"\"{query}\" → \"{resolved}\""
                )
            return resolved

        compare_match2 = re.search(r'(?:和|跟|与)\s*(.+?)\s*比', query)
        if compare_match2 and '对比' not in query:
            target = compare_match2.group(1).strip()
            supplement = self._build_supplement()
            if supplement:
                resolved = f"对比{target}的{supplement}"
                logger.info(
                    f"[ConversationSession] 指代消解[对比] | "
                    f"\"{query}\" → \"{resolved}\""
                )
            return resolved

        then_match = re.search(r'那\s*(.+?)\s*呢', query)
        if then_match:
            target = then_match.group(1).strip()
            supplement = self._build_supplement()
            if supplement:
                resolved = f"查询{target}的{supplement}"
                logger.info(
                    f"[ConversationSession] 指代消解[继承] | "
                    f"\"{query}\" → \"{resolved}\""
                )
            return resolved

        extreme_match = re.search(r'(最多|最少|最高|最低|最好|最差|最大|最小)', query)
        if extreme_match:
            supplement = self._build_supplement()
            if supplement:
                resolved = f"{query}（指{supplement}中的）"
                logger.info(
                    f"[ConversationSession] 指代消解[极值] | "
                    f"\"{query}\" → \"{resolved}\""
                )
            return resolved

        return resolved

    def is_expired(self, max_minutes: int = 30) -> bool:
        elapsed = (datetime.now() - self.last_active).total_seconds() / 60
        return elapsed > max_minutes

    def _build_business_context_block(self) -> str:
        parts = []
        if self.last_report:
            parts.append(f"上次查询的报表: {self.last_report}")
        if self.last_dataset:
            parts.append(f"上次使用的数据集: {self.last_dataset}")
        if self.last_params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.last_params.items())
            parts.append(f"上次使用的参数: {param_str}")
        if self.last_result_summary:
            parts.append(f"上次结果摘要: {self.last_result_summary}")
        return "\n".join(parts)

    def _build_supplement(self) -> str:
        segments = []
        if self.last_report:
            segments.append(self.last_report)
        if self.last_dataset:
            segments.append(self.last_dataset)
        if self.last_params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.last_params.items())
            segments.append(param_str)
        return " - ".join(segments) if segments else ""


class ConversationManager:
    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create(self, session_id: str = "") -> ConversationSession:
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id)
            logger.info(
                f"[ConversationManager] 创建新会话 | session={session_id}"
            )

        session = self._sessions[session_id]
        session.last_active = datetime.now()
        return session

    def cleanup_expired(self, max_minutes: int = 30):
        expired_ids = [
            sid for sid, session in self._sessions.items()
            if session.is_expired(max_minutes)
        ]
        for sid in expired_ids:
            del self._sessions[sid]

        if expired_ids:
            logger.info(
                f"[ConversationManager] 清理过期会话 | "
                f"数量={len(expired_ids)} | 剩余={len(self._sessions)}"
            )

    @property
    def session_count(self) -> int:
        return len(self._sessions)


_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
