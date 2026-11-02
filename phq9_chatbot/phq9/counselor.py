"""The counsellor: a RAG chain with per-session conversation memory."""
from __future__ import annotations
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from .knowledge import CounselingKnowledge
from .prompts import COUNSELOR_SYSTEM_PROMPT


class Counselor:
    """reply(text, session_id) -> the counsellor's answer, remembering earlier turns of that session."""

    def __init__(self, knowledge: CounselingKnowledge, llm):
        self.knowledge = knowledge
        self.llm = llm
        self._histories: dict[str, ChatMessageHistory] = {}
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", COUNSELOR_SYSTEM_PROMPT),
            ("placeholder", "{history}"),
            ("human", "{input}"),
        ])
        rag_chain = (
            RunnablePassthrough.assign(context=lambda x: self.knowledge.context_for(x["input"]))
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return RunnableWithMessageHistory(rag_chain, self.history, input_messages_key="input", history_messages_key="history")

    # -- memory ------------------------------------------------------------------------------------------------
    def history(self, session_id: str) -> ChatMessageHistory:
        return self._histories.setdefault(session_id, ChatMessageHistory())

    def transcript(self, session_id: str) -> str:
        """'Client: ... / Counselor: ...' lines, the input to the scorer."""
        lines = []
        for msg in self.history(session_id).messages:
            role = "Client" if msg.type == "human" else "Counselor"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    # -- talking -----------------------------------------------------------------------------------------------
    def reply(self, text: str, session_id: str) -> str:
        return self.chain.invoke({"input": text}, {"configurable": {"session_id": session_id}})
