"""
RAG Pipeline Module
────────────────────
Architecture:
  1. Every incoming message is first classified by the LLM as either
     GENERAL (casual chat) or DOCUMENT (needs retrieval).
  2. GENERAL  → answered directly by the LLM, no vector store touched.
  3. DOCUMENT → full RAG flow: retrieve context → grounded answer only.

This two-step approach is the only reliable way to handle both general
conversation and strict document Q&A in the same chatbot.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    TOP_K_RETRIEVAL,
    MEMORY_WINDOW,
)
from src.prompts import build_qa_prompt, CONDENSE_QUESTION_PROMPT

logger = logging.getLogger(__name__)

# ─── Intent Classifier Prompt ────────────────────────────────────────────────
# A tight, single-purpose prompt that returns ONLY "GENERAL" or "DOCUMENT".
# Uses a separate LLM call so it is 100% independent from the RAG chain.

_CLASSIFIER_SYSTEM = """\
You are an intent classifier for a document chatbot.
Classify the user message into exactly one of these two categories:

GENERAL  — greetings, small talk, casual questions, questions about the AI
           itself, emotional exchanges, compliments, anything that is NOT
           asking for information that would be found in an uploaded document.
           Examples: "hi", "how are you", "fine how about you",
                     "how was your day", "what is your name",
                     "are you capable of chatting with me other than documents",
                     "thanks", "bye", "you're great", "what can you do"

DOCUMENT — questions asking about facts, data, policies, procedures, or any
           content that could be found in the uploaded documents.
           Examples: "what is the leave policy", "summarise the document",
                     "how many vacation days do employees get"

Reply with ONE word only — either GENERAL or DOCUMENT. No explanation."""


# ─── General Chat System Prompt ──────────────────────────────────────────────

_GENERAL_SYSTEM = """\
You are DocChat AI — a friendly, helpful assistant.
Respond naturally and warmly to the user's message.
Keep your reply concise and conversational.
Do NOT mention documents unless the user brings them up."""


class RAGPipeline:
    """
    Two-stage pipeline:
      Stage 1 — Intent classification  (LLM, no retrieval)
      Stage 2 — Answer generation
                  GENERAL  → direct LLM call
                  DOCUMENT → ConversationalRetrievalChain (FAISS + LLM)
    """

    def __init__(
        self,
        vector_store: FAISS,
        role: str = "Assistant",
        instruction: str = "",
    ) -> None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "Add it to your .env file: GROQ_API_KEY=your_key_here\n"
                "Get a free key at: https://console.groq.com"
            )

        self.vector_store = vector_store
        self.role = role
        self.instruction = instruction

        # ── LLM ──────────────────────────────────────────────────────────────
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        # ── Memory ────────────────────────────────────────────────────────────
        self.memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # ── Retriever ─────────────────────────────────────────────────────────
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RETRIEVAL},
        )

        # ── RAG Chain ─────────────────────────────────────────────────────────
        self.chain = self._build_chain()

        logger.info(
            f"✅ RAGPipeline ready | model={GROQ_MODEL} | role={role} | "
            f"top_k={TOP_K_RETRIEVAL} | memory_window={MEMORY_WINDOW}"
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _build_chain(self) -> ConversationalRetrievalChain:
        qa_prompt = build_qa_prompt(self.role, self.instruction)
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=False,
        )

    def _classify_intent(self, question: str) -> str:
        """
        Ask the LLM whether this message is GENERAL chat or a DOCUMENT query.
        Returns the string "GENERAL" or "DOCUMENT".
        Falls back to "DOCUMENT" on any error so the RAG chain stays the
        safe default.
        """
        try:
            response = self.llm.invoke([
                SystemMessage(content=_CLASSIFIER_SYSTEM),
                HumanMessage(content=question),
            ])
            result = response.content.strip().upper()
            # Accept the label even if the model adds punctuation / extra words
            if "GENERAL" in result:
                return "GENERAL"
            return "DOCUMENT"
        except Exception as e:
            logger.warning(f"Intent classifier failed ({e}), defaulting to DOCUMENT.")
            return "DOCUMENT"

    def _answer_general(self, question: str) -> Dict[str, Any]:
        """Direct LLM answer for general / casual conversation."""
        try:
            response = self.llm.invoke([
                SystemMessage(content=_GENERAL_SYSTEM),
                HumanMessage(content=question),
            ])
            return {
                "answer": response.content,
                "sources": [],
                "num_retrieved": 0,
            }
        except Exception as e:
            logger.error(f"General chat LLM call failed: {e}")
            return {
                "answer": "Hi there! How can I help you today?",
                "sources": [],
                "num_retrieved": 0,
            }

    def _answer_document(self, question: str) -> Dict[str, Any]:
        """Full RAG answer — retrieves context then answers strictly from it."""
        result = self.chain.invoke({"question": question})

        answer: str = result.get("answer", "No answer generated.")
        source_docs: List = result.get("source_documents", [])

        seen: set = set()
        sources: List[str] = []
        for doc in source_docs:
            name = doc.metadata.get("source", "Unknown source")
            if name not in seen:
                seen.add(name)
                sources.append(name)

        return {
            "answer": answer,
            "sources": sources,
            "num_retrieved": len(source_docs),
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        """
        Route the question through the correct pipeline stage.

        Stage 1 — classify intent (GENERAL vs DOCUMENT).
        Stage 2 — answer accordingly:
                    GENERAL  → direct LLM  (no vector store)
                    DOCUMENT → RAG chain   (strict context-only answer)
        """
        question = question.strip()
        if not question:
            return {"answer": "Please enter a valid question.",
                    "sources": [], "num_retrieved": 0}

        intent = self._classify_intent(question)
        logger.info(f"Intent classified as '{intent}' for: '{question}'")

        if intent == "GENERAL":
            return self._answer_general(question)
        else:
            try:
                return self._answer_document(question)
            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                raise RuntimeError(f"Query error: {e}") from e

    def reset_memory(self) -> None:
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    def update_settings(self, role: str, instruction: str) -> None:
        self.role = role
        self.instruction = instruction
        self.chain = self._build_chain()
        logger.info(f"RAGPipeline settings updated: role={role}")