"""
Prompt Engineering Module
─────────────────────────
This prompt is ONLY ever called for DOCUMENT questions.
General conversation is handled before this prompt is reached
(see RAGPipeline._classify_intent in rag_pipeline.py).
"""

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ─── Role Descriptions ───────────────────────────────────────────────────────

ROLE_DESCRIPTIONS: dict = {
    "Assistant": "a helpful and knowledgeable assistant",
    "Teacher": (
        "a patient and educational teacher who explains concepts clearly, "
        "uses examples, and ensures understanding"
    ),
    "Expert": (
        "a precise domain expert who provides authoritative, well-structured, "
        "and technically accurate answers"
    ),
    "Analyst": (
        "an analytical thinker who breaks down information systematically, "
        "identifies patterns, and provides data-driven insights"
    ),
    "Summarizer": (
        "a concise summarizer who extracts key points, main ideas, and "
        "essential information efficiently"
    ),
}

# ─── Condense-Question Prompt ────────────────────────────────────────────────

_CONDENSE_TEMPLATE = """\
Given the conversation history below and a follow-up question,
rephrase the follow-up into a standalone question that can be fully
understood without reading the chat history.

Rules:
- If the follow-up is already standalone, return it unchanged.
- Do NOT answer the question — only rephrase it.
- Keep the rephrased question concise and specific.

Chat History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=_CONDENSE_TEMPLATE,
)

# ─── QA Prompt Builder ───────────────────────────────────────────────────────

def build_qa_prompt(role: str = "Assistant", instruction: str = "") -> ChatPromptTemplate:
    """
    Strict document-only QA prompt.
    This is ONLY called for messages already classified as DOCUMENT questions.
    """
    role_desc = ROLE_DESCRIPTIONS.get(role, f"a helpful {role} assistant")

    instruction_block = (
        f"\nSPECIAL INSTRUCTION:\n{instruction.strip()}\n"
        if instruction and instruction.strip()
        else ""
    )

    system_template = f"""\
You are {role_desc}. Answer questions using ONLY the document context below.
{instruction_block}
RULES:
1. Answer ONLY from the CONTEXT section. Never use outside knowledge.
2. If the answer is not in the context, say exactly:
   "Sorry, I don't know based on the provided documents."
3. Never fabricate or infer facts not present in the context.
4. Briefly cite the relevant part when answering.
5. Maintain the tone appropriate for your role: {role}.

CONTEXT (from uploaded documents):
{{context}}
"""

    human_template = "Question: {question}\n\nAnswer:"

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])


# ─── Summary Prompt ──────────────────────────────────────────────────────────

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""\
You are a precise document summarizer. Summarize the following document content.

Requirements:
- Include the main topic, key findings, and important details.
- Use clear, well-structured language.
- Base ONLY on the provided text — do not add external information.

Document Content:
{text}

Summary:""",
)