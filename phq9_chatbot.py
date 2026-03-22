"""
PHQ-9 Depression Assessment Chatbot
====================================
A conversational AI chatbot that estimates a client's depression level
based on the PHQ-9 (Patient Health Questionnaire-9) by having a natural
conversation, then scoring the transcript with an LLM.

Usage:
    python phq9_chatbot.py

Requires:
    - Ollama running with `llama3:instruct` and `nomic-embed-text` pulled
    - train.csv in the same directory
"""

import csv
import json
import re
from typing import List

import pandas as pd
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---------------------------------------------------------------------------
# PHQ-9 Item Definitions
# ---------------------------------------------------------------------------
PHQ9_ITEMS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself, or that you are a failure, or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed, or being fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself in some way",
]

SEVERITY_LEVELS = [
    (0, 4, "Minimal depression"),
    (5, 9, "Mild depression"),
    (10, 14, "Moderate depression"),
    (15, 19, "Moderately severe depression"),
    (20, 27, "Severe depression"),
]


def get_severity(total_score: int) -> str:
    for low, high, label in SEVERITY_LEVELS:
        if low <= total_score <= high:
            return label
    return "Unknown"


# ---------------------------------------------------------------------------
# RAG Setup  (reuses the project's existing pattern)
# ---------------------------------------------------------------------------
def load_csv_to_documents(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                content = (
                    f"Context: {row['Context']}\n"
                    f"Suggested Response: {row['Response']}"
                )
                doc = Document(page_content=content, metadata={"source": "csv_data"})
                documents.append(doc)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    return documents


def setup_rag(csv_path: str = "train.csv", max_docs: int = 2000):
    print("Loading counseling data...")
    docs = load_csv_to_documents(csv_path)
    docs = docs[:max_docs]

    print("Building vector store (this may take a moment)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print(f"Vector store ready with {len(docs)} documents.\n")
    return retriever


# ---------------------------------------------------------------------------
# Chain Setup
# ---------------------------------------------------------------------------
COUNSELOR_SYSTEM_PROMPT = """\
You are a compassionate mental health counselor conducting an intake assessment.
Your goal is to understand the client's mental state by having a warm, natural conversation.

You must gently explore ALL of the following PHQ-9 domains over the course of the conversation.
Do NOT ask them as a checklist. Weave them naturally into the dialogue, one or two at a time:

1. Interest or pleasure in doing things
2. Feeling down, depressed, or hopeless
3. Sleep problems (too much or too little)
4. Energy level / fatigue
5. Appetite changes (too much or too little)
6. Self-worth / feelings of failure
7. Concentration difficulties
8. Psychomotor changes (moving/speaking slowly, or restlessness)
9. Thoughts of self-harm or being better off dead

Guidelines:
- Be empathetic and validating. Never be judgmental.
- Ask open-ended questions. Let the client share at their own pace.
- If the client brings up a topic naturally, acknowledge it before moving on.
- Do NOT mention "PHQ-9" or "assessment" — keep it conversational.
- Keep responses concise (2-4 sentences). Do not lecture.
- When you sense a domain has been sufficiently discussed, gently transition.
- Use the retrieved counseling context below to inform your responses:

{context}
"""

SCORER_SYSTEM_PROMPT = """\
You are a clinical psychologist analyzing a counseling conversation transcript.
Your task is to score each PHQ-9 item based on what the client expressed.

PHQ-9 scoring per item:
  0 = Not at all
  1 = Several days
  2 = More than half the days
  3 = Nearly every day

The 9 items are:
1. Little interest or pleasure in doing things
2. Feeling down, depressed, or hopeless
3. Trouble falling or staying asleep, or sleeping too much
4. Feeling tired or having little energy
5. Poor appetite or overeating
6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down
7. Trouble concentrating on things
8. Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless
9. Thoughts that you would be better off dead, or of hurting yourself

IMPORTANT: Respond ONLY with valid JSON in this exact format, nothing else:
{{
  "scores": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "reasoning": ["reason for item 1", "reason for item 2", ..., "reason for item 9"],
  "summary": "A brief clinical summary of the client's presentation."
}}
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def setup_chains(retriever):
    llm = ChatOllama(model="llama3:instruct", temperature=0.3)

    # --- Counselor chain (with RAG + chat history) ---
    counselor_prompt = ChatPromptTemplate.from_messages([
        ("system", COUNSELOR_SYSTEM_PROMPT),
        ("placeholder", "{message}"),
        ("human", "{input}"),
    ])

    counselor_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | counselor_prompt
        | llm
        | StrOutputParser()
    )

    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_chain = RunnableWithMessageHistory(
        counselor_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="message",
    )

    # --- Scorer chain (no RAG, no history — one-shot analysis) ---
    scorer_prompt = ChatPromptTemplate.from_messages([
        ("system", SCORER_SYSTEM_PROMPT),
        ("human", "Here is the full conversation transcript:\n\n{transcript}"),
    ])

    scorer_llm = ChatOllama(model="llama3:instruct", temperature=0)
    scorer_chain = scorer_prompt | scorer_llm | StrOutputParser()

    return conversational_chain, scorer_chain, store


# ---------------------------------------------------------------------------
# Conversation Loop
# ---------------------------------------------------------------------------
SESSION_ID = "phq9_session"


def run_conversation(conversational_chain) -> List[str]:
    print("=" * 60)
    print("  PHQ-9 Depression Assessment Chatbot")
    print("=" * 60)
    print("  I'm here to talk with you about how you've been feeling.")
    print("  Type 'exit' when you'd like to end the conversation.")
    print("=" * 60)
    print()

    # Opening message from the counselor
    opening = conversational_chain.invoke(
        {"input": "Hello, I've been wanting to talk to someone about how I've been feeling lately."},
        {"configurable": {"session_id": SESSION_ID}},
    )
    print(f"Counselor: {opening}\n")

    user_inputs = []
    turn = 0

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        user_inputs.append(user_input)
        turn += 1

        if user_input.lower() == "exit":
            if turn < 5:
                print(
                    "\nCounselor: We've only just started talking. "
                    "Could you share a little more so I can better understand "
                    "how you're doing? (Type 'exit' again to end anyway.)\n"
                )
                confirm = input("You: ").strip()
                if confirm.lower() != "exit":
                    user_inputs.append(confirm)
                    response = conversational_chain.invoke(
                        {"input": confirm},
                        {"configurable": {"session_id": SESSION_ID}},
                    )
                    print(f"\nCounselor: {response}\n")
                    continue
            print("\nCounselor: Thank you for sharing with me today. "
                  "Let me put together a summary for you.\n")
            break

        response = conversational_chain.invoke(
            {"input": user_input},
            {"configurable": {"session_id": SESSION_ID}},
        )
        print(f"\nCounselor: {response}\n")

    return user_inputs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def build_transcript(store) -> str:
    if SESSION_ID not in store:
        return ""
    messages = store[SESSION_ID].messages
    lines = []
    for msg in messages:
        role = "Client" if msg.type == "human" else "Counselor"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def score_conversation(scorer_chain, transcript: str) -> dict:
    print("Analyzing conversation for PHQ-9 scoring...\n")
    raw = scorer_chain.invoke({"transcript": transcript})

    # Try to extract JSON from the LLM response
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if "scores" in result and len(result["scores"]) == 9:
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: return empty scores if parsing fails
    print("Warning: Could not parse scorer output. Showing raw response:")
    print(raw)
    return {
        "scores": [0] * 9,
        "reasoning": ["Unable to parse"] * 9,
        "summary": raw,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(result: dict):
    scores = result["scores"]
    reasoning = result.get("reasoning", [""] * 9)
    summary = result.get("summary", "")
    total = sum(scores)
    severity = get_severity(total)

    print("=" * 60)
    print("  PHQ-9 DEPRESSION ASSESSMENT REPORT")
    print("=" * 60)
    print()
    print(f"{'#':<4} {'PHQ-9 Item':<55} {'Score':<6}")
    print("-" * 65)

    for i, (item, score) in enumerate(zip(PHQ9_ITEMS, scores), 1):
        print(f"{i:<4} {item:<55} {score}/3")
        if reasoning[i - 1] and reasoning[i - 1] != "Unable to parse":
            print(f"     Reason: {reasoning[i - 1]}")

    print("-" * 65)
    print(f"{'TOTAL SCORE:':<60} {total}/27")
    print(f"{'SEVERITY:':<60} {severity}")
    print()

    if summary:
        print("CLINICAL SUMMARY:")
        print(summary)
    print()

    print("NOTE: This is an AI-assisted estimate, NOT a clinical diagnosis.")
    print("Please consult a licensed mental health professional for proper evaluation.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main - plays a role to connect/use all above fucntions  
# ---------------------------------------------------------------------------
def main():
    retriever = setup_rag()
    conversational_chain, scorer_chain, store = setup_chains(retriever)

    run_conversation(conversational_chain)

    transcript = build_transcript(store)
    if not transcript:
        print("No conversation recorded. Exiting.")
        return

    result = score_conversation(scorer_chain, transcript)
    print_report(result)


if __name__ == "__main__":
    main()
