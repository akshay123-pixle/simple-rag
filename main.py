import logfire
from app.guardrails import initialize_rails
import sys

from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from langchain_groq import ChatGroq
from app.guardrails.rails import guard

def ask_kubernetes_doc(
    question: str = "What is Kubernetes and what are its core components?",
    top_k: int = 5,
) -> str:
    """
    Simple function to query Kubernetes documents:
    1. Embeds the question and searches Qdrant vector database.
    2. Builds context from retrieved documents.
    3. Sends context + question to LLM (Groq / Llama-3.3-70b).
    4. Returns the LLM response.
    """
    rail_fired, rail_response = guard(question)
    if rail_fired:
        # thread_id= "default_user"
        # logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
        print("🛡️ Request blocked by guardrails")
        return {
                "question": question,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }
    print(f"\n[Question]: {question}")
    print("[Vector DB]: Searching Qdrant...")

    # Step 1: Search vector DB
    retrieved_docs = search_enterprise_knowledge(
        query=question,
        limit=top_k,
    )

    if not retrieved_docs:
        print("[Vector DB]: No relevant context found.")
        return "No relevant documentation found."

    print(f"[Vector DB]: Found {len(retrieved_docs)} relevant context chunks.")

    # Step 2: Build truncated context (capping text to prevent Groq 11k+ token bloat)
    truncated_chunks = []
    for doc in retrieved_docs[:2]:
        content = doc.get("content", "")
        if len(content) > 1500:
            content = content[:1500] + "... [truncated]"
        truncated_chunks.append(content)

    context_text = "\n\n---\n\n".join(truncated_chunks)

    # Step 3: Query LLM (with automatic fallback to llama-3.1-8b-instant if 70b daily quota is hit)
    print("[LLM]: Generating response...")

    prompt = f"""
You are an assistant answering questions based on Kubernetes documentation.

Answer the user's question accurately using ONLY the provided context.

--- CONTEXT ---
{context_text}

--- QUESTION ---
{question}

--- ANSWER ---
"""

    try:
        llm = ChatGroq(
            model_name=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.2,
        )
        response = llm.invoke(prompt)
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e):
            print("[LLM]: llama-3.3-70b daily limit reached. Falling back to llama-3.1-8b-instant...")
            fallback_llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                groq_api_key=settings.GROQ_API_KEY,
                temperature=0.2,
            )
            response = fallback_llm.invoke(prompt)
        else:
            raise e

    answer = response.content.strip()

    print(f"\n[Answer]:\n{answer}\n")
    return answer


def main():
    initialize_rails()
    # If a question is passed via command line, use it.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        # Otherwise, prompt the user interactively.
        question = input(
            "Enter your Kubernetes question (press Enter for default): "
        ).strip()

        if not question:
            question = (
                "What is Kubernetes and what are its core components?"
            )

    ask_kubernetes_doc(question)


if __name__ == "__main__":
    main()