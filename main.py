import sys

from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from langchain_groq import ChatGroq


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

    # Step 2: Build context
    context_text = "\n\n---\n\n".join(
        doc["content"] for doc in retrieved_docs
    )

    # Step 3: Query LLM
    print("[LLM]: Generating response...")

    llm = ChatGroq(
        model_name=settings.GROQ_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.2,
    )

    prompt = f"""
You are an assistant answering questions based on Kubernetes documentation.

Answer the user's question accurately using ONLY the provided context.

--- CONTEXT ---
{context_text}

--- QUESTION ---
{question}

--- ANSWER ---
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    print(f"\n[Answer]:\n{answer}\n")
    return answer


def main():
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