from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from langchain_groq import ChatGroq


def ask_kubernetes_doc(question: str = "What is Kubernetes and what are its core components?", top_k: int = 5) -> str:
    """
    Simple function to query Kubernetes documents:
    1. Embeds the question and searches Qdrant vector database.
    2. Builds context from retrieved documents.
    3. Sends context + question to LLM (Groq / Llama-3.3-70b).
    4. Returns the LLM response.
    """
    print(f"\n[Question]: {question}")
    print("[Vector DB]: Searching Qdrant...")

    # Step 1: Embed query and search vector DB
    top_k_val = min(top_k, 3)
    retrieved_docs = search_enterprise_knowledge(query=question, limit=top_k_val)

    if not retrieved_docs:
        print("[Vector DB]: No relevant context found.")
        return "No relevant documentation found."

    print(f"[Vector DB]: Found {len(retrieved_docs)} relevant context chunks.")

    # Step 2: Construct context from retrieved documents (truncate to prevent token limit)
    truncated_chunks = []
    for doc in retrieved_docs[:3]:
        content = doc.get("content", "")
        if len(content) > 1500:
            content = content[:1500] + "... [truncated]"
        truncated_chunks.append(content)

    context_text = "\n\n---\n\n".join(truncated_chunks)

    # Step 3: Initialize LLM and pass question + context
    print("[LLM]: Generating response...")
    llm = ChatGroq(
        model_name=settings.GROQ_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.2,
    )

    prompt = (
        "You are an assistant answering questions based on Kubernetes documentation.\n"
        "Answer the user's question accurately using ONLY the provided context.\n\n"
        f"--- CONTEXT ---\n{context_text}\n\n"
        f"--- QUESTION ---\n{question}\n\n"
        "--- ANSWER ---"
    )

    response = llm.invoke(prompt)
    answer = response.content.strip()

    print(f"\n[Answer]:\n{answer}\n")
    return answer


def main():
    ask_kubernetes_doc("What is Kubernetes and what are its core components?")


if __name__ == "__main__":
    main()


