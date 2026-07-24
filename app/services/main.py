from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from langchain_groq import ChatGroq

app = FastAPI(title="Enterprise Assistant API")


class QueryRequest(BaseModel):
    q: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict] = []


@app.get("/")
def home():
    return {"status": "online", "message": "Enterprise RAG API is live."}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Executes the RAG query flow:
    1. Embeds question & searches Qdrant vector DB.
    2. Constructs prompt context.
    3. Queries LLM (Groq Llama-3.3-70b).
    4. Returns response + sources to client.
    """
    q = request.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query string 'q' cannot be empty.")

    try:
        # Step 1: Vector DB Search (top 3 chunks to stay well within Groq TPM limits)
        top_k_val = min(request.top_k or 3, 3)
        retrieved_docs = search_enterprise_knowledge(query=q, limit=top_k_val)

        if not retrieved_docs:
            return QueryResponse(
                question=q,
                answer="No relevant documentation found in the vector database.",
                sources=[]
            )

        # Step 2: Format & Truncate Context to prevent Groq Token Limit (413 TPM error)
        # Capping total context to ~4000 characters (~1000 tokens)
        truncated_chunks = []
        for doc in retrieved_docs[:3]:
            content = doc.get("content", "")
            if len(content) > 1500:
                content = content[:1500] + "... [truncated]"
            truncated_chunks.append(content)

        context_text = "\n\n---\n\n".join(truncated_chunks)
        sources = [
            {"source": doc.get("source", "Unknown"), "score": round(doc.get("score", 0.0), 4)}
            for doc in retrieved_docs
        ]

        # Step 3: LLM Generation
        llm = ChatGroq(
            model_name=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.2,
        )

        prompt = (
            "You are an assistant answering questions based on Kubernetes documentation.\n"
            "Answer the user's question accurately using ONLY the provided context.\n\n"
            f"--- CONTEXT ---\n{context_text}\n\n"
            f"--- QUESTION ---\n{q}\n\n"
            "--- ANSWER ---"
        )

        response = llm.invoke(prompt)
        answer_text = response.content.strip()

        return QueryResponse(
            question=q,
            answer=answer_text,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend execution error: {str(e)}")

