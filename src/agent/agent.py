from src.rag.retriever_store import get_retriever
from src.llm.generator import run_llm_agent

def agent_decide(query):
    query = query.lower()
    if "summarize" in query:
        return {"type": "tool", "name": "summarize_doc"}
    elif "support" in query:
        return {"type": "tool", "name": "get_support_info"}
    elif "return" in query or "policy" in query:
        return {"type": "retrieval"}
    else:
        return {"type": "llm"}

def run_agent_with_debug(query: str, session_id: str = "default"):
    retriever = get_retriever()
    contexts = retriever.retrieve(query)
    answer, metadata = run_llm_agent(query, contexts, session_id)
    return {
        "answer": answer,
        "retrieved_chunks": contexts,
        "metadata": metadata
    }
