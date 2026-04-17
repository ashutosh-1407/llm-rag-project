from fastapi import FastAPI
from src.rag.loader import load_pdf
from src.rag.chunker import chunk_text
from src.rag.retriever import Retriever
from src.llm.generator import run_rule_based_agent, run_llm_agent
from src.agent.agent import agent_decide
from src.tools.registry import TOOL_MAP
from src.utils.helper import rerank, logger
from pathlib import Path
import time


app = FastAPI()

# load once at startup
parent_dir = Path(__file__).resolve().parent.parent
data_path = parent_dir / "data/company_policy.pdf"
document = load_pdf(data_path)
chunks = chunk_text(document)
retriever = Retriever(chunks)

@app.get("/")
def get_welcome_page():
    return "Welcome to Ashu's LLM + RAG PROJECT!!"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ask_rule_based_agent")
def ask_rule_based_agent(query: str):
    decision = agent_decide(query)
    if decision["type"] == "tool":
        tool_fn = TOOL_MAP[decision["name"]]
        result = tool_fn()
        return {"answer": result}
    elif decision["type"] == "retrieval":
        contexts = retriever.retrieve(query, k=5)
        contexts = rerank(contexts)
        answer = run_rule_based_agent(query, contexts, [])
        return {"answer": answer}
    else:
        answer = run_rule_based_agent(query, [], [])
        return {"answer": answer}

@app.get("/ask_llm_agent")
def ask_llm_agent(query: str, session_id: str = "default"):
    start_time = time.time()
    try:
        logger.info(f"Incoming query | session_id={session_id} | query={query}")
        contexts = retriever.retrieve(query, k=5)
        contexts = rerank(contexts)
        answer, metadata = run_llm_agent(query, contexts, session_id)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"Query served | session_id={session_id} | "
            f"latency_ms={latency_ms} | "
            f"route={metadata.get('route')} | "
            f"tool={metadata.get('tool')} | "
            f"retrieved_k={metadata.get('retrieved_k')}"
        )
        return {
            "answer": answer,
            "metadata": {
                "latency_ms": latency_ms,
                **metadata
            }
        }
    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.exception(f"Query failed | session_id={session_id} | latency_ms={latency_ms}")
        return {
            "error": str(e),
            "metadata": {
                "latency_ms": latency_ms
            }
        }
