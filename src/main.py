from fastapi import FastAPI
from src.rag.loader import load_pdf
from src.rag.chunker import chunk_text
from src.rag.retriever import Retriever
from src.llm.generator import generate_answer_with_memory, run_agent
from src.agent.agent import agent_decide
from src.tools.registry import TOOL_MAP
from src.utils.helper import rerank
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
app = FastAPI()
sessions = {}

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

@app.get("/ask")
def ask_question(session_id: str, query: str):
    try:
        logging.info(f"Query: {query}")
        if session_id not in sessions:
            sessions[session_id] = []
        history = sessions[session_id]
        contexts = retriever.retrieve(query, k=5)
        contexts = rerank(contexts)
        # answer = generate_answer(query, contexts)
        answer = generate_answer_with_memory(query, contexts, history)
        history.append({"query": query, "answer": answer})
        logging.info(f"Answer: {answer}")
        return {
            "answer": answer,
            "sources": contexts
        }
    except Exception as e:
        return {
            "error": str(e)
        }

@app.get("/ask_rule_based_agent")
def ask_agent(query: str):
    decision = agent_decide(query)
    if decision["type"] == "tool":
        tool_fn = TOOL_MAP[decision["name"]]
        result = tool_fn()
        return {"answer": result}
    elif decision["type"] == "retrieval":
        contexts = retriever.retrieve(query, k=5)
        contexts = rerank(contexts)
        answer = generate_answer_with_memory(query, contexts, [])
        return {"answer": answer}
    else:
        answer = generate_answer_with_memory(query, [], [])
        return {"answer": answer}

@app.get("/ask_llm_agent")
def ask_llm(query: str):
    contexts = retriever.retrieve(query, k=5)
    contexts = rerank(contexts)
    answer = run_agent(query, contexts)
    return {"answer": answer}
