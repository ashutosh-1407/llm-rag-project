from backend.src.rag.loader import load_pdf
from backend.src.rag.chunker import chunk_text
from backend.src.rag.retriever import Retriever
from pathlib import Path


_retriever = None
parent_dir = Path(__file__).resolve().parent.parent.parent
data_path = parent_dir / "data/company_policy.pdf"

def get_retriever():
    global _retriever
    if _retriever is None:
        document = load_pdf(data_path)
        chunks = chunk_text(document)
        _retriever = Retriever(chunks)
    return _retriever
