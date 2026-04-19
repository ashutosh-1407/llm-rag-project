from backend.src.observability.logger import setup_logger


logger = setup_logger()

def rerank(contexts):
    # simple heuristic: shorter chunks often more focused
    return sorted(contexts, key=lambda x: len(x))[:3]
