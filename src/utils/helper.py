def rerank(contexts):
    # simple heuristic: shorter chunks often more focused
    return sorted(contexts, key=lambda x: len(x))[:3]
