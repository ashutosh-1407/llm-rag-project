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
