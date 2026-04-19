from collections import defaultdict


_SESSIONS = defaultdict(list)

def get_history(session_id: str):
    return _SESSIONS[session_id]

def append_turn(session_id: str, query: str, answer: str):
    _SESSIONS[session_id].append({
        "query": query,
        "answer": answer
    })

def clear_history(session_id):
    _SESSIONS[session_id] = []
