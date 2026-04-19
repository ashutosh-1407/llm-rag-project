from collections import Counter


REQUEST_COUNTS = Counter()
ERROR_COUNTS = Counter()
ROUTE_COUNTS = Counter()

def record_requests():
    REQUEST_COUNTS["total"] += 1

def record_error():
    ERROR_COUNTS["total"] += 1

def record_route(route: str): 
    ROUTE_COUNTS[route] += 1

def get_metrics():
    return {
        "requests": dict(REQUEST_COUNTS),
        "error": dict(ERROR_COUNTS),
        "routes": dict(ROUTE_COUNTS)
    }
