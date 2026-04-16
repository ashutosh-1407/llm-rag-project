from src.tools.tools import summarize_doc, get_support_info


tools = [
    {
        "type": "function",
        "function": {
            "name": "summarize_doc",
            "description": "Summarize the entire document",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_support_info",
            "description": "Get support contact details",
            "parameters": {}
        }
    }
]

TOOL_MAP = {
    "summarize_doc": summarize_doc,
    "get_support_info": get_support_info
}
