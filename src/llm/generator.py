from dotenv import load_dotenv
from openai import OpenAI
from src.tools.registry import tools, TOOL_MAP
from src.utils.constants import OPENAI_MODEL_NAME
from src.utils.helper import logger
import os


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# simple in-memory session store
SESSIONS = {}

def run_rule_based_agent(query, contexts, history):
    combined_context = "\n\n".join(contexts)

    # prompt with history and using tools with rule-based agent
    prompt = f"""
    You are a helpful assistant.

    Conversation so far:
    {history}

    Context:
    {contexts}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        tools=tools
    )

    return response.choices[0].message.content

# llm decides if tools needs to be run
def run_llm_agent(query, contexts, session_id: str = "default"):
    logger.info(f"Fetching history | session_id={session_id}")
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
    history = SESSIONS[session_id]
    history_text = "\n".join(
        f"User: {item['query']}\nAssistant: {item['answer']}"
        for item in history[-3:]
    )
    if history != []:
        logger.info(f"Found history | history text={history_text}")
    else:
        logger.info("No history found.")

    context_text = "\n\n".join(contexts)
    
    system_prompt = f"""
    You are a helpful assistant.

    Use the provided context to answer the user's question.
    Use tools if they are helpful.
    If the answer is not in the context and no tool applies, say you don't know.

    Conversation history:
    {history_text if history_text else "No prior conversation."}

    Context:
    {context_text if context_text else "No retrieved context."}
    """.strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    logger.info(f"Running llm based agent with tools | query={query} | session_id={session_id}")

    # Step 1 - Ask LLM - it may call a tool
    response = client.chat.completions.create(
        model = OPENAI_MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0
    )

    message = response.choices[0].message

    route = "direct_llm"
    tools_used = None

    # Step 2 - Did LLM call a tool?
    if message.tool_calls:
        route = "tool"
        
        # Step 3 - Send tool result back to LLM
        messages.append(message)
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tools_used = tool_name
            if tool_name not in TOOL_MAP:
                tool_result = f"Tool '{tool_name}' is not implemented."
            else:
                tool_result = TOOL_MAP[tool_name]()
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

        # Step 4 - Final response
        final_response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages,
            temperature=0
        )

        answer = final_response.choices[0].message.content
    else:
        # no tool needed
        # if context exists, this is effectively retrieval-based answering
        if contexts:
            answer = message.content
    
    # fallback safety
    if not answer:
        answer = "Sorry, I couldn't generate an answer"
    
    history.append({
        "query": query,
        "answer": answer
    })

    metadata = {
        "route": route,
        "tool": tools_used,
        "retrieved_k": len(contexts),
        "retrieved_chunks": contexts
    }
    
    # Step 5 - No tool needed
    return answer, metadata
