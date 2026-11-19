import sys
from pathlib import Path
from typing import TypedDict, List

# Ensure the directory containing this script is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from tools import (
    get_order_status,
    issue_refund,
    parse_handoff,
    extract_order_id,
)
from prompts import TRIAGE_PROMPT, ORDERS_PROMPT, REFUNDS_PROMPT

# Guard rails constants
MAX_HOPS = 5
SAFETY_NET = {"triage": "orders"}


class AgentState(TypedDict):
    request: str
    active: str
    log: List[str]
    hops: int


def get_llm():
    return ChatOllama(model="llama3.2", temperature=0)


# Module-level LLM instance (can be replaced in tests)
llm = get_llm()


def triage_agent(state: AgentState) -> dict:
    hops = state.get("hops", 0) + 1
    messages = [
        SystemMessage(content=TRIAGE_PROMPT),
        HumanMessage(content=state["request"]),
    ]

    try:
        response = llm.invoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        raw_text = f"Hello! Welcome to our store. Let me connect you with orders.\nHANDOFF: orders"

    default_target = SAFETY_NET.get("triage", "orders")
    clean_msg, target = parse_handoff(raw_text, default=default_target)

    # Guard rail: an agent cannot hand off to itself
    if target == "triage":
        target = "orders"

    # Guard rail: MAX_HOPS
    if hops >= MAX_HOPS:
        target = "done"

    new_log = state.get("log", []) + [f"[triage] {clean_msg}"]
    return {
        "active": target,
        "hops": hops,
        "log": new_log,
    }


def orders_agent(state: AgentState) -> dict:
    hops = state.get("hops", 0) + 1

    # Plain code extracts 4-digit order id and runs get_order_status
    order_id = extract_order_id(state.get("request", ""))
    if order_id is None:
        for entry in reversed(state.get("log", [])):
            order_id = extract_order_id(entry)
            if order_id:
                break

    if order_id is not None:
        tool_result = get_order_status.invoke({"order_id": order_id})
    else:
        tool_result = "No 4-digit order ID found in the customer message."

    messages = [
        SystemMessage(content=ORDERS_PROMPT),
        HumanMessage(
            content=f"Customer message: {state.get('request', '')}\nTOOL RESULT: {tool_result}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        raw_text = f"{tool_result}\nHANDOFF: DONE"

    default_target = SAFETY_NET.get("orders", "done")
    clean_msg, target = parse_handoff(raw_text, default=default_target)

    # Guard rail: an agent cannot hand off to itself
    if target == "orders":
        target = "done"

    # Guard rail: MAX_HOPS
    if hops >= MAX_HOPS:
        target = "done"

    new_log = state.get("log", []) + [f"[orders] {clean_msg}"]
    return {
        "active": target,
        "hops": hops,
        "log": new_log,
    }


def refunds_agent(state: AgentState) -> dict:
    hops = state.get("hops", 0) + 1

    # Plain code runs issue_refund
    order_id = extract_order_id(state.get("request", ""))
    if order_id is None:
        for entry in reversed(state.get("log", [])):
            order_id = extract_order_id(entry)
            if order_id:
                break

    if order_id is not None:
        tool_result = issue_refund.invoke({"order_id": order_id})
    else:
        tool_result = "No 4-digit order ID found to issue refund."

    messages = [
        SystemMessage(content=REFUNDS_PROMPT),
        HumanMessage(
            content=f"Customer message: {state.get('request', '')}\nTOOL RESULT: {tool_result}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        raw_text = f"{tool_result}\nHANDOFF: DONE"

    default_target = SAFETY_NET.get("refunds", "done")
    clean_msg, target = parse_handoff(raw_text, default=default_target)

    # Guard rail: an agent cannot hand off to itself
    if target == "refunds":
        target = "done"

    # Guard rail: MAX_HOPS
    if hops >= MAX_HOPS:
        target = "done"

    new_log = state.get("log", []) + [f"[refunds] {clean_msg}"]
    return {
        "active": target,
        "hops": hops,
        "log": new_log,
    }


def next_agent(state: AgentState) -> str:
    """Follows the active handoff target or routes to done if max hops reached or target is terminal."""
    if state.get("hops", 0) >= MAX_HOPS:
        return "done"
    target = state.get("active", "done")
    if target not in {"triage", "orders", "refunds"}:
        return "done"
    return target


def create_swarm_graph():
    builder = StateGraph(AgentState)

    # Add agents
    builder.add_node("triage", triage_agent)
    builder.add_node("orders", orders_agent)
    builder.add_node("refunds", refunds_agent)

    # Start at triage
    builder.add_edge(START, "triage")

    # Add conditional edges for EVERY agent
    for agent_name in ["triage", "orders", "refunds"]:
        builder.add_conditional_edges(
            agent_name,
            next_agent,
            {
                "triage": "triage",
                "orders": "orders",
                "refunds": "refunds",
                "done": END,
            },
        )

    return builder.compile()


app = create_swarm_graph()


def run_cli():
    print("=" * 60)
    print("🛍️  Swarm Agent CLI (LangGraph + Ollama llama3.2)")
    print("Type your message below. Type 'exit' or 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            customer_input = input("\nCustomer: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not customer_input:
            continue
        if customer_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        initial_state: AgentState = {
            "request": customer_input,
            "active": "triage",
            "log": [],
            "hops": 0,
        }

        for event in app.stream(initial_state):
            for node_name, node_state in event.items():
                if node_state.get("log"):
                    last_msg = node_state["log"][-1]
                    print(last_msg)
                target = node_state.get("active", "done")
                print(f"↪ hands off to: {target}")


if __name__ == "__main__":
    run_cli()
