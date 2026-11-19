import re
from typing import Tuple, Optional
from langchain_core.tools import tool

# Fake offline data
ORDERS: dict[int, str] = {
    5001: "delivered",
    5002: "shipped",
    5003: "refunded",
}

VALID_HANDOFFS = {"orders", "refunds", "triage", "done"}


def _lookup_order_status(order_id: int) -> str:
    order_id = int(order_id)
    if order_id in ORDERS:
        return f"Order {order_id} is currently '{ORDERS[order_id]}'."
    return f"Order {order_id} not found."


@tool
def get_order_status(order_id: int) -> str:
    """Fetch the delivery status of a given 4-digit order ID."""
    return _lookup_order_status(order_id)


def _process_refund(order_id: int) -> str:
    order_id = int(order_id)
    if order_id not in ORDERS:
        return f"Order {order_id} not found."
    current_status = ORDERS[order_id]
    if current_status == "delivered":
        ORDERS[order_id] = "refunded"
        return f"Refund successfully issued for order {order_id}. Status updated to 'refunded'."
    elif current_status == "refunded":
        return f"Order {order_id} has already been refunded."
    else:
        return f"Cannot issue refund: Order {order_id} is '{current_status}'. Only 'delivered' orders can be refunded."


@tool
def issue_refund(order_id: int) -> str:
    """Issue a refund for an order. Only 'delivered' orders can be refunded. Marks status as refunded."""
    return _process_refund(order_id)


def parse_handoff(text: str, default: str = "done") -> Tuple[str, str]:
    """
    Parses an agent reply to separate the customer message from the HANDOFF line.
    Returns (message_without_handoff_line, target) where target is in:
    {'orders', 'refunds', 'triage', 'done'}.
    """
    normalized_default = default.strip().lower()
    if normalized_default not in VALID_HANDOFFS:
        normalized_default = "done"

    target = normalized_default
    clean_lines = []

    for line in text.strip().splitlines():
        match = re.search(r"HANDOFF:\s*([a-zA-Z_]+)", line, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().lower()
            if candidate in VALID_HANDOFFS:
                target = candidate
        else:
            clean_lines.append(line)

    clean_message = "\n".join(clean_lines).strip()
    return clean_message, target


def extract_order_id(text: str) -> Optional[int]:
    """Plain code to extract a 4-digit order ID from text."""
    match = re.search(r"\b\d{4}\b", text)
    if match:
        return int(match.group(0))
    return None
