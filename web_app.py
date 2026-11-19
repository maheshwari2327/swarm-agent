import sys
from pathlib import Path

# Ensure the directory containing this script is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import tools
from agent import app, AgentState, MAX_HOPS

# Streamlit Page Config
st.set_page_config(
    page_title="Swarm Multi-Agent Shop",
    page_icon="🛍️",
    layout="wide",
)

# Custom CSS for styling handoff pills and badges
st.markdown(
    """
    <style>
    .handoff-badge {
        display: inline-block;
        padding: 4px 10px;
        margin: 6px 0;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
    }
    .handoff-done {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    .agent-header {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Live mock database state & controls
with st.sidebar:
    st.title("⚙️ Swarm Control Panel")
    st.markdown("### 📦 Order Database (Mock Data)")
    
    # Display table of current orders
    for oid, status in sorted(tools.ORDERS.items()):
        badge_color = "green" if status == "delivered" else "orange" if status == "shipped" else "gray"
        st.write(f"**Order #{oid}**: `:{badge_color}[{status}]`")

    if st.button("🔄 Reset Orders Database", use_container_width=True):
        tools.ORDERS = {
            5001: "delivered",
            5002: "shipped",
            5003: "refunded",
        }
        st.success("Orders reset to initial state!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🛡️ Guard Rails")
    st.markdown(f"- **Max Hops**: `{MAX_HOPS}`")
    st.markdown("- **Self-Handoff**: `Blocked`")
    st.markdown("- **Safety Net**: `triage -> orders`")

    st.markdown("---")
    st.markdown("### 💡 Quick Test Prompts")
    quick_prompts = [
        "Can I get a refund for order 5001?",
        "Where is my order 5002 right now?",
        "Can I get a refund for order 5002?",
        "Hi there, how are you today?",
    ]
    for qp in quick_prompts:
        if st.button(qp, key=qp, use_container_width=True):
            st.session_state["prompt_input"] = qp
            st.rerun()

# Main Chat View
st.title("🛍️ Decentralized Swarm Multi-Agent")
st.caption("No supervisor: Autonomous hand-offs between Triage, Orders, and Refunds via LangGraph.")

# Initialize message history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous conversations
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        if chat["role"] == "user":
            st.markdown(chat["content"])
        else:
            for step in chat["steps"]:
                agent_name = step["agent"]
                content = step["content"]
                target = step["target"]

                agent_emojis = {
                    "triage": "🛎️ Front Desk (Triage)",
                    "orders": "📦 Orders Agent",
                    "refunds": "💳 Refunds Agent",
                }
                label = agent_emojis.get(agent_name, agent_name.capitalize())
                st.markdown(f"**{label}**")
                st.markdown(content)
                if target == "done":
                    st.markdown(f"<span class='handoff-badge handoff-done'>✅ Resolved (DONE)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='handoff-badge'>↪ Hands off to: {target}</span>", unsafe_allow_html=True)

# User Input handling (from chat_input or quick prompt button)
user_prompt = st.chat_input("Ask about an order, request a refund, or say hello...")
if "prompt_input" in st.session_state and st.session_state["prompt_input"]:
    user_prompt = st.session_state.pop("prompt_input")

if user_prompt:
    # 1. Display user query
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Execute Swarm Graph
    steps = []
    initial_state: AgentState = {
        "request": user_prompt,
        "active": "triage",
        "log": [],
        "hops": 0,
    }

    with st.chat_message("assistant"):
        status_box = st.status("Swarm processing...", expanded=True)
        with status_box:
            for event in app.stream(initial_state):
                for node_name, node_state in event.items():
                    log_entry = node_state["log"][-1] if node_state.get("log") else ""
                    # Remove the [agent] prefix from log for clean display
                    if log_entry.startswith(f"[{node_name}] "):
                        clean_content = log_entry[len(f"[{node_name}] "):]
                    else:
                        clean_content = log_entry

                    target = node_state.get("active", "done")

                    steps.append({
                        "agent": node_name,
                        "content": clean_content,
                        "target": target,
                    })

                    agent_emojis = {
                        "triage": "🛎️ Front Desk (Triage)",
                        "orders": "📦 Orders Agent",
                        "refunds": "💳 Refunds Agent",
                    }
                    label = agent_emojis.get(node_name, node_name.capitalize())
                    st.markdown(f"**{label}**")
                    st.markdown(clean_content)
                    if target == "done":
                        st.markdown(f"<span class='handoff-badge handoff-done'>✅ Resolved (DONE)</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='handoff-badge'>↪ Hands off to: {target}</span>", unsafe_allow_html=True)
            status_box.update(label="Conversation complete", state="complete", expanded=True)

    # Save to history
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    st.session_state.chat_history.append({"role": "assistant", "steps": steps})
    st.rerun()
