# Swarm Agent (Hand-off Multi-Agent CLI)

A decentralized multi-agent customer service swarm built using **LangChain**, **LangGraph**, and local **Ollama (`llama3.2`)**.

## Architecture & Concept

In a Swarm architecture, there is **no centralized supervisor agent**. Instead:
1. Each specialized agent executes its focused task.
2. The agent produces its response message for the customer followed by a handoff declaration:
   ```text
   <Message to customer>
   HANDOFF: <agent_name | DONE>
   ```
3. A LangGraph conditional edge parses the handoff target and routes execution directly to the designated agent (or terminates at `END`).

```
                    ┌─────────────────────────┐
                    │          START          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      triage_agent       │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │ HANDOFF: orders               │ HANDOFF: DONE
                 ▼                               ▼
    ┌─────────────────────────┐             ┌─────────┐
    │      orders_agent       │             │   END   │
    └────────────┬────────────┘             └─────────┘
                 │                               ▲
                 ├───────────────────────────────┤
                 │ HANDOFF: refunds              │ HANDOFF: DONE
                 ▼                               │
    ┌─────────────────────────┐                  │
    │      refunds_agent      │──────────────────┘
    └─────────────────────────┘  HANDOFF: DONE
```

---

## Agents

| Agent | Responsibility | Handoff Rules |
| :--- | :--- | :--- |
| **`triage`** | Greets the customer; routes order/delivery/damage/refund requests to orders, or responds to small talk. | `orders` or `DONE` |
| **`orders`** | Extracts 4-digit order ID via plain code, fetches status with `get_order_status`, explains status to customer. | If refund requested & status is `delivered` → `refunds`, else `DONE` |
| **`refunds`**| Executes `issue_refund` via plain code, explains the result to the customer. | Always `DONE` |

---

## Guard Rails

- **`MAX_HOPS = 5`**: Prevents infinite loops between agents.
- **No Self-Handoff**: Agents cannot hand off back to themselves.
- **`SAFETY_NET = {'triage': 'orders'}`**: Ensures that if the `triage` agent output fails to produce a valid handoff tag, it safely defaults to `orders`.

---

## Setup & Requirements

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Local Ollama LLM
Ensure Ollama is running with `llama3.2`:
```bash
ollama run llama3.2
```

---

## Running the Application

### 1. Web UI (Streamlit)
To launch the interactive web interface:
```bash
streamlit run web_app.py
```
This opens `http://localhost:8501` with:
- Live multi-agent chat interface with hand-off status badges.
- Sidebar showing live mock order status updates in real time.
- Database reset button and quick-test prompt buttons.

### 2. Command Line Interface (CLI)
Start the CLI loop:
```bash
python main.py
```
Or directly:
```bash
python agent.py
```

### Running Tests
Run the comprehensive unit test suite:
```bash
python -m unittest test_swarm.py
```

---

## File Structure

```text
swarm agent/
├── prompts.py         # System prompts and handoff instructions
├── tools.py           # Offline order data, tools, parser, ID extractor
├── agent.py           # StateGraph definition, guard rails, agent nodes, CLI
├── main.py            # CLI entry point
├── test_swarm.py      # Unit test suite
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```
