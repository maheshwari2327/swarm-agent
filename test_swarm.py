import unittest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

import tools
from tools import (
    ORDERS,
    get_order_status,
    issue_refund,
    parse_handoff,
    extract_order_id,
)
import agent
from agent import (
    AgentState,
    triage_agent,
    orders_agent,
    refunds_agent,
    next_agent,
    create_swarm_graph,
    MAX_HOPS,
    SAFETY_NET,
)


class TestTools(unittest.TestCase):
    def setUp(self):
        # Reset ORDERS before each test
        tools.ORDERS = {
            5001: "delivered",
            5002: "shipped",
            5003: "refunded",
        }

    def test_get_order_status(self):
        res1 = get_order_status.invoke({"order_id": 5001})
        self.assertIn("delivered", res1)

        res2 = get_order_status.invoke({"order_id": 5002})
        self.assertIn("shipped", res2)

        res_unknown = get_order_status.invoke({"order_id": 9999})
        self.assertIn("not found", res_unknown)

    def test_issue_refund(self):
        # 5001 is delivered -> refund allowed
        res1 = issue_refund.invoke({"order_id": 5001})
        self.assertIn("successfully issued", res1)
        self.assertEqual(tools.ORDERS[5001], "refunded")

        # 5002 is shipped -> refund not allowed
        res2 = issue_refund.invoke({"order_id": 5002})
        self.assertIn("shipped", res2)
        self.assertIn("Cannot issue refund", res2)

        # 5003 is already refunded
        res3 = issue_refund.invoke({"order_id": 5003})
        self.assertIn("already been refunded", res3)

    def test_parse_handoff(self):
        # Standard case
        msg, target = parse_handoff("Welcome to our store!\nHANDOFF: orders")
        self.assertEqual(msg, "Welcome to our store!")
        self.assertEqual(target, "orders")

        # Markdown or uppercase
        msg, target = parse_handoff("Refund processed.\n**HANDOFF: DONE**")
        self.assertEqual(msg, "Refund processed.")
        self.assertEqual(target, "done")

        # Missing handoff uses default
        msg, target = parse_handoff("No handoff here.", default="orders")
        self.assertEqual(msg, "No handoff here.")
        self.assertEqual(target, "orders")

    def test_extract_order_id(self):
        self.assertEqual(extract_order_id("Status of 5001 please"), 5001)
        self.assertEqual(extract_order_id("Order #5002 has arrived"), 5002)
        self.assertIsNone(extract_order_id("No numbers here"))
        self.assertIsNone(extract_order_id("Only 123 is 3 digits"))


class TestSwarmAgentFlow(unittest.TestCase):
    def setUp(self):
        tools.ORDERS = {
            5001: "delivered",
            5002: "shipped",
            5003: "refunded",
        }
        self.original_llm = agent.llm

    def tearDown(self):
        agent.llm = self.original_llm

    def test_guardrail_self_handoff_and_safety_net(self):
        # Test triage missing handoff triggers SAFETY_NET
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = AIMessage(content="Hello there! No handoff line.")

        state: AgentState = {
            "request": "Where is my order?",
            "active": "triage",
            "log": [],
            "hops": 0,
        }
        out = triage_agent(state)
        # Should fall back to SAFETY_NET['triage'] which is 'orders'
        self.assertEqual(out["active"], "orders")
        self.assertEqual(out["hops"], 1)

    def test_guardrail_max_hops(self):
        state: AgentState = {
            "request": "Help!",
            "active": "triage",
            "log": [],
            "hops": MAX_HOPS,
        }
        self.assertEqual(next_agent(state), "done")

    def test_full_swarm_graph_small_talk(self):
        app = create_swarm_graph()
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = AIMessage(
            content="Hello! Hope you are having a wonderful day!\nHANDOFF: DONE"
        )

        initial_state: AgentState = {
            "request": "Hello, how are you?",
            "active": "triage",
            "log": [],
            "hops": 0,
        }

        events = list(app.stream(initial_state))
        self.assertEqual(len(events), 1)
        self.assertIn("triage", events[0])
        self.assertEqual(events[0]["triage"]["active"], "done")

    def test_full_swarm_graph_refund_flow(self):
        app = create_swarm_graph()
        agent.llm = MagicMock()

        # Step 1: Triage hands off to orders
        # Step 2: Orders sees delivered + refund request -> hands off to refunds
        # Step 3: Refunds handles refund -> hands off to done
        agent.llm.invoke.side_effect = [
            AIMessage(content="I can help route that.\nHANDOFF: orders"),
            AIMessage(content="Order 5001 is delivered. Forwarding to refunds.\nHANDOFF: refunds"),
            AIMessage(content="I processed your refund for order 5001.\nHANDOFF: DONE"),
        ]

        initial_state: AgentState = {
            "request": "I received order 5001 and it's damaged. I want my money back.",
            "active": "triage",
            "log": [],
            "hops": 0,
        }

        events = list(app.stream(initial_state))
        self.assertEqual(len(events), 3)

        self.assertIn("triage", events[0])
        self.assertEqual(events[0]["triage"]["active"], "orders")

        self.assertIn("orders", events[1])
        self.assertEqual(events[1]["orders"]["active"], "refunds")

        self.assertIn("refunds", events[2])
        self.assertEqual(events[2]["refunds"]["active"], "done")

        # Check order 5001 was marked as refunded
        self.assertEqual(tools.ORDERS[5001], "refunded")


if __name__ == "__main__":
    unittest.main()
