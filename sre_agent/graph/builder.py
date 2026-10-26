"""Assembles the top-level incident graph.

    START → ingest_alert → recall_memory → triage
    triage ─(known signature)→ runbook ─(confirmed)→ plan_remediation
       └──(unknown)──────────→ generate_hypotheses ←──────────────┐
    generate_hypotheses ═Send═▶ investigate ×N (parallel)          │
    investigate → score_hypotheses ─(low confidence, rounds left)──┘
                        ├─(stuck)→ escalate ⏸ ─(hint)→ generate_hypotheses
                        └─(confident)→ plan_remediation → human_approval ⏸
    human_approval ─approve→ execute_remediation → verify
                   ├─reject→ plan_remediation
                   └─reinvestigate→ generate_hypotheses
    verify ─(healthy)→ write_postmortem → save_memory → END
           └─(unhealthy)→ rollback → score_hypotheses | escalate
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer, RetryPolicy
from langgraph.store.base import BaseStore

from sre_agent.graph import nodes
from sre_agent.graph.context import AgentContext
from sre_agent.graph.investigator import build_investigator
from sre_agent.graph.runbook import build_runbook
from sre_agent.state import IncidentInput, IncidentState

LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0)


def build_graph(checkpointer: Checkpointer = None, store: BaseStore | None = None):
    graph = StateGraph(IncidentState, context_schema=AgentContext, input_schema=IncidentInput)

    graph.add_node("ingest_alert", nodes.ingest_alert)
    graph.add_node("recall_memory", nodes.recall_memory)
    graph.add_node("triage", nodes.triage, retry_policy=LLM_RETRY)
    graph.add_node("runbook", build_runbook())
    graph.add_node("generate_hypotheses", nodes.generate_hypotheses, retry_policy=LLM_RETRY)
    graph.add_node("investigate", build_investigator())
    graph.add_node("score_hypotheses", nodes.score_hypotheses, retry_policy=LLM_RETRY)
    graph.add_node("plan_remediation", nodes.plan_remediation, retry_policy=LLM_RETRY)
    graph.add_node("human_approval", nodes.human_approval)
    graph.add_node("execute_remediation", nodes.execute_remediation)
    graph.add_node("verify", nodes.verify)
    graph.add_node("rollback", nodes.rollback)
    graph.add_node("escalate", nodes.escalate)
    graph.add_node("write_postmortem", nodes.write_postmortem, retry_policy=LLM_RETRY)
    graph.add_node("save_memory", nodes.save_memory)

    graph.add_edge(START, "ingest_alert")
    graph.add_edge("ingest_alert", "recall_memory")
    graph.add_edge("recall_memory", "triage")
    graph.add_conditional_edges("runbook", nodes.after_runbook, ["plan_remediation", "generate_hypotheses"])
    graph.add_conditional_edges("generate_hypotheses", nodes.dispatch_investigations, ["investigate", "score_hypotheses"])
    graph.add_edge("investigate", "score_hypotheses")
    graph.add_edge("execute_remediation", "verify")
    graph.add_edge("write_postmortem", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=checkpointer, store=store, name="sre_incident_agent")
