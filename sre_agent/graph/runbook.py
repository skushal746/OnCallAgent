"""Runbook fast-path subgraph.

When triage matches the alert to a past incident from long-term memory, this
subgraph re-runs only the probes that confirmed that incident and asks the LLM
whether the live evidence shows the same failure signature. A confirmed match
skips the full parallel investigation; a mismatch falls back to it.

    probe_signature ──▶ compare_signature ──▶ END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from sre_agent import prompts
from sre_agent.environment import get_environment
from sre_agent.graph.context import AgentContext
from sre_agent.models import RunbookCheck
from sre_agent.state import RunbookInput, RunbookOutput, RunbookState

RUNBOOK_HYPOTHESIS_ID = "H0.1"


def probe_signature(state: RunbookState) -> dict:
    env = get_environment(state["environment"], state["actions"])
    probes = state["matched_incident"].get("signature_probes", [])[:4]
    evidence = [
        {
            "id": f"E-{RUNBOOK_HYPOTHESIS_ID}-{i}",
            "hypothesis_id": RUNBOOK_HYPOTHESIS_ID,
            "tool": probe["tool"],
            "service": probe["service"],
            "query": probe.get("query", ""),
            "reasoning": f"re-running probe from {state['matched_incident']['incident_id']}",
            "output": env.run_tool(probe["tool"], probe["service"], probe.get("query", "")),
        }
        for i, probe in enumerate(probes, start=1)
    ]
    return {"evidence": evidence, "tool_calls_used": len(evidence)}


def compare_signature(state: RunbookState, runtime: Runtime[AgentContext]) -> dict:
    past = state["matched_incident"]
    system, user = prompts.runbook_check(past, state["alert"], state.get("evidence", []))
    check = runtime.context.reasoner.structured(RunbookCheck, system=system, user=user, name="runbook_check")
    confidence = max(0.0, min(1.0, check.confidence))
    if not check.matches or confidence < runtime.context.confidence_threshold:
        return {"runbook_confirmed": False, "scoring_rationale": f"Runbook mismatch: {check.reasoning}"}

    root = past["root_cause"]
    hypothesis = {
        "id": RUNBOOK_HYPOTHESIS_ID,
        "round": 0,
        "title": root["title"],
        "category": root["category"],
        "service": root["service"],
        "rationale": f"Same signature as {past['incident_id']}",
        "status": "supported",
        "confidence": confidence,
        "findings": check.reasoning,
    }
    return {
        "runbook_confirmed": True,
        "hypotheses": [hypothesis],
        "leading_hypothesis_id": RUNBOOK_HYPOTHESIS_ID,
        "overall_confidence": confidence,
        "scoring_rationale": check.reasoning,
    }


def build_runbook():
    graph = StateGraph(
        RunbookState,
        context_schema=AgentContext,
        input_schema=RunbookInput,
        output_schema=RunbookOutput,
    )
    graph.add_node("probe_signature", probe_signature)
    graph.add_node("compare_signature", compare_signature)
    graph.add_edge(START, "probe_signature")
    graph.add_edge("probe_signature", "compare_signature")
    graph.add_edge("compare_signature", END)
    return graph.compile(name="runbook")
