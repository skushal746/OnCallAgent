"""Nodes of the top-level incident graph.

Nodes that make a routing decision return `Command(goto=..., update=...)` so
the decision and the state change it implies are made in one place, with the
runtime context (thresholds, approval mode) available.

Nodes containing `interrupt()` re-run from the top when resumed, so they do
nothing with side effects before the interrupt.
"""

import uuid
from typing import Literal

from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command, Send, interrupt

from sre_agent import memory, prompts
from sre_agent.environment import get_environment
from sre_agent.graph.context import AgentContext
from sre_agent.models import (
    HypothesesOutput,
    RemediationPlanOutput,
    ScoringOutput,
    TriageOutput,
)
from sre_agent.state import IncidentState

RUNBOOK_MATCH_THRESHOLD = 0.6


def _event(node: str, message: str) -> list[dict]:
    return [{"node": node, "message": message}]


def _hypothesis(state: IncidentState, hypothesis_id: str | None) -> dict | None:
    return next((h for h in state.get("hypotheses", []) if h["id"] == hypothesis_id), None)


def _known_facts(state: IncidentState) -> str:
    lines = [f"Triage: {state.get('triage', {}).get('summary', '')}"]
    lines += [
        f"{h['id']} ({h['category']} @ {h['service']}) was {h['status']}: {h['findings']}"
        for h in state.get("hypotheses", [])
        if h["status"] != "open" and h.get("findings")
    ]
    lines += [f"Human hint: {hint}" for hint in state.get("human_hints", [])]
    return "\n".join(lines)


# ------------------------------------------------------------------ intake


def ingest_alert(state: IncidentState, runtime: Runtime[AgentContext]) -> dict:
    alert = state["alert"]
    return {
        "incident_id": state.get("incident_id") or f"INC-{uuid.uuid4().hex[:8]}",
        "status": "triaging",
        "investigation_round": 0,
        "max_rounds": runtime.context.max_investigation_rounds,
        "remediation_attempts": 0,
        "timeline": _event("ingest_alert", f"{alert['service']} {alert['signal']}={alert['value']} (> {alert['threshold']})"),
    }


def recall_memory(state: IncidentState, runtime: Runtime[AgentContext]) -> dict:
    similar = memory.recall(runtime.store, state["alert"])
    return {
        "similar_incidents": similar,
        "timeline": _event("recall_memory", f"{len(similar)} past incident(s) with the same alert"),
    }


def triage(state: IncidentState, runtime: Runtime[AgentContext]) -> Command[Literal["runbook", "generate_hypotheses"]]:
    env = get_environment(state["environment"])
    similar = state.get("similar_incidents", [])
    system, user = prompts.triage(state["alert"], env.topology_summary(), similar)
    result = runtime.context.reasoner.structured(TriageOutput, system=system, user=user, name="triage")

    matched = next((s for s in similar if s["incident_id"] == result.matched_incident_id), None)
    if matched and result.match_confidence < RUNBOOK_MATCH_THRESHOLD:
        matched = None
    update = {
        "triage": result.model_dump(),
        "matched_incident": matched,
        "status": "investigating",
        "timeline": _event(
            "triage",
            f"severity={result.severity}; {result.summary}"
            + (f" Possible repeat of {matched['incident_id']}." if matched else ""),
        ),
    }
    return Command(goto="runbook" if matched else "generate_hypotheses", update=update)


def after_runbook(state: IncidentState) -> Literal["plan_remediation", "generate_hypotheses"]:
    return "plan_remediation" if state.get("runbook_confirmed") else "generate_hypotheses"


# ----------------------------------------------------------- investigation


def generate_hypotheses(state: IncidentState, runtime: Runtime[AgentContext]) -> dict:
    round_no = state.get("investigation_round", 0) + 1
    env = get_environment(state["environment"])
    services = env.services()
    system, user = prompts.generate_hypotheses(
        alert=state["alert"],
        triage_summary=state.get("triage", {}).get("summary", ""),
        topology=env.topology_summary(),
        hypotheses=state.get("hypotheses", []),
        actions=state.get("actions", []),
        missing_evidence=state.get("scoring_rationale", ""),
        hints=state.get("human_hints", []),
        round_no=round_no,
        services=services,
    )
    drafts = runtime.context.reasoner.structured(
        HypothesesOutput, system=system, user=user, name=f"generate_hypotheses:r{round_no}"
    ).hypotheses
    hypotheses = [
        {
            "id": f"H{round_no}.{i}",
            "round": round_no,
            "title": d.title,
            "category": d.category,
            "service": d.service,
            "rationale": d.rationale,
            "status": "open",
            "confidence": 0.0,
            "findings": "",
        }
        for i, d in enumerate([d for d in drafts if d.service in services][:4], start=1)
    ]
    return {
        "hypotheses": hypotheses,
        "investigation_round": round_no,
        "status": "investigating",
        "timeline": _event(
            "generate_hypotheses",
            f"round {round_no}: " + "; ".join(f"{h['id']} {h['category']}@{h['service']}" for h in hypotheses),
        ),
    }


def dispatch_investigations(state: IncidentState) -> list[Send] | Literal["score_hypotheses"]:
    """Fan out: one investigator subgraph per hypothesis opened this round."""
    round_no = state["investigation_round"]
    known_facts = _known_facts(state)
    sends = [
        Send(
            "investigate",
            {
                "hypothesis": h,
                "alert": state["alert"],
                "environment": state["environment"],
                "actions": state.get("actions", []),
                "known_facts": known_facts,
            },
        )
        for h in state.get("hypotheses", [])
        if h["status"] == "open" and h["round"] == round_no
    ]
    return sends or "score_hypotheses"


def score_hypotheses(
    state: IncidentState, runtime: Runtime[AgentContext]
) -> Command[Literal["plan_remediation", "generate_hypotheses", "escalate"]]:
    """Fan in: correlate every branch's evidence and decide whether we know enough."""
    ctx = runtime.context
    system, user = prompts.score(
        state["alert"],
        state.get("hypotheses", []),
        state.get("evidence", []),
        state.get("actions", []),
        state.get("human_hints", []),
    )
    result = ctx.reasoner.structured(ScoringOutput, system=system, user=user, name="score_hypotheses")
    leading = _hypothesis(state, result.leading_hypothesis_id)
    confidence = max(0.0, min(1.0, result.confidence)) if leading else 0.0
    update = {
        "leading_hypothesis_id": leading["id"] if leading else None,
        "overall_confidence": confidence,
        "scoring_rationale": f"{result.reasoning}\nStill missing: {result.missing_evidence}",
    }
    summary = f"leading={update['leading_hypothesis_id']} confidence={confidence:.2f}"

    if leading and confidence >= ctx.confidence_threshold:
        return Command(goto="plan_remediation", update={**update, "timeline": _event("score_hypotheses", summary + " → remediate")})

    out_of_budget = state.get("tool_calls_used", 0) >= ctx.tool_call_budget
    if state["investigation_round"] < state["max_rounds"] and not out_of_budget:
        return Command(
            goto="generate_hypotheses",
            update={**update, "timeline": _event("score_hypotheses", summary + " → investigate further")},
        )
    reason = "tool-call budget exhausted" if out_of_budget else f"no confident root cause after {state['investigation_round']} rounds"
    return Command(
        goto="escalate",
        update={**update, "escalation_reason": reason, "timeline": _event("score_hypotheses", summary + f" → escalate ({reason})")},
    )


# ------------------------------------------------------------- remediation


def plan_remediation(state: IncidentState, runtime: Runtime[AgentContext]) -> Command[Literal["human_approval", "escalate"]]:
    hypothesis = _hypothesis(state, state["leading_hypothesis_id"])
    env = get_environment(state["environment"])
    tried = state.get("actions", [])
    system, user = prompts.plan_remediation(
        state["alert"],
        hypothesis,
        [e for e in state.get("evidence", []) if e["hypothesis_id"] == hypothesis["id"]],
        tried,
        state.get("plan_feedback", []),
        env.topology_summary(),
    )
    plan = runtime.context.reasoner.structured(RemediationPlanOutput, system=system, user=user, name="plan_remediation")

    failed = {(a["type"], a["target"]) for a in tried if a["status"] in ("reverted", "failed")}
    actions = [
        a.model_dump()
        for a in plan.actions
        if a.target in env.services() and (a.type, a.target) not in failed
    ][:3]
    if not actions:
        return Command(
            goto="escalate",
            update={
                "escalation_reason": "planner produced no new valid remediation action",
                "timeline": _event("plan_remediation", "no valid untried actions → escalate"),
            },
        )
    plan_record = {
        "attempt": state.get("remediation_attempts", 0) + 1,
        "hypothesis_id": hypothesis["id"],
        "actions": actions,
        "risk": plan.risk,
        "expected_outcome": plan.expected_outcome,
        "rollback_strategy": plan.rollback_strategy,
    }
    return Command(
        goto="human_approval",
        update={
            "plan": plan_record,
            "status": "awaiting_approval",
            "timeline": _event(
                "plan_remediation",
                f"attempt {plan_record['attempt']} ({plan.risk} risk): "
                + ", ".join(f"{a['type']} {a['target']}" for a in actions),
            ),
        },
    )


def human_approval(
    state: IncidentState, runtime: Runtime[AgentContext]
) -> Command[Literal["execute_remediation", "plan_remediation", "generate_hypotheses", "__end__"]]:
    plan = state["plan"]
    mode = runtime.context.approval_mode
    if mode == "auto" or (mode == "auto_low_risk" and plan["risk"] == "low"):
        decision = {"decision": "approve", "feedback": f"auto-approved ({mode})"}
    else:
        decision = interrupt(
            {
                "type": "approval",
                "incident_id": state["incident_id"],
                "hypothesis": _hypothesis(state, state["leading_hypothesis_id"]),
                "confidence": state.get("overall_confidence", 0.0),
                "plan": plan,
                "options": ["approve", "reject", "reinvestigate", "abort"],
            }
        )
    if isinstance(decision, str):
        decision = {"decision": decision}
    choice = decision.get("decision", "reject")
    feedback = decision.get("feedback", "")
    log = _event("human_approval", f"{choice}" + (f": {feedback}" if feedback else ""))

    if choice == "approve":
        return Command(goto="execute_remediation", update={"status": "remediating", "timeline": log})
    if choice == "reinvestigate":
        return Command(
            goto="generate_hypotheses",
            update={
                "human_hints": [feedback or "Approver doubts the diagnosis; consider other causes."],
                "max_rounds": state["max_rounds"] + 1,
                "hypotheses": [{"id": state["leading_hypothesis_id"], "status": "inconclusive"}],
                "timeline": log,
            },
        )
    if choice == "abort":
        return Command(goto=END, update={"status": "aborted", "timeline": log})
    return Command(
        goto="plan_remediation",
        update={"plan_feedback": [feedback or "Plan rejected without comment; propose a different approach."], "timeline": log},
    )


def execute_remediation(state: IncidentState) -> dict:
    plan = state["plan"]
    env = get_environment(state["environment"], state.get("actions", []))
    records = []
    for i, action in enumerate(plan["actions"], start=1):
        ok, message = env.apply_action(action["type"], action["target"], action.get("parameter", ""))
        records.append(
            {
                "id": f"A{plan['attempt']}.{i}",
                "attempt": plan["attempt"],
                "type": action["type"],
                "target": action["target"],
                "parameter": action.get("parameter", ""),
                "reason": action["reason"],
                "status": "applied" if ok else "failed",
                "result": message,
            }
        )
    return {
        "actions": records,
        "remediation_attempts": plan["attempt"],
        "timeline": _event("execute_remediation", "; ".join(f"{r['id']} {r['status']}: {r['result']}" for r in records)),
    }


def verify(state: IncidentState) -> Command[Literal["write_postmortem", "rollback"]]:
    alert = state["alert"]
    env = get_environment(state["environment"], state.get("actions", []))
    health = env.check_health(alert["service"], alert["signal"])
    message = f"{health['signal']}={health['value']} (threshold {health['threshold']}) → {'healthy' if health['healthy'] else 'still unhealthy'}"
    if health["healthy"]:
        return Command(
            goto="write_postmortem",
            update={
                "verification": dict(health),
                "status": "resolved",
                "hypotheses": [{"id": state["leading_hypothesis_id"], "status": "confirmed"}],
                "timeline": _event("verify", message),
            },
        )
    return Command(goto="rollback", update={"verification": dict(health), "timeline": _event("verify", message)})


def rollback(state: IncidentState, runtime: Runtime[AgentContext]) -> Command[Literal["score_hypotheses", "escalate"]]:
    """Compensate the failed attempt, then go back to diagnosis."""
    attempt = state["remediation_attempts"]
    verification = state["verification"]
    reverted = [
        {**a, "status": "reverted", "result": f"{a['result']}; reverted — {verification['signal']} stayed at {verification['value']}"}
        for a in state.get("actions", [])
        if a["attempt"] == attempt and a["status"] == "applied"
    ]
    update = {
        "actions": reverted,
        "hypotheses": [{"id": state["leading_hypothesis_id"], "status": "fix_failed"}],
        "status": "investigating",
        "timeline": _event("rollback", f"reverted {len(reverted)} action(s) from attempt {attempt}"),
    }
    if attempt >= state.get("max_attempts", runtime.context.max_remediation_attempts):
        return Command(goto="escalate", update={**update, "escalation_reason": f"{attempt} remediation attempt(s) failed"})
    return Command(goto="score_hypotheses", update=update)


def escalate(state: IncidentState, runtime: Runtime[AgentContext]) -> Command[Literal["generate_hypotheses", "__end__"]]:
    reason = state.get("escalation_reason", "investigation stalled")
    if runtime.context.unattended:
        return Command(goto=END, update={"status": "escalated", "timeline": _event("escalate", f"{reason}; unattended → stop")})

    response = interrupt(
        {
            "type": "escalation",
            "incident_id": state["incident_id"],
            "reason": reason,
            "hypotheses": state.get("hypotheses", []),
            "leading_hypothesis_id": state.get("leading_hypothesis_id"),
            "confidence": state.get("overall_confidence", 0.0),
            "options": ["hint", "abort"],
        }
    )
    if isinstance(response, str):
        response = {"action": "hint", "hint": response} if response not in ("abort", "") else {"action": "abort"}
    if response.get("action") == "hint" and response.get("hint"):
        return Command(
            goto="generate_hypotheses",
            update={
                "human_hints": [response["hint"]],
                "max_rounds": state["investigation_round"] + 1,
                "max_attempts": max(state.get("remediation_attempts", 0) + 1, runtime.context.max_remediation_attempts),
                "timeline": _event("escalate", f"human hint: {response['hint']}"),
            },
        )
    return Command(goto=END, update={"status": "escalated", "timeline": _event("escalate", f"{reason}; handed to human")})


# --------------------------------------------------------------- wrap-up


def write_postmortem(state: IncidentState, runtime: Runtime[AgentContext]) -> dict:
    hypothesis = _hypothesis(state, state["leading_hypothesis_id"])
    system, user = prompts.postmortem(state, hypothesis)
    return {"postmortem": runtime.context.reasoner.text(system=system, user=user, name="write_postmortem")}


def save_memory(state: IncidentState, runtime: Runtime[AgentContext]) -> dict:
    hypothesis = _hypothesis(state, state["leading_hypothesis_id"])
    probes, seen = [], set()
    for e in state.get("evidence", []):
        key = (e["tool"], e["service"], e["query"])
        if e["hypothesis_id"] == hypothesis["id"] and key not in seen:
            seen.add(key)
            probes.append({"tool": e["tool"], "service": e["service"], "query": e["query"]})
    memory.remember(
        runtime.store,
        {
            "incident_id": state["incident_id"],
            "alert_service": state["alert"]["service"],
            "alert_signal": state["alert"]["signal"],
            "root_cause": {
                "title": hypothesis["title"],
                "category": hypothesis["category"],
                "service": hypothesis["service"],
                "findings": hypothesis.get("findings", ""),
            },
            "fix": [
                {"type": a["type"], "target": a["target"], "parameter": a["parameter"]}
                for a in state.get("actions", [])
                if a["status"] == "applied"
            ],
            "signature_probes": probes[:4],
        },
    )
    return {"timeline": _event("save_memory", f"stored {state['incident_id']} with {len(probes[:4])} signature probe(s)")}
