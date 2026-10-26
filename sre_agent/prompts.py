"""Prompt builders. Each takes plain data and returns (system, user) strings."""

import json

from sre_agent.models import ACTION_DESCRIPTIONS, TOOL_DESCRIPTIONS

SRE_SYSTEM = (
    "You are a senior site reliability engineer on call for a microservice platform. "
    "You reason from evidence, not assumptions: you separate symptoms (where the alert fires) "
    "from causes (where the fault lives), you notice red herrings such as unrelated deploys, "
    "and you quote concrete numbers from tool output. Be concise."
)


CATEGORY_GUIDE = """Root-cause categories — classify by the UNDERLYING cause, not the symptom:
- bad_deploy: a recent release introduced the fault (even if it shows up as OOM, leaks or errors)
- resource_exhaustion: CPU/memory/disk limits hit WITHOUT a code change causing it
- consumer_lag: a message consumer is stuck or falling behind
- dependency_failure: an external or downstream dependency is failing or slow
- db_connection_pool: pool/connection misconfiguration not caused by a recent release
- config_error: a configuration or feature-flag change
- traffic_spike: load grew beyond provisioned capacity
- network: DNS, TLS, packet loss, partitions
- unknown: none of the above"""


def _alert(alert: dict) -> str:
    return (
        f"ALERT [{alert['severity']}] {alert['service']} / {alert['signal']}: {alert['description']}\n"
        f"Measured value {alert['value']} vs threshold {alert['threshold']} (same units) — the alert IS firing; "
        "do not second-guess whether it is real."
    )


def _hypotheses(hypotheses: list[dict]) -> str:
    if not hypotheses:
        return "(none yet)"
    return "\n".join(
        f"- {h['id']} [{h['status']}, conf={h['confidence']:.2f}] {h['category']} @ {h['service']}: {h['title']}"
        + (f"\n    findings: {h['findings']}" if h.get("findings") else "")
        for h in hypotheses
    )


def _evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "(no evidence gathered yet)"
    return "\n\n".join(
        f"[{e['id']}] {e['tool']}(service={e['service']!r}, query={e['query']!r})\n{e['output']}" for e in evidence
    )


def _actions(actions: list[dict]) -> str:
    if not actions:
        return "(none)"
    return "\n".join(
        f"- {a['id']} {a['type']} {a['target']} {a['parameter']} -> {a['status']}: {a['result']}" for a in actions
    )


def triage(alert: dict, topology: str, similar: list[dict]) -> tuple[str, str]:
    memory = (
        "\n".join(
            f"- id={s['incident_id']}: root cause {s['root_cause']['category']} @ {s['root_cause']['service']} — "
            f"{s['root_cause']['title']}. Key findings: {s['root_cause']['findings']}. "
            f"Fixed by: {', '.join(a['type'] + ' ' + a['target'] for a in s['fix'])}"
            for s in similar
        )
        or "(no past incidents for this alert)"
    )
    user = f"""{_alert(alert)}

SERVICE TOPOLOGY
{topology}

PAST INCIDENTS WITH THE SAME ALERT (long-term memory)
{memory}

Triage this alert. Only set matched_incident_id if a past incident is very likely the same failure
mode — it will be verified against live evidence before being trusted."""
    return SRE_SYSTEM, user


def generate_hypotheses(
    alert: dict,
    triage_summary: str,
    topology: str,
    hypotheses: list[dict],
    actions: list[dict],
    missing_evidence: str,
    hints: list[str],
    round_no: int,
    services: list[str],
) -> tuple[str, str]:
    hint_block = "\n".join(f"- {h}" for h in hints) or "(none)"
    user = f"""{_alert(alert)}
Triage: {triage_summary}

SERVICE TOPOLOGY
{topology}

HYPOTHESES SO FAR
{_hypotheses(hypotheses)}

REMEDIATIONS ALREADY TRIED
{_actions(actions)}

WHAT THE LAST SCORING SAID IS STILL MISSING
{missing_evidence or "(n/a)"}

HINTS FROM THE HUMAN ON CALL
{hint_block}

This is investigation round {round_no}. Propose 2-4 NEW, distinct, testable root-cause hypotheses.
Do not repeat refuted ones. A hypothesis whose fix failed may be refined, not restated.

{CATEGORY_GUIDE}

The root cause may live upstream or downstream of the alerting service.
`service` must be one of: {", ".join(services)}"""
    return SRE_SYSTEM, user


def investigator_step(
    hypothesis: dict, alert: dict, topology: str, known_facts: str, evidence: list[dict], steps_left: int
) -> tuple[str, str]:
    tools = "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())
    user = f"""{_alert(alert)}

You are investigating ONE hypothesis:
{hypothesis['id']}: {hypothesis['category']} @ {hypothesis['service']} — {hypothesis['title']}
Rationale: {hypothesis['rationale']}

SERVICE TOPOLOGY
{topology}

FACTS ESTABLISHED IN EARLIER ROUNDS
{known_facts or "(none)"}

TOOLS
{tools}

EVIDENCE YOU HAVE GATHERED SO FAR
{_evidence(evidence)}

You have {steps_left} tool call(s) left. Pick the single most discriminating next query — one whose
result would most change your belief in this hypothesis. Do not repeat a query. Choose `finish`
as soon as the evidence clearly confirms or refutes the hypothesis."""
    return SRE_SYSTEM, user


def assess_hypothesis(hypothesis: dict, alert: dict, evidence: list[dict]) -> tuple[str, str]:
    user = f"""{_alert(alert)}

HYPOTHESIS {hypothesis['id']}: {hypothesis['category']} @ {hypothesis['service']} — {hypothesis['title']}

EVIDENCE
{_evidence(evidence)}

Judge this hypothesis on the evidence alone. "supported" means the evidence points at this as the
cause, "refuted" means the evidence contradicts it, "inconclusive" means it was not tested well enough.
Confidence is your probability that this hypothesis is THE root cause."""
    return SRE_SYSTEM, user


def score(alert: dict, hypotheses: list[dict], evidence: list[dict], actions: list[dict], hints: list[str]) -> tuple[str, str]:
    user = f"""{_alert(alert)}

HYPOTHESES (with per-branch verdicts)
{_hypotheses(hypotheses)}

ALL EVIDENCE
{_evidence(evidence)}

REMEDIATIONS ALREADY TRIED
{_actions(actions)}

HUMAN HINTS
{chr(10).join(hints) or "(none)"}

Correlate the evidence ACROSS branches. Pick the single most likely root cause and give an honest
overall confidence — be well calibrated: if two hypotheses are both plausible, confidence should be
low. A hypothesis whose fix failed is weaker unless the evidence shows the fix itself was wrong."""
    return SRE_SYSTEM, user


def runbook_check(past: dict, alert: dict, evidence: list[dict]) -> tuple[str, str]:
    user = f"""{_alert(alert)}

A PAST INCIDENT with the same alert was caused by:
{past['root_cause']['category']} @ {past['root_cause']['service']} — {past['root_cause']['title']}
Key findings then: {past['root_cause']['findings']}

We re-ran the probes that confirmed it last time. CURRENT EVIDENCE:
{_evidence(evidence)}

Does the current evidence show the SAME failure signature? Similar symptoms with a different cause
is NOT a match."""
    return SRE_SYSTEM, user


def plan_remediation(
    alert: dict,
    hypothesis: dict,
    evidence: list[dict],
    actions: list[dict],
    feedback: list[str],
    topology: str,
) -> tuple[str, str]:
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in ACTION_DESCRIPTIONS.items())
    user = f"""{_alert(alert)}

DIAGNOSED ROOT CAUSE
{hypothesis['category']} @ {hypothesis['service']} — {hypothesis['title']}
Findings: {hypothesis.get('findings', '')}

SUPPORTING EVIDENCE
{_evidence(evidence)}

SERVICE TOPOLOGY (external dependencies and databases cannot be acted on directly)
{topology}

AVAILABLE ACTIONS
{catalog}

ALREADY TRIED (do not repeat failed actions)
{_actions(actions)}

FEEDBACK FROM THE HUMAN APPROVER ON EARLIER PLANS
{chr(10).join(f"- {f}" for f in feedback) or "(none)"}

Propose the minimal remediation that removes the root cause, not just the symptom. Usually ONE
action is right; add another only if the evidence shows it is independently needed. Prefer
reversible, low-blast-radius actions. Beware actions that could amplify the failure."""
    return SRE_SYSTEM, user


def postmortem(state: dict, hypothesis: dict) -> tuple[str, str]:
    timeline = "\n".join(f"- [{t['node']}] {t['message']}" for t in state.get("timeline", []))
    user = f"""Write a blameless postmortem in Markdown for incident {state['incident_id']}.

{_alert(state['alert'])}

ROOT CAUSE: {hypothesis['category']} @ {hypothesis['service']} — {hypothesis['title']}
Findings: {hypothesis.get('findings', '')}

INVESTIGATION TIMELINE
{timeline}

ALL HYPOTHESES CONSIDERED
{_hypotheses(state.get('hypotheses', []))}

REMEDIATION ACTIONS
{_actions(state.get('actions', []))}

FINAL HEALTH CHECK
{json.dumps(state.get('verification'))}

Sections: Summary, Impact, Root cause, Detection & investigation (include ruled-out hypotheses),
Resolution, Action items (concrete, owner-agnostic). Keep it under 400 words."""
    return SRE_SYSTEM, user
