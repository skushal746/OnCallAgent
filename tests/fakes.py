"""Scripted stand-in for the LLM so graph mechanics can be tested deterministically."""

import re
from collections import defaultdict
from collections.abc import Callable

from sre_agent.models import (
    ActionDraft,
    HypothesesOutput,
    HypothesisAssessment,
    HypothesisDraft,
    InvestigatorStep,
    RemediationPlanOutput,
    RunbookCheck,
    ScoringOutput,
    TriageOutput,
)


class ScriptedReasoner:
    """Dispatches on the call `name` (e.g. "assess:H1.2"); handlers get (call_index, user_prompt)."""

    def __init__(self, handlers: dict[str, Callable[[int, str], object]]):
        self.handlers = handlers
        self.calls: dict[str, int] = defaultdict(int)

    def _dispatch(self, name: str, user: str):
        key = next(k for k in sorted(self.handlers, key=len, reverse=True) if name.startswith(k))
        index = self.calls[key]
        self.calls[key] += 1
        return self.handlers[key](index, user)

    def structured(self, schema, *, system, user, name):
        result = self._dispatch(name, user)
        assert isinstance(result, schema), f"{name}: expected {schema.__name__}, got {type(result).__name__}"
        return result

    def text(self, *, system, user, name):
        return self._dispatch(name, user)


def evidence_count(prompt: str) -> int:
    return len(re.findall(r"^\[E-H", prompt, flags=re.MULTILINE))


def oom_handlers(plans: list[list[tuple[str, str]]] | None = None, score_confidence: float = 0.9) -> dict:
    """Handlers that solve the bad_deploy_oom scenario. `plans` scripts successive remediation plans."""
    plans = plans or [[("rollback_deploy", "user-service")]]

    def triage(_, user):
        match = re.search(r"id=(INC-\w+)", user)
        return TriageOutput(
            severity="high",
            affected_services=["user-service", "api-gateway"],
            summary="user-service is returning 5xx.",
            matched_incident_id=match.group(1) if match else None,
            match_confidence=0.9 if match else 0.0,
        )

    def hypotheses(_, __):
        return HypothesesOutput(
            hypotheses=[
                HypothesisDraft(title="v2.4.1 leaks memory", category="bad_deploy", service="user-service", rationale="recent deploy"),
                HypothesisDraft(title="redis-sessions down", category="dependency_failure", service="redis-sessions", rationale="session store"),
            ]
        )

    def plan_h1(_, user):
        steps = [
            InvestigatorStep(reasoning="check deploys", tool="get_recent_deploys", service="user-service"),
            InvestigatorStep(reasoning="check pods", tool="describe_service", service="user-service"),
        ]
        n = evidence_count(user)
        return steps[n] if n < len(steps) else InvestigatorStep(reasoning="enough", tool="finish")

    def plan_h2(_, user):
        if evidence_count(user) == 0:
            return InvestigatorStep(reasoning="check redis", tool="get_metrics", service="redis-sessions")
        return InvestigatorStep(reasoning="enough", tool="finish")

    def remediation(index, _):
        actions = plans[min(index, len(plans) - 1)]
        return RemediationPlanOutput(
            actions=[ActionDraft(type=t, target=s, reason="scripted") for t, s in actions],
            risk="medium",
            expected_outcome="5xx back under threshold",
            rollback_strategy="roll forward",
        )

    supported = HypothesisAssessment(verdict="supported", confidence=0.9, findings="OOMKilled x14 after v2.4.1")
    refuted = HypothesisAssessment(verdict="refuted", confidence=0.05, findings="redis hit ratio 0.97")
    per_round = {}
    for r in range(1, 4):  # every round proposes the same pair, numbered H<r>.1 / H<r>.2
        per_round |= {
            f"plan_step:H{r}.1": plan_h1,
            f"plan_step:H{r}.2": plan_h2,
            f"assess:H{r}.1": lambda _, __: supported,
            f"assess:H{r}.2": lambda _, __: refuted,
        }

    return {
        **per_round,
        "triage": triage,
        "generate_hypotheses": hypotheses,
        "score_hypotheses": lambda _, __: ScoringOutput(
            leading_hypothesis_id="H1.1", confidence=score_confidence, reasoning="OOM after deploy", missing_evidence=""
        ),
        "plan_remediation": remediation,
        "runbook_check": lambda _, __: RunbookCheck(matches=True, confidence=0.9, reasoning="same OOM signature"),
        "write_postmortem": lambda _, __: "# Postmortem\nRolled back user-service.",
    }
