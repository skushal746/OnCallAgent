"""Graph state schemas and reducers.

Every list of records that parallel branches or subgraphs write to uses
`merge_by_id`, so a branch can both append new records and update existing
ones (e.g. an investigator flips its hypothesis from "open" to "refuted")
without clobbering what sibling branches wrote in the same superstep.
"""

import operator
from typing import Annotated, Any, TypedDict

from sre_agent.models import (
    Alert,
    EnvironmentRef,
    Evidence,
    Hypothesis,
    RemediationAction,
    TimelineEvent,
)


def merge_by_id(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    merged = {item["id"]: item for item in left or []}
    for item in right or []:
        merged[item["id"]] = {**merged.get(item["id"], {}), **item}
    return list(merged.values())


class IncidentInput(TypedDict):
    alert: Alert
    environment: EnvironmentRef


class IncidentState(TypedDict, total=False):
    incident_id: str
    alert: Alert
    environment: EnvironmentRef
    status: str

    # triage + long-term memory
    triage: dict[str, Any]
    similar_incidents: list[dict[str, Any]]
    matched_incident: dict[str, Any] | None
    runbook_confirmed: bool

    # investigation
    hypotheses: Annotated[list[Hypothesis], merge_by_id]
    evidence: Annotated[list[Evidence], merge_by_id]
    investigation_round: int
    max_rounds: int
    tool_calls_used: Annotated[int, operator.add]
    leading_hypothesis_id: str | None
    overall_confidence: float
    scoring_rationale: str
    human_hints: Annotated[list[str], operator.add]
    escalation_reason: str

    # remediation
    plan: dict[str, Any] | None
    plan_feedback: Annotated[list[str], operator.add]
    actions: Annotated[list[RemediationAction], merge_by_id]
    remediation_attempts: int
    max_attempts: int
    verification: dict[str, Any] | None

    # wrap-up
    timeline: Annotated[list[TimelineEvent], operator.add]
    postmortem: str | None


# ---------------------------------------------------- investigator subgraph


class InvestigatorInput(TypedDict):
    hypothesis: Hypothesis
    alert: Alert
    environment: EnvironmentRef
    actions: list[RemediationAction]
    known_facts: str


class InvestigatorOutput(TypedDict):
    hypotheses: list[Hypothesis]
    evidence: list[Evidence]
    tool_calls_used: int


class InvestigatorState(InvestigatorInput, total=False):
    pending_step: dict[str, Any] | None
    evidence: Annotated[list[Evidence], merge_by_id]
    tool_calls_used: Annotated[int, operator.add]
    hypotheses: list[Hypothesis]


# --------------------------------------------------------- runbook subgraph


class RunbookInput(TypedDict):
    alert: Alert
    environment: EnvironmentRef
    actions: list[RemediationAction]
    matched_incident: dict[str, Any]


class RunbookOutput(TypedDict, total=False):
    runbook_confirmed: bool
    hypotheses: list[Hypothesis]
    evidence: list[Evidence]
    tool_calls_used: int
    leading_hypothesis_id: str | None
    overall_confidence: float
    scoring_rationale: str


class RunbookState(RunbookInput, total=False):
    runbook_confirmed: bool
    hypotheses: list[Hypothesis]
    evidence: Annotated[list[Evidence], merge_by_id]
    tool_calls_used: Annotated[int, operator.add]
    leading_hypothesis_id: str | None
    overall_confidence: float
    scoring_rationale: str
