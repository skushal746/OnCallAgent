"""Domain vocabulary shared by the graph, the environment and the LLM schemas.

State items are plain TypedDicts so checkpoints serialize cleanly; the pydantic
models at the bottom are only used as structured-output schemas for the LLM.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

RootCauseCategory = Literal[
    "bad_deploy",
    "resource_exhaustion",
    "consumer_lag",
    "dependency_failure",
    "db_connection_pool",
    "config_error",
    "traffic_spike",
    "network",
    "unknown",
]

ActionType = Literal[
    "restart_service",
    "scale_service",
    "rollback_deploy",
    "increase_connection_pool",
    "reset_consumer_group",
    "toggle_feature_flag",
    "failover_dependency",
]

ToolName = Literal[
    "get_logs",
    "get_metrics",
    "get_traces",
    "get_kafka_lag",
    "get_recent_deploys",
    "describe_service",
    "finish",
]

HypothesisStatus = Literal["open", "supported", "refuted", "inconclusive", "fix_failed", "confirmed"]

IncidentStatus = Literal[
    "triaging",
    "investigating",
    "awaiting_approval",
    "remediating",
    "resolved",
    "escalated",
    "aborted",
]

TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_logs": "Recent log lines for `service`. `query` is an optional keyword filter (e.g. 'ERROR', 'timeout').",
    "get_metrics": "Key metrics for `service`: latency, error rate, CPU, memory, restarts, pool/connection usage.",
    "get_traces": "Sampled slow or failing traces passing through `service`, with per-span timings.",
    "get_kafka_lag": "Consumer-group lag per partition. `service` is the consuming service.",
    "get_recent_deploys": "Deploys in the last 24h. `service` is a service name or 'all'.",
    "describe_service": "Pod status, replicas, autoscaling, resource limits, feature flags and recent events for `service`.",
    "finish": "Stop gathering evidence; you have enough to judge the hypothesis.",
}

ACTION_DESCRIPTIONS: dict[str, str] = {
    "restart_service": "Rolling restart of all pods of `target`.",
    "scale_service": "Add replicas to `target` (raises autoscaler max if needed).",
    "rollback_deploy": "Roll `target` back to its previous version.",
    "increase_connection_pool": "Raise the DB connection pool size of `target`.",
    "reset_consumer_group": "Move the Kafka consumer group of `target` past the current offset (skips stuck messages).",
    "toggle_feature_flag": "Flip a feature flag on `target` (put the flag name in `parameter`).",
    "failover_dependency": "Route `target`'s calls for a dependency to its configured fallback.",
}


# ---------------------------------------------------------------- state items


class Alert(TypedDict):
    service: str
    signal: str
    description: str
    threshold: float
    value: float
    severity: str


class EnvironmentRef(TypedDict):
    kind: Literal["simulated"]
    scenario_id: str


class Hypothesis(TypedDict):
    id: str
    round: int
    title: str
    category: str
    service: str
    rationale: str
    status: str
    confidence: float
    findings: str


class Evidence(TypedDict):
    id: str
    hypothesis_id: str
    tool: str
    service: str
    query: str
    reasoning: str
    output: str


class RemediationAction(TypedDict):
    id: str
    attempt: int
    type: str
    target: str
    parameter: str
    reason: str
    status: Literal["applied", "failed", "reverted"]
    result: str


class TimelineEvent(TypedDict):
    node: str
    message: str


# ----------------------------------------------------- LLM structured outputs


class TriageOutput(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    affected_services: list[str] = Field(description="Services plausibly involved, upstream and downstream of the alert.")
    summary: str = Field(description="Two-sentence summary of what is going wrong and the user impact.")
    matched_incident_id: str | None = Field(
        default=None,
        description="id of a past incident from memory whose signature clearly matches this alert, else null.",
    )
    match_confidence: float = Field(default=0.0, description="0-1 confidence in the matched incident.")


class HypothesisDraft(BaseModel):
    title: str = Field(description="One-line statement of the suspected root cause.")
    category: RootCauseCategory
    service: str = Field(description="The service where the root cause lives (must be a known service).")
    rationale: str = Field(description="Why this is plausible given what is known so far.")


class HypothesesOutput(BaseModel):
    hypotheses: list[HypothesisDraft] = Field(description="2-4 distinct, testable root-cause hypotheses.")


class InvestigatorStep(BaseModel):
    reasoning: str = Field(description="What you expect to learn and how it would confirm or refute the hypothesis.")
    tool: ToolName
    service: str = Field(default="", description="Service to query ('all' allowed for get_recent_deploys).")
    query: str = Field(default="", description="Optional keyword filter for get_logs.")


class HypothesisAssessment(BaseModel):
    verdict: Literal["supported", "refuted", "inconclusive"]
    confidence: float = Field(description="0-1 confidence that THIS hypothesis is the root cause.")
    findings: str = Field(description="The decisive facts from the evidence, with concrete numbers.")


class ScoringOutput(BaseModel):
    leading_hypothesis_id: str | None = Field(description="id of the most likely root cause, or null if none is credible.")
    confidence: float = Field(description="0-1 overall confidence that the leading hypothesis is the true root cause.")
    reasoning: str = Field(description="How the evidence across hypotheses fits together, including what rules others out.")
    missing_evidence: str = Field(default="", description="What would still need checking to be sure.")


class RunbookCheck(BaseModel):
    matches: bool = Field(description="True only if the current evidence shows the same failure signature.")
    confidence: float
    reasoning: str


class ActionDraft(BaseModel):
    type: ActionType
    target: str = Field(description="Service to act on (must be a known service).")
    parameter: str = Field(default="", description="Action parameter, e.g. the feature flag name.")
    reason: str


class RemediationPlanOutput(BaseModel):
    actions: list[ActionDraft] = Field(description="1-3 actions, smallest blast radius first.")
    risk: Literal["low", "medium", "high"]
    expected_outcome: str
    rollback_strategy: str
