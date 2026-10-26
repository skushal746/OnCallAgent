from dataclasses import dataclass
from typing import Literal

from sre_agent.config import Settings
from sre_agent.llm import Reasoner

ApprovalMode = Literal["ask", "auto_low_risk", "auto"]


@dataclass
class AgentContext:
    """Run-scoped dependencies and limits (LangGraph runtime context, never checkpointed)."""

    reasoner: Reasoner
    confidence_threshold: float = 0.75
    max_investigation_rounds: int = 3
    max_tool_steps: int = 4
    max_remediation_attempts: int = 2
    tool_call_budget: int = 40
    approval_mode: ApprovalMode = "ask"
    # When True, escalations end the run instead of pausing for a human (used by eval).
    unattended: bool = False

    @classmethod
    def from_settings(cls, settings: Settings, reasoner: Reasoner, **overrides) -> "AgentContext":
        return cls(
            reasoner=reasoner,
            confidence_threshold=settings.confidence_threshold,
            max_investigation_rounds=settings.max_investigation_rounds,
            max_tool_steps=settings.max_tool_steps,
            max_remediation_attempts=settings.max_remediation_attempts,
            tool_call_budget=settings.tool_call_budget,
            **overrides,
        )
