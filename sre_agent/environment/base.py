"""The interface the agent uses to observe and act on a production system.

`SimulatedEnvironment` implements it from scenario files. A live
implementation would back the same methods with kubectl, Prometheus, Loki,
Tempo/Jaeger and the Kafka admin API.
"""

from typing import Protocol, TypedDict


class HealthCheck(TypedDict):
    service: str
    signal: str
    value: float
    threshold: float
    healthy: bool


class Environment(Protocol):
    def services(self) -> list[str]: ...

    def topology_summary(self) -> str: ...

    def run_tool(self, tool: str, service: str, query: str = "") -> str: ...

    def check_health(self, service: str, signal: str) -> HealthCheck: ...

    def apply_action(self, action_type: str, target: str, parameter: str = "") -> tuple[bool, str]: ...
