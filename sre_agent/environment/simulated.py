"""Scenario-driven fake production environment.

A scenario YAML describes a service topology, a hidden root cause, which
remediation actions fix it, which make it worse, and what every observability
tool returns in each world state ("broken", "healed", "worse"). Anything a
scenario does not specify falls back to plausible healthy output, so the agent
has to find the real signal among a lot of normal-looking noise.
"""

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml

from sre_agent.config import get_settings
from sre_agent.environment.base import HealthCheck

WorldStatus = Literal["broken", "healed", "worse"]

MAX_OUTPUT_CHARS = 2500


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    difficulty: str
    alert: dict[str, Any]
    topology: dict[str, dict[str, Any]]
    root_cause: dict[str, Any]
    fixes: list[dict[str, str]]
    harmful: list[dict[str, str]] = field(default_factory=list)
    deploys: list[dict[str, str]] = field(default_factory=list)
    signals: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def accepted_root_cause_services(self) -> set[str]:
        return {self.root_cause["service"], *self.root_cause.get("also_accept_services", [])}

    @property
    def accepted_root_cause_categories(self) -> set[str]:
        return {self.root_cause["category"], *self.root_cause.get("also_accept_categories", [])}


def _scenarios_dir() -> Path:
    return get_settings().scenarios_dir


@cache
def load_scenario(scenario_id: str) -> Scenario:
    path = _scenarios_dir() / f"{scenario_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no scenario '{scenario_id}' in {_scenarios_dir()}")
    return Scenario(**yaml.safe_load(path.read_text()))


def list_scenarios() -> list[Scenario]:
    return [load_scenario(p.stem) for p in sorted(_scenarios_dir().glob("*.yaml"))]


def _matches(action: dict[str, str], rule: dict[str, str]) -> bool:
    return action["type"] == rule["type"] and action["target"] == rule["target"]


class SimulatedEnvironment:
    def __init__(self, scenario: Scenario, actions: list[dict]):
        self.scenario = scenario
        self.applied = [a for a in actions if a.get("status") == "applied"]

    # ------------------------------------------------------------ world model

    @property
    def status(self) -> WorldStatus:
        if any(_matches(a, rule) for a in self.applied for rule in self.scenario.harmful):
            return "worse"
        if any(_matches(a, rule) for a in self.applied for rule in self.scenario.fixes):
            return "healed"
        return "broken"

    def services(self) -> list[str]:
        return list(self.scenario.topology)

    def topology_summary(self) -> str:
        lines = []
        for name, spec in self.scenario.topology.items():
            parts = [f"kind={spec.get('kind', 'service')}"]
            if spec.get("depends_on"):
                parts.append(f"calls={','.join(spec['depends_on'])}")
            if spec.get("consumes"):
                parts.append(f"consumes={','.join(spec['consumes'])}")
            if spec.get("produces"):
                parts.append(f"produces={','.join(spec['produces'])}")
            lines.append(f"- {name}: {' '.join(parts)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ tools

    def run_tool(self, tool: str, service: str, query: str = "") -> str:
        if tool == "get_recent_deploys":
            return self._deploys(service)
        if service not in self.scenario.topology:
            return f"error: unknown service '{service}'. Known services: {', '.join(self.services())}"
        handlers = {
            "get_logs": self._logs,
            "get_metrics": self._metrics,
            "get_traces": self._traces,
            "get_kafka_lag": self._kafka_lag,
            "describe_service": self._describe,
        }
        if tool not in handlers:
            return f"error: unknown tool '{tool}'"
        return handlers[tool](service, query)[:MAX_OUTPUT_CHARS]

    def _signal(self, tool: str, service: str) -> str | None:
        spec = self.scenario.signals.get(tool, {}).get(service)
        if spec is None or isinstance(spec, str):
            return spec
        status = self.status
        if status == "worse":
            return spec.get("worse") or spec.get("broken")
        if status == "healed":
            return spec.get("healed")
        return spec.get("broken")

    def _logs(self, service: str, query: str) -> str:
        text = self._signal("get_logs", service) or (
            f"INFO  {service} request completed status=200 duration_ms=41\n"
            f"INFO  {service} request completed status=200 duration_ms=38\n"
            f"INFO  {service} health check ok\n"
            f"(no WARN/ERROR lines for {service} in the last 30m)"
        )
        if query:
            hits = [line for line in text.splitlines() if query.lower() in line.lower()]
            if hits:
                return "\n".join(hits)
            return f"(no log lines matching '{query}' for {service}; showing latest)\n{text}"
        return text

    def _metrics(self, service: str, _: str) -> str:
        return self._signal("get_metrics", service) or (
            f"{service} (last 30m): p50_latency_ms=22 p99_latency_ms=95 error_rate=0.1% "
            "cpu=31% memory=46% of limit restarts_1h=0"
        )

    def _traces(self, service: str, _: str) -> str:
        return self._signal("get_traces", service) or f"No slow (>1s) or error traces through {service} in the sample."

    def _kafka_lag(self, service: str, _: str) -> str:
        if signal := self._signal("get_kafka_lag", service):
            return signal
        consumes = self.scenario.topology[service].get("consumes")
        if not consumes:
            return f"{service} is not a Kafka consumer."
        return "\n".join(f"group={service} topic={t} total_lag=0 (all partitions caught up)" for t in consumes)

    def _describe(self, service: str, _: str) -> str:
        if signal := self._signal("describe_service", service):
            return signal
        spec = self.scenario.topology[service]
        return (
            f"{service}: version={spec.get('version', 'unknown')} replicas={spec.get('replicas', 2)}/"
            f"{spec.get('replicas', 2)} ready, restarts_1h=0, limits cpu={spec.get('cpu_limit', '1')} "
            f"memory={spec.get('memory_limit', '1Gi')}, no warning events"
        )

    def _deploys(self, service: str) -> str:
        deploys = [d for d in self.scenario.deploys if service in ("all", "", d["service"])]
        if not deploys:
            return f"No deploys in the last 24h for {service or 'any service'}."
        return "\n".join(f"{d['at']}  {d['service']} {d['version']}  — {d['change']}" for d in deploys)

    # ----------------------------------------------------- health and actions

    def check_health(self, service: str, signal: str) -> HealthCheck:
        alert = self.scenario.alert
        value = alert[f"{self.status}_value"] if self.status != "broken" else alert["value"]
        return HealthCheck(
            service=service,
            signal=signal,
            value=value,
            threshold=alert["threshold"],
            healthy=value < alert["threshold"],
        )

    def apply_action(self, action_type: str, target: str, parameter: str = "") -> tuple[bool, str]:
        if target not in self.scenario.topology:
            return False, f"unknown target '{target}'"
        if self.scenario.topology[target].get("kind") == "external":
            return False, f"'{target}' is an external dependency and cannot be changed directly"
        detail = f" ({parameter})" if parameter else ""
        return True, f"{action_type} on {target}{detail} completed"
