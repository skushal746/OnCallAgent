from sre_agent.environment.base import Environment, HealthCheck
from sre_agent.environment.simulated import SimulatedEnvironment, load_scenario, list_scenarios


def get_environment(ref: dict, actions: list[dict] | None = None) -> Environment:
    """Rebuild the environment for a graph state.

    The world is event-sourced: it is derived from the scenario plus the
    remediation actions recorded in state. Re-running a node after a resume,
    or forking from an old checkpoint, therefore sees exactly the world that
    existed at that point — no hidden mutable state outside the checkpoint.
    """
    if ref["kind"] == "simulated":
        return SimulatedEnvironment(load_scenario(ref["scenario_id"]), actions or [])
    raise ValueError(f"unknown environment kind: {ref['kind']}")


__all__ = ["Environment", "HealthCheck", "SimulatedEnvironment", "get_environment", "list_scenarios", "load_scenario"]
