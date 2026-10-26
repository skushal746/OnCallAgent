from sre_agent.environment import SimulatedEnvironment, list_scenarios, load_scenario


def _action(type_: str, target: str, status: str = "applied") -> dict:
    return {"type": type_, "target": target, "status": status}


def test_world_status_is_derived_from_actions():
    scenario = load_scenario("db_connection_leak")
    assert SimulatedEnvironment(scenario, []).status == "broken"
    assert SimulatedEnvironment(scenario, [_action("rollback_deploy", "inventory-service")]).status == "healed"
    assert SimulatedEnvironment(scenario, [_action("increase_connection_pool", "inventory-service")]).status == "worse"
    reverted = [_action("increase_connection_pool", "inventory-service", status="reverted")]
    assert SimulatedEnvironment(scenario, reverted).status == "broken"


def test_tool_outputs_follow_world_status():
    scenario = load_scenario("db_connection_leak")
    broken = SimulatedEnvironment(scenario, []).run_tool("get_metrics", "inventory-service")
    healed = SimulatedEnvironment(scenario, [_action("rollback_deploy", "inventory-service")]).run_tool(
        "get_metrics", "inventory-service"
    )
    assert "50/50" in broken and "6/50" in healed


def test_health_check_and_unknowns():
    env = SimulatedEnvironment(load_scenario("traffic_spike"), [])
    assert not env.check_health("search-service", "http_p99_latency_ms")["healthy"]
    assert "unknown service" in env.run_tool("get_logs", "nope")
    assert env.apply_action("restart_service", "nope")[0] is False


def test_log_query_filters_lines():
    env = SimulatedEnvironment(load_scenario("bad_deploy_oom"), [])
    out = env.run_tool("get_logs", "user-service", "OutOfMemoryError")
    assert out and all("OutOfMemoryError" in line for line in out.splitlines())


def test_all_scenarios_load_and_fixes_heal():
    for scenario in list_scenarios():
        fix = scenario.fixes[0]
        env = SimulatedEnvironment(scenario, [_action(fix["type"], fix["target"])])
        assert env.check_health(scenario.alert["service"], scenario.alert["signal"])["healthy"], scenario.id
        assert scenario.root_cause["service"] in scenario.topology, scenario.id
