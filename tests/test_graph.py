import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from sre_agent import memory
from sre_agent.environment import load_scenario
from sre_agent.graph import AgentContext, build_graph
from sre_agent.graph.timetravel import find_checkpoint, fork, short_id
from tests import fakes as tests_fakes
from tests.fakes import ScriptedReasoner, oom_handlers


def _input(scenario_id: str = "bad_deploy_oom") -> dict:
    scenario = load_scenario(scenario_id)
    alert = {k: scenario.alert[k] for k in ("service", "signal", "description", "threshold", "value", "severity")}
    return {"alert": alert, "environment": {"kind": "simulated", "scenario_id": scenario_id}}


def _run(reasoner, *, store=None, checkpointer=None, **ctx):
    graph = build_graph(checkpointer=checkpointer or InMemorySaver(), store=store or InMemoryStore())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    context = AgentContext(reasoner=reasoner, **{"approval_mode": "auto", **ctx})
    result = graph.invoke(_input(), config, context=context)
    return graph, config, context, result


def test_parallel_investigation_resolves_incident():
    reasoner = ScriptedReasoner(oom_handlers())
    store = InMemoryStore()
    _, _, _, result = _run(reasoner, store=store)

    assert result["status"] == "resolved"
    statuses = {h["id"]: h["status"] for h in result["hypotheses"]}
    # both parallel branches wrote their verdicts without clobbering each other
    assert statuses == {"H1.1": "confirmed", "H1.2": "refuted"}
    assert result["tool_calls_used"] == 3
    assert [(a["type"], a["status"]) for a in result["actions"]] == [("rollback_deploy", "applied")]
    assert result["verification"]["healthy"]
    assert result["postmortem"].startswith("# Postmortem")

    saved = memory.recall(store, result["alert"])
    assert len(saved) == 1
    assert {p["tool"] for p in saved[0]["signature_probes"]} == {"get_recent_deploys", "describe_service"}


def test_failed_fix_is_rolled_back_and_retried():
    plans = [[("restart_service", "user-service")], [("rollback_deploy", "user-service")]]
    reasoner = ScriptedReasoner(oom_handlers(plans=plans))
    _, _, _, result = _run(reasoner)

    assert result["status"] == "resolved"
    assert result["remediation_attempts"] == 2
    assert [(a["type"], a["status"]) for a in result["actions"]] == [
        ("restart_service", "reverted"),
        ("rollback_deploy", "applied"),
    ]
    assert reasoner.calls["score_hypotheses"] == 2  # rollback routes back through scoring


def test_repeated_failures_escalate_when_unattended():
    plans = [[("restart_service", "user-service")], [("scale_service", "user-service")]]
    reasoner = ScriptedReasoner(oom_handlers(plans=plans))
    _, _, _, result = _run(reasoner, unattended=True, max_remediation_attempts=2)

    assert result["status"] == "escalated"
    assert all(a["status"] == "reverted" for a in result["actions"])


def test_low_confidence_loops_then_escalates():
    reasoner = ScriptedReasoner(oom_handlers(score_confidence=0.4))
    _, _, _, result = _run(reasoner, unattended=True, max_investigation_rounds=2)

    assert result["status"] == "escalated"
    assert result["investigation_round"] == 2
    assert reasoner.calls["generate_hypotheses"] == 2


def test_human_approval_interrupt_and_resume():
    reasoner = ScriptedReasoner(oom_handlers(plans=[[("restart_service", "user-service")], [("rollback_deploy", "user-service")]]))
    graph, config, context, result = _run(reasoner, approval_mode="ask")

    assert result["__interrupt__"][0].value["type"] == "approval"
    assert graph.get_state(config).next == ("human_approval",)

    result = graph.invoke(Command(resume={"decision": "reject", "feedback": "restarts won't fix a leak"}), config, context=context)
    plan = result["__interrupt__"][0].value["plan"]
    assert plan["actions"][0]["type"] == "rollback_deploy"

    result = graph.invoke(Command(resume="approve"), config, context=context)
    assert result["status"] == "resolved"
    assert result["plan_feedback"] == ["restarts won't fix a leak"]


def test_escalation_hint_resumes_investigation():
    reasoner = ScriptedReasoner(oom_handlers(score_confidence=0.4))
    graph, config, context, result = _run(reasoner, max_investigation_rounds=1)
    assert result["__interrupt__"][0].value["type"] == "escalation"

    reasoner.handlers["score_hypotheses"] = oom_handlers()["score_hypotheses"]
    result = graph.invoke(Command(resume={"action": "hint", "hint": "look at the 09:58 deploy"}), config, context=context)

    assert result["status"] == "resolved"
    assert result["human_hints"] == ["look at the 09:58 deploy"]
    assert result["investigation_round"] == 2


def test_memory_enables_runbook_fast_path():
    store = InMemoryStore()
    _run(ScriptedReasoner(oom_handlers()), store=store)

    second = ScriptedReasoner(oom_handlers())
    _, _, _, result = _run(second, store=store)

    assert result["status"] == "resolved"
    assert result["runbook_confirmed"]
    assert second.calls["generate_hypotheses"] == 0
    assert result["tool_calls_used"] == 2  # only the saved signature probes
    assert result["leading_hypothesis_id"] == "H0.1"


def test_fork_from_past_checkpoint():
    reasoner = ScriptedReasoner(oom_handlers())
    graph, config, context, result = _run(reasoner)
    assert result["status"] == "resolved"

    before_plan = next(s for s in graph.get_state_history(config) if s.next == ("plan_remediation",))
    checkpoint = find_checkpoint(graph, config, short_id(before_plan.config))
    replay = graph.invoke(None, fork(graph, checkpoint, {"plan_feedback": ["prefer scaling"]}), context=context)

    assert replay["status"] == "resolved"
    assert replay["plan_feedback"] == ["prefer scaling"]
    assert graph.get_state(config).values["plan_feedback"] == ["prefer scaling"]  # fork is now the thread head


def test_repeated_query_ends_investigation():
    handlers = oom_handlers()
    deploys = tests_fakes.InvestigatorStep(reasoning="again", tool="get_recent_deploys", service="")
    handlers["plan_step:H1.1"] = lambda _, __: deploys  # keeps asking for the same thing
    reasoner = ScriptedReasoner(handlers)
    _, _, _, result = _run(reasoner)

    h1_evidence = [e for e in result["evidence"] if e["hypothesis_id"] == "H1.1"]
    assert [(e["tool"], e["service"]) for e in h1_evidence] == [("get_recent_deploys", "all")]
    assert reasoner.calls["plan_step:H1.1"] == 2
