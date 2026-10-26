"""Per-hypothesis investigator subgraph.

One instance runs for every open hypothesis, in parallel, via `Send`. Each
instance has private state (its own evidence trail and pending step) and runs
a bounded plan -> act loop before judging its hypothesis:

    plan_step ──(tool)──▶ run_tool ──▶ plan_step ...
        └──(finish | step budget spent)──▶ assess ──▶ END
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from sre_agent import prompts
from sre_agent.environment import get_environment
from sre_agent.graph.context import AgentContext
from sre_agent.models import HypothesisAssessment, InvestigatorStep
from sre_agent.state import InvestigatorInput, InvestigatorOutput, InvestigatorState


def _query_key(step: dict) -> tuple[str, str, str]:
    return step["tool"], step["service"].strip().lower(), step["query"].strip().lower()


def plan_step(state: InvestigatorState, runtime: Runtime[AgentContext]) -> Command[Literal["run_tool", "assess"]]:
    ctx = runtime.context
    evidence = state.get("evidence", [])
    steps_left = ctx.max_tool_steps - len(evidence)
    if steps_left <= 0:
        return Command(goto="assess")

    env = get_environment(state["environment"], state["actions"])
    system, user = prompts.investigator_step(
        state["hypothesis"], state["alert"], env.topology_summary(), state["known_facts"], evidence, steps_left
    )
    step = ctx.reasoner.structured(InvestigatorStep, system=system, user=user, name=f"plan_step:{state['hypothesis']['id']}")
    if step.tool == "get_recent_deploys" and not step.service:
        step.service = "all"
    # Smaller models tend to re-issue a query they already ran; treat that as "nothing new to learn".
    if step.tool == "finish" or _query_key(step.model_dump()) in {_query_key(e) for e in evidence}:
        return Command(goto="assess")
    return Command(goto="run_tool", update={"pending_step": step.model_dump()})


def run_tool(state: InvestigatorState) -> dict:
    step = state["pending_step"]
    env = get_environment(state["environment"], state["actions"])
    output = env.run_tool(step["tool"], step["service"], step["query"])
    hypothesis_id = state["hypothesis"]["id"]
    record = {
        "id": f"E-{hypothesis_id}-{len(state.get('evidence', [])) + 1}",
        "hypothesis_id": hypothesis_id,
        "tool": step["tool"],
        "service": step["service"],
        "query": step["query"],
        "reasoning": step["reasoning"],
        "output": output,
    }
    return {"evidence": [record], "tool_calls_used": 1, "pending_step": None}


def assess(state: InvestigatorState, runtime: Runtime[AgentContext]) -> dict:
    hypothesis = state["hypothesis"]
    system, user = prompts.assess_hypothesis(hypothesis, state["alert"], state.get("evidence", []))
    verdict = runtime.context.reasoner.structured(
        HypothesisAssessment, system=system, user=user, name=f"assess:{hypothesis['id']}"
    )
    confidence = max(0.0, min(1.0, verdict.confidence))
    if verdict.verdict == "refuted" and confidence > 0.5:
        # the model reported confidence in its verdict, not in the hypothesis
        confidence = 1.0 - confidence
    updated = {
        **hypothesis,
        "status": verdict.verdict,
        "confidence": confidence,
        "findings": verdict.findings,
    }
    return {"hypotheses": [updated]}


def build_investigator():
    graph = StateGraph(
        InvestigatorState,
        context_schema=AgentContext,
        input_schema=InvestigatorInput,
        output_schema=InvestigatorOutput,
    )
    graph.add_node("plan_step", plan_step)
    graph.add_node("run_tool", run_tool)
    graph.add_node("assess", assess)
    graph.add_edge(START, "plan_step")
    graph.add_edge("run_tool", "plan_step")
    graph.add_edge("assess", END)
    return graph.compile(name="investigator")
