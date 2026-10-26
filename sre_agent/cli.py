"""Command-line interface: run incidents, answer interrupts, inspect and fork threads, evaluate."""

import time
import uuid
from contextlib import contextmanager
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from sre_agent import memory
from sre_agent.config import get_settings
from sre_agent.environment import list_scenarios, load_scenario
from sre_agent.graph import AgentContext, build_graph
from sre_agent.graph.context import ApprovalMode
from sre_agent.graph.timetravel import find_checkpoint, fork, short_id
from sre_agent.llm import build_reasoner
from sre_agent.persistence import open_persistence

load_dotenv()
app = typer.Typer(help="Autonomous SRE incident investigator (LangGraph).", no_args_is_help=True)
console = Console()

ApprovalOpt = Annotated[str, typer.Option("--approval", help="ask | auto_low_risk | auto")]


def _thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _incident_input(scenario_id: str) -> dict:
    scenario = load_scenario(scenario_id)
    alert = {k: scenario.alert[k] for k in ("service", "signal", "description", "threshold", "value", "severity")}
    return {"alert": alert, "environment": {"kind": "simulated", "scenario_id": scenario_id}}


@contextmanager
def _session(approval: str = "ask", *, with_llm: bool = True):
    settings = get_settings()
    with open_persistence(settings) as (checkpointer, store):
        graph = build_graph(checkpointer=checkpointer, store=store)
        context = (
            AgentContext.from_settings(settings, build_reasoner(settings), approval_mode=_approval_mode(approval))
            if with_llm
            else None
        )
        yield graph, context, store


def _approval_mode(value: str) -> ApprovalMode:
    if value not in ("ask", "auto_low_risk", "auto"):
        raise typer.BadParameter("approval must be one of: ask, auto_low_risk, auto")
    return value  # type: ignore[return-value]


# ------------------------------------------------------------ rendering


def _render_update(namespace: tuple[str, ...], node: str, update: dict[str, Any] | None) -> None:
    if not update:
        return
    if namespace:  # inside the investigator / runbook subgraph
        branch = namespace[0].split(":")[0]
        for e in update.get("evidence", []):
            first = e["output"].strip().splitlines()[0] if e["output"].strip() else ""
            console.print(f"    [dim]{branch}[/] [cyan]{e['hypothesis_id']}[/] {e['tool']}({e['service']}) → [dim]{first[:110]}[/]")
        for h in update.get("hypotheses", []):
            color = {"supported": "green", "refuted": "red"}.get(h["status"], "yellow")
            console.print(f"    [dim]{branch}[/] [cyan]{h['id']}[/] [{color}]{h['status']}[/] ({h['confidence']:.2f}) {h.get('findings', '')[:120]}")
        return
    for event in update.get("timeline", []):
        console.print(f"[bold magenta]{event['node']:>20}[/]  {event['message']}")


def _stream(graph, payload, config, context) -> None:
    for namespace, chunk in graph.stream(payload, config, context=context, stream_mode="updates", subgraphs=True):
        for node, update in chunk.items():
            if node != "__interrupt__":
                _render_update(namespace, node, update)


def _ask_approval(value: dict) -> dict:
    plan, hyp = value["plan"], value["hypothesis"]
    table = Table(show_header=True, header_style="bold")
    for col in ("action", "target", "parameter", "reason"):
        table.add_column(col)
    for a in plan["actions"]:
        table.add_row(a["type"], a["target"], a.get("parameter", ""), a["reason"])
    console.print(
        Panel.fit(
            f"[bold]Diagnosis[/] {hyp['category']} @ {hyp['service']}: {hyp['title']}\n"
            f"[bold]Confidence[/] {value['confidence']:.2f}   [bold]Risk[/] {plan['risk']}   [bold]Attempt[/] {plan['attempt']}\n"
            f"[bold]Expected[/] {plan['expected_outcome']}\n[bold]Rollback[/] {plan['rollback_strategy']}",
            title=f"Approval needed — {value['incident_id']}",
            border_style="yellow",
        )
    )
    console.print(table)
    choice = Prompt.ask("Decision", choices=["approve", "reject", "reinvestigate", "abort"], default="approve")
    feedback = Prompt.ask("Feedback for the agent", default="") if choice in ("reject", "reinvestigate") else ""
    return {"decision": choice, "feedback": feedback}


def _ask_escalation(value: dict) -> dict:
    lines = "\n".join(
        f"  {h['id']} [{h['status']}] {h['category']} @ {h['service']}: {h['title']}" for h in value["hypotheses"]
    )
    console.print(Panel.fit(f"[bold]{escape(value['reason'])}[/]\n\n{escape(lines)}", title=f"Escalation — {value['incident_id']}", border_style="red"))
    hint = Prompt.ask("Give the agent a hint to keep investigating (empty to stop)", default="")
    return {"action": "hint", "hint": hint} if hint else {"action": "abort"}


def _drive(graph, payload, config, context, interactive: bool) -> None:
    """Run until the thread finishes, answering interrupts at the terminal."""
    while True:
        _stream(graph, payload, config, context)
        state = graph.get_state(config)
        if not state.interrupts:
            _summarize(state.values)
            return
        value = state.interrupts[0].value
        if not interactive:
            thread_id = config["configurable"]["thread_id"]
            console.print(f"\n[yellow]Paused for {value['type']}.[/] Resume with: [bold]sre-agent resume {thread_id}[/]")
            return
        answer = _ask_approval(value) if value["type"] == "approval" else _ask_escalation(value)
        payload = Command(resume=answer)


def _summarize(values: dict) -> None:
    status = values.get("status")
    color = {"resolved": "green", "escalated": "red", "aborted": "red"}.get(status, "yellow")
    leading = next((h for h in values.get("hypotheses", []) if h["id"] == values.get("leading_hypothesis_id")), None)
    console.print(
        Panel.fit(
            f"[bold]Status[/] [{color}]{status}[/]\n"
            f"[bold]Root cause[/] {leading['category'] + ' @ ' + leading['service'] + ': ' + leading['title'] if leading else '—'}\n"
            f"[bold]Rounds[/] {values.get('investigation_round', 0)}   [bold]Tool calls[/] {values.get('tool_calls_used', 0)}   "
            f"[bold]Remediation attempts[/] {values.get('remediation_attempts', 0)}",
            title=values.get("incident_id", "incident"),
        )
    )
    if values.get("postmortem"):
        console.print(Markdown(values["postmortem"]))


# ------------------------------------------------------------- commands


@app.command()
def scenarios() -> None:
    """List the simulated incident scenarios."""
    table = Table("id", "difficulty", "alert", "title")
    for s in list_scenarios():
        table.add_row(s.id, s.difficulty, f"{s.alert['service']} / {s.alert['signal']}", s.title)
    console.print(table)


@app.command()
def run(
    scenario: str,
    thread_id: Annotated[str | None, typer.Option(help="Defaults to <scenario>-<random>")] = None,
    approval: ApprovalOpt = "ask",
    interactive: Annotated[bool, typer.Option(help="Answer interrupts here; --no-interactive pauses and exits.")] = True,
) -> None:
    """Fire a scenario's alert and investigate it."""
    thread_id = thread_id or f"{scenario}-{uuid.uuid4().hex[:6]}"
    console.print(f"[bold]Thread[/] {thread_id}")
    with _session(approval) as (graph, context, _):
        _drive(graph, _incident_input(scenario), _thread(thread_id), context, interactive)


@app.command()
def resume(thread_id: str, approval: ApprovalOpt = "ask") -> None:
    """Continue a paused thread (works across processes thanks to the checkpointer)."""
    with _session(approval) as (graph, context, _):
        config = _thread(thread_id)
        state = graph.get_state(config)
        if not state.interrupts:
            console.print("Thread is not waiting on a human." if not state.next else f"Continuing at {state.next}.")
            if state.next:
                _drive(graph, None, config, context, True)
            return
        value = state.interrupts[0].value
        answer = _ask_approval(value) if value["type"] == "approval" else _ask_escalation(value)
        _drive(graph, Command(resume=answer), config, context, True)


@app.command()
def status(thread_id: str) -> None:
    """Show where a thread is and what it believes."""
    with _session(with_llm=False) as (graph, _, _):
        state = graph.get_state(_thread(thread_id))
        if not state.values:
            raise typer.BadParameter(f"no thread {thread_id}")
        console.print(f"[bold]Next[/] {state.next or 'done'}")
        for h in state.values.get("hypotheses", []):
            console.print(escape(f"  {h['id']} [{h['status']}] ({h['confidence']:.2f}) {h['category']} @ {h['service']}: {h['title']}"))
        _summarize(state.values)


@app.command()
def history(thread_id: str) -> None:
    """List a thread's checkpoints (newest first) for replay / forking."""
    with _session(with_llm=False) as (graph, _, _):
        table = Table("checkpoint", "step", "next", "status", "round", "leading")
        for snap in graph.get_state_history(_thread(thread_id)):
            v = snap.values
            table.add_row(
                short_id(snap.config),
                str(snap.metadata.get("step")),
                ", ".join(snap.next) or "—",
                v.get("status", ""),
                str(v.get("investigation_round", "")),
                f"{v.get('leading_hypothesis_id') or ''} {v.get('overall_confidence', 0):.2f}" if v.get("leading_hypothesis_id") else "",
            )
        console.print(table)


@app.command("fork")
def fork_cmd(
    thread_id: str,
    checkpoint: Annotated[str, typer.Argument(help="Checkpoint id from `history`")],
    hint: Annotated[str | None, typer.Option(help="Inject a human hint into the forked state")] = None,
    feedback: Annotated[str | None, typer.Option(help="Inject approver feedback into the forked state")] = None,
    approval: ApprovalOpt = "ask",
) -> None:
    """Time-travel: resume a thread from a past checkpoint, optionally editing state first."""
    values = {k: [v] for k, v in (("human_hints", hint), ("plan_feedback", feedback)) if v}
    with _session(approval) as (graph, context, _):
        config = _thread(thread_id)
        forked = fork(graph, find_checkpoint(graph, config, checkpoint), values or None)
        console.print(f"Forking {thread_id} at {checkpoint}" + (f" with {values}" if values else ""))
        _drive(graph, None, forked, context, True)


@app.command("memory")
def memory_cmd(clear: Annotated[bool, typer.Option(help="Delete all remembered incidents")] = False) -> None:
    """Show (or clear) long-term incident memory."""
    with _session(with_llm=False) as (_, _, store):
        if clear:
            console.print(f"Deleted {memory.forget_all(store)} incident(s).")
            return
        table = Table("incident", "alert", "root cause", "fix", "probes")
        for m in memory.list_all(store):
            rc = m["root_cause"]
            table.add_row(
                m["incident_id"],
                f"{m['alert_service']} / {m['alert_signal']}",
                f"{rc['category']} @ {rc['service']}",
                ", ".join(f"{a['type']} {a['target']}" for a in m["fix"]),
                str(len(m["signature_probes"])),
            )
        console.print(table)


@app.command()
def graph(xray: Annotated[bool, typer.Option(help="Expand subgraphs")] = True) -> None:
    """Print the graph as Mermaid."""
    print(build_graph().get_graph(xray=xray).draw_mermaid())


@app.command("eval")
def evaluate(
    only: Annotated[list[str] | None, typer.Option("--scenario", help="Limit to these scenario ids")] = None,
    warm: Annotated[bool, typer.Option(help="Run each scenario twice with shared memory to measure the runbook fast path")] = False,
) -> None:
    """Run scenarios unattended and score root-cause accuracy and resolution."""
    settings = get_settings()
    reasoner = build_reasoner(settings)
    context = AgentContext.from_settings(settings, reasoner, approval_mode="auto", unattended=True)
    table = Table("scenario", "pass", "status", "diagnosis", "correct", "rounds", "tools", "fixes", "secs")
    results = []
    for scenario in list_scenarios():
        if only and scenario.id not in only:
            continue
        store = InMemoryStore()
        for run_no in (1, 2) if warm else (1,):
            graph = build_graph(checkpointer=InMemorySaver(), store=store)
            started = time.monotonic()
            try:
                values = graph.invoke(_incident_input(scenario.id), _thread(str(uuid.uuid4())), context=context)
                error = None
            except Exception as exc:  # keep evaluating other scenarios
                values, error = {}, f"{type(exc).__name__}: {exc}"
            leading = next((h for h in values.get("hypotheses", []) if h["id"] == values.get("leading_hypothesis_id")), None)
            correct = bool(
                leading
                and leading["service"] in scenario.accepted_root_cause_services
                and leading["category"] in scenario.accepted_root_cause_categories
            )
            resolved = values.get("status") == "resolved"
            results.append((correct, resolved))
            table.add_row(
                scenario.id,
                str(run_no),
                values.get("status", f"[red]{error}[/]"),
                f"{leading['category']} @ {leading['service']}" if leading else "—",
                "[green]yes[/]" if correct else "[red]no[/]",
                str(values.get("investigation_round", 0)),
                str(values.get("tool_calls_used", 0)),
                str(values.get("remediation_attempts", 0)),
                f"{time.monotonic() - started:.0f}",
            )
            console.print(f"done: {scenario.id} (pass {run_no})")
    console.print(table)
    if results:
        n = len(results)
        console.print(
            f"root cause accuracy {sum(c for c, _ in results)}/{n}   resolved {sum(r for _, r in results)}/{n}   model {settings.llm_model}"
        )


if __name__ == "__main__":
    app()
