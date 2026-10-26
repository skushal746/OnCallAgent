"""Replay and fork incident threads from any past checkpoint."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command


def short_id(checkpoint_config: RunnableConfig) -> str:
    """Checkpoint ids are time-ordered, so prefixes collide; the random tail is unique."""
    return checkpoint_config["configurable"]["checkpoint_id"][-12:]


def find_checkpoint(graph: CompiledStateGraph, config: RunnableConfig, checkpoint_id: str) -> RunnableConfig:
    """Look up a checkpoint by its full id or its `short_id`."""
    for snapshot in graph.get_state_history(config):
        if snapshot.config["configurable"]["checkpoint_id"].endswith(checkpoint_id):
            return snapshot.config
    raise KeyError(f"checkpoint {checkpoint_id} not found in thread {config['configurable']['thread_id']}")


def fork(graph: CompiledStateGraph, checkpoint_config: RunnableConfig, values: dict[str, Any] | None = None) -> RunnableConfig:
    """Return a config that resumes from `checkpoint_config`, optionally with edited state.

    A plain `update_state(values)` would re-derive the next node from static
    edges, which drops routing decided by `Command(goto=...)`. Re-issuing the
    pending node as an explicit goto keeps the fork on the original path.
    """
    if not values:
        return checkpoint_config
    pending = graph.get_state(checkpoint_config).next
    if len(pending) != 1:
        raise ValueError(f"can only edit-and-fork at a checkpoint with one pending node, found {pending or 'none'}")
    node = pending[0]
    return graph.update_state(checkpoint_config, Command(goto=node, update=values), as_node=node)
