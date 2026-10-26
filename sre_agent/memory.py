"""Long-term incident memory, kept in the LangGraph store across threads.

Each resolved incident is saved with the probes (tool + service + query)
that confirmed its root cause. When the same alert fires again, the runbook
subgraph re-runs just those probes instead of a full investigation.
"""

from typing import Any

from langgraph.store.base import BaseStore

NAMESPACE = ("incidents",)


def recall(store: BaseStore | None, alert: dict, limit: int = 5) -> list[dict[str, Any]]:
    if store is None:
        return []
    items = store.search(
        NAMESPACE,
        filter={"alert_service": alert["service"], "alert_signal": alert["signal"]},
        limit=limit,
    )
    return [item.value for item in items]


def remember(store: BaseStore | None, record: dict[str, Any]) -> None:
    if store is not None:
        store.put(NAMESPACE, record["incident_id"], record)


def list_all(store: BaseStore, limit: int = 100) -> list[dict[str, Any]]:
    return [item.value for item in store.search(NAMESPACE, limit=limit)]


def forget_all(store: BaseStore) -> int:
    items = store.search(NAMESPACE, limit=1000)
    for item in items:
        store.delete(NAMESPACE, item.key)
    return len(items)
