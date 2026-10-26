"""Checkpointer + store factory.

The checkpointer makes threads durable (pause at an interrupt, resume from
another process, replay or fork any past checkpoint). The store holds
long-term incident memory shared across threads.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from sre_agent.config import Settings


@contextmanager
def open_persistence(settings: Settings) -> Iterator[tuple[BaseCheckpointSaver, BaseStore]]:
    if settings.persistence == "memory":
        yield InMemorySaver(), InMemoryStore()

    elif settings.persistence == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.store.sqlite import SqliteStore

        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        path = str(settings.sqlite_path)
        with SqliteSaver.from_conn_string(path) as saver, SqliteStore.from_conn_string(path) as store:
            saver.setup()
            store.setup()
            yield saver, store

    elif settings.persistence == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore

        with (
            PostgresSaver.from_conn_string(settings.postgres_url) as saver,
            PostgresStore.from_conn_string(settings.postgres_url) as store,
        ):
            saver.setup()
            store.setup()
            yield saver, store

    else:
        raise ValueError(f"unknown persistence backend: {settings.persistence}")
