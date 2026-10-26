"""Thin, provider-agnostic wrapper around a LangChain chat model.

Nodes talk to a `Reasoner` instead of a chat model directly. That keeps the
graph independent of provider quirks and lets tests swap in a scripted fake.
"""

from typing import Protocol, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from sre_agent.config import Settings

T = TypeVar("T", bound=BaseModel)


class Reasoner(Protocol):
    def structured(self, schema: type[T], *, system: str, user: str, name: str) -> T: ...

    def text(self, *, system: str, user: str, name: str) -> str: ...


class ChatReasoner:
    def __init__(self, model: BaseChatModel, retries: int = 2):
        self._model = model
        self._retries = retries

    def structured(self, schema: type[T], *, system: str, user: str, name: str) -> T:
        runnable = self._model.with_structured_output(schema)
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
        error: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                result = runnable.invoke(messages, config={"run_name": name})
                if result is None:
                    raise ValueError("model returned no structured output")
                return result if isinstance(result, schema) else schema.model_validate(result)
            except (ValidationError, OutputParserException, ValueError) as exc:
                error = exc
                messages = [
                    *messages,
                    HumanMessage(f"Your previous answer was invalid ({exc}). Answer again, following the schema exactly."),
                ]
        raise RuntimeError(f"{name}: no valid structured output after {self._retries + 1} attempts") from error

    def text(self, *, system: str, user: str, name: str) -> str:
        reply = self._model.invoke([SystemMessage(system), HumanMessage(user)], config={"run_name": name})
        return reply.text


def build_reasoner(settings: Settings) -> ChatReasoner:
    kwargs = {}
    if settings.llm_model.startswith("ollama:"):
        kwargs = {"num_ctx": settings.ollama_num_ctx, "reasoning": settings.ollama_reasoning}
    return ChatReasoner(init_chat_model(settings.llm_model, temperature=settings.llm_temperature, **kwargs))
