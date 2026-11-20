"""O esforço `none` precisa chegar ao corpo da requisição em endpoints OpenAI-compatíveis.

Um Ollama servindo Qwen3.5 é endpoint OpenAI-compatível: o cliente OpenAI só encaminha
`reasoning_effort` para os modelos de reasoning dele, então o switch vai por `extra_body`.
Sem ele o agente devolve resposta vazia (o orçamento de tokens inteiro vai para o campo
`reasoning`).
"""

from __future__ import annotations

import pytest

from paid_media_agent.assembly import resolve_model
from paid_media_agent.config import ModelConfig


def test_effort_none_is_sent_in_the_body_for_openai_compatible_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")

    llm = resolve_model(
        ModelConfig.parse("openai:qwen3.5:9b", base_url="http://127.0.0.1:11434/v1"),
        effort="none",
    )

    assert getattr(llm, "reasoning_effort", None) is None
    assert (getattr(llm, "extra_body", None) or {}) == {"reasoning_effort": "none"}


def test_effort_none_is_accepted_by_the_settings_schema() -> None:
    from paid_media_agent.config import Settings

    settings = Settings(paid_media_model_effort="none")

    assert settings.paid_media_model_effort == "none"
