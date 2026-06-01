from typing import Any

from ollama import AsyncClient

from odyssey.config import load_settings


_settings = None


def get_client() -> AsyncClient:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return AsyncClient(host=_settings.ollama_host)


def get_fast_model() -> str:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings.fast_model


def get_deep_model() -> str:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings.deep_model


def get_embedding_model() -> str:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings.embedding_model


async def chat_completion(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.7,
    include_tools: bool = True,
) -> dict[str, Any]:
    client = get_client()
    model = model or get_fast_model()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
    }
    if include_tools:
        from odyssey.tools.registry import get_tool_schemas
        schemas = get_tool_schemas()
        if schemas:
            kwargs["tools"] = schemas
    return await client.chat(**kwargs)


async def check_connection() -> bool:
    try:
        client = get_client()
        await client.list()
        return True
    except Exception:
        return False


async def embed(text: str) -> list[float]:
    client = get_client()
    model = get_embedding_model()
    response = await client.embed(model=model, input=text)
    return response["embeddings"][0]
