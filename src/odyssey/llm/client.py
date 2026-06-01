from typing import Any
import json

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


async def generate(
    system: str,
    prompt: str,
    model: str | None = None,
    temperature: float = 0.7,
    format: str | None = None,
) -> str:
    client = get_client()
    model = model or get_fast_model()
    response = await client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature},
        format=format,
    )
    return response["message"]["content"]


async def generate_structured(
    system: str,
    prompt: str,
    output_model: type[Any],
    model: str | None = None,
    temperature: float = 0.2,
) -> Any:
    client = get_client()
    model = model or get_fast_model()
    schema = _pydantic_to_json_schema(output_model)
    response = await client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature},
        format=schema,
    )
    raw = response["message"]["content"]
    return output_model.model_validate_json(raw)


async def embed(text: str) -> list[float]:
    client = get_client()
    model = get_embedding_model()
    response = await client.embed(model=model, input=text)
    return response["embeddings"][0]


def _pydantic_to_json_schema(model: type[Any]) -> dict[str, Any]:
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    schema.pop("title", None)
    if definitions:
        _resolve_refs(schema, definitions)
    return {"type": "object", "properties": schema.get("properties", {})}


def _resolve_refs(schema: dict, definitions: dict) -> None:
    if not isinstance(schema, dict):
        return
    ref = schema.pop("$ref", None)
    if ref:
        key = ref.split("/")[-1]
        resolved = definitions.get(key, {})
        schema.update(resolved)
    for v in schema.values():
        if isinstance(v, dict):
            _resolve_refs(v, definitions)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _resolve_refs(item, definitions)


async def check_connection() -> bool:
    try:
        client = get_client()
        await client.list()
        return True
    except Exception:
        return False
