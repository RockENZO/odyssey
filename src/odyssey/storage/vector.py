from typing import Any
import uuid

import chromadb
from chromadb.errors import NotFoundError

from odyssey.config import get_data_dir


_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        persist_dir = str(get_data_dir() / "chroma")
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def get_or_create_collection(name: str):
    client = _get_client()
    try:
        return client.get_collection(name)
    except NotFoundError:
        return client.create_collection(name)


def add_to_collection(
    collection_name: str,
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]] | None = None,
    ids: list[str] | None = None,
) -> list[str]:
    collection = get_or_create_collection(collection_name)
    doc_ids = ids or [str(uuid.uuid4()) for _ in texts]
    collection.add(
        ids=doc_ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return doc_ids


def query_collection(
    collection_name: str,
    query_embedding: list[float],
    n_results: int = 5,
) -> list[dict[str, Any]]:
    collection = get_or_create_collection(collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    output = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "document": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
            })
    return output


def list_collections() -> list[str]:
    client = _get_client()
    collections = client.list_collections()
    return [c.name for c in collections]
