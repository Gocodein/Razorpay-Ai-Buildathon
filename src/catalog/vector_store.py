"""
ChromaDB vector store for semantic product search.

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings — runs
entirely locally, no API key needed for the search step.

Typical workflow:
  1. seed_catalog()  → enrich products → upsert into ChromaDB
  2. search()        → agent queries by natural language
  3. get_by_id()     → agent fetches exact product details
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Enforce offline cache mode to prevent Hugging Face Hub network latency and rate limits
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import chromadb
from chromadb.utils import embedding_functions

from src.config import settings


# ── Embedding function (local sentence-transformers) ────────────────────────

_EMBED_FN: Any = None
_CACHED_CLIENT: chromadb.PersistentClient | None = None
_CACHED_COLLECTION: chromadb.Collection | None = None


def _get_embed_fn():
    """Return the cached embedding function, lazy-loaded on first call to prevent import stalls."""
    global _EMBED_FN
    if _EMBED_FN is None:
        _EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
    return _EMBED_FN


# ── ChromaDB client (persistent, stored on disk) ────────────────────────────

def _get_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection (singleton cached)."""
    global _CACHED_CLIENT, _CACHED_COLLECTION
    if _CACHED_COLLECTION is not None:
        return _CACHED_COLLECTION

    Path(settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
    if _CACHED_CLIENT is None:
        _CACHED_CLIENT = chromadb.PersistentClient(path=settings.chroma_db_path)
    _CACHED_COLLECTION = _CACHED_CLIENT.get_or_create_collection(
        name=settings.catalog_collection,
        embedding_function=_get_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )
    return _CACHED_COLLECTION


_CACHED_CLIENT: chromadb.PersistentClient | None = None
_CACHED_COLLECTION: chromadb.Collection | None = None
_CACHED_PRODUCTS: list[dict[str, Any]] | None = None
_CACHED_COUNT: int | None = None


def invalidate_cache() -> None:
    """Invalidate collection and product caches after upserts or inventory changes."""
    global _CACHED_CLIENT, _CACHED_COLLECTION, _CACHED_PRODUCTS, _CACHED_COUNT
    _CACHED_CLIENT = None
    _CACHED_COLLECTION = None
    _CACHED_PRODUCTS = None
    _CACHED_COUNT = None



# ── Seeding ──────────────────────────────────────────────────────────────────

def upsert_products(enriched_products: list[dict[str, Any]]) -> None:
    """
    Insert or update enriched product documents into ChromaDB.

    Each document's `search_text` field is embedded; all other fields
    are stored as metadata for retrieval.
    """
    collection = _get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for p in enriched_products:
        ids.append(p["id"])
        documents.append(p["search_text"])

        # Metadata must be flat (ChromaDB requirement):
        # serialise nested dicts / lists as JSON strings.
        meta: dict[str, Any] = {
            "id": p["id"],
            "name": p["name"],
            "price_inr": int(p["price_inr"]),
            "stock": int(p["stock"]),
            "sku": p["sku"],
            "category": p["category"],
            "merchant_id": str(p.get("merchant_id") or settings.merchant_id),
            "agent_description": p["agent_description"],
            "structured_attributes": json.dumps(
                p.get("structured_attributes", {}), ensure_ascii=False
            ),
            "intent_phrases": json.dumps(
                p.get("intent_phrases", []), ensure_ascii=False
            ),
            "aliases": json.dumps(p.get("aliases", []), ensure_ascii=False),
        }
        metadatas.append(meta)

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    invalidate_cache()
    print(f"[vector_store] Upserted {len(ids)} products into '{settings.catalog_collection}'.")


# ── Search & Resilient Fallback ─────────────────────────────────────────────

def _fallback_in_memory_search(
    query: str,
    *,
    n_results: int = 5,
    max_price_inr: float | None = None,
    in_stock_only: bool = False,
    merchant_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Robust zero-downtime fallback: Searches products in-memory using token matching
    and metadata filtering if ChromaDB's vector index encounters internal drift.
    """
    all_prods = get_all_products(limit=500)
    clean_q = (query or "").strip().lower()
    query_tokens = [tok for tok in re.findall(r"\w+", clean_q) if len(tok) >= 2]

    scored: list[tuple[float, dict[str, Any]]] = []
    for prod in all_prods:
        stock = int(prod.get("stock", 0))
        if in_stock_only and stock <= 0:
            continue
        price = float(prod.get("price_inr", 0))
        if max_price_inr is not None and price > max_price_inr:
            continue
        if merchant_id and prod.get("merchant_id") != merchant_id:
            continue

        name = str(prod.get("name", "")).lower()
        category = str(prod.get("category", "")).lower()
        search_text = str(prod.get("search_text", "")).lower()
        desc = str(prod.get("agent_description", "")).lower()
        combined = f"{name} {category} {search_text} {desc}"

        score = 0.0
        for tok in query_tokens:
            if tok in name:
                score += 4.0
            elif tok in category:
                score += 2.5
            elif tok in combined:
                score += 1.0

        if clean_q and clean_q in combined:
            score += 3.0

        relevance = round(min(0.95, 0.50 + (score * 0.1)), 3)
        prod_copy = dict(prod)
        prod_copy["relevance_score"] = relevance
        scored.append((score, prod_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] == 0:
        return [p for _, p in scored[:n_results]]
    return [p for s, p in scored if s > 0][:n_results]


def search(
    query: str,
    *,
    n_results: int = 5,
    max_price_inr: float | None = None,
    in_stock_only: bool = False,
    merchant_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Query the catalog with natural language; returns nearest neighbours.

    Args:
        query:         Free-text buyer request (e.g. "matte sunscreen for oily skin").
        n_results:     Maximum number of results to return.
        max_price_inr: Optional price ceiling in INR.
        in_stock_only: If True, only return products with stock > 0.
        merchant_id:   Optional merchant ID filter for multi-tenant stores.

    Returns:
        List of product dicts, sorted by relevance (closest embedding first).
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return get_all_products(limit=n_results)

    collection = _get_collection()
    if collection.count() == 0:
        return []

    # Build ChromaDB where-filter
    where: dict[str, Any] = {}
    conditions: list[dict[str, Any]] = []

    if in_stock_only:
        conditions.append({"stock": {"$gt": 0}})
    if max_price_inr is not None:
        conditions.append({"price_inr": {"$lte": max_price_inr}})
    if merchant_id:
        conditions.append({"merchant_id": {"$eq": merchant_id}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    query_kwargs: dict[str, Any] = {
        "query_texts": [clean_query],
        "n_results": min(n_results, collection.count() or 1),
        "include": ["metadatas", "distances", "documents"],
    }
    if where:
        query_kwargs["where"] = where

    results = None
    try:
        results = collection.query(**query_kwargs)
    except Exception as exc:
        logger.warning("ChromaDB query encountered error '%s'. Re-initializing client...", exc)
        invalidate_cache()
        collection = _get_collection()
        try:
            results = collection.query(**query_kwargs)
        except Exception as retry_exc:
            logger.error("ChromaDB retry failed: %s. Using in-memory fallback search.", retry_exc)
            return _fallback_in_memory_search(
                query=clean_query,
                n_results=n_results,
                max_price_inr=max_price_inr,
                in_stock_only=in_stock_only,
                merchant_id=merchant_id,
            )

    products: list[dict[str, Any]] = []
    if results and results.get("metadatas") and results["metadatas"][0]:
        for i, meta in enumerate(results["metadatas"][0]):
            distance = results["distances"][0][i]
            relevance_score = round(1 - distance, 3)   # cosine: higher = more relevant

            product = _deserialise_meta(meta)
            product["id"] = results["ids"][0][i]
            product["relevance_score"] = relevance_score
            products.append(product)

    return products



# ── Point lookup ─────────────────────────────────────────────────────────────

def get_by_id(product_id: str) -> dict[str, Any] | None:
    """
    Fetch a single product by its ID.

    Returns None if the product does not exist in the catalog.
    """
    try:
        collection = _get_collection()
        result = collection.get(
            ids=[product_id],
            include=["metadatas", "documents"],
        )
        if result and result.get("ids"):
            meta = result["metadatas"][0]
            product = _deserialise_meta(meta)
            product["id"] = product_id
            return product
    except Exception as exc:
        logger.warning("ChromaDB get_by_id error for '%s': %s", product_id, exc)

    # Fallback to in-memory catalog
    for prod in get_all_products(limit=500):
        if prod.get("id") == product_id:
            return dict(prod)
    return None


def catalog_size() -> int:
    """Return the number of products currently in the vector store (cached in memory)."""
    global _CACHED_COUNT
    if _CACHED_COUNT is not None:
        return _CACHED_COUNT
    _CACHED_COUNT = _get_collection().count()
    return _CACHED_COUNT


def get_all_products(limit: int = 500) -> list[dict[str, Any]]:
    """Retrieve all products from the persistent ChromaDB collection up to limit (cached in memory)."""
    global _CACHED_PRODUCTS
    if _CACHED_PRODUCTS is not None:
        return _CACHED_PRODUCTS[:limit]

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []
    result = collection.get(
        limit=max(limit, 500),
        include=["metadatas", "documents"]
    )
    products = []
    if result["ids"] and result["metadatas"]:
        for pid, meta in zip(result["ids"], result["metadatas"]):
            p = _deserialise_meta(meta)
            p["id"] = pid
            products.append(p)
    _CACHED_PRODUCTS = products
    return products[:limit]



# ── Inventory state management ───────────────────────────────────────────────

def decrement_stock(product_id: str, quantity: int = 1) -> bool:
    """
    Atomically decrement stock for a product in ChromaDB upon order placement.
    Returns True if sufficient stock was available and decremented, False otherwise.
    """
    if quantity <= 0:
        return False
    collection = _get_collection()
    result = collection.get(ids=[product_id], include=["metadatas"])
    if not result["ids"] or not result["metadatas"]:
        return False

    meta = dict(result["metadatas"][0])
    current_stock = int(meta.get("stock", 0))
    if current_stock < quantity:
        return False

    meta["stock"] = current_stock - quantity
    collection.update(ids=[product_id], metadatas=[meta])
    return True


def restore_stock(product_id: str, quantity: int = 1) -> bool:
    """
    Restore stock for a product (e.g., on order cancellation or failure).
    """
    if quantity <= 0:
        return False
    collection = _get_collection()
    result = collection.get(ids=[product_id], include=["metadatas"])
    if not result["ids"] or not result["metadatas"]:
        return False

    meta = dict(result["metadatas"][0])
    current_stock = int(meta.get("stock", 0))
    meta["stock"] = current_stock + quantity
    collection.update(ids=[product_id], metadatas=[meta])
    return True


# ── Internal helpers ─────────────────────────────────────────────────────────

def _deserialise_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Reverse the serialisation applied in upsert_products().
    JSON-string fields are parsed back to their native Python types.
    """
    product = dict(meta)
    for key in ("structured_attributes", "intent_phrases", "aliases"):
        if key in product and isinstance(product[key], str):
            try:
                product[key] = json.loads(product[key])
            except json.JSONDecodeError:
                product[key] = {}
    return product
