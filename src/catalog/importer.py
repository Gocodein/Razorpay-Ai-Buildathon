"""
Universal Merchant Catalog Importer — CSV/JSON → ChromaDB in < 5 minutes.

Auto-detects and imports product catalogs from:
  - BigBasket datasets (category, sub_category, brand, sale_price, product, description)
  - Flipkart datasets (product_name, category, actual_price, discount_price, brand)
  - Shopify / WooCommerce exports (Title, Variant Price, Body HTML, Type, Vendor)
  - Generic / Custom CSV & JSON formats

Usage:
    python -m src.catalog.importer --file my_catalog.csv
    python -m src.catalog.importer --file data/bigbasket_sample.csv --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import warnings
from pathlib import Path

# Suppress Hugging Face unauthenticated rate-limit advisory notice
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ── Ensure project root is in sys.path ────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table

from src.catalog.enrichment import enrich_catalog
from src.catalog.vector_store import upsert_products, catalog_size

console = Console(legacy_windows=False)


def _clean_price(val: any) -> int:
    """Parse and clean numeric price from messy string formats (e.g., '₹1,299.00', 'Rs. 499', 'INR 150.50')."""
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return max(0, int(val))
    val_str = str(val).strip()
    if not val_str:
        return 0
    match = re.search(r"(\d+(?:,\d+)*(?:\.\d+)?)", val_str)
    if not match:
        return 0
    num_str = match.group(1).replace(",", "")
    try:
        return max(0, int(float(num_str)))
    except (ValueError, TypeError):
        return 0


def _normalize_key(k: str) -> str:
    """Normalize header key: lowercase, stripped, spaces/hyphens to underscore."""
    return re.sub(r"[^a-z0-9]", "_", k.strip().lower()).strip("_")


def _find_field(row: dict[str, any], candidate_keys: list[str]) -> any:
    """Search for the first matching key in a normalized row dictionary."""
    normalized_row = {_normalize_key(k): v for k, v in row.items() if k is not None}
    for candidate in candidate_keys:
        cand_norm = _normalize_key(candidate)
        if cand_norm in normalized_row and normalized_row[cand_norm] is not None:
            val = str(normalized_row[cand_norm]).strip()
            if val and val.lower() != "nan" and val.lower() != "null":
                return val
    return ""


def _parse_row(row: dict, index: int) -> dict:
    """Intelligently map a raw dictionary row to the standard Agentic Commerce Product schema."""
    # 1. Product Name
    name = _find_field(row, [
        "product", "product_name", "title", "name", "item_name", "item"
    ])
    if not name:
        name = f"Product #{index + 1}"

    # 2. Price (INR)
    price_raw = _find_field(row, [
        "sale_price", "discount_price", "price_inr", "price", "mrp",
        "market_price", "actual_price", "variant_price", "amount"
    ])
    price_inr = _clean_price(price_raw)

    # 3. Category & Sub-category
    cat = _find_field(row, ["category", "type", "product_type", "department"]) or "General"
    sub_cat = _find_field(row, ["sub_category", "sub_type", "subcategory"])
    if sub_cat and sub_cat.lower() not in cat.lower():
        cat = f"{cat} > {sub_cat}"

    # 4. Brand
    brand = _find_field(row, ["brand", "vendor", "brand_name", "manufacturer"])

    # 5. Description
    description = _find_field(row, [
        "description", "product_description", "body_html", "details", "desc", "about"
    ])
    if not description:
        description = f"{name} by {brand}. Category: {cat}." if brand else f"{name}. Category: {cat}."

    # 6. ID & SKU
    raw_id = _find_field(row, ["id", "ean_code", "sku", "product_id", "item_id", "barcode"])
    product_id = raw_id if raw_id else f"SKU_{index + 1:04d}"
    sku = f"SKU-{product_id}"

    # 7. Stock
    stock_raw = _find_field(row, ["stock", "inventory", "quantity", "qty", "stock_qty"])
    stock = int(_clean_price(stock_raw)) if stock_raw else 50  # Default 50 units for testing

    # 8. Tags / Keywords
    raw_tags = _find_field(row, ["tags", "keywords", "aliases", "intent_phrases"])
    tags = []
    if raw_tags:
        tags = [t.strip() for t in re.split(r"[,;|\n]", raw_tags) if t.strip()]

    return {
        "id": product_id,
        "name": name,
        "description": description,
        "price_inr": price_inr,
        "category": cat,
        "brand": brand,
        "stock": stock,
        "sku": sku,
        "tags": tags,
    }


def _load_csv(path: Path, limit: int | None = None) -> list[dict]:
    """Load and normalize products from any CSV file."""
    products = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and len(products) >= limit:
                break
            product = _parse_row(row, i)
            products.append(product)
    return products


def _load_json(path: Path, limit: int | None = None) -> list[dict]:
    """Load and normalize products from a JSON file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    if isinstance(data, dict) and "products" in data:
        data = data["products"]
    elif isinstance(data, dict) and "data" in data:
        data = data["data"]

    products = []
    for i, item in enumerate(data):
        if limit and len(products) >= limit:
            break
        product = _parse_row(item, i)
        products.append(product)
    return products


async def import_catalog(file_path: str, limit: int | None = None, verbose: bool = True) -> int:
    """
    Import, auto-enrich, and index a merchant's product catalog.

    Args:
        file_path: Path to CSV or JSON file.
        limit:     Max items to import (optional).
        verbose:   Print summary table when done.

    Returns:
        Number of products imported.
    """
    path = Path(file_path)

    if not path.exists():
        # Check parent directory, workspace root, or known dataset locations
        alt_paths = [
            Path("..") / file_path,
            ROOT_DIR.parent / file_path,
            ROOT_DIR / file_path,
            Path("..") / "bigbasket_data" / "BigBasket_Products.csv",
            Path("..") / "bigbasket_data" / "BigBasket Products.csv",
            Path("..") / "bigbasket" / "BigBasket_Products.csv",
            Path("..") / "bigbasket" / "BigBasket Products.csv",
            ROOT_DIR.parent / "bigbasket_data" / "BigBasket_Products.csv",
            ROOT_DIR.parent / "bigbasket_data" / "BigBasket Products.csv",
            Path("data") / path.name,
        ]

        found = False
        for alt in alt_paths:
            if alt.exists() and alt.is_file():
                path = alt.resolve()
                found = True
                break
        if not found:
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            return 0

    ext = path.suffix.lower()
    if ext == ".csv":
        raw = _load_csv(path, limit=limit)
    elif ext in (".json", ".jsonl"):
        raw = _load_json(path, limit=limit)
    else:
        console.print(f"[red]Error: Unsupported file type '{ext}'. Use .csv or .json[/red]")
        return 0


    if not raw:
        console.print("[yellow]Warning: No products found in the file.[/yellow]")
        return 0

    if verbose:
        console.rule("[bold cyan]🛒 Merchant AI Readability — Catalog Ingestion")
        console.print(
            f"\n[bold]Source:[/bold] `{path.name}`  |  "
            f"[bold]Products Parsed:[/bold] {len(raw)}  |  "
            f"[bold]Vector DB Target:[/bold] ChromaDB\n"
        )

    console.print("[dim]⚡ Generating semantic intent phrases & Hinglish aliases…[/dim]")
    enriched = await enrich_catalog(raw)

    console.print(f"[dim]📦 Indexing {len(enriched)} embeddings into ChromaDB collection…[/dim]")
    upsert_products(enriched)

    if verbose:
        _print_summary(enriched)

    return len(enriched)


def import_catalog_sync(file_path: str | Path, limit: int | None = None, verbose: bool = False) -> int:
    """
    Synchronously run catalog import without asyncio event loop conflicts.
    Guarantees reliable execution inside Streamlit, notebooks, and background threads.
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, import_catalog(str(file_path), limit=limit, verbose=verbose))
        return future.result()


def _print_summary(enriched: list[dict]) -> None:
    """Print import results as a rich table."""
    table = Table(title="✨ Enriched & Agent-Ready Catalog", show_lines=True, expand=True)
    table.add_column("ID / SKU", style="cyan", width=12)
    table.add_column("Product Name", style="bold", width=30)
    table.add_column("Price (₹)", justify="right", width=10)
    table.add_column("Stock", justify="right", width=8)
    table.add_column("Category", width=18)
    table.add_column("AI Intent Phrases & Aliases", width=34)

    display_sample = enriched[:15]  # Show first 15 for readability
    for p in display_sample:
        intents = p.get("intent_phrases", [])
        aliases = p.get("aliases", [])
        combined = (intents[:2] + aliases[:2]) if (intents or aliases) else ["auto-indexed"]
        intents_str = " · ".join(combined)
        stock = int(p.get("stock", 0))
        stock_str = f"[red]{stock}[/red]" if stock == 0 else f"[green]{stock}[/green]"
        table.add_row(
            p["id"][:12],
            p["name"][:28],
            f"₹{p['price_inr']}",
            stock_str,
            p.get("category", "—")[:16],
            f"[dim]{intents_str[:32]}[/dim]",
        )

    console.print(table)
    if len(enriched) > 15:
        console.print(f"[dim]... and {len(enriched) - 15} more products successfully indexed.[/dim]")

    console.print(
        f"\n[bold green]✓ {len(enriched)} products successfully onboarded and discoverable! "
        f"(Total Vector DB size: {catalog_size()})[/bold green]\n"
    )


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal Merchant Catalog Importer — Auto-onboard CSV/JSON catalogs into ChromaDB."
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to CSV or JSON file (BigBasket, Flipkart, Shopify, or Custom)."
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Optional max limit on items to import (e.g. 50 or 100)."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output."
    )
    args = parser.parse_args()

    count = asyncio.run(import_catalog(args.file, limit=args.limit, verbose=not args.quiet))
    if count > 0:
        console.print(f"[bold cyan]Done! Merchant catalog is now 100% AI-agent-readable. 🚀[/bold cyan]\n")
    else:
        console.print("[red]Import failed. Check file format and path.[/red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
