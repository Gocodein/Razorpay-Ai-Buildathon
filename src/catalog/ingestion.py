"""
Catalog ingestion pipeline.

Orchestrates the one-time (or periodic) process of:
  1. Loading raw merchant product data
  2. Enriching each product via Claude (structured attributes, intent phrases, etc.)
  3. Upserting enriched documents into ChromaDB for semantic search

Run once before starting the MCP server:
    python -m src.catalog.ingestion

In production this would pull from Razorpay's Items API or the
merchant's own ERP/PIM. For the buildathon demo it uses the
sample_merchant.py fixture.
"""

import asyncio
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.catalog.enrichment import enrich_catalog
from src.catalog.vector_store import upsert_products, catalog_size
from src.demo.sample_merchant import SAMPLE_CATALOG

console = Console(legacy_windows=False)


async def run_ingestion(products=None, verbose: bool = True) -> int:
    """
    Enrich and ingest products into ChromaDB.

    Args:
        products: List of raw product dicts. Defaults to SAMPLE_CATALOG.
        verbose:  Print a summary table when done.

    Returns:
        Number of products ingested.
    """
    raw = products or SAMPLE_CATALOG

    if verbose:
        console.rule("[bold cyan]Merchant AI Readability — Catalog Ingestion")
        console.print(
            f"\n[dim]Found [bold]{len(raw)}[/bold] products to enrich & index.[/dim]\n"
        )

    t0 = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Enriching with Claude…", total=None)
        enriched = await enrich_catalog(raw)
        progress.update(task, description="Upserting into ChromaDB…")
        upsert_products(enriched)

    elapsed = time.perf_counter() - t0

    if verbose:
        _print_summary(enriched, elapsed)

    return len(enriched)


def _print_summary(enriched: list[dict], elapsed: float) -> None:
    table = Table(title="Enriched Catalog", show_lines=True)
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Product", style="bold", width=32)
    table.add_column("Price (₹)", justify="right", width=10)
    table.add_column("Stock", justify="right", width=7)
    table.add_column("Intent phrases", width=40)

    for p in enriched:
        intents = p.get("intent_phrases", [])
        intents_str = " · ".join(intents[:3])
        stock_str = (
            f"[red]{p['stock']}[/red]"
            if p["stock"] == 0
            else f"[green]{p['stock']}[/green]"
        )
        table.add_row(
            p["id"],
            p["name"],
            f"₹{p['price_inr']}",
            stock_str,
            f"[dim]{intents_str}[/dim]",
        )

    console.print(table)
    console.print(
        f"\n[bold green]✓ {len(enriched)} products indexed in ChromaDB "
        f"(total catalog size: {catalog_size()}) "
        f"— {elapsed:.1f}s[/bold green]\n"
    )


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_ingestion())
