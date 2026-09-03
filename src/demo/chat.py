"""
Interactive Chat Mode — Real-time AI Shopping Agent REPL.

Provides a live terminal chat interface where judges and testers can
freely converse with the Google Gemini-powered shopping agent in
natural English or Hinglish.

Run:
    python -m src.demo.chat

Features:
  - Multi-turn conversational context preserved across turns
  - Live tool call display (search, inventory, order, audit)
  - Real-time budget meter after each action
  - Full audit trail printed on session exit (Ctrl+C or 'exit')
  - Automatic model failover cascade (3.5-flash-lite -> 3.1-flash-lite -> 3.5-flash)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

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

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.audit import logger as audit
from src.catalog import vector_store as vs
from src.catalog.ingestion import run_ingestion
from src.config import settings
from src.payment import razorpay_client as rzp

console = Console(legacy_windows=False)

# ── Tool functions (same as run_demo.py) ────────────────────────────────────


def _search_products(query, max_results=5, max_price_inr=None, in_stock_only=True, session_id=""):
    results = vs.search(query=query, n_results=max_results,
                        max_price_inr=max_price_inr, in_stock_only=in_stock_only)
    audit.log_event(session_id=session_id, tool_name="catalog_search_products",
                    inputs={"query": query, "max_price_inr": max_price_inr},
                    outcome="success", details={"count": len(results)})
    return {"results": results, "count": len(results)}


def _check_inventory(product_id, quantity=1, session_id=""):
    product = vs.get_by_id(product_id)
    if not product:
        return {"available": False, "error": "not_found"}
    stock = int(product.get("stock", 0))
    available = stock >= quantity
    response = {"product_id": product_id, "name": product.get("name"),
                "stock": stock, "available": available,
                "price_inr": product.get("price_inr"),
                "budget_remaining_inr": audit.remaining_budget_inr(session_id)}
    if not available:
        alts = vs.search(product.get("name", ""), n_results=3, in_stock_only=True)
        alts = [a for a in alts if a.get("id") != product_id][:2]
        response["alternatives"] = alts
        response["message"] = "Out of stock. See alternatives."
    audit.log_event(session_id=session_id, tool_name="catalog_check_inventory",
                    inputs={"product_id": product_id, "quantity": quantity},
                    outcome="success" if available else "out_of_stock",
                    details=response)
    return response


def _create_order(product_id, quantity=1, buyer_id="buyer", session_id="", confirmed=True):
    product = vs.get_by_id(product_id)
    if not product or not confirmed:
        return {"status": "not_confirmed"}
    return rzp.create_order(
        product_id=product_id,
        product_name=product.get("name", product_id),
        quantity=quantity,
        unit_price_inr=int(product["price_inr"]),
        buyer_id=buyer_id,
        session_id=session_id,
    )


def _cancel_order(order_id="", reason="Buyer requested cancellation", session_id=""):
    return rzp.cancel_order(
        order_id=order_id,
        session_id=session_id,
        reason=reason,
    )


def _get_audit_trail(session_id):
    return {
        "events": audit.get_session_events(session_id),
        "total_spent_inr": audit.session_spent_inr(session_id),
        "remaining_budget_inr": audit.remaining_budget_inr(session_id),
    }


TOOL_REGISTRY = {
    "catalog_search_products": _search_products,
    "catalog_check_inventory": _check_inventory,
    "catalog_create_order": _create_order,
    "catalog_cancel_order": _cancel_order,
    "catalog_get_audit_trail": _get_audit_trail,
}

# ── Gemini tool declarations ────────────────────────────────────────────────

GEMINI_TOOLS = [{
    "function_declarations": [
        {
            "name": "catalog_search_products",
            "description": "Semantic search over the merchant product catalog by buyer intent.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "Natural-language buyer intent"},
                    "max_results": {"type": "INTEGER", "description": "Max results to return"},
                    "max_price_inr": {"type": "INTEGER", "description": "Price ceiling in INR"},
                    "in_stock_only": {"type": "BOOLEAN", "description": "Filter in-stock items only"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "catalog_check_inventory",
            "description": "Check stock before creating an order. Returns alternatives if out of stock.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "product_id": {"type": "STRING", "description": "Product ID (e.g. PRD_001)"},
                    "quantity": {"type": "INTEGER", "description": "Units requested"},
                },
                "required": ["product_id"],
            },
        },
        {
            "name": "catalog_create_order",
            "description": "Create a Razorpay order and return a UPI payment link.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "product_id": {"type": "STRING", "description": "Product ID"},
                    "quantity": {"type": "INTEGER", "description": "Units to order"},
                    "buyer_id": {"type": "STRING", "description": "Buyer name or UPI ID"},
                    "confirmed": {"type": "BOOLEAN", "description": "Must be true after buyer confirms"},
                },
                "required": ["product_id", "buyer_id", "confirmed"],
            },
        },
        {
            "name": "catalog_cancel_order",
            "description": "Cancel an existing order, restore stock, and restore the session budget.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "order_id": {"type": "STRING", "description": "Order ID to cancel (e.g. order_xxx)"},
                    "reason": {"type": "STRING", "description": "Reason for cancellation"},
                },
            },
        },
        {
            "name": "catalog_get_audit_trail",
            "description": "Return the complete audit log for this session.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "session_id": {"type": "STRING", "description": "Session token"},
                },
                "required": [],
            },
        },
    ]
}]


# ── Budget display ──────────────────────────────────────────────────────────


def _print_budget(session_id: str) -> None:
    """Display a compact inline budget meter."""
    spent = audit.session_spent_inr(session_id)
    limit = settings.agent_spending_limit_inr
    remaining = max(0.0, limit - spent)
    pct = min(spent / limit, 1.0) if limit > 0 else 0
    bar_len = 30
    filled = int(pct * bar_len)
    empty = bar_len - filled

    if pct < 0.5:
        color = "green"
    elif pct < 0.8:
        color = "yellow"
    else:
        color = "red"

    bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
    console.print(
        f"  💰 Budget: {bar} ₹{spent:.0f}/₹{limit} "
        f"([bold {color}]₹{remaining:.0f} remaining[/bold {color}])"
    )


# ── Audit trail printer ────────────────────────────────────────────────────


def _print_audit(session_id: str) -> None:
    """Print the full session audit trail as a rich table."""
    events = audit.get_session_events(session_id)
    total = audit.session_spent_inr(session_id)

    if not events:
        console.print("[dim]No events recorded in this session.[/dim]")
        return

    console.rule("[bold magenta]📋 Session Audit Trail")

    table = Table(show_lines=True, expand=True)
    table.add_column("#", width=3)
    table.add_column("Timestamp", width=22)
    table.add_column("Tool", width=28)
    table.add_column("Outcome", width=16)
    table.add_column("₹ Amount", justify="right", width=10)
    table.add_column("Cumulative ₹", justify="right", width=13)

    for i, ev in enumerate(events, 1):
        outcome_color = {
            "success": "green",
            "out_of_stock": "yellow",
            "payment_created": "cyan",
            "limit_exceeded": "red",
            "failure": "red",
        }.get(ev["outcome"], "white")

        table.add_row(
            str(i),
            ev["ts"][:19].replace("T", " "),
            ev["tool_name"],
            f"[{outcome_color}]{ev['outcome']}[/{outcome_color}]",
            f"₹{ev['amount_inr']:.0f}" if ev["amount_inr"] else "—",
            f"₹{ev['cumulative_spend_inr']:.0f}",
        )

    console.print(table)
    remaining = audit.remaining_budget_inr(session_id)
    console.print(
        f"\n[bold]Total spent:[/bold] ₹{total:.0f}  |  "
        f"[bold]Remaining:[/bold] ₹{remaining:.0f}  |  "
        f"[bold]Events logged:[/bold] {len(events)}\n"
    )


# ── Catalog browser ─────────────────────────────────────────────────────────


def _print_catalog() -> None:
    """Print all products in the catalog as a formatted table."""
    console.rule("[bold cyan]📦 Merchant Catalog — GreenLeaf Organics")
    table = Table(show_lines=True, expand=True)
    table.add_column("ID", width=8, style="cyan")
    table.add_column("Product", width=30, style="bold")
    table.add_column("₹ Price", justify="right", width=8)
    table.add_column("Stock", justify="right", width=7)
    table.add_column("Category", width=14)

    from src.demo.sample_merchant import SAMPLE_CATALOG
    for p in SAMPLE_CATALOG:
        product = vs.get_by_id(p["id"])
        stock = int(product.get("stock", p["stock"])) if product else p["stock"]
        stock_str = f"[red]{stock}[/red]" if stock == 0 else f"[green]{stock}[/green]"
        table.add_row(
            p["id"], p["name"], f"₹{p['price_inr']}", stock_str, p["category"]
        )

    console.print(table)
    console.print()


# ── Gemini agent loop ───────────────────────────────────────────────────────


async def _agent_turn(
    user_message: str,
    contents: list[dict],
    session_id: str,
    client: httpx.AsyncClient,
) -> list[dict]:
    """
    Process a single user turn through the Gemini agent loop.
    The agent may make multiple tool calls before responding with text.
    Returns the updated contents list.
    """
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    system_text = f"""You are an autonomous AI shopping agent for '{settings.merchant_name}'.
Your Session ID: {session_id}

Rules:
1. Always call catalog_check_inventory BEFORE calling catalog_create_order.
2. If a product is out of stock (available=false), present the returned alternatives to the buyer and NEVER place an order for out-of-stock items.
3. Inquire/confirm product details and total price with the buyer before setting confirmed=true in catalog_create_order.
4. If the buyer declines or says do not place the order, politely acknowledge that no order was placed and NEVER call catalog_create_order.
5. If the buyer asks to order an alternative suggestion, check inventory for that alternative and place the order with confirmed=true.
6. Always provide clear, factual descriptions without hype.
7. When quoting prices, use the ₹ symbol.
8. If the buyer says something unrelated to shopping, respond helpfully but steer back to the catalog.
"""

    candidate_models = [settings.gemini_model, "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    unique_models = [m for m in candidate_models if m]

    headers = {"x-goog-api-key": settings.gemini_api_key}

    for _subturn in range(8):  # Max 8 tool-call sub-turns
        payload = {
            "system_instruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "tools": GEMINI_TOOLS,
        }

        resp = None
        for model_name in unique_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            for attempt in range(2):
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                    if resp.status_code == 200:
                        break
                    if resp.status_code in (429, 503):
                        console.print(f"  [dim yellow]⚠ Model {model_name} returned {resp.status_code}, trying next…[/dim yellow]")
                        await asyncio.sleep(1.0 * (attempt + 1))
                except Exception:
                    await asyncio.sleep(0.5)
            if resp and resp.status_code == 200:
                break

        if not resp or resp.status_code != 200:
            code = resp.status_code if resp else "timeout"
            console.print(f"[red]⚠ All Gemini models unavailable (last code: {code}). Please try again.[/red]")
            return contents

        candidate = resp.json().get("candidates", [{}])[0]
        candidate_content = candidate.get("content", {})
        parts = candidate_content.get("parts", [])

        has_func_call = False
        func_responses = []

        for p in parts:
            if "text" in p and p["text"].strip():
                console.print()
                console.print(Panel(
                    Markdown(p["text"].strip()),
                    title="[bold blue]🤖 AI Agent",
                    border_style="blue",
                    padding=(1, 2),
                ))
                _print_budget(session_id)

            elif "functionCall" in p:
                has_func_call = True
                fn_call = p["functionCall"]
                fn_name = fn_call.get("name")
                fn_args = dict(fn_call.get("args", {}))
                fn_args["session_id"] = session_id

                display_args = {k: v for k, v in fn_args.items() if k != "session_id"}
                console.print(
                    f"  [dim yellow]🔧 tool call:[/dim yellow] [yellow]{fn_name}[/yellow] "
                    f"[dim]{json.dumps(display_args, ensure_ascii=False)}[/dim]"
                )

                func = TOOL_REGISTRY.get(fn_name)
                if func:
                    res = func(**fn_args)
                else:
                    res = {"error": f"Unknown tool: {fn_name}"}

                # Compact result display
                res_str = json.dumps(res, ensure_ascii=False)
                if len(res_str) > 200:
                    res_str = res_str[:197] + "…"
                console.print(f"  [dim green]✓ result:[/dim green] [dim]{res_str}[/dim]")

                fn_resp = {
                    "name": fn_name,
                    "response": {"output": res},
                }
                if fn_call.get("id"):
                    fn_resp["id"] = fn_call.get("id")
                func_responses.append({"functionResponse": fn_resp})

        if not has_func_call:
            contents.append(candidate_content)
            break

        contents.append(candidate_content)
        contents.append({"role": "user", "parts": func_responses})
        await asyncio.sleep(0.3)

    return contents


# ── Main REPL ───────────────────────────────────────────────────────────────


async def main() -> None:
    """Interactive chat REPL."""

    # ── Header ──────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]🛒 Merchant AI Readability — Interactive Chat Mode")
    console.print(
        f"\n[bold]Merchant:[/bold] {settings.merchant_name}  |  "
        f"[bold]Session Budget:[/bold] ₹{settings.agent_spending_limit_inr}\n"
    )
    console.print(
        "[dim]Chat with the AI shopping agent in natural language (English / Hinglish).\n"
        "Commands:  [bold]/catalog[/bold] — browse products  |  "
        "[bold]/audit[/bold] — view audit trail  |  "
        "[bold]/budget[/bold] — check budget  |  "
        "[bold]exit[/bold] or [bold]Ctrl+C[/bold] — quit\n[/dim]"
    )

    # ── Check for Gemini API key ─────────────────────────────────────────────
    if not settings.gemini_api_key:
        console.print("[red]⚠ No GEMINI_API_KEY configured in .env. Please set it and restart.[/red]")
        return

    # ── Seed catalog if empty ───────────────────────────────────────────────
    if vs.catalog_size() == 0:
        console.print("[yellow]Seeding catalog…[/yellow]")
        await run_ingestion(verbose=False)
        console.print(f"[green]✓ Catalog ready: {vs.catalog_size()} products indexed.[/green]")

    session_id = f"chat_{uuid.uuid4().hex[:12]}"
    contents: list[dict] = []

    console.print(f"[dim]Session ID: {session_id}[/dim]\n")
    console.rule()

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                user_input = console.input("\n[bold green]You ❯[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n")
                break

            if not user_input:
                continue

            # ── Slash commands & Exit Handling ──────────────────────────────
            cleaned_input = user_input.lower().strip()
            if cleaned_input in ("exit", "quit", "/quit", "/exit", "okay exit", "ok exit", "bye", "goodbye", "stop"):
                break
            elif cleaned_input == "/catalog":
                _print_catalog()
                continue
            elif cleaned_input == "/audit":
                _print_audit(session_id)
                continue
            elif cleaned_input == "/budget":
                _print_budget(session_id)
                continue
            elif cleaned_input == "/help":
                console.print(
                    "[dim]Commands: /catalog, /audit, /budget, /help, exit[/dim]"
                )
                continue

            # ── Send to Gemini agent ────────────────────────────────────────
            try:
                contents = await _agent_turn(user_input, contents, session_id, client)
            except (KeyboardInterrupt, asyncio.CancelledError):
                break

    # ── Session summary on exit ─────────────────────────────────────────────
    console.print()
    _print_audit(session_id)
    console.print("[bold cyan]Thanks for shopping with GreenLeaf Organics! 🌿[/bold cyan]\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass

