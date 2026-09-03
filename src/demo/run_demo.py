"""
End-to-end agentic commerce demo.

Demonstrates the full Track-1 flow:
  1. Seed catalog into ChromaDB (if empty)
  2. Run two demo scenarios via Autonomous Tool-Calling Agent Engine (Gemini / Claude / Local):
      Scenario A — Happy path: find, inventory check, budget validation, Razorpay order
      Scenario B — Graceful failure: out-of-stock → in-stock alternatives offered
  3. Print the complete immutable SQLite audit trail for both sessions

Run:
    python -m src.demo.run_demo

Requirements:
    GEMINI_API_KEY / ANTHROPIC_API_KEY — for cloud LLM reasoning (or runs Local Engine)
    RAZORPAY_KEY_ID     — test-mode (orders won't charge real money)
    RAZORPAY_KEY_SECRET
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.audit import logger as audit
from src.catalog import vector_store
from src.catalog.ingestion import run_ingestion
from src.config import settings

console = Console(legacy_windows=False)

# ── MCP tools that Claude can call ───────────────────────────────────────────
# We define them as Anthropic tool schemas so we can run the demo without
# standing up a separate MCP server process (the logic is imported directly).

from src.catalog import vector_store as vs
from src.payment import razorpay_client as rzp


def _search_products(query, max_results=5, max_price_inr=None, in_stock_only=False, session_id=""):
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


def _get_audit_trail(session_id):
    return {
        "events": audit.get_session_events(session_id),
        "total_spent_inr": audit.session_spent_inr(session_id),
        "remaining_budget_inr": audit.remaining_budget_inr(session_id),
    }


# ── Tool dispatcher ────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "catalog_search_products": _search_products,
    "catalog_check_inventory": _check_inventory,
    "catalog_create_order": _create_order,
    "catalog_get_audit_trail": _get_audit_trail,
}

TOOL_SCHEMAS = [
    {
        "name": "catalog_search_products",
        "description": "Semantic search over the merchant catalog by buyer intent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
                "max_price_inr": {"type": "integer"},
                "in_stock_only": {"type": "boolean", "default": True},
                "session_id": {"type": "string"},
            },
            "required": ["query", "session_id"],
        },
    },
    {
        "name": "catalog_check_inventory",
        "description": "Check stock before creating an order. Returns alternatives if out of stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "quantity": {"type": "integer", "default": 1},
                "session_id": {"type": "string"},
            },
            "required": ["product_id", "session_id"],
        },
    },
    {
        "name": "catalog_create_order",
        "description": "Create a Razorpay order and return a UPI payment link.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "quantity": {"type": "integer", "default": 1},
                "buyer_id": {"type": "string"},
                "session_id": {"type": "string"},
                "confirmed": {"type": "boolean"},
            },
            "required": ["product_id", "buyer_id", "session_id", "confirmed"],
        },
    },
    {
        "name": "catalog_get_audit_trail",
        "description": "Return the complete audit log for this session.",
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
]


# ── Agentic conversation loop ─────────────────────────────────────────────────

def _is_api_key_valid() -> bool:
    key = (settings.anthropic_api_key or "").strip()
    return bool(key and not key.startswith("sk-ant-xxx") and len(key) > 20)


async def _run_simulated_scenario(
    scenario_name: str,
    turns: list[str],
    session_id: str,
) -> None:
    """
    Deterministic simulated agent loop used when running offline / without API key.
    """
    console.rule(f"[bold cyan]Scenario: {scenario_name} (Simulated Agent)")
    
    first_msg = turns[0] if turns else ""
    console.print(f"[bold green]Buyer:[/bold green] {first_msg}\n")

    if "sunscreen" in first_msg.lower():
        # Turn 1: Search & Check
        console.print(Panel(
            "Hello Priya! I'll find an effective, non-greasy sunscreen for oily skin under ₹400 for you right away.",
            title="[bold blue]Agent", border_style="blue"
        ))
        s_input = {"query": "sunscreen oily skin under 400", "max_price_inr": 400, "session_id": session_id}
        console.print(f"  [dim yellow]→ tool call:[/dim yellow] [yellow]catalog_search_products[/yellow] [dim]{json.dumps(s_input)}[/dim]")
        s_res = _search_products(**s_input)
        console.print(f"  [dim green]← result:[/dim green] [dim]{json.dumps(s_res, ensure_ascii=False)[:180]}…[/dim]\n")

        prd_id = s_res["results"][0]["id"] if s_res.get("results") else "PRD_001"
        inv_input = {"product_id": prd_id, "quantity": 1, "session_id": session_id}
        console.print(f"  [dim yellow]→ tool call:[/dim yellow] [yellow]catalog_check_inventory[/yellow] [dim]{json.dumps(inv_input)}[/dim]")
        inv_res = _check_inventory(**inv_input)
        console.print(f"  [dim green]← result:[/dim green] [dim]{json.dumps(inv_res, ensure_ascii=False)[:180]}…[/dim]\n")

        console.print(Panel(
            "We have **Aloe Vera Sunscreen SPF 50** at ₹349 in stock (119 units).\n\n"
            "Would you like me to go ahead and place the order for you at priya@okaxis?",
            title="[bold blue]Agent", border_style="blue"
        ))

        # Turn 2 if present
        if len(turns) > 1:
            second_msg = turns[1]
            console.print(f"\n[bold green]Buyer:[/bold green] {second_msg}\n")
            if any(w in second_msg.lower() for w in ["yes", "please", "confirm", "order"]):
                # Order confirmed
                ord_input = {"product_id": prd_id, "quantity": 1, "buyer_id": "priya@okaxis", "session_id": session_id, "confirmed": True}
                console.print(f"  [dim yellow]→ tool call:[/dim yellow] [yellow]catalog_create_order[/yellow] [dim]{json.dumps(ord_input)}[/dim]")
                ord_res = _create_order(**ord_input)
                console.print(f"  [dim green]← result:[/dim green] [dim]{json.dumps(ord_res, ensure_ascii=False)[:180]}…[/dim]\n")

                console.print(Panel(
                    f"🎉 Your order has been initiated!\n\n"
                    f"• **Product:** Aloe Vera Sunscreen SPF 50 (1 unit)\n"
                    f"• **Amount:** ₹{ord_res.get('amount_inr', 349)}\n"
                    f"• **Payment Link:** [cyan]{ord_res.get('payment_link')}[/cyan]\n"
                    f"• **Remaining Session Budget:** ₹{ord_res.get('remaining_budget_inr')}\n\n"
                    f"Please complete your UPI payment using the link above.",
                    title="[bold blue]Agent", border_style="blue"
                ))
            else:
                # Order declined
                console.print(Panel(
                    "Understood! No order has been placed and no charge has been made. Let me know if you need anything else!",
                    title="[bold blue]Agent", border_style="blue"
                ))

    else:
        # Out-of-stock scenario (Neem face wash)
        console.print(Panel(
            "Hello Rahul! Checking stock for Neem & Tulsi Face Wash (PRD_003) immediately.",
            title="[bold blue]Agent", border_style="blue"
        ))
        inv_input = {"product_id": "PRD_003", "quantity": 1, "session_id": session_id}
        console.print(f"  [dim yellow]→ tool call:[/dim yellow] [yellow]catalog_check_inventory[/yellow] [dim]{json.dumps(inv_input)}[/dim]")
        inv_res = _check_inventory(**inv_input)
        console.print(f"  [dim green]← result:[/dim green] [dim]{json.dumps(inv_res, ensure_ascii=False)[:180]}…[/dim]\n")

        alts = inv_res.get("alternatives", [])
        alt_lines = "\n".join([f"• {a.get('name')} — ₹{a.get('price_inr')}" for a in alts])
        console.print(Panel(
            f"⚠️ **Notice**: The Neem & Tulsi Face Wash is currently **out of stock** (0 units available).\n\n"
            f"To protect your payment, I have **not placed an order**.\n\n"
            f"Here are top-rated in-stock alternatives from our skincare catalog:\n"
            f"{alt_lines}\n\n"
            f"Would you like me to reserve one of these for you instead?",
            title="[bold blue]Agent", border_style="blue"
        ))

        # Turn 2 if buyer chooses an alternative
        if len(turns) > 1:
            second_msg = turns[1]
            console.print(f"\n[bold green]Buyer:[/bold green] {second_msg}\n")
            alt_id = "PRD_007"  # Rose Water Facial Toner
            inv_input2 = {"product_id": alt_id, "quantity": 1, "session_id": session_id}
            console.print(f"  [dim yellow]→ tool call:[/dim yellow] [yellow]catalog_check_inventory[/yellow] [dim]{json.dumps(inv_input2)}[/dim]")
            inv_res2 = _check_inventory(**inv_input2)
            console.print(f"  [dim green]← result:[/dim green] [dim]{json.dumps(inv_res2, ensure_ascii=False)[:180]}…[/dim]\n")

            ord_input = {"product_id": alt_id, "quantity": 1, "buyer_id": "rahul@okhdfcbank", "session_id": session_id, "confirmed": True}
            console.print(f"  [dim yellow]→ tool call:[/dim yellow] [yellow]catalog_create_order[/yellow] [dim]{json.dumps(ord_input)}[/dim]")
            ord_res = _create_order(**ord_input)
            console.print(f"  [dim green]← result:[/dim green] [dim]{json.dumps(ord_res, ensure_ascii=False)[:180]}…[/dim]\n")

            console.print(Panel(
                f"🎉 Your alternative order has been initiated!\n\n"
                f"• **Product:** Rose Water Facial Toner 200ml (1 unit)\n"
                f"• **Amount:** ₹{ord_res.get('amount_inr', 249)}\n"
                f"• **Payment Link:** [cyan]{ord_res.get('payment_link')}[/cyan]\n"
                f"• **Remaining Session Budget:** ₹{ord_res.get('remaining_budget_inr')}\n\n"
                f"Please complete your UPI payment using the link above.",
                title="[bold blue]Agent", border_style="blue"
            ))


import httpx


# ── Google Gemini Agentic Loop ──────────────────────────────────────────────

async def _run_gemini_scenario(
    scenario_name: str,
    turns: list[str],
    session_id: str,
) -> None:
    """
    Live multi-turn agent execution using Google Gemini with native tool calling.
    """
    console.rule(f"[bold cyan]Scenario: {scenario_name} (Google Gemini Live Agent)")

    gemini_tools = [{
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
                        "confirmed": {"type": "BOOLEAN", "description": "Must be set to true when buyer confirmed"},
                    },
                    "required": ["product_id", "buyer_id", "confirmed"],
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

    system_text = f"""You are an autonomous AI shopping agent for '{settings.merchant_name}'.
Your Session ID: {session_id}

Rules:
1. Always call catalog_check_inventory BEFORE calling catalog_create_order.
2. If a product is out of stock (available=false), present the returned alternatives to the buyer and NEVER place an order for out-of-stock items.
3. Inquire/confirm product details and total price with the buyer before setting confirmed=true in catalog_create_order.
4. If the buyer declines or says do not place the order, politely acknowledge that no order was placed and NEVER call catalog_create_order.
5. If the buyer asks to order an alternative suggestion, check inventory for that alternative and place the order with confirmed=true.
6. Always provide clear, factual descriptions without hype.
"""
    contents: list[dict] = []
    candidate_models = [settings.gemini_model, "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    unique_models = [m for m in candidate_models if m]

    headers = {"x-goog-api-key": settings.gemini_api_key}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for turn_idx, user_query in enumerate(turns):
            if turn_idx > 0:
                console.print(f"\n[bold green]Buyer:[/bold green] {user_query}\n")
            else:
                console.print(f"[bold green]Buyer:[/bold green] {user_query}\n")
            
            contents.append({"role": "user", "parts": [{"text": user_query}]})

            for _subturn in range(6):
                payload = {
                    "system_instruction": {"parts": [{"text": system_text}]},
                    "contents": contents,
                    "tools": gemini_tools,
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
                                await asyncio.sleep(1.0 * (attempt + 1))
                        except Exception:
                            await asyncio.sleep(0.5)
                    if resp and resp.status_code == 200:
                        break

                if not resp or resp.status_code != 200:
                    code = resp.status_code if resp else "timeout"
                    console.print(f"[yellow]Gemini API returned code {code} — falling back to local runner[/yellow]")
                    await _run_simulated_scenario(scenario_name, turns, session_id)
                    return

                candidate = resp.json().get("candidates", [{}])[0]
                candidate_content = candidate.get("content", {})
                parts = candidate_content.get("parts", [])

                has_func_call = False
                func_responses = []

                for p in parts:
                    if "text" in p and p["text"].strip():
                        console.print(Panel(p["text"].strip(), title="[bold blue]Gemini Agent", border_style="blue"))
                    elif "functionCall" in p:
                        has_func_call = True
                        fn_call = p["functionCall"]
                        fn_name = fn_call.get("name")
                        fn_args = dict(fn_call.get("args", {}))
                        fn_args["session_id"] = session_id

                        console.print(
                            f"  [dim yellow]→ tool call:[/dim yellow] [yellow]{fn_name}[/yellow] "
                            f"[dim]{json.dumps({k: v for k, v in fn_args.items() if k != 'session_id'})}[/dim]"
                        )

                        func = TOOL_REGISTRY.get(fn_name)
                        if func:
                            res = func(**fn_args)
                        else:
                            res = {"error": f"Unknown tool: {fn_name}"}

                        console.print(
                            f"  [dim green]← result:[/dim green] "
                            f"[dim]{json.dumps(res, ensure_ascii=False)[:180]}…[/dim]\n"
                        )

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
                await asyncio.sleep(0.5)

            await asyncio.sleep(1.0)


async def run_scenario(
    scenario_name: str,
    turns: list[str] | str,
    session_id: str,
) -> None:
    turn_list = [turns] if isinstance(turns, str) else turns

    # ── 1. Check for Gemini Provider ─────────────────────────────────────────
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        await _run_gemini_scenario(scenario_name, turn_list, session_id)
        return

    # ── 2. Fallback to Local Deterministic Engine ─────────────────────────────
    await _run_simulated_scenario(scenario_name, turn_list, session_id)


# ── Audit summary printer ─────────────────────────────────────────────────────

def _print_audit(session_id: str, label: str) -> None:
    events = audit.get_session_events(session_id)
    total = audit.session_spent_inr(session_id)

    console.rule(f"[bold magenta]Audit Trail — {label}")

    table = Table(show_lines=True, expand=True)
    table.add_column("#", width=3)
    table.add_column("Timestamp", width=26)
    table.add_column("Tool", width=30)
    table.add_column("Outcome", width=18)
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
            ev["ts"][:23],
            ev["tool_name"],
            f"[{outcome_color}]{ev['outcome']}[/{outcome_color}]",
            f"₹{ev['amount_inr']:.0f}" if ev["amount_inr"] else "—",
            f"₹{ev['cumulative_spend_inr']:.0f}",
        )

    console.print(table)
    console.print(
        f"[bold]Total spent:[/bold] ₹{total:.0f}  |  "
        f"[bold]Events logged:[/bold] {len(events)}\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    console.rule("[bold cyan]Merchant AI Readability — Multi-Turn Agentic Checkout Demo")
    console.print(
        f"[dim]Merchant: {settings.merchant_name} | "
        f"Session budget: ₹{settings.agent_spending_limit_inr}[/dim]\n"
    )

    # Seed catalog if needed
    if vector_store.catalog_size() == 0:
        console.print("[yellow]Seeding catalog…[/yellow]")
        await run_ingestion()

    # ── Scenario 1: Buyer Confirms Order (Happy Path Checkout) ────────────────
    session_1 = f"demo_confirm_{uuid.uuid4().hex[:8]}"
    await run_scenario(
        scenario_name="Scenario 1: Buyer Confirms Order (Happy Path Checkout)",
        turns=[
            "Hi! I need a good sunscreen for oily skin, preferably under ₹400. Can you find one for me? My UPI ID is priya@okaxis.",
            "Yes, confirmed! Please place the order for 1 unit of Aloe Vera Sunscreen for me at priya@okaxis.",
        ],
        session_id=session_1,
    )
    _print_audit(session_1, "Scenario 1 — Order Confirmed")

    console.print("\n")
    await asyncio.sleep(2.0)

    # ── Scenario 2: Buyer Declines Order (Safety & Consent Enforcement) ───────
    session_2 = f"demo_decline_{uuid.uuid4().hex[:8]}"
    await run_scenario(
        scenario_name="Scenario 2: Buyer Declines Order (Safety Verification)",
        turns=[
            "Hi! I need a good sunscreen for oily skin, preferably under ₹400. Can you check one for me? My UPI ID is priya@okaxis.",
            "No, please do not place the order. I changed my mind.",
        ],
        session_id=session_2,
    )
    _print_audit(session_2, "Scenario 2 — Order Declined (₹0 Spent)")

    console.print("\n")
    await asyncio.sleep(2.0)

    # ── Scenario 3: Out-of-Stock → Alternative Suggestion Accepted & Ordered ─
    session_3 = f"demo_alt_{uuid.uuid4().hex[:8]}"
    await run_scenario(
        scenario_name="Scenario 3: Out-of-Stock → Alternative Accepted & Purchased",
        turns=[
            "I want to buy the Neem & Tulsi face wash. Please order 1 unit for me — my name is Rahul, UPI ID rahul@okhdfcbank.",
            "Since Neem & Tulsi is out of stock, please go ahead and order 1 unit of the Rose Water Facial Toner for me at rahul@okhdfcbank instead!",
        ],
        session_id=session_3,
    )
    _print_audit(session_3, "Scenario 3 — Alternative Ordered")


if __name__ == "__main__":
    asyncio.run(main())
