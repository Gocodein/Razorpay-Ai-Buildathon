"""
Merchant AI Readability — MCP Tool Server

Exposes 6 tools that make any Razorpay merchant's catalog discoverable
and purchasable by an AI agent (Claude, ChatGPT, etc.):

    catalog_search_products   — Semantic search ("sunscreen under ₹500")
    catalog_get_product       — Fetch full product details by ID
    catalog_check_inventory   — Real-time stock check before committing
    catalog_create_order      — Initiate a Razorpay UPI order
    catalog_payment_status    — Check if a payment completed
    catalog_get_audit_trail   — Retrieve the session audit log

The "every money action must have an audit trail and one handled failure"
bar from Razorpay is satisfied by:
  - audit.logger recording every tool call with outcome + amount
  - catalog_check_inventory returning out_of_stock + alternatives before
    catalog_create_order is ever reached

Run with:
    python -m src.mcp_server.server
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Optional

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        from mcp.server import Server as FastMCP
from pydantic import BaseModel, ConfigDict, Field

from src.audit import logger as audit
from src.catalog import vector_store
from src.catalog.ingestion import run_ingestion
from src.config import settings
from src.payment import razorpay_client as rzp


# ── Lifespan: seed catalog on first boot if ChromaDB is empty ────────────────

@asynccontextmanager
async def _lifespan(app: Any = None):
    if vector_store.catalog_size() == 0:
        print("[server] ChromaDB is empty — running catalog ingestion…")
        await run_ingestion(verbose=False)
        print(f"[server] Catalog ready: {vector_store.catalog_size()} products indexed.")
    else:
        print(f"[server] Catalog loaded: {vector_store.catalog_size()} products.")
    yield


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "merchant_ai_readability_mcp",
    lifespan=_lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1 — Semantic product search
# ═══════════════════════════════════════════════════════════════════════════════

class SearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description=(
            "Natural-language buyer intent, e.g. "
            "'sunscreen under 500 rupees for oily skin' or "
            "'something for hair fall'."
        ),
        min_length=1,
        max_length=300,
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of products to return (1–10).",
        ge=1,
        le=10,
    )
    max_price_inr: Optional[int] = Field(
        default=None,
        description="Price ceiling in INR. Omit to return all prices.",
        ge=1,
    )
    in_stock_only: bool = Field(
        default=True,
        description="If true, only return products that have stock available.",
    )
    session_id: str = Field(
        default="demo_session_001",
        description="Unique agent session token — used for audit trail and spend tracking.",
    )


@mcp.tool(
    name="catalog_search_products",
    annotations={
        "title": "Search Merchant Product Catalog",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def catalog_search_products(params: SearchInput) -> str:
    """
    Semantic search over the merchant's product catalog.

    Use this first when a buyer describes what they need. Returns ranked
    products with price, stock status, and a buyer-facing description.

    Args:
        params (SearchInput): Validated input with query, filters, and session_id.

    Returns:
        str: JSON array of matching products sorted by relevance.
             Each item has: id, name, price_inr, stock, relevance_score,
             agent_description, structured_attributes, intent_phrases.
    """
    results = vector_store.search(
        query=params.query,
        n_results=params.max_results,
        max_price_inr=params.max_price_inr,
        in_stock_only=params.in_stock_only,
    )

    audit.log_event(
        session_id=params.session_id,
        tool_name="catalog_search_products",
        inputs=params.model_dump(),
        outcome="success",
        details={"result_count": len(results)},
    )

    if not results:
        return json.dumps({
            "results": [],
            "message": (
                "No products found matching your query"
                + (f" under ₹{params.max_price_inr}" if params.max_price_inr else "")
                + ". Try broadening your search."
            ),
        })

    return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2 — Fetch full product details
# ═══════════════════════════════════════════════════════════════════════════════

class GetProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_id: str = Field(
        ...,
        description="Product ID as returned by catalog_search_products (e.g. 'PRD_001').",
        min_length=3,
    )
    session_id: str = Field(..., min_length=4)


@mcp.tool(
    name="catalog_get_product",
    annotations={
        "title": "Get Full Product Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def catalog_get_product(params: GetProductInput) -> str:
    """
    Retrieve complete details for a specific product by its ID.

    Use after catalog_search_products to show the buyer full information
    before confirming a purchase.

    Args:
        params (GetProductInput): product_id and session_id.

    Returns:
        str: JSON object with all product fields, or an error message.
    """
    product = vector_store.get_by_id(params.product_id)

    audit.log_event(
        session_id=params.session_id,
        tool_name="catalog_get_product",
        inputs=params.model_dump(),
        outcome="success" if product else "failure",
        details={"found": product is not None},
    )

    if product is None:
        return json.dumps({
            "error": f"Product '{params.product_id}' not found in catalog.",
            "suggestion": "Use catalog_search_products to find valid product IDs.",
        })

    product["id"] = params.product_id
    return json.dumps(product, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3 — Inventory check (ALWAYS call before create_order)
# ═══════════════════════════════════════════════════════════════════════════════

class InventoryInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_id: str = Field(..., min_length=3)
    quantity: int = Field(default=1, description="Units the buyer wants.", ge=1, le=100)
    session_id: str = Field(..., min_length=4)


@mcp.tool(
    name="catalog_check_inventory",
    annotations={
        "title": "Check Product Inventory",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def catalog_check_inventory(params: InventoryInput) -> str:
    """
    Check whether the requested quantity is available before creating an order.

    Always call this before catalog_create_order.
    If the product is out of stock, this tool returns alternatives so the
    agent can offer the buyer a graceful substitute (handled-failure demo).

    Args:
        params (InventoryInput): product_id, quantity, session_id.

    Returns:
        str: JSON with:
             available (bool), stock_count, alternatives (if out of stock),
             spending_budget_remaining_inr.
    """
    product = vector_store.get_by_id(params.product_id)

    if product is None:
        audit.log_event(
            session_id=params.session_id,
            tool_name="catalog_check_inventory",
            inputs=params.model_dump(),
            outcome="failure",
            details={"reason": "product_not_found"},
        )
        return json.dumps({"error": f"Product '{params.product_id}' not found."})

    stock = int(product.get("stock", 0))
    available = stock >= params.quantity
    remaining_budget = audit.remaining_budget_inr(params.session_id)

    outcome = "success" if available else "out_of_stock"

    response: dict[str, Any] = {
        "product_id": params.product_id,
        "product_name": product.get("name"),
        "requested_quantity": params.quantity,
        "stock_count": stock,
        "available": available,
        "spending_budget_remaining_inr": remaining_budget,
    }

    if not available:
        # ── Graceful failure: suggest alternatives ────────────────────────────
        response["message"] = (
            f"'{product.get('name')}' is out of stock "
            f"(only {stock} units available). "
            f"Here are some alternatives from the same category:"
        )
        alternatives = vector_store.search(
            query=product.get("name", ""),
            n_results=3,
            in_stock_only=True,
        )
        # Exclude the out-of-stock product itself from alternatives
        alternatives = [a for a in alternatives if a.get("id") != params.product_id]
        response["alternatives"] = alternatives[:2]
    else:
        price_inr = int(product.get("price_inr", 0))
        order_total = price_inr * params.quantity
        response["message"] = f"{stock} units in stock. Ready to order."
        response["estimated_order_total_inr"] = order_total

        if order_total > remaining_budget:
            response["budget_warning"] = (
                f"This order (₹{order_total}) exceeds your remaining "
                f"session budget of ₹{remaining_budget:.0f}. "
                f"The order will not go through."
            )

    audit.log_event(
        session_id=params.session_id,
        tool_name="catalog_check_inventory",
        inputs=params.model_dump(),
        outcome=outcome,
        details=response,
    )

    return json.dumps(response, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 4 — Create order (money action — fully audited)
# ═══════════════════════════════════════════════════════════════════════════════

class CreateOrderInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_id: str = Field(..., min_length=3)
    quantity: int = Field(default=1, ge=1, le=100)
    buyer_id: str = Field(
        ...,
        description=(
            "Buyer identifier — name, UPI ID, or phone number. "
            "Used for audit trail and payment link personalisation."
        ),
        min_length=2,
    )
    session_id: str = Field(
        ...,
        description=(
            "Agent session token. The session-level spending limit "
            f"(₹{settings.agent_spending_limit_inr}) is enforced here."
        ),
        min_length=4,
    )
    buyer_name: Optional[str] = Field(
        default="",
        description="Buyer's real customer name (e.g. 'Sagar Shaw').",
    )
    buyer_upi: Optional[str] = Field(
        default="",
        description="Buyer's UPI ID (e.g. 'sagar@slice', 'sagar@okhdfcbank').",
    )
    confirmed: bool = Field(
        ...,
        description=(
            "Must be explicitly set to true by the agent after the buyer "
            "has confirmed the purchase. Prevents accidental orders."
        ),
    )


@mcp.tool(
    name="catalog_create_order",
    annotations={
        "title": "Create Razorpay Order (Money Action)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def catalog_create_order(params: CreateOrderInput) -> str:
    """
    Create a Razorpay order and return a UPI payment link.

    Prerequisites (agent must do these first):
      1. catalog_search_products → find the product
      2. catalog_check_inventory → confirm it is in stock
      3. Show buyer the price and get explicit confirmation
      4. Set confirmed=true in this call

    Enforces the NPCI UAP session spending limit automatically.
    Every call is written to the immutable audit log.

    Args:
        params (CreateOrderInput): product_id, quantity, buyer_id,
                                   session_id, confirmed.

    Returns:
        str: JSON with order_id, payment_link, amount_inr,
             remaining_budget_inr, status, message.
    """
    # ── Safety gate: buyer must have confirmed ────────────────────────────────
    if not params.confirmed:
        return json.dumps({
            "status": "not_confirmed",
            "message": (
                "The buyer has not confirmed this purchase. "
                "Show them the product name, price, and quantity, "
                "then set confirmed=true only after they agree."
            ),
        })

    # ── Fetch product to get price ────────────────────────────────────────────
    product = vector_store.get_by_id(params.product_id)
    if product is None:
        return json.dumps({
            "status": "failure",
            "message": f"Product '{params.product_id}' not found.",
        })

    result = rzp.create_order(
        product_id=params.product_id,
        product_name=product.get("name", params.product_id),
        quantity=params.quantity,
        unit_price_inr=int(product["price_inr"]),
        buyer_id=params.buyer_id,
        session_id=params.session_id,
        buyer_name=params.buyer_name or "",
        buyer_upi=params.buyer_upi or "",
    )

    return json.dumps(result, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 5 — Payment status
# ═══════════════════════════════════════════════════════════════════════════════

class PaymentStatusInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    order_id: str = Field(
        ...,
        description="Razorpay order ID returned by catalog_create_order.",
        min_length=6,
    )
    session_id: str = Field(..., min_length=4)


@mcp.tool(
    name="catalog_payment_status",
    annotations={
        "title": "Check Razorpay Payment Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def catalog_payment_status(params: PaymentStatusInput) -> str:
    """
    Check whether the buyer has completed payment for a Razorpay order.

    Call this after catalog_create_order to confirm the transaction
    before informing the buyer that the purchase is complete.

    Args:
        params (PaymentStatusInput): order_id, session_id.

    Returns:
        str: JSON with status ('paid' | 'pending' | 'authorized' | 'error'),
             message, and payment attempts list.
    """
    result = rzp.get_payment_status(
        order_id=params.order_id,
        session_id=params.session_id,
    )
    return json.dumps(result, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 6 — Cancel Order & Restore Budget
# ═══════════════════════════════════════════════════════════════════════════════

class CancelOrderInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    order_id: str = Field(
        default="",
        description="Order ID to cancel (e.g. order_xxx). If omitted, the latest order is cancelled.",
    )
    session_id: str = Field(
        ...,
        description="Session ID where the order was placed.",
        min_length=4,
    )
    reason: str = Field(
        default="Buyer requested cancellation",
        description="Reason for order cancellation.",
    )


@mcp.tool(
    name="catalog_cancel_order",
    annotations={
        "title": "Cancel Order & Restore Budget (Money Action)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def catalog_cancel_order(params: CancelOrderInput) -> str:
    """
    Cancel an existing order, restore inventory in ChromaDB,
    and refund the session spending budget in the immutable SQLite audit log.

    Args:
        params (CancelOrderInput): order_id, session_id, reason.

    Returns:
        str: JSON with status, order_id, refunded amount, and updated budget.
    """
    result = rzp.cancel_order(
        order_id=params.order_id,
        session_id=params.session_id,
        reason=params.reason,
    )
    return json.dumps(result, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 7 — Retrieve Session Audit Trail
# ═══════════════════════════════════════════════════════════════════════════════

class AuditInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: str = Field(
        ...,
        description="Session ID to retrieve audit events for.",
        min_length=4,
    )


@mcp.tool(
    name="catalog_get_audit_trail",
    annotations={
        "title": "Retrieve Session Audit Trail",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def catalog_get_audit_trail(params: AuditInput) -> str:
    """
    Return all actions taken by this agent session, in chronological order.

    Use this to provide a complete, explainable record of what the agent
    did on behalf of the buyer — satisfying Razorpay's audit-trail bar.

    Args:
        params (AuditInput): session_id.

    Returns:
        str: JSON list of events with timestamps, tool names,
             outcomes, amounts, and cumulative spend.
    """
    events = audit.get_session_events(params.session_id)
    total_spent = audit.session_spent_inr(params.session_id)

    return json.dumps(
        {
            "session_id": params.session_id,
            "event_count": len(events),
            "total_spent_inr": total_spent,
            "remaining_budget_inr": audit.remaining_budget_inr(params.session_id),
            "spending_limit_inr": settings.agent_spending_limit_inr,
            "events": events,
        },
        ensure_ascii=False,
        default=str,
    )



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # stdio transport for local use with Claude Desktop / MCP Inspector
    mcp.run()
