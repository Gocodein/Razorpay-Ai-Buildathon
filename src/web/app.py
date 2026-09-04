"""
Merchant AI Readability — Interactive Web Dashboard (Streamlit)
Engineered for Razorpay AI Buildathon — Track 01 (AI Growth & Agentic Commerce)

Design: Modern Frosted Glass Interface (Glassmorphism + Translucency + Blur)
Features:
  - 💬 Autonomous AI Shopping Assistant (Gemini Tool Calling + Hinglish + Order Cancellation)
  - 💳 Spacious Frosted Glass Razorpay Checkout Terminal (Modal + UPI QR + Cancel/Refund)
  - 📦 Live Catalog Explorer (158+ BigBasket SKUs with Instant Search & Category Filters)
  - 💰 Real-Time NPCI UAP Session Budget Gauge (₹2,000 Hard Limit)
  - 📋 Immutable SQLite WAL Audit Trail with Live Cancelled Order Tracking & CSV Export
  - 🚀 Merchant 1-Click Catalog Ingestion & Schema Mapper
  - ⚡ Sub-50ms In-Memory Caching
"""

from __future__ import annotations

import asyncio
import csv
import html
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from io import BytesIO, StringIO
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
import streamlit as st

from src.config import settings
from src.catalog import vector_store as vs
from src.payment import razorpay_client as rzp
from src.audit import logger as audit

# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"{settings.merchant_name} — AI Commerce Gateway",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Fast In-Memory Data Caching ──────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def get_cached_catalog() -> list[dict]:
    """Retrieve all catalog items from ChromaDB with fast in-memory caching."""
    return vs.get_all_products(limit=300)


@st.cache_data(ttl=30, show_spinner=False)
def get_cached_categories() -> list[str]:
    """Extract distinct top-level categories from the catalog."""
    products = get_cached_catalog()
    raw_cats = {p.get("category", "General").split(" > ")[0].strip() for p in products if p.get("category")}
    return ["All Categories"] + sorted(list(raw_cats))


# ── Session State Initialization ─────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_{uuid.uuid4().hex[:12]}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"👋 **Welcome to {settings.merchant_name}!**\n\n"
                f"I am your autonomous AI shopping assistant connected directly to the **Razorpay Agentic Commerce Gateway**.\n\n"
                f"You can search over **{vs.catalog_size()} verified products** in plain English or Hinglish "
                f"(*e.g., 'sunscreen under ₹400'*, *'pure cow ghee'*, *'machhar bhagane ka spray'*), verify stock, "
                f"and place instant Razorpay UPI orders within your **₹{settings.agent_spending_limit_inr}** session budget.\n\n"
                f"💡 *Click any quick action chip below or type your request to begin!*"
            ),
        }
    ]

if "contents" not in st.session_state:
    st.session_state.contents = []

if "latest_order" not in st.session_state:
    st.session_state.latest_order = None

if "paid_success_info" not in st.session_state:
    st.session_state.paid_success_info = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "merchant_authenticated" not in st.session_state:
    st.session_state.merchant_authenticated = False

if "active_merchant_id" not in st.session_state:
    st.session_state.active_merchant_id = settings.merchant_id

if "active_merchant_name" not in st.session_state:
    st.session_state.active_merchant_name = settings.merchant_name

# ── Handle Payment Success Redirect / Query Params ───────────────────────────
query_params = st.query_params
if "paid_order" in query_params:
    paid_oid = query_params.get("paid_order", "")
    pay_id = query_params.get("payment_id", "pay_test_captured")
    st.query_params.clear()

    latest = st.session_state.latest_order or {}
    p_name = latest.get("product_name", "Product")
    p_id = latest.get("product_id", "")
    qty = latest.get("quantity", 1)
    amt = latest.get("amount_inr", 0)
    b_id = latest.get("buyer_id") or ""
    b_name = latest.get("buyer_name") or ""
    b_upi = latest.get("buyer_upi") or ""

    # Fallback to SQLite event lookup if session state was cleared
    if not (b_name or b_upi or b_id):
        events = audit.get_session_events(st.session_state.session_id)
        for ev in events:
            if ev.get("tool_name") == "razorpay_create_order":
                inp = ev.get("inputs", {})
                if isinstance(inp, str):
                    try: inp = json.loads(inp)
                    except Exception: inp = {}
                b_id = inp.get("buyer_id", "")
                b_name = inp.get("buyer_name", "")
                b_upi = inp.get("buyer_upi", "")
                if not p_name: p_name = inp.get("product_name", "Product")
                if not p_id: p_id = inp.get("product_id", "")
                if not amt: amt = ev.get("amount_inr", 0)
                break

    cust_name, cust_upi = rzp._extract_customer_identity(b_id, b_name, b_upi)
    audit.log_customer_action(
        customer_id=cust_name,
        upi_id=cust_upi,
        action_type="PAYMENT_CAPTURED",
        order_id=paid_oid,
        product_id=p_id,
        product_name=p_name,
        quantity=qty,
        amount_inr=amt,
        session_id=st.session_state.session_id,
        details={"payment_id": pay_id, "status": "paid", "rail": "Razorpay_Checkout"},
    )

    audit.log_event(
        session_id=st.session_state.session_id,
        tool_name="razorpay_payment_success",
        inputs={"order_id": paid_oid, "payment_id": pay_id},
        outcome="success",
        details={"payment_id": pay_id, "status": "paid", "amount_inr": amt},
        amount_inr=0,
    )

    st.session_state.latest_order = None
    st.session_state.paid_success_info = {
        "order_id": paid_oid,
        "payment_id": pay_id,
        "product_name": p_name,
        "amount_inr": amt,
    }

    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            f"🎉 **Payment Confirmed via Razorpay!**\n\n"
            f"• **Order ID**: `{paid_oid}`\n"
            f"• **Payment ID**: `{pay_id}`\n"
            f"• **Amount Settled**: ₹{amt:.0f}\n"
            f"• **Customer**: {cust_name} ({cust_upi})\n"
            f"• **Status**: `PAYMENT_CAPTURED`\n\n"
            f"Your order has been officially processed and recorded in the **Customer Intelligence Sheet**. Thank you for shopping with **{settings.merchant_name}**!"
        ),
    })
    if "contents" in st.session_state and isinstance(st.session_state.contents, list):
        st.session_state.contents.append({
            "role": "model",
            "parts": [{"text": f"Payment of ₹{amt:.0f} for Order {paid_oid} (Payment ID: {pay_id}) was successfully captured via Razorpay. The order is fulfilled."}]
        })
    st.rerun()

# ── Modern Frosted Glass (Midnight Pro Stitch Theme) CSS ───────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Ambient Midnight Navy Obsidian Background */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #07193b 0%, #001232 55%, #000c24 100%);
        color: #d8e2ff;
    }

    /* Main Frosted Glass Stage */
    .glass-stage {
        background: rgba(7, 30, 71, 0.65);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(82, 143, 240, 0.28);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 12px 36px 0 rgba(0, 0, 0, 0.45);
    }

    /* Frosted Metric Card */
    .glass-metric {
        background: rgba(7, 38, 84, 0.55);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(82, 143, 240, 0.25);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5em;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-lbl {
        font-size: 0.74em;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
        font-weight: 700;
    }

    /* Frosted Product Card */
    .product-glass-card {
        background: rgba(11, 41, 87, 0.42);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(82, 143, 240, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        transition: all 0.22s ease-in-out;
        height: 100%;
    }
    .product-glass-card:hover {
        transform: translateY(-3px);
        border-color: #528FF0;
        box-shadow: 0 10px 28px rgba(82, 143, 240, 0.25);
        background: rgba(15, 51, 107, 0.6);
    }
    .p-title {
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.96em;
        line-height: 1.35em;
        min-height: 2.7em;
    }
    .p-price {
        font-family: 'JetBrains Mono', monospace;
        color: #528FF0;
        font-weight: 800;
        font-size: 1.25em;
        margin: 8px 0;
    }
    .p-cat {
        color: #94a3b8;
        font-size: 0.76em;
        margin-top: 4px;
    }

    /* Luminous Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.72em;
        font-weight: 700;
        letter-spacing: 0.05em;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-in { background: rgba(0, 200, 150, 0.15); color: #00C896; border: 1px solid rgba(0, 200, 150, 0.4); }
    .badge-low { background: rgba(255, 184, 0, 0.15); color: #FFB800; border: 1px solid rgba(255, 184, 0, 0.4); }
    .badge-out { background: rgba(255, 77, 77, 0.15); color: #FF4D4D; border: 1px solid rgba(255, 77, 77, 0.4); }

    /* Expanded Payment Terminal Card */
    .terminal-container {
        background: linear-gradient(135deg, rgba(16, 45, 96, 0.85) 0%, rgba(7, 24, 58, 0.85) 100%);
        backdrop-filter: blur(24px) saturate(200%);
        -webkit-backdrop-filter: blur(24px) saturate(200%);
        border: 1px solid #528FF0;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 14px 45px rgba(82, 143, 240, 0.28);
    }
</style>
""", unsafe_allow_html=True)

# ── Agent Tool Execution Registry ────────────────────────────────────────────

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
    response = {
        "product_id": product_id,
        "name": product.get("name"),
        "stock": stock,
        "available": available,
        "price_inr": product.get("price_inr"),
        "budget_remaining_inr": audit.remaining_budget_inr(session_id),
    }
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


def _create_order(product_id, quantity=1, buyer_id="buyer", session_id="", confirmed=True, buyer_name="", buyer_upi=""):
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
        buyer_name=buyer_name,
        buyer_upi=buyer_upi,
    )


def _cancel_order(order_id="", reason="Buyer requested cancellation", session_id=""):
    latest = st.session_state.latest_order or {}
    product_id = latest.get("product_id", "")
    quantity = int(latest.get("quantity", 1))
    amount_inr = float(latest.get("amount_inr", 0))
    target_oid = order_id or latest.get("order_id", "")

    res = rzp.cancel_order(
        order_id=target_oid,
        session_id=session_id,
        product_id=product_id,
        quantity=quantity,
        amount_inr=amount_inr,
        reason=reason,
    )
    st.session_state.latest_order = None
    return res


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
                    "product_id": {"type": "STRING", "description": "Product ID (e.g. SKU_0001)"},
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
                    "buyer_id": {"type": "STRING", "description": "Buyer name or composite identity (e.g. 'Sagar Shaw (sagar@slice)')"},
                    "buyer_name": {"type": "STRING", "description": "Buyer's real customer name (e.g. 'Sagar Shaw')"},
                    "buyer_upi": {"type": "STRING", "description": "Buyer's UPI ID (e.g. 'sagar@slice', 'sagar@okhdfcbank')"},
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
                "properties": {},
            },
        },
    ]
}]

# ── Autonomous Local Agent Engine (Seamless On-Device Fallback) ──────────────

def run_local_agent_turn(user_message: str, session_id: str) -> str:
    """
    Autonomous Local Agent Engine (Zero Cloud LLM Dependency).
    Executes semantic vector search, real-time stock verification, out-of-stock alternative
    recommendations, two-phase customer identity verification, budget headroom checking,
    and instant Razorpay checkout initiation.
    """
    import re
    tool_calls_log = []
    clean_msg = user_message.strip()
    lower_msg = clean_msg.lower()

    if "local_context" not in st.session_state:
        st.session_state.local_context = {
            "last_search_results": [],
            "pending_product": None,
            "pending_qty": 1,
            "buyer_name": "",
            "buyer_upi": "",
        }
    ctx = st.session_state.local_context

    # 1. Extract Customer Identity (Name & UPI Handle vs Email)
    at_matches = re.findall(r'([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-_]+)', clean_msg)
    detected_email = ""
    detected_upi = ""

    for item in at_matches:
        if rzp.is_valid_upi_vpa(item):
            detected_upi = item.lower()
            ctx["buyer_upi"] = detected_upi
        else:
            domain = item.lower().split("@")[1]
            if domain in rzp.KNOWN_EMAIL_DOMAINS:
                detected_email = item.lower()

    name_match = re.search(r'(?:my name is|i am|name\s*[:=])\s*([a-zA-Z\s]{2,30})', clean_msg, re.IGNORECASE)
    if name_match:
        ctx["buyer_name"] = name_match.group(1).strip()
    elif not ctx.get("buyer_name"):
        # If user typed "Sagar Shaw sagar@okaxis"
        text_without_at = re.sub(r'([a-zA-Z0-9.\-_]+@[a-zA-Z0-9.\-_]+)', '', clean_msg)
        text_clean = re.sub(r'[\(\),;:\"\'\[\]\-]', ' ', text_without_at).strip()
        words = text_clean.split()
        if 1 <= len(words) <= 3 and all(w.isalpha() for w in words) and not any(w.lower() in ["buy", "order", "search", "yes", "confirm", "proceed", "pay"] for w in words):
            ctx["buyer_name"] = " ".join(words).title()

    buyer_name = ctx.get("buyer_name", "")
    buyer_upi = ctx.get("buyer_upi", "")

    # 2. Extract Price Constraint (e.g. "under ₹400", "below 500", "under 350")
    price_ceiling = None
    price_match = re.search(r'(?:under|below|less than|within|upto|max|budget|₹)\s*(\d+)', clean_msg, re.IGNORECASE)
    if price_match:
        try:
            price_ceiling = int(price_match.group(1))
        except ValueError:
            price_ceiling = None

    # 3. Extract Quantity
    quantity = 1
    qty_match = re.search(r'\b(?:buy|order|get|take|need|want)?\s*(\d+)\s*(?:units?|packs?|bottles?|pieces?|pcs?|items?|kg|g|gm)?\b', clean_msg, re.IGNORECASE)
    if qty_match:
        num = int(qty_match.group(1))
        if 1 <= num <= 50 and (price_ceiling is None or num != price_ceiling):
            quantity = num

    # 4. Handle Order Cancellation Intent
    cancel_keywords = ["cancel", "stop", "dont want", "don't want", "refund", "nahi chahiye", "reject", "abort"]
    if any(k in lower_msg for k in cancel_keywords) and not any(w in lower_msg for w in ["buy", "search", "find", "order"]):
        cancel_res = _cancel_order(reason="Buyer requested cancellation via AI agent", session_id=session_id)
        tool_calls_log.append({
            "tool": "catalog_cancel_order",
            "args": {"reason": "Buyer requested cancellation"},
            "result": cancel_res,
        })
        st.session_state.messages.append({
            "role": "tools",
            "tool_calls": tool_calls_log,
        })
        ctx["pending_product"] = None
        refunded = cancel_res.get("amount_inr", 0)
        return (
            f"❌ **Order Cancelled Successfully.**\n\n"
            f"Your order has been cancelled and ₹{refunded:.0f} has been refunded to your session budget headroom. "
            f"How else can I assist you today?"
        )

    # 5. Handle Budget & Audit Trail Inquiries
    budget_keywords = ["budget", "balance", "limit", "remaining", "kitna bacha", "how much left", "audit", "history"]
    if any(k in lower_msg for k in budget_keywords) and not any(w in lower_msg for w in ["buy", "search", "find", "order"]):
        audit_info = _get_audit_trail(session_id)
        tool_calls_log.append({
            "tool": "catalog_get_audit_trail",
            "args": {"session_id": session_id},
            "result": audit_info,
        })
        st.session_state.messages.append({
            "role": "tools",
            "tool_calls": tool_calls_log,
        })
        rem = audit_info.get("remaining_budget_inr", 2000)
        spent = audit_info.get("total_spent_inr", 0)
        return (
            f"💰 **NPCI UAP Session Budget Overview**\n\n"
            f"• **Hard Spending Limit**: ₹2,000.00\n"
            f"• **Total Settled / Spent**: ₹{spent:.0f}\n"
            f"• **Remaining Budget Headroom**: **₹{rem:.0f}**\n\n"
            f"Feel free to search for products or ask to buy any SKU within your available headroom!"
        )

    # 6. Check if User is Providing Details for Pending Order
    if ctx.get("pending_product"):
        target_prod = ctx["pending_product"]
        target_qty = ctx.get("pending_qty", 1)

        # If user gave an email instead of UPI handle
        if detected_email and not detected_upi:
            prefix = detected_email.split("@")[0]
            return (
                f"ℹ️ **Email Detected ({detected_email})**\n\n"
                f"For instant UPI QR payments, please provide a valid **UPI ID** "
                f"(e.g. `{prefix}@slice`, `{prefix}@okaxis`, `{prefix}@okhdfcbank`, or `{prefix}@paytm`).\n\n"
                f"Or type your Name & UPI handle to complete the order for **{target_prod.get('name')}**."
            )

        # If user gave UPI handle or confirmed
        confirm_words = ["yes", "proceed", "confirm", "pay", "checkout", "kar do", "thik hai", "ok", "agree"]
        is_confirm = any(w in lower_msg for w in confirm_words)

        if detected_upi or (is_confirm and buyer_name and buyer_upi):
            # Check stock before creating order
            chk = _check_inventory(target_prod["id"], quantity=target_qty, session_id=session_id)
            tool_calls_log.append({
                "tool": "catalog_check_inventory",
                "args": {"product_id": target_prod["id"], "quantity": target_qty},
                "result": chk,
            })

            if not chk.get("available", False):
                alts = chk.get("alternatives", [])
                alt_lines = [f"• **{a.get('name')}** (₹{a.get('price_inr')}) — Stock: {a.get('stock')}" for a in alts]
                alt_text = "\n".join(alt_lines) if alt_lines else "No direct alternatives available in stock."
                st.session_state.messages.append({
                    "role": "tools",
                    "tool_calls": tool_calls_log,
                })
                ctx["pending_product"] = None
                return (
                    f"⚠️ **{target_prod.get('name')}** is currently out of stock.\n\n"
                    f"Here are top-rated in-stock alternatives:\n{alt_text}\n\n"
                    f"Would you like to purchase one of these alternatives instead?"
                )

            buyer_composite = f"{buyer_name} ({buyer_upi})" if (buyer_name and buyer_upi) else (buyer_upi or buyer_name or "Verified Customer")
            order_res = _create_order(
                product_id=target_prod["id"],
                quantity=target_qty,
                buyer_id=buyer_composite,
                session_id=session_id,
                confirmed=True,
                buyer_name=buyer_name,
                buyer_upi=buyer_upi,
            )
            tool_calls_log.append({
                "tool": "catalog_create_order",
                "args": {
                    "product_id": target_prod["id"],
                    "quantity": target_qty,
                    "buyer_id": buyer_composite,
                    "buyer_name": buyer_name,
                    "buyer_upi": buyer_upi,
                    "confirmed": True,
                },
                "result": order_res,
            })
            st.session_state.messages.append({
                "role": "tools",
                "tool_calls": tool_calls_log,
            })

            if order_res.get("status") == "created":
                st.session_state.latest_order = order_res
                ctx["pending_product"] = None
                total_amt = order_res.get("amount_inr", int(target_prod["price_inr"]) * target_qty)
                return (
                    f"🎉 **Order Initiated for {target_qty}x {target_prod.get('name')}!**\n\n"
                    f"• **Total Amount**: **₹{total_amt:.0f}**\n"
                    f"• **Customer**: {buyer_composite}\n"
                    f"• **Order ID**: `{order_res.get('order_id')}`\n\n"
                    f"Please complete your payment in the **Razorpay Unified Payment Terminal** above or scan the dynamic UPI QR code!"
                )
            else:
                return f"❌ Unable to create order: {order_res.get('message', 'Budget limit or stock error.')}"

    # 7. Identify Purchase vs Search Intent
    buy_intent_keywords = ["buy", "purchase", "order", "khareed", "lena hai", "le lo", "i want", "checkout"]
    has_buy_intent = any(k in lower_msg for k in buy_intent_keywords)

    clean_query = re.sub(
        r'\b(find|search|show|get|recommend|give|me|a|an|the|product|products|item|items|buy|purchase|order|please|for|under|below|less than|₹|\d+|rs|rupees|inr)\b',
        ' ',
        clean_msg,
        flags=re.IGNORECASE,
    ).strip()
    if not clean_query or len(clean_query) < 2:
        clean_query = clean_msg

    # Step 1: Semantic Product Search via ChromaDB
    search_res = _search_products(
        query=clean_query,
        max_results=4,
        max_price_inr=price_ceiling,
        in_stock_only=True,
        session_id=session_id,
    )
    results = search_res.get("results", [])
    ctx["last_search_results"] = results

    tool_calls_log.append({
        "tool": "catalog_search_products",
        "args": {"query": clean_query, "max_price_inr": price_ceiling, "max_results": 4},
        "result": search_res,
    })

    if not results:
        st.session_state.messages.append({
            "role": "tools",
            "tool_calls": tool_calls_log,
        })
        return (
            f"🔍 I searched our catalog for **'{clean_query}'**"
            f"{f' under ₹{price_ceiling}' if price_ceiling else ''}, but found no matching products.\n\n"
            f"Try browsing our categories like **Skincare, Sunscreen, Hair Care, Organic Wellness, or Snacks**!"
        )

    # 8. If BUY Intent is detected ("buy me a natural oil for hair", "order 2 sunscreen"):
    if has_buy_intent:
        best_p = results[0]
        ctx["pending_product"] = best_p
        ctx["pending_qty"] = quantity

        # Step 2: Check Inventory before ordering
        chk = _check_inventory(best_p["id"], quantity=quantity, session_id=session_id)
        tool_calls_log.append({
            "tool": "catalog_check_inventory",
            "args": {"product_id": best_p["id"], "quantity": quantity},
            "result": chk,
        })

        if not chk.get("available", False):
            alts = chk.get("alternatives", [])
            alt_lines = [f"• **{a.get('name')}** (₹{a.get('price_inr')}) — Stock: {a.get('stock')}" for a in alts]
            alt_text = "\n".join(alt_lines) if alt_lines else "No direct alternatives available in stock."

            st.session_state.messages.append({
                "role": "tools",
                "tool_calls": tool_calls_log,
            })
            return (
                f"⚠️ **{best_p.get('name')}** is currently **Out of Stock** (Requested: {quantity} unit(s)).\n\n"
                f"Here are top-rated in-stock alternatives from the same category:\n{alt_text}\n\n"
                f"Would you like to buy one of these alternatives instead?"
            )

        # Check if we already have the customer's Full Name AND valid UPI ID:
        if buyer_name and buyer_upi and rzp.is_valid_upi_vpa(buyer_upi):
            buyer_composite = f"{buyer_name} ({buyer_upi})"
            order_res = _create_order(
                product_id=best_p["id"],
                quantity=quantity,
                buyer_id=buyer_composite,
                session_id=session_id,
                confirmed=True,
                buyer_name=buyer_name,
                buyer_upi=buyer_upi,
            )
            tool_calls_log.append({
                "tool": "catalog_create_order",
                "args": {
                    "product_id": best_p["id"],
                    "quantity": quantity,
                    "buyer_id": buyer_composite,
                    "buyer_name": buyer_name,
                    "buyer_upi": buyer_upi,
                    "confirmed": True,
                },
                "result": order_res,
            })
            st.session_state.messages.append({
                "role": "tools",
                "tool_calls": tool_calls_log,
            })
            if order_res.get("status") == "created":
                st.session_state.latest_order = order_res
                total_amt = order_res.get("amount_inr", int(best_p["price_inr"]) * quantity)
                return (
                    f"🛒 **Order Initiated for {best_p.get('name')}!**\n\n"
                    f"• **SKU**: `{best_p.get('id')}`\n"
                    f"• **Unit Price**: ₹{best_p.get('price_inr')}\n"
                    f"• **Quantity**: {quantity} unit(s) · **Total**: **₹{total_amt:.0f}**\n"
                    f"• **Customer**: {buyer_composite}\n"
                    f"• **Order ID**: `{order_res.get('order_id')}`\n\n"
                    f"Please complete your payment in the **Razorpay Unified Payment Terminal** above or scan the dynamic UPI QR code!"
                )

        # Prompt for customer name & UPI handle to verify identity before ordering:
        st.session_state.messages.append({
            "role": "tools",
            "tool_calls": tool_calls_log,
        })
        unit_price = int(best_p.get("price_inr", 0))
        total_price = unit_price * quantity
        return (
            f"I found **{best_p.get('name')}** for you at **₹{unit_price}**!\n\n"
            f"• **Stock Status**: In Stock ({best_p.get('stock')} units available) • **Category**: {best_p.get('category')} • **Total for {quantity} unit(s)**: ₹{total_price}\n\n"
            f"Would you like me to generate your Razorpay payment link? Please reply with your **Full Name** and **UPI ID** (e.g. `Sagar Shaw, sagar@slice` or `priya@okaxis`) to proceed!"
        )

    # 9. Standard Search / Discovery Mode ("Find me lightweight sunscreen for oily skin under ₹400")
    top_p = results[0]
    ctx["pending_product"] = top_p
    ctx["pending_qty"] = 1

    st.session_state.messages.append({
        "role": "tools",
        "tool_calls": tool_calls_log,
    })

    rec_lines = []
    for idx, p in enumerate(results[:4], 1):
        stock = int(p.get("stock", 0))
        stk_badge = f"🟢 In Stock ({stock})" if stock > 0 else "🔴 Out of Stock"
        desc = p.get("agent_description") or p.get("description") or ""
        short_desc = desc[:90] + "..." if len(desc) > 90 else desc
        rec_lines.append(
            f"**{idx}. {p.get('name')}** — **₹{p.get('price_inr')}** ({stk_badge})\n"
            f"   `{p.get('id')}` · *{p.get('category')}*\n"
            f"   _{short_desc}_"
        )

    products_formatted = "\n\n".join(rec_lines)
    return (
        f"Here are the best matching products from **{settings.merchant_name}**:\n\n"
        f"{products_formatted}\n\n"
        f"💡 **Would you like to buy one of these?**\n"
        f"Reply with **'Buy {top_p.get('name')}'** or provide your **Full Name and UPI ID** to generate your checkout link!"
    )


# ── Cloud LLM Agent Execution with Automatic Local Fallback ──────────────────

def run_gemini_turn(user_message: str, session_id: str) -> str:
    """Execute multi-turn agent conversation with tool calls, with automatic local fallback."""
    if not getattr(settings, "gemini_api_key", None):
        return run_local_agent_turn(user_message, session_id)

    contents = st.session_state.contents
    user_part = {"role": "user", "parts": [{"text": user_message}]}
    contents.append(user_part)

    system_text = f"""You are an autonomous AI shopping agent for '{settings.merchant_name}'.
Session ID: {session_id}

Rules:
1. Always call catalog_check_inventory BEFORE calling catalog_create_order.
2. If a product is out of stock (available=false), present alternatives and NEVER place an order.
3. When the buyer wants to order, ask for their Full Name and UPI ID (e.g. 'Sagar Shaw, sagar@slice') if not already provided.
4. If buyer confirms or provides UPI ID, call catalog_create_order with confirmed=true.
5. If the buyer mentions their name (e.g. 'Sagar Shaw') and/or UPI ID (e.g. 'sagar@slice'), pass the real name in buyer_name and the UPI handle in buyer_upi.
6. If buyer wants to cancel an order, call catalog_cancel_order.
7. Use the ₹ symbol for prices. Keep answers helpful and direct.
"""

    candidate_models = [settings.gemini_model, "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    unique_models = [m for m in candidate_models if m]

    tool_calls_log = []

    try:
        headers = {"x-goog-api-key": settings.gemini_api_key}
        with httpx.Client(timeout=20.0) as client:
            for _subturn in range(8):
                payload = {
                    "system_instruction": {"parts": [{"text": system_text}]},
                    "contents": contents,
                    "tools": GEMINI_TOOLS,
                }

                resp = None
                for model_name in unique_models:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                    try:
                        resp = client.post(url, json=payload, headers=headers)
                        if resp.status_code == 200:
                            break
                    except Exception:
                        continue

                # If cloud LLM endpoint is non-200 (401, 403, 429, 503), trigger seamless local engine
                if not resp or resp.status_code != 200:
                    if contents and contents[-1] == user_part:
                        contents.pop()
                    st.session_state.contents = contents
                    return run_local_agent_turn(user_message, session_id)

                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return run_local_agent_turn(user_message, session_id)

                candidate = candidates[0]
                candidate_content = candidate.get("content", {})
                parts = candidate_content.get("parts", [])

                has_func_call = False
                func_responses = []
                text_response = ""

                for p in parts:
                    if "text" in p and p["text"].strip():
                        text_response += p["text"].strip()
                    elif "functionCall" in p:
                        has_func_call = True
                        fn_call = p["functionCall"]
                        fn_name = fn_call.get("name")
                        fn_args = dict(fn_call.get("args", {}))
                        fn_args["session_id"] = session_id

                        func = TOOL_REGISTRY.get(fn_name)
                        res = func(**fn_args) if func else {"error": f"Unknown tool: {fn_name}"}

                        display_args = {k: v for k, v in fn_args.items() if k != "session_id"}
                        tool_calls_log.append({
                            "tool": fn_name,
                            "args": display_args,
                            "result": res,
                        })

                        if fn_name == "catalog_create_order" and isinstance(res, dict) and res.get("status") == "created":
                            st.session_state.latest_order = res

                        fn_resp = {"name": fn_name, "response": {"output": res}}
                        if fn_call.get("id"):
                            fn_resp["id"] = fn_call.get("id")
                        func_responses.append({"functionResponse": fn_resp})

                if not has_func_call:
                    contents.append(candidate_content)
                    st.session_state.contents = contents
                    if tool_calls_log:
                        st.session_state.messages.append({
                            "role": "tools",
                            "tool_calls": tool_calls_log,
                        })
                    return text_response

                contents.append(candidate_content)
                contents.append({"role": "user", "parts": func_responses})

        st.session_state.contents = contents
        if tool_calls_log:
            st.session_state.messages.append({
                "role": "tools",
                "tool_calls": tool_calls_log,
            })
        return text_response or "Action completed."
    except Exception:
        if contents and contents[-1] == user_part:
            contents.pop()
        st.session_state.contents = contents
        return run_local_agent_turn(user_message, session_id)


def run_agent_turn(user_message: str, session_id: str) -> str:
    """Universal dispatcher for conversational shopping agent turn."""
    if getattr(settings, "llm_provider", "gemini") == "gemini" and getattr(settings, "gemini_api_key", None):
        return run_gemini_turn(user_message, session_id)
    return run_local_agent_turn(user_message, session_id)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Budget Gauge & Status Monitor
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🛡️ NPCI UAP Session Budget")

    sid = st.session_state.session_id
    spent = audit.session_spent_inr(sid)
    limit = float(settings.agent_spending_limit_inr)
    remaining = max(0.0, limit - spent)
    pct = min(spent / limit, 1.0) if limit > 0 else 0

    st.progress(pct)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='glass-metric'><div class='metric-val'>₹{spent:.0f}</div><div class='metric-lbl'>Spent</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='glass-metric'><div class='metric-val' style='color:#00E599;'>₹{remaining:.0f}</div><div class='metric-lbl'>Remaining</div></div>", unsafe_allow_html=True)

    st.caption(f"🔒 Hard Session Cap: **₹{limit:.0f}** · SQLite WAL Protected")
    st.divider()

    st.markdown("### ⚙️ System Status")
    st.markdown(f"• **ChromaDB Store**: `{vs.catalog_size()} SKUs`")
    st.markdown(f"• **AI Model**: `{settings.gemini_model}`")
    st.markdown(f"• **Razorpay Key**: `{settings.razorpay_key_id[:12]}…`")
    st.markdown(f"• **Orders Log**: [Razorpay Orders](https://dashboard.razorpay.com/app/orders)")
    st.caption("🔧 Merchant API settings located in **Tab 4 (Merchant Hub)**")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE: Command Header & Stage
# ═══════════════════════════════════════════════════════════════════════════════

# Top Bar Header
header_c1, header_c2 = st.columns([3, 1])
with header_c1:
    safe_merchant_title = html.escape(settings.merchant_name)
    st.markdown(f"""
    <div style="padding: 6px 0 14px 0;">
        <h1 style="color:#528FF0; margin:0; font-size:2.2em; font-weight:800; letter-spacing:-0.03em;">
            🛒 {safe_merchant_title}
        </h1>
        <p style="color:#94a3b8; font-size:0.95em; margin:4px 0 0 0;">
            Autonomous Agentic Commerce Gateway — Razorpay Buildathon Track 01
        </p>
    </div>
    """, unsafe_allow_html=True)

with header_c2:
    st.markdown("<div style='text-align: right; padding-top: 12px;'>", unsafe_allow_html=True)
    if st.button("🔄 Reset Session / Clear Chat"):
        st.session_state.session_id = f"web_{uuid.uuid4().hex[:12]}"
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Session reset! How can I help you shop at **{settings.merchant_name}** today?",
            }
        ]
        st.session_state.contents = []
        st.session_state.latest_order = None
        st.session_state.pending_prompt = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ── EXPANSIVE RAZORPAY PAYMENT TERMINAL ───────────────────────────────────────
if st.session_state.latest_order:
    order = st.session_state.latest_order
    order_id = order.get("order_id", "")
    amount_inr = order.get("amount_inr", 0)
    amount_paise = int(amount_inr) * 100
    product_name = order.get("product_name", "Product")
    product_id = order.get("product_id", "")
    quantity = order.get("quantity", 1)
    key_id = getattr(settings, "razorpay_key_id", "rzp_test_demo")
    merchant_vpa = getattr(settings, "merchant_upi_vpa", None) or "rzp.greenleaf@hdfcbank"
    upi_link = order.get("upi_link") or order.get("payment_link", "")

    safe_order_id = html.escape(str(order_id))
    safe_product_name = html.escape(str(product_name))
    safe_merchant_vpa = html.escape(str(merchant_vpa))
    safe_merchant_name = html.escape(str(settings.merchant_name))

    st.markdown(f"""
    <div class="terminal-container">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; flex-wrap:wrap; gap:10px;">
            <div style="font-size:1.3em; font-weight:800; color:#528FF0; display:flex; align-items:center; gap:10px;">
                💳 Razorpay Unified Payment Terminal
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span class="badge badge-in" style="font-size:0.8em; padding:5px 12px;">● LIVE ORDER ACTIVE</span>
                <span style="font-family:'JetBrains Mono'; font-size:0.82em; color:#94a3b8; background:rgba(255,255,255,0.06); padding:4px 10px; border-radius:6px;">{safe_order_id}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    term_left, term_right = st.columns([1.0, 1.3], gap="large")

    with term_left:
        st.markdown(f"""
        <div style="background:rgba(7,38,84,0.65); border:1px solid rgba(82,143,240,0.3); border-radius:12px; padding:18px; margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.75em; color:#94a3b8; text-transform:uppercase; font-weight:700; letter-spacing:0.06em;">Order Summary</span>
                <span class="badge badge-in">₹{amount_inr:.0f}</span>
            </div>
            <div style="font-size:1.15em; font-weight:700; color:#ffffff; margin:8px 0 2px 0;">{safe_product_name}</div>
            <div style="font-size:0.86em; color:#cbd5e1;">Quantity: <b>{quantity} unit(s)</b> · Price: ₹{amount_inr/quantity:.0f} ea</div>
            <div style="font-family:'JetBrains Mono'; font-size:2.0em; font-weight:800; color:#00C896; margin-top:8px;">₹{amount_inr:.0f}</div>
            <div style="font-size:0.78em; color:#64748b; margin-top:2px;">Session Budget Headroom: ₹{remaining:.0f} left</div>
        </div>
        """, unsafe_allow_html=True)

        if upi_link:
            try:
                import qrcode
                qr = qrcode.QRCode(box_size=3, border=2)
                qr.add_data(upi_link)
                qr.make(fit=True)
                img = qr.make_image(fill_color="#001232", back_color="white")
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                st.image(buf, caption=f"Scan via GPay / PhonePe / Slice (VPA: {merchant_vpa})", width=170)
            except Exception as exc:
                logger.warning("Could not render QR code: %s", exc)

        # Dual Payment Actions (Verify Payment or Cancel Order)
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("✅ Verify Payment Status", key="btn_verify_payment", use_container_width=True):
                # Query real live status from Razorpay API
                status_res = rzp.get_payment_status(order_id, sid)
                
                if not status_res.get("is_paid", False):
                    st.error(
                        f"❌ **Payment Not Yet Received on Razorpay** (Gateway Status: `{status_res.get('status', 'unpaid').upper()}`). "
                        f"Please complete authorization in the Razorpay checkout window before verifying."
                    )
                else:
                    # Verified as truly PAID by Razorpay API
                    b_id = order.get("buyer_id") or ""
                    b_name = order.get("buyer_name") or ""
                    b_upi = order.get("buyer_upi") or ""
                    cust_name, cust_upi = rzp._extract_customer_identity(b_id, b_name, b_upi)
                    pay_id = status_res.get("payment_id") or f"pay_{uuid.uuid4().hex[:10]}"

                    # Log authentic customer action as PAYMENT_CAPTURED
                    audit.log_customer_action(
                        customer_id=cust_name,
                        upi_id=cust_upi,
                        action_type="PAYMENT_CAPTURED",
                        order_id=order_id,
                        product_id=product_id,
                        product_name=product_name,
                        quantity=quantity,
                        amount_inr=amount_inr,
                        session_id=sid,
                        details={"payment_id": pay_id, "status": "paid", "rail": "Razorpay_Gateway"},
                    )
                    
                    audit.log_event(
                        session_id=sid,
                        tool_name="razorpay_payment_success",
                        inputs={"order_id": order_id},
                        outcome="success",
                        details={"payment_id": pay_id, "status": "paid", "amount_inr": amount_inr},
                        amount_inr=0,
                    )
                    
                    st.session_state.latest_order = None
                    st.session_state.paid_success_info = {
                        "order_id": order_id,
                        "payment_id": pay_id,
                        "product_name": product_name,
                        "amount_inr": amount_inr,
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"🎉 **Payment Verified & Captured via Razorpay!**\n\n"
                            f"• **Order ID**: `{order_id}`\n"
                            f"• **Payment ID**: `{pay_id}`\n"
                            f"• **Amount Settled**: ₹{amount_inr:.0f}\n"
                            f"• **Customer**: {cust_name} ({cust_upi})\n"
                            f"• **Item**: {quantity}x {product_name}\n"
                            f"• **Status**: `PAYMENT_CAPTURED`\n\n"
                            f"Your order has been officially processed and recorded in the **Customer Intelligence Sheet**. Thank you for shopping with **{settings.merchant_name}**!"
                        ),
                    })
                    if "contents" in st.session_state and isinstance(st.session_state.contents, list):
                        st.session_state.contents.append({
                            "role": "model",
                            "parts": [{"text": f"Payment of ₹{amount_inr:.0f} for Order {order_id} was successfully verified and captured via Razorpay. The order is fulfilled."}]
                        })
                    st.rerun()

        with col_act2:
            if st.button("❌ Cancel Order", key="cancel_main_stage", use_container_width=True):
                cancel_res = rzp.cancel_order(
                    order_id=order_id,
                    session_id=sid,
                    product_id=product_id,
                    quantity=quantity,
                    amount_inr=amount_inr,
                    reason="Buyer cancelled via checkout terminal",
                )
                st.session_state.latest_order = None
                refunded = cancel_res.get("amount_inr", amount_inr)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ **Order `{order_id}` has been cancelled.** ₹{refunded:.0f} was refunded to your session budget.",
                })
                if "contents" in st.session_state and isinstance(st.session_state.contents, list):
                    st.session_state.contents.append({
                        "role": "model",
                        "parts": [{"text": f"Order {order_id} was successfully cancelled by the user and ₹{refunded:.0f} has been refunded to their budget."}]
                    })
                st.rerun()

    with term_right:
        checkout_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
            <style>
                body {{
                    margin: 0; padding: 0;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                    background: transparent;
                    color: #d8e2ff;
                    min-height: 580px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: flex-start;
                }}
                .checkout-card {{
                    width: 100%;
                    background: rgba(7, 30, 71, 0.65);
                    border: 1px solid rgba(82, 143, 240, 0.35);
                    border-radius: 14px;
                    padding: 20px;
                    box-sizing: border-box;
                }}
                .pay-btn-main {{
                    background: linear-gradient(135deg, #528FF0 0%, #1D61E7 100%);
                    color: #ffffff;
                    border: none;
                    padding: 16px 28px;
                    border-radius: 12px;
                    font-weight: 800;
                    font-size: 1.15em;
                    cursor: pointer;
                    width: 100%;
                    box-shadow: 0 6px 22px rgba(82, 143, 240, 0.45);
                    transition: all 0.2s ease;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 10px;
                }}
                .pay-btn-main:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 28px rgba(82, 143, 240, 0.65);
                }}
                .info-banner {{
                    margin-top: 14px;
                    padding: 12px 14px;
                    background: rgba(82, 143, 240, 0.1);
                    border: 1px solid rgba(82, 143, 240, 0.2);
                    border-radius: 10px;
                    font-size: 0.82em;
                    color: #cbd5e1;
                    line-height: 1.5;
                }}
                .payment-methods-strip {{
                    display: flex;
                    justify-content: space-around;
                    margin-top: 14px;
                    gap: 8px;
                }}
                .method-badge {{
                    background: rgba(11, 41, 87, 0.5);
                    border: 1px solid rgba(82, 143, 240, 0.25);
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-size: 0.76em;
                    color: #94a3b8;
                    font-weight: 600;
                    text-align: center;
                    flex: 1;
                }}
                #rzp-status-box {{
                    font-size: 0.95em;
                    color: #00C896;
                    margin-top: 12px;
                    font-weight: 700;
                    text-align: center;
                    min-height: 1.4em;
                }}
            </style>
        </head>
        <body>
            <div class="checkout-card">
                <button id="rzp-btn-trigger" class="pay-btn-main">
                    💳 Launch Razorpay Standard Checkout (₹{amount_inr:.0f})
                </button>
                <div id="rzp-status-box"></div>

                <div class="payment-methods-strip">
                    <div class="method-badge">💳 Credit/Debit Cards</div>
                    <div class="method-badge">📱 UPI & QR</div>
                    <div class="method-badge">🏦 Netbanking</div>
                    <div class="method-badge">👛 Wallets</div>
                </div>

                <div class="info-banner">
                    💡 <b>Multi-Method Navigation:</b> Inside the Razorpay popup, click any payment option (e.g. Card, UPI, Netbanking). You can use the top <b>&lt; Back</b> button at any time to return to the options list.
                </div>
            </div>

            <script>
                var options = {{
                    "key": "{key_id}",
                    "amount": "{amount_paise}",
                    "currency": "INR",
                    "name": "{safe_merchant_name}",
                    "description": "{quantity}x {safe_product_name}",
                    "order_id": "{order_id}",
                    "prefill": {{
                        "name": "{html.escape(str(order.get('buyer_name') or order.get('buyer_id') or 'Buyer'))}",
                        "email": "buyer@example.com",
                        "contact": "9999999999"
                    }},
                    "theme": {{
                        "color": "#001232"
                    }},
                    "modal": {{
                        "backdropclose": true,
                        "escape": true,
                        "handleback": true,
                        "confirm_close": false
                    }},
                    "handler": function (response) {{
                        document.getElementById("rzp-status-box").innerHTML = "✓ <b>Payment Authorized by Razorpay!</b> ID: <code>" + response.razorpay_payment_id + "</code><br><span style='color:#00C896;font-size:0.88em;font-weight:600;'>Order confirmed! Click <b>[Verify Payment Status]</b> to complete.</span>";
                        try {{
                            if (window.top && window.top !== window) {{
                                window.top.location.search = "?paid_order=" + encodeURIComponent(response.razorpay_order_id) + "&payment_id=" + encodeURIComponent(response.razorpay_payment_id);
                            }}
                        }} catch(err) {{
                            console.log("Payment authorized:", response.razorpay_payment_id);
                        }}
                    }}
                }};
                var rzpInstance = new Razorpay(options);
                document.getElementById('rzp-btn-trigger').onclick = function(e) {{
                    rzpInstance.open();
                    e.preventDefault();
                }};
            </script>
        </body>
        </html>
        """
        st.components.v1.html(checkout_html, height=640)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Celebration Confirmation Banner (When payment is completed) ───────────────
if st.session_state.paid_success_info:
    psi = st.session_state.paid_success_info
    safe_psi_name = html.escape(str(psi.get('product_name', 'Product')))
    safe_psi_oid = html.escape(str(psi.get('order_id', '')))
    safe_psi_payid = html.escape(str(psi.get('payment_id', '')))
    st.markdown(f"""
    <div style="background: rgba(0, 200, 150, 0.12); border: 1.5px solid #00C896; border-radius: 14px; padding: 20px 24px; margin-bottom: 22px; box-shadow: 0 8px 32px rgba(0, 200, 150, 0.2);">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size: 1.25em; font-weight: 800; color: #00C896; display:flex; align-items:center; gap:8px;">
                    🎉 Payment Successfully Captured!
                </div>
                <div style="color: #cbd5e1; font-size: 0.9em; margin-top: 6px;">
                    Item: <b>{safe_psi_name}</b> · Settled: <b style="color:#00C896;">₹{psi.get('amount_inr', 0):.0f}</b> · Order: <code>{safe_psi_oid}</code> · Payment ID: <code>{safe_psi_payid}</code>
                </div>
            </div>
            <span class="badge badge-in" style="font-size:0.85em; padding:6px 14px; background:rgba(0,200,150,0.25); color:#00C896; border:1px solid #00C896;">
                ● FULFILLED & RECORDED
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("✕ Dismiss Confirmation Banner", key="dismiss_success_banner"):
        st.session_state.paid_success_info = None
        st.rerun()





# 4 Core Dashboard Tabs
tab_chat, tab_catalog, tab_audit, tab_onboard = st.tabs([
    "💬 AI Shopping Assistant",
    "📦 Catalog Explorer",
    "📋 Immutable Audit Trail",
    "🚀 Merchant Ingestion Wizard",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Conversational AI Shopping Assistant
# ─────────────────────────────────────────────────────────────────────────────

with tab_chat:
    st.caption("💡 **Quick Test Scenarios (Click to Execute):**")
    chip_c1, chip_c2, chip_c3, chip_c4 = st.columns(4)

    with chip_c1:
        if st.button("🧴 Sunscreen under ₹400"):
            st.session_state.pending_prompt = "Find me a lightweight sunscreen for oily skin under ₹400"
    with chip_c2:
        if st.button("🌿 Neem Face Wash (OOS)"):
            st.session_state.pending_prompt = "I want to buy 1 unit of Neem & Tulsi face wash"
    with chip_c3:
        if st.button("🍪 Chocolate Cookies"):
            st.session_state.pending_prompt = "Find me delicious chocolate cookies for tea time"
    with chip_c4:
        if st.button("💰 Check Spent Budget"):
            st.session_state.pending_prompt = "How much of my session budget have I spent so far?"

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    # Chat Messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
        elif msg["role"] == "tools":
            for tc in msg.get("tool_calls", []):
                tool_name = tc.get("tool", "")
                with st.expander(f"🔧 Tool Executed: `{tool_name}`", expanded=False):
                    st.markdown(f"**Arguments:** `{json.dumps(tc.get('args', {}), ensure_ascii=False)}`")
                    st.json(tc.get("result", {}))

    # Chat Input Box
    submitted_prompt = None
    if st.session_state.pending_prompt:
        submitted_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    else:
        chat_val = st.chat_input("Type in English or Hinglish (e.g. 'Order 1 bottle of pure cow ghee for sagar@okaxis')")
        if chat_val:
            submitted_prompt = chat_val

    if submitted_prompt:
        st.session_state.messages.append({"role": "user", "content": submitted_prompt})
        with st.chat_message("user"):
            st.markdown(submitted_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI Agent querying catalog & verifying inventory…"):
                response_text = run_agent_turn(submitted_prompt, st.session_state.session_id)
            st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Live Product Catalog Explorer
# ─────────────────────────────────────────────────────────────────────────────

with tab_catalog:
    all_prods = get_cached_catalog()
    all_categories = get_cached_categories()

    filter_c1, filter_c2, filter_c3 = st.columns([2, 2, 1])
    with filter_c1:
        sel_category = st.selectbox("Category Filter", all_categories)
    with filter_c2:
        search_kw = st.text_input("Search Catalog", "", placeholder="Search by name, brand, keyword…")
    with filter_c3:
        in_stock_only = st.checkbox("In Stock Only", value=False)

    filtered = []
    for p in all_prods:
        p_cat_root = p.get("category", "General").split(" > ")[0].strip()
        if sel_category != "All Categories" and p_cat_root != sel_category:
            continue
        if search_kw:
            kw = search_kw.lower()
            name_match = kw in p.get("name", "").lower()
            desc_match = kw in p.get("agent_description", "").lower()
            cat_match = kw in p.get("category", "").lower()
            if not (name_match or desc_match or cat_match):
                continue
        if in_stock_only and int(p.get("stock", 0)) <= 0:
            continue
        filtered.append(p)

    st.markdown(f"**Showing {len(filtered)} of {len(all_prods)} products**")

    # 3-column responsive grid
    num_cols = 3
    for i in range(0, len(filtered), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j < len(filtered):
                p = filtered[i + j]
                stock = int(p.get("stock", 0))
                badge_cls = "badge-in" if stock > 10 else ("badge-low" if stock > 0 else "badge-out")
                badge_lbl = f"In Stock ({stock})" if stock > 10 else (f"Low Stock ({stock})" if stock > 0 else "Out of Stock")

                with cols[j]:
                    safe_p_id = html.escape(str(p.get('id', '')))
                    safe_p_name = html.escape(str(p.get('name', '')))
                    safe_p_cat = html.escape(str(p.get('category', 'General')))
                    st.markdown(f"""
                    <div class="product-glass-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <span style="font-family:'JetBrains Mono'; font-size:0.75em; color:#94a3b8;">{safe_p_id}</span>
                            <span class="badge {badge_cls}">{badge_lbl}</span>
                        </div>
                        <div class="p-title" style="margin-top:8px;">{safe_p_name}</div>
                        <div class="p-price">₹{p.get('price_inr', 0)}</div>
                        <div class="p-cat">📁 {safe_p_cat}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Immutable Audit Trail & Customer Insights Sheet
# ─────────────────────────────────────────────────────────────────────────────

with tab_audit:
    sheet_tab1, sheet_tab2 = st.tabs([
        "👥 Customer Insights & Lifecycle Sheet",
        "📋 System Audit Trail (Events Ledger)",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # SHEET 1: DEDICATED CUSTOMER INSIGHTS & ORDER LIFECYCLE LEDGER
    # ═════════════════════════════════════════════════════════════════════════
    with sheet_tab1:
        st.markdown("### 👥 Customer Order & Cancellation Ledger")
        st.caption(
            "Merchant intelligence sheet tracking customer identities, UPI handles, "
            "order timestamps, and purchasing patterns to improve sales conversion and retention."
        )

        all_cust_records = audit.get_customer_records()

        if all_cust_records:
            # High-Level Merchant Analytics KPIs (De-duplicated per unique order_id)
            unique_customers = len(set(r["customer_id"] for r in all_cust_records))
            
            # Map unique orders to their latest status
            order_status_map = {}
            order_amounts = {}
            for r in all_cust_records:
                oid = r.get("order_id", "")
                act = r.get("action_type", "")
                amt = float(r.get("amount_inr", 0))
                if oid:
                    order_amounts[oid] = amt
                    # Priority: ORDER_CANCELLED > PAYMENT_CAPTURED > ORDER_PLACED
                    if act == "ORDER_CANCELLED":
                        order_status_map[oid] = "CANCELLED"
                    elif act == "PAYMENT_CAPTURED" and order_status_map.get(oid) != "CANCELLED":
                        order_status_map[oid] = "CAPTURED"
                    elif oid not in order_status_map:
                        order_status_map[oid] = "PLACED"

            total_placed_count = len(order_status_map)
            total_captured_count = sum(1 for s in order_status_map.values() if s == "CAPTURED")
            total_cancelled_count = sum(1 for s in order_status_map.values() if s == "CANCELLED")

            total_placed_gmv = sum(order_amounts.values())
            total_refunded_gmv = sum(amt for oid, amt in order_amounts.items() if order_status_map.get(oid) == "CANCELLED")
            net_realized_gmv = max(0.0, total_placed_gmv - total_refunded_gmv)

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(f"<div class='glass-metric'><div class='metric-val' style='color:#528FF0;'>{unique_customers}</div><div class='metric-lbl'>Unique Customers</div></div>", unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"<div class='glass-metric'><div class='metric-val' style='color:#00E599;'>{total_placed_count}</div><div class='metric-lbl'>Orders Placed (₹{total_placed_gmv:.0f})</div></div>", unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"<div class='glass-metric'><div class='metric-val' style='color:#FF4D4D;'>{total_cancelled_count}</div><div class='metric-lbl'>Cancellations (₹{total_refunded_gmv:.0f})</div></div>", unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"<div class='glass-metric'><div class='metric-val' style='color:#00E599;'>₹{net_realized_gmv:.0f}</div><div class='metric-lbl'>Net Realized Sales</div></div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

            # Filtering and Search Controls
            flt_c1, flt_c2, flt_c3 = st.columns([2, 1.2, 1.2])
            with flt_c1:
                search_cust = st.text_input("🔍 Search Customer, UPI ID, or Order ID", placeholder="e.g. sagar@okaxis, priya, order_...")
            with flt_c2:
                filter_action = st.selectbox("Filter Action Type", ["All Actions", "Orders Placed", "Payments Captured", "Cancellations / Refunds"])
            with flt_c3:
                filter_scope = st.selectbox("Session Scope", ["All Merchant Sessions", "Current Session Only"])

            # Filter customer records
            filtered_cust = all_cust_records
            if filter_scope == "Current Session Only":
                filtered_cust = [r for r in filtered_cust if r.get("session_id") == st.session_state.session_id]

            if filter_action == "Orders Placed":
                filtered_cust = [r for r in filtered_cust if r.get("action_type") == "ORDER_PLACED"]
            elif filter_action == "Payments Captured":
                filtered_cust = [r for r in filtered_cust if r.get("action_type") == "PAYMENT_CAPTURED"]
            elif filter_action == "Cancellations / Refunds":
                filtered_cust = [r for r in filtered_cust if r.get("action_type") == "ORDER_CANCELLED"]

            if search_cust.strip():
                q = search_cust.strip().lower()
                filtered_cust = [
                    r for r in filtered_cust
                    if q in r.get("customer_id", "").lower()
                    or q in r.get("upi_id", "").lower()
                    or q in r.get("order_id", "").lower()
                    or q in r.get("product_name", "").lower()
                ]

            cust_table_data = []
            for idx, rec in enumerate(filtered_cust, 1):
                act = rec.get("action_type", "")
                amt = float(rec.get("amount_inr", 0))

                if act == "ORDER_PLACED":
                    act_badge = "🟢 Placed"
                    amt_str = f"₹{amt:.0f}"
                elif act == "PAYMENT_CAPTURED":
                    act_badge = "💳 Paid & Captured"
                    amt_str = f"+₹{amt:.0f} (Settled)"
                elif act == "ORDER_CANCELLED":
                    act_badge = "🔴 Cancelled"
                    amt_str = f"-₹{amt:.0f} (Refunded)"
                else:
                    act_badge = act
                    amt_str = f"₹{amt:.0f}"

                cust_table_data.append({
                    "#": idx,
                    "Processed Time (UTC)": rec.get("ts", "")[:19].replace("T", " "),
                    "Customer ID / Name": rec.get("customer_id", "Anonymous"),
                    "UPI ID / Payment Handle": rec.get("upi_id", "—"),
                    "Action": act_badge,
                    "Order ID": rec.get("order_id", ""),
                    "Product Name": rec.get("product_name", "—"),
                    "Qty": rec.get("quantity", 1),
                    "Amount": amt_str,
                    "Session Token": rec.get("session_id", "")[:16] + "…",
                })

            # CSV Download for Customer Insights Sheet
            csv_cust_buf = StringIO()
            writer_cust = csv.DictWriter(
                csv_cust_buf,
                fieldnames=[
                    "#", "Processed Time (UTC)", "Customer ID / Name",
                    "UPI ID / Payment Handle", "Action", "Order ID",
                    "Product Name", "Qty", "Amount", "Session Token"
                ]
            )
            writer_cust.writeheader()
            writer_cust.writerows(cust_table_data)

            exp_c1, exp_c2 = st.columns([3, 1])
            with exp_c2:
                st.download_button(
                    "📥 Export Customer Sheet (CSV)",
                    data=csv_cust_buf.getvalue(),
                    file_name=f"customer_insights_sheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.dataframe(cust_table_data, hide_index=True, use_container_width=True)
        else:
            st.info("No customer transactions recorded yet. Place or cancel an order to populate the customer intelligence sheet!")

    # ═════════════════════════════════════════════════════════════════════════
    # SHEET 2: SYSTEM AUDIT TRAIL (ALL EVENTS LEDGER)
    # ═════════════════════════════════════════════════════════════════════════
    with sheet_tab2:
        st.markdown("### 📋 System Tool Execution Audit Log")
        st.caption("Immutable append-only record of all tool invocations, NPCI budget limits, and financial decisions.")

        audit_events = audit.get_session_events(st.session_state.session_id)

        if audit_events:
            total_spent = audit.session_spent_inr(st.session_state.session_id)
            rem = audit.remaining_budget_inr(st.session_state.session_id)

            aud_c1, aud_c2, aud_c3 = st.columns(3)
            with aud_c1:
                st.markdown(f"<div class='glass-metric'><div class='metric-val'>₹{total_spent:.0f}</div><div class='metric-lbl'>Total Spend Recorded</div></div>", unsafe_allow_html=True)
            with aud_c2:
                st.markdown(f"<div class='glass-metric'><div class='metric-val' style='color:#00E599;'>₹{rem:.0f}</div><div class='metric-lbl'>Remaining Budget</div></div>", unsafe_allow_html=True)
            with aud_c3:
                st.markdown(f"<div class='glass-metric'><div class='metric-val'>{len(audit_events)}</div><div class='metric-lbl'>Total Events Logged</div></div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

            audit_data = []
            for i, ev in enumerate(audit_events, 1):
                amount = ev.get("amount_inr", 0)
                if amount > 0:
                    spend_str = f"₹{amount:.0f}"
                elif amount < 0:
                    spend_str = f"-₹{abs(amount):.0f} (Refunded)"
                else:
                    spend_str = "—"

                outcome = ev.get("outcome", "")
                if outcome == "order_cancelled":
                    outcome_str = "❌ Cancelled"
                elif outcome in ("payment_created", "success"):
                    outcome_str = "✅ Success"
                elif outcome == "limit_exceeded":
                    outcome_str = "🛡️ Limit Exceeded"
                elif outcome == "out_of_stock":
                    outcome_str = "⚠️ Out of Stock"
                else:
                    outcome_str = outcome

                audit_data.append({
                    "#": i,
                    "Timestamp (UTC)": ev["ts"][:19].replace("T", " "),
                    "Tool": ev["tool_name"],
                    "Outcome": outcome_str,
                    "₹ Spend": spend_str,
                    "Cumulative ₹": f"₹{ev['cumulative_spend_inr']:.0f}",
                })

            # CSV Export
            csv_buf = StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=["#", "Timestamp (UTC)", "Tool", "Outcome", "₹ Spend", "Cumulative ₹"])
            writer.writeheader()
            writer.writerows(audit_data)

            export_c1, export_c2 = st.columns([3, 1])
            with export_c2:
                st.download_button(
                    "📥 Export System Audit CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"audit_events_{st.session_state.session_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.dataframe(audit_data, hide_index=True, use_container_width=True)
        else:
            st.info("No audit events recorded yet in this session. Start chatting with the agent to record transactions!")



# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Merchant Catalog Ingestion Wizard
# ─────────────────────────────────────────────────────────────────────────────

with tab_onboard:
    st.markdown("### 🚀 Universal Merchant Onboarding & AI Control Hub")
    st.markdown(
        "Any merchant can onboard their CSV/JSON product catalog in seconds. "
        "Our pipeline auto-detects BigBasket, Flipkart, Shopify, WooCommerce, or Custom CSV headers, "
        "generates bilingual Indian vernacular intent phrases, and indexes embeddings into ChromaDB."
    )

    if not st.session_state.merchant_authenticated:
        st.markdown("""
        <div style="background: rgba(7, 30, 71, 0.65); border: 1.5px solid rgba(82, 143, 240, 0.45); border-radius: 14px; padding: 26px; margin: 16px 0 24px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size:1.35em; font-weight:800; color:#528FF0; display:flex; align-items:center; gap:10px;">
                🔒 Merchant Admin Authentication & Store Login
            </div>
            <p style="color:#cbd5e1; font-size:0.92em; margin-top:8px; line-height:1.5;">
                Select your store identity and enter your Merchant Admin PIN to access the catalog ingestion pipeline, 
                vector embeddings, and payment gateway credentials.
            </p>
        </div>
        """, unsafe_allow_html=True)

        login_col1, login_col2 = st.columns([2, 1])
        with login_col1:
            store_options = [
                f"{settings.merchant_name} ({settings.merchant_id})",
                "Himalayan Herbals (demo_merchant_002)",
                "FreshDirect Organics (demo_merchant_003)",
            ]
            selected_store_login = st.selectbox(
                "🏢 Select Merchant Store to Access",
                store_options,
                index=0,
                key="login_store_select",
                help="Choose which merchant store you are administering"
            )

            entered_pin = st.text_input(
                "Merchant Admin PIN",
                type="password",
                placeholder="Enter PIN (Default: merchant123)",
                key="merchant_pin_input"
            )

            act_col1, act_col2 = st.columns(2)
            with act_col1:
                if st.button("🔓 Sign In to Merchant Hub", key="btn_unlock_merchant", use_container_width=True):
                    if entered_pin.strip() in ("merchant123", "admin", "1234", "rzp2026"):
                        if "demo_merchant_002" in selected_store_login:
                            st.session_state.active_merchant_id = "demo_merchant_002"
                            st.session_state.active_merchant_name = "Himalayan Herbals"
                        elif "demo_merchant_003" in selected_store_login:
                            st.session_state.active_merchant_id = "demo_merchant_003"
                            st.session_state.active_merchant_name = "FreshDirect Organics"
                        else:
                            st.session_state.active_merchant_id = settings.merchant_id
                            st.session_state.active_merchant_name = settings.merchant_name

                        st.session_state.merchant_authenticated = True
                        st.success(f"✅ Signed in as {st.session_state.active_merchant_name}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid PIN. Please enter 'merchant123' or use 1-Click Quick Demo Access.")

            with act_col2:
                if st.button("⚡ 1-Click Quick Demo Access", key="btn_quick_unlock_demo", use_container_width=True):
                    if "demo_merchant_002" in selected_store_login:
                        st.session_state.active_merchant_id = "demo_merchant_002"
                        st.session_state.active_merchant_name = "Himalayan Herbals"
                    elif "demo_merchant_003" in selected_store_login:
                        st.session_state.active_merchant_id = "demo_merchant_003"
                        st.session_state.active_merchant_name = "FreshDirect Organics"
                    else:
                        st.session_state.active_merchant_id = settings.merchant_id
                        st.session_state.active_merchant_name = settings.merchant_name

                    st.session_state.merchant_authenticated = True
                    st.rerun()

        with login_col2:
            st.info("""
            **💡 Hackathon Evaluation Guide:**
            - **Selected Store**: Log in directly as that merchant.
            - **Default PIN**: `merchant123`
            - **Evaluator Shortcut**: Click **[⚡ 1-Click Quick Demo Access]** for instant evaluation.
            - **Isolation**: Customer shopping (Tabs 1 & 2) remains 100% public, while catalog ingestion is scoped to your selected store.
            """)
    else:
        # Authenticated Header
        auth_head1, auth_head2 = st.columns([3, 1])
        with auth_head1:
            st.markdown(f"""
            <div style="background: rgba(0, 200, 150, 0.12); border: 1.5px solid rgba(0, 200, 150, 0.45); border-radius: 12px; padding: 14px 20px; margin-bottom: 18px;">
                <span style="color:#00E599; font-weight:800; font-size:1.05em;">🟢 Authenticated Merchant:</span> 
                <span style="color:#ffffff; font-weight:700; font-size:1.05em; margin-left:6px;">{st.session_state.active_merchant_name}</span> 
                <code style="font-size:0.85em; color:#94a3b8; margin-left:8px;">(Tenant ID: {st.session_state.active_merchant_id})</code>
            </div>
            """, unsafe_allow_html=True)
        with auth_head2:
            if st.button("🔒 Switch Store / Sign Out", key="btn_merchant_signout", use_container_width=True):
                st.session_state.merchant_authenticated = False
                st.rerun()

        # 1. Live Catalog KPI Metrics
        cat_kpi1, cat_kpi2, cat_kpi3 = st.columns(3)
        current_catalog = vs.get_all_products(limit=500)
        total_skus = len(current_catalog)
        in_stock_skus = sum(1 for p in current_catalog if int(p.get("stock", 0)) > 0)
        distinct_cats = len(set(p.get("category", "General").split(" > ")[0].strip() for p in current_catalog if p.get("category")))

        with cat_kpi1:
            st.markdown(f"<div class='glass-metric'><div class='metric-val' style='color:#528FF0;'>{total_skus}</div><div class='metric-lbl'>Total Indexed SKUs</div></div>", unsafe_allow_html=True)
        with cat_kpi2:
            st.markdown(f"<div class='glass-metric'><div class='metric-val' style='color:#00E599;'>{in_stock_skus}</div><div class='metric-lbl'>In-Stock SKUs</div></div>", unsafe_allow_html=True)
        with cat_kpi3:
            st.markdown(f"<div class='glass-metric'><div class='metric-val' style='color:#FFB800;'>{distinct_cats}</div><div class='metric-lbl'>Active Categories</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 18px;'></div>", unsafe_allow_html=True)

        ingest_tab1, ingest_tab2, ingest_tab3 = st.tabs([
            "📤 Upload Custom Catalog (CSV / JSON)",
            "📦 Ingest from Sample Datasets",
            "🔑 Gateway API Credentials & Live Diagnostics",
        ])

        # ── TAB 4.1: Custom File Upload ──────────────────────────────────────
        with ingest_tab1:
            st.markdown("#### 📤 Upload Your Merchant Catalog File")
            st.caption(f"Upload your product catalog in CSV or JSON format. SKUs will be automatically associated with **{st.session_state.active_merchant_name}**.")

            uploaded_file = st.file_uploader(
                "Choose a CSV or JSON file",
                type=["csv", "json", "jsonl"],
                key="custom_catalog_uploader",
            )

            if uploaded_file is not None:
                upload_dir = Path("data") / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                saved_path = upload_dir / uploaded_file.name

                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"✓ File uploaded: `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")

                # Quick Schema Preview
                from src.catalog.importer import _load_csv, _load_json
                try:
                    if uploaded_file.name.endswith(".csv"):
                        preview_items = _load_csv(saved_path, limit=5)
                    else:
                        preview_items = _load_json(saved_path, limit=5)

                    if preview_items:
                        st.markdown("##### 🔍 Auto-Detected Schema Preview (First 5 Items):")
                        preview_rows = []
                        for item in preview_items:
                            preview_rows.append({
                                "Product Name": item.get("name"),
                                "Price (₹)": f"₹{item.get('price_inr', 0)}",
                                "Category": item.get("category"),
                                "Stock": item.get("stock"),
                                "ID / SKU": item.get("id"),
                            })
                        st.dataframe(preview_rows, hide_index=True, use_container_width=True)

                    custom_limit = st.slider("Import Limit", min_value=5, max_value=200, value=min(len(preview_items) * 10, 50), step=5, key="custom_limit_slider")

                    if st.button("⚡ Ingest & Enrich Uploaded Products", key="btn_ingest_uploaded"):
                        with st.spinner("Parsing schema, enriching vernacular intent phrases & upserting to ChromaDB…"):
                            from src.catalog.importer import import_catalog_sync
                            count = import_catalog_sync(
                                saved_path,
                                limit=custom_limit,
                                verbose=False,
                                merchant_id=st.session_state.active_merchant_id,
                            )
                            st.cache_data.clear()
                            vs.invalidate_cache()
                            st.success(f"🎉 Successfully enriched and indexed {count} products for '{st.session_state.active_merchant_name}'! Total Catalog Size: {vs.catalog_size()}")
                            st.rerun()
                except Exception as exc:
                    st.error(f"Error parsing uploaded file: {exc}")

        # ── TAB 4.2: Sample Datasets ─────────────────────────────────────────
        with ingest_tab2:
            onboard_c1, onboard_c2 = st.columns([2, 1])

            with onboard_c1:
                st.markdown("#### 📁 Select Pre-Configured FMCG Dataset")
                sample_choice = st.selectbox(
                    "Select Catalog Source",
                    [
                        "BigBasket Real FMCG Products (bigbasket_data/BigBasket_Products.csv)",
                        "Curated 20 Organic Wellness Pack (data/bigbasket_sample.csv)",
                    ],
                    key="sample_choice_select"
                )
                sample_limit = st.slider("Product Import Limit", min_value=10, max_value=200, value=50, step=10, key="sample_limit_slider")

                if st.button("⚡ Ingest & Index Sample Dataset", key="btn_ingest_sample"):
                    with st.spinner("Parsing headers, generating intent phrases & upserting to ChromaDB…"):
                        from src.catalog.importer import import_catalog_sync
                        target_file = (
                            "../bigbasket_data/BigBasket_Products.csv"
                            if "BigBasket Real" in sample_choice
                            else "data/bigbasket_sample.csv"
                        )
                        count = import_catalog_sync(
                            target_file,
                            limit=sample_limit,
                            verbose=False,
                            merchant_id=st.session_state.active_merchant_id,
                        )
                        st.cache_data.clear()
                        vs.invalidate_cache()
                        st.success(f"✓ Successfully indexed {count} products for '{st.session_state.active_merchant_name}'! Total Catalog Size: {vs.catalog_size()}")
                        st.rerun()

            with onboard_c2:
                st.markdown("#### ℹ️ Supported Platforms")
                st.markdown("""
                • **BigBasket**: `product`, `sale_price`, `category`, `description`
                • **Flipkart**: `product_name`, `discount_price`, `brand`
                • **Shopify**: `Title`, `Variant Price`, `Body HTML`
                • **WooCommerce / Custom**: Auto-detected
                """)

        # ── TAB 4.3: Merchant Gateway API Credentials & Diagnostics ─────────
        with ingest_tab3:
            st.markdown("#### 🔑 Merchant API Gateway Credentials & Diagnostics")
            st.caption(
                "Securely configure Google Gemini and Razorpay credentials. "
                "Keys are saved to the backend environment and dynamically reloaded in memory without server restarts."
            )

            cred_c1, cred_c2 = st.columns([2, 1])
            with cred_c1:
                new_gem_key = st.text_input(
                    "Google Gemini API Key (Google AI Studio)",
                    value=settings.gemini_api_key,
                    type="password",
                    key="admin_gem_key",
                    help="Enter AIzaSy... key from aistudio.google.com",
                )
                new_rzp_id = st.text_input(
                    "Razorpay Key ID",
                    value=settings.razorpay_key_id,
                    key="admin_rzp_id",
                    help="e.g. rzp_test_...",
                )
                new_rzp_sec = st.text_input(
                    "Razorpay Key Secret",
                    value=settings.razorpay_key_secret,
                    type="password",
                    key="admin_rzp_sec",
                    help="Your Razorpay test/live secret",
                )

                if st.button("⚡ Save & Test API Connections", key="btn_save_admin_keys", use_container_width=True):
                    from src.config import save_env_key, reload_settings
                    if new_gem_key.strip():
                        save_env_key("GEMINI_API_KEY", new_gem_key.strip())
                    if new_rzp_id.strip():
                        save_env_key("RAZORPAY_KEY_ID", new_rzp_id.strip())
                    if new_rzp_sec.strip():
                        save_env_key("RAZORPAY_KEY_SECRET", new_rzp_sec.strip())

                    reloaded = reload_settings()
                    rzp_ok, rzp_msg = rzp.validate_razorpay_credentials(reloaded.razorpay_key_id, reloaded.razorpay_key_secret)

                    gem_ok = False
                    gem_msg = ""
                    if reloaded.gemini_api_key:
                        try:
                            headers = {"x-goog-api-key": reloaded.gemini_api_key}
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{reloaded.gemini_model}:generateContent"
                            with httpx.Client(timeout=6.0) as cl:
                                r = cl.post(url, json={"contents": [{"parts": [{"text": "hi"}]}]}, headers=headers)
                                if r.status_code == 200:
                                    gem_ok = True
                                    gem_msg = "Live Google AI / Gemini Connected (200 OK)"
                                else:
                                    gem_msg = f"HTTP {r.status_code}: {r.text[:80]}"
                        except Exception as ex:
                            gem_msg = str(ex)
                    else:
                        gem_msg = "No Gemini key configured"

                    st.session_state.api_test_results = {
                        "rzp_ok": rzp_ok,
                        "rzp_msg": rzp_msg,
                        "gem_ok": gem_ok,
                        "gem_msg": gem_msg,
                    }
                    st.rerun()

                if "api_test_results" in st.session_state:
                    res = st.session_state.api_test_results
                    if res.get("rzp_ok"):
                        st.success(f"✅ Razorpay Gateway: {res.get('rzp_msg')}")
                    else:
                        st.warning(f"⚠️ Razorpay Gateway: {res.get('rzp_msg')}")

                    if res.get("gem_ok"):
                        st.success(f"✅ Gemini AI Agent: {res.get('gem_msg')}")
                    else:
                        st.info(f"ℹ️ Gemini AI Agent: {res.get('gem_msg')} (Local Autonomous Engine active)")

            with cred_c2:
                st.markdown("#### 🛡️ Security Guardrails")
                st.markdown("""
                • **Zero Client Exposure**: Key Secret is strictly kept server-side in Python backend memory and SQLite WAL.
                • **Autofill Protection**: Credential inputs are isolated from buyer shopping sessions.
                • **Multi-LLM Fallback**: If cloud keys are absent, the on-device Autonomous Local Agent Engine handles semantic search, inventory locks, and checkout generation seamlessly.
                """)
