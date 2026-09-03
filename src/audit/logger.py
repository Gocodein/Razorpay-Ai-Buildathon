"""
Audit trail logger.

Every action taken by an AI agent through the MCP server is recorded
in a local SQLite database.  The log is append-only (no UPDATE/DELETE)
to satisfy the "every money action must have an audit trail" requirement
in Razorpay's Track-1 bar.

Schema
──────
events
  id          TEXT  PRIMARY KEY  (UUID)
  ts          TEXT               (ISO-8601 UTC timestamp)
  session_id  TEXT               (AI agent session token)
  tool_name   TEXT               (MCP tool that was called)
  inputs      TEXT               (JSON — tool call parameters)
  outcome     TEXT               (success | failure | out_of_stock | limit_exceeded)
  details     TEXT               (JSON — tool response summary)
  amount_inr  REAL               (₹ amount if a payment was involved, else 0)
  cumulative_spend_inr  REAL     (running session total after this event)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings


# ── DB bootstrap ─────────────────────────────────────────────────────────────

def _init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=10.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        # 1. System Events Audit Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id                    TEXT PRIMARY KEY,
                ts                    TEXT NOT NULL,
                session_id            TEXT NOT NULL,
                tool_name             TEXT NOT NULL,
                inputs                TEXT NOT NULL,
                outcome               TEXT NOT NULL,
                details               TEXT NOT NULL,
                amount_inr            REAL NOT NULL DEFAULT 0,
                cumulative_spend_inr  REAL NOT NULL DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON events (session_id)"
        )

        # 2. Dedicated Customer Insights & Lifecycle Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_records (
                id            TEXT PRIMARY KEY,
                ts            TEXT NOT NULL,
                customer_id   TEXT NOT NULL,
                upi_id        TEXT,
                action_type   TEXT NOT NULL,
                order_id      TEXT NOT NULL,
                product_id    TEXT,
                product_name  TEXT,
                quantity      INTEGER DEFAULT 1,
                amount_inr    REAL DEFAULT 0,
                session_id    TEXT NOT NULL,
                details       TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_customer_id ON customer_records (customer_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cust_order ON customer_records (order_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cust_session ON customer_records (session_id)"
        )
        conn.commit()


@contextmanager
def _conn():
    _init_db(settings.audit_log_path)
    conn = sqlite3.connect(settings.audit_log_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ── Core System Event Writer ──────────────────────────────────────────────────

def log_event(
    *,
    session_id: str,
    tool_name: str,
    inputs: dict[str, Any],
    outcome: str,
    details: dict[str, Any],
    amount_inr: float = 0.0,
) -> str:
    """
    Record one agent action in an append-only audit trail.

    Returns the event UUID so callers can cross-reference.
    """
    event_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    with _conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        # Compute exact sum of previous successful payments
        row = conn.execute(
            "SELECT COALESCE(SUM(amount_inr), 0) FROM events WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        prev_cumulative = float(row[0]) if row else 0.0
        cumulative = prev_cumulative + amount_inr

        conn.execute(
            """
            INSERT INTO events
              (id, ts, session_id, tool_name, inputs, outcome, details,
               amount_inr, cumulative_spend_inr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                ts,
                session_id,
                tool_name,
                json.dumps(inputs, ensure_ascii=False),
                outcome,
                json.dumps(details, ensure_ascii=False),
                amount_inr,
                cumulative,
            ),
        )
        conn.commit()

    return event_id


# ── Customer Lifecycle & Merchant Insights Writer ─────────────────────────────

def log_customer_action(
    *,
    customer_id: str,
    upi_id: str = "",
    action_type: str,  # 'ORDER_PLACED' | 'ORDER_CANCELLED'
    order_id: str,
    product_id: str = "",
    product_name: str = "",
    quantity: int = 1,
    amount_inr: float = 0.0,
    session_id: str,
    details: dict[str, Any] | None = None,
) -> str:
    """
    Record a customer order or cancellation event in the dedicated customer records ledger.
    Stores customer identity, UPI handle, processing time, and transaction metrics.
    """
    record_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    clean_details = json.dumps(details or {}, ensure_ascii=False)

    with _conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO customer_records
              (id, ts, customer_id, upi_id, action_type, order_id,
               product_id, product_name, quantity, amount_inr, session_id, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                ts,
                customer_id or "Anonymous Buyer",
                upi_id or customer_id or "upi@handle",
                action_type,
                order_id,
                product_id,
                product_name,
                quantity,
                amount_inr,
                session_id,
                clean_details,
            ),
        )
        conn.commit()

    return record_id


# ── Spend tracking ────────────────────────────────────────────────────────────

def session_spent_inr(session_id: str) -> float:
    """Return total ₹ spent by this agent session so far."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount_inr), 0) FROM events WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return float(row[0]) if row else 0.0


def remaining_budget_inr(session_id: str) -> float:
    """Return remaining ₹ budget for this session (never negative)."""
    spent = session_spent_inr(session_id)
    return max(0.0, float(settings.agent_spending_limit_inr) - spent)


def can_spend(session_id: str, amount_inr: float) -> tuple[bool, float]:
    """
    Check if the requested amount can be spent within the session limit.
    Returns (can_spend_bool, remaining_budget_inr).
    """
    remaining = remaining_budget_inr(session_id)
    return (remaining >= amount_inr, remaining)


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def get_session_events(session_id: str) -> list[dict[str, Any]]:
    """Return all events for a session, oldest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_events(limit: int = 500) -> list[dict[str, Any]]:
    """Return all historical audit events across all sessions, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_customer_records(
    session_id: str | None = None,
    customer_id: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """
    Return customer order/cancellation records for merchants to view customer details,
    order habits, UPI handles, and lifecycle analytics.
    """
    with _conn() as conn:
        query = "SELECT * FROM customer_records"
        params: list[Any] = []
        conditions: list[str] = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if customer_id:
            conditions.append("customer_id = ?")
            params.append(customer_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]


def get_event(event_id: str) -> dict[str, Any] | None:
    """Fetch a single event by its UUID."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return dict(row) if row else None


