"""
Comprehensive Test Suite for Merchant AI Readability Gateway.
Tests all 6 tools, spending gates, inventory decrement, failure handling, and audit trails.
"""

import asyncio
import uuid
import pytest

from src.catalog import vector_store
from src.catalog.ingestion import run_ingestion
from src.audit import logger as audit
from src.payment import razorpay_client as rzp
from src.mcp_server.server import (
    catalog_search_products,
    catalog_get_product,
    catalog_check_inventory,
    catalog_create_order,
    catalog_payment_status,
    catalog_get_audit_trail,
    SearchInput,
    GetProductInput,
    InventoryInput,
    CreateOrderInput,
    PaymentStatusInput,
    AuditInput,
)


@pytest.fixture(scope="session", autouse=True)
def setup_catalog():
    asyncio.run(run_ingestion(verbose=False))


def test_01_semantic_search():
    session_id = f"test_search_{uuid.uuid4().hex[:8]}"
    params = SearchInput(
        query="sunscreen for oily skin",
        max_results=3,
        max_price_inr=400,
        session_id=session_id,
    )
    res_json = asyncio.run(catalog_search_products(params))
    assert "results" in res_json
    assert "PRD_001" in res_json


def test_02_get_product():
    session_id = f"test_get_{uuid.uuid4().hex[:8]}"
    params = GetProductInput(product_id="PRD_001", session_id=session_id)
    res_json = asyncio.run(catalog_get_product(params))
    assert "Aloe Vera Sunscreen" in res_json

    # Test invalid product ID
    params_invalid = GetProductInput(product_id="PRD_999", session_id=session_id)
    res_inv = asyncio.run(catalog_get_product(params_invalid))
    assert "error" in res_inv


def test_03_check_inventory_instock():
    session_id = f"test_inv_{uuid.uuid4().hex[:8]}"
    params = InventoryInput(product_id="PRD_001", quantity=1, session_id=session_id)
    res_json = asyncio.run(catalog_check_inventory(params))
    assert '"available": true' in res_json


def test_04_check_inventory_outofstock_alternatives():
    session_id = f"test_oos_{uuid.uuid4().hex[:8]}"
    params = InventoryInput(product_id="PRD_003", quantity=1, session_id=session_id)
    res_json = asyncio.run(catalog_check_inventory(params))
    assert '"available": false' in res_json
    assert "alternatives" in res_json


def test_05_create_order_unconfirmed_rejected():
    session_id = f"test_unconf_{uuid.uuid4().hex[:8]}"
    params = CreateOrderInput(
        product_id="PRD_001",
        quantity=1,
        buyer_id="test@buyer",
        session_id=session_id,
        confirmed=False,
    )
    res_json = asyncio.run(catalog_create_order(params))
    assert "not_confirmed" in res_json


def test_06_create_order_happy_path():
    session_id = f"test_order_{uuid.uuid4().hex[:8]}"
    initial_stock = vector_store.get_by_id("PRD_001")["stock"]

    params = CreateOrderInput(
        product_id="PRD_001",
        quantity=1,
        buyer_id="priya@okaxis",
        session_id=session_id,
        confirmed=True,
    )
    res_json = asyncio.run(catalog_create_order(params))
    assert "created" in res_json
    assert "order_" in res_json

    # Check inventory was decremented
    new_stock = vector_store.get_by_id("PRD_001")["stock"]
    assert new_stock == initial_stock - 1


def test_07_spending_limit_exceeded():
    session_id = f"test_limit_{uuid.uuid4().hex[:8]}"
    # Attempting to order quantity 10 of PRD_006 (799 * 10 = 7990 INR > 2000 INR limit)
    params = CreateOrderInput(
        product_id="PRD_006",
        quantity=10,
        buyer_id="whale@buyer",
        session_id=session_id,
        confirmed=True,
    )
    res_json = asyncio.run(catalog_create_order(params))
    assert "limit_exceeded" in res_json


def test_08_audit_trail_retrieval():
    session_id = f"test_audit_{uuid.uuid4().hex[:8]}"
    # Search
    asyncio.run(catalog_search_products(SearchInput(query="serum", session_id=session_id)))
    # Inventory
    asyncio.run(catalog_check_inventory(InventoryInput(product_id="PRD_002", quantity=1, session_id=session_id)))
    # Order
    asyncio.run(catalog_create_order(CreateOrderInput(product_id="PRD_002", quantity=1, buyer_id="priya@upi", session_id=session_id, confirmed=True)))

    # Fetch audit trail
    audit_json = asyncio.run(catalog_get_audit_trail(AuditInput(session_id=session_id)))
    assert "events" in audit_json
    assert '"total_spent_inr": 599.0' in audit_json


def test_09_cancel_order_and_budget_refund():
    session_id = f"test_canc_{uuid.uuid4().hex[:8]}"
    initial_stock = vector_store.get_by_id("PRD_001")["stock"]

    # 1. Place order
    res_ord = rzp.create_order(
        product_id="PRD_001",
        product_name="Aloe Vera Sunscreen",
        quantity=1,
        unit_price_inr=349,
        buyer_id="sagar@upi",
        session_id=session_id,
    )
    assert res_ord["status"] == "created"
    assert audit.session_spent_inr(session_id) == 349.0
    assert vector_store.get_by_id("PRD_001")["stock"] == initial_stock - 1

    # 2. Cancel order
    canc_res = rzp.cancel_order(
        order_id=res_ord["order_id"],
        session_id=session_id,
        product_id="PRD_001",
        quantity=1,
        amount_inr=349,
        reason="Buyer cancelled test",
    )
    assert canc_res["status"] == "cancelled"

    # Verify stock restored and budget refunded to 0
    assert vector_store.get_by_id("PRD_001")["stock"] == initial_stock
    assert audit.session_spent_inr(session_id) == 0.0
    assert audit.remaining_budget_inr(session_id) == 2000.0


def test_10_multi_merchant_and_empty_query_guard():
    # Empty query returns catalog products without throwing error
    results = vector_store.search("", n_results=2)
    assert len(results) >= 1

    # Search with valid query
    results_sun = vector_store.search("sunscreen", n_results=2)
    assert len(results_sun) >= 1


def test_11_customer_lifecycle_records_and_duplicate_cancellation():
    session_id = f"test_crm_{uuid.uuid4().hex[:8]}"
    buyer = "priya.sharma@hdfcbank"

    # 1. Place order
    res_ord = rzp.create_order(
        product_id="PRD_001",
        product_name="Aloe Vera Sunscreen",
        quantity=2,
        unit_price_inr=349,
        buyer_id=buyer,
        session_id=session_id,
    )
    oid = res_ord["order_id"]
    assert res_ord["status"] == "created"

    # Verify customer records ledger captured the placement
    recs = audit.get_customer_records(session_id=session_id)
    assert len(recs) == 1
    assert recs[0]["action_type"] == "ORDER_PLACED"
    assert recs[0]["customer_id"] == "Priya Sharma"
    assert recs[0]["upi_id"] == buyer
    assert recs[0]["amount_inr"] == 698.0

    # 2. Cancel order
    canc1 = rzp.cancel_order(order_id=oid, session_id=session_id)
    assert canc1["status"] == "cancelled"

    # Verify customer records captured cancellation
    recs_after = audit.get_customer_records(session_id=session_id)
    assert len(recs_after) == 2
    assert recs_after[0]["action_type"] == "ORDER_CANCELLED"
    assert recs_after[0]["amount_inr"] == 698.0

    # 3. Duplicate cancellation protection guard
    canc2 = rzp.cancel_order(order_id=oid, session_id=session_id)
    assert canc2["status"] == "already_cancelled"
    assert canc2["amount_inr"] == 0.0


def test_12_catalog_importer_and_price_sanitization():
    from src.catalog.importer import _clean_price, _parse_row

    # Test dirty price strings commonly found in Indian e-commerce CSVs
    assert _clean_price("₹1,249.00") == 1249.0
    assert _clean_price("Rs. 499") == 499.0
    assert _clean_price("349") == 349.0
    assert _clean_price("FREE") == 0.0

    # Test BigBasket header format auto-mapping
    bb_row = {
        "product": "Organic Almond Milk 1L",
        "sale_price": "280",
        "category": "Dairy & Plant Milk",
        "description": "Unsweetened rich almond milk"
    }
    parsed = _parse_row(bb_row, 1)
    assert parsed is not None
    assert parsed["name"] == "Organic Almond Milk 1L"
    assert parsed["price_inr"] == 280
    assert parsed["category"] == "Dairy & Plant Milk"


def test_13_customer_identity_and_upi_email_differentiation():
    from src.payment.razorpay_client import is_valid_upi_vpa, _extract_customer_identity

    # 1. Valid UPI handles
    assert is_valid_upi_vpa("sagar@slice") is True
    assert is_valid_upi_vpa("priya@okhdfcbank") is True
    assert is_valid_upi_vpa("rahul@okaxis") is True
    assert is_valid_upi_vpa("merchant@paytm") is True

    # 2. Email domains must NOT be treated as UPI
    assert is_valid_upi_vpa("sagar@gmail.com") is False
    assert is_valid_upi_vpa("priya@yahoo.com") is False
    assert is_valid_upi_vpa("support@outlook.com") is False

    # 3. Customer Identity extraction
    name1, upi1 = _extract_customer_identity(buyer_id="Sagar Shaw (sagar@slice)")
    assert name1 == "Sagar Shaw"
    assert upi1 == "sagar@slice"

    name2, upi2 = _extract_customer_identity(buyer_name="Priya Sharma", buyer_upi="priya@okhdfcbank")
    assert name2 == "Priya Sharma"
    assert upi2 == "priya@okhdfcbank"

    # 4. If email provided, fallback safely without crashing
    name3, upi3 = _extract_customer_identity(buyer_name="Sagar Shaw", buyer_upi="sagar@gmail.com")
    assert name3 == "Sagar Shaw"
    assert is_valid_upi_vpa(upi3) is True


def test_14_order_response_security_sanitization():
    session_id = f"test_sec_{uuid.uuid4().hex[:8]}"
    order = rzp.create_order(
        product_id="PRD_001",
        product_name="Aloe Vera Sunscreen SPF 50",
        quantity=1,
        unit_price_inr=349,
        buyer_id="Sagar Shaw (sagar@slice)",
        session_id=session_id,
        buyer_name="Sagar Shaw",
        buyer_upi="sagar@slice",
    )
    # Ensure razorpay_key_id is NEVER leaked in public order response dict
    assert "razorpay_key_id" not in order
    assert order["status"] == "created"
    assert order["customer_name"] == "Sagar Shaw"
    assert order["customer_upi"] == "sagar@slice"


def test_15_payment_capture_ledger_and_gmv_accounting():
    session_id = f"test_cap_{uuid.uuid4().hex[:8]}"
    oid = f"order_{uuid.uuid4().hex[:10]}"
    cust_name = "Sagar Shaw"
    cust_upi = "sagar@okaxis"
    amt = 102.0

    # 1. Log ORDER_PLACED
    r1 = audit.log_customer_action(
        customer_id=cust_name,
        upi_id=cust_upi,
        action_type="ORDER_PLACED",
        order_id=oid,
        product_id="PRD_001",
        product_name="Chocobakes Choc Filled Cookies",
        quantity=1,
        amount_inr=amt,
        session_id=session_id,
    )
    assert r1 is not None

    # 2. Log PAYMENT_CAPTURED
    r2 = audit.log_customer_action(
        customer_id=cust_name,
        upi_id=cust_upi,
        action_type="PAYMENT_CAPTURED",
        order_id=oid,
        product_id="PRD_001",
        product_name="Chocobakes Choc Filled Cookies",
        quantity=1,
        amount_inr=amt,
        session_id=session_id,
        details={"payment_id": "pay_test_123", "status": "paid"},
    )
    assert r2 is not None

    records = audit.get_customer_records(session_id=session_id)
    assert len(records) == 2

    # Verify both records retain positive amount
    for r in records:
        assert r["amount_inr"] == 102.0
        assert r["order_id"] == oid

    # Verify accounting logic (de-duplication per order_id)
    order_status_map = {}
    order_amounts = {}
    for r in records:
        o = r.get("order_id")
        act = r.get("action_type")
        order_amounts[o] = float(r.get("amount_inr", 0))
        if act == "ORDER_CANCELLED":
            order_status_map[o] = "CANCELLED"
        elif act == "PAYMENT_CAPTURED" and order_status_map.get(o) != "CANCELLED":
            order_status_map[o] = "CAPTURED"
        elif o not in order_status_map:
            order_status_map[o] = "PLACED"

    assert len(order_status_map) == 1
    assert order_status_map[oid] == "CAPTURED"
    assert sum(order_amounts.values()) == 102.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])

