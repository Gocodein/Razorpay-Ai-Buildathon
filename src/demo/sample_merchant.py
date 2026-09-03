"""
Sample merchant product catalog for demo / testing.

In production this would be fetched via Razorpay APIs or the merchant's
own inventory system. For the buildathon demo we seed this data into
ChromaDB so the AI agent can search it semantically.
"""

from typing import TypedDict


class RawProduct(TypedDict):
    id: str
    name: str
    description: str
    price_inr: int          # integer paise-free price for simplicity
    category: str
    stock: int              # units in stock
    sku: str
    tags: list[str]


SAMPLE_CATALOG: list[RawProduct] = [
    {
        "id": "PRD_001",
        "name": "Aloe Vera Sunscreen SPF 50",
        "description": "Lightweight, non-greasy sunscreen with SPF 50+. "
                       "Suitable for all skin types including oily and sensitive. "
                       "Water-resistant for 80 minutes. Paraben-free, vegan.",
        "price_inr": 349,
        "category": "Skincare",
        "stock": 120,
        "sku": "SKN-SUN-001",
        "tags": ["sunscreen", "spf50", "aloe vera", "vegan", "oily skin", "summer"],
    },
    {
        "id": "PRD_002",
        "name": "Vitamin C Face Serum 30ml",
        "description": "10% stabilised Vitamin C serum with hyaluronic acid. "
                       "Brightens skin, reduces dark spots. For daily morning use. "
                       "Dermatologist-tested. Fragrance-free.",
        "price_inr": 599,
        "category": "Skincare",
        "stock": 85,
        "sku": "SKN-SRM-002",
        "tags": ["vitamin c", "serum", "brightening", "dark spots", "hyaluronic acid"],
    },
    {
        "id": "PRD_003",
        "name": "Neem & Tulsi Face Wash 150ml",
        "description": "Ayurvedic face wash with neem, tulsi and green tea extracts. "
                       "Controls acne, removes excess oil without over-drying. "
                       "SLS-free, suitable for daily use.",
        "price_inr": 199,
        "category": "Skincare",
        "stock": 0,          # ← deliberately out of stock (tests failure handling)
        "sku": "SKN-FW-003",
        "tags": ["face wash", "neem", "tulsi", "acne", "ayurvedic", "oily skin"],
    },
    {
        "id": "PRD_004",
        "name": "Cold-Pressed Coconut Oil 500ml",
        "description": "100% pure, cold-pressed extra-virgin coconut oil. "
                       "Multi-use: hair, skin and cooking. No additives, hexane-free. "
                       "FSSAI certified.",
        "price_inr": 449,
        "category": "Hair & Body",
        "stock": 200,
        "sku": "HB-OIL-004",
        "tags": ["coconut oil", "hair oil", "cold pressed", "cooking", "natural", "organic"],
    },
    {
        "id": "PRD_005",
        "name": "Charcoal Detox Face Mask 100g",
        "description": "Activated charcoal mask that deep-cleans pores, "
                       "removes blackheads and excess sebum. Clay-based with kaolin. "
                       "Use twice a week. All skin types.",
        "price_inr": 299,
        "category": "Skincare",
        "stock": 60,
        "sku": "SKN-MSK-005",
        "tags": ["charcoal", "face mask", "pore cleansing", "blackheads", "detox", "clay"],
    },
    {
        "id": "PRD_006",
        "name": "Biotin Hair Growth Gummies 60ct",
        "description": "Vegan gummies with Biotin 10000mcg, Zinc and Folic Acid. "
                       "Supports hair growth and reduces hair fall. "
                       "Strawberry flavour. FSSAI approved.",
        "price_inr": 799,
        "category": "Supplements",
        "stock": 45,
        "sku": "SUPP-BIO-006",
        "tags": ["biotin", "hair growth", "hair fall", "gummies", "vegan", "supplements"],
    },
    {
        "id": "PRD_007",
        "name": "Rose Water Facial Toner 200ml",
        "description": "Pure Bulgarian rose water toner. Balances skin pH, "
                       "hydrates and preps skin before moisturiser. "
                       "Alcohol-free, suitable for sensitive skin.",
        "price_inr": 249,
        "category": "Skincare",
        "stock": 150,
        "sku": "SKN-TNR-007",
        "tags": ["rose water", "toner", "sensitive skin", "hydration", "ph balance"],
    },
    {
        "id": "PRD_008",
        "name": "Shea Butter Body Lotion 300ml",
        "description": "Rich, non-greasy body lotion with Shea Butter, "
                       "Vitamin E and Almond Oil. 48-hour moisturisation. "
                       "For dry to very dry skin. Dermatologist tested.",
        "price_inr": 399,
        "category": "Hair & Body",
        "stock": 95,
        "sku": "HB-LOT-008",
        "tags": ["body lotion", "shea butter", "dry skin", "moisturiser", "vitamin e"],
    },
]
