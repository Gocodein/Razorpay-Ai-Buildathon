# Merchant AI Readability Agent
### Razorpay AI Buildathon — Track 01: AI Growth & Agentic Commerce

> **"Build an agent that makes a merchant transactable by an AI buyer end-to-end."**

---

## What this builds

Most Indian D2C merchants on Razorpay are **invisible to AI agents** like Claude or
ChatGPT. When a buyer asks an AI assistant "buy me sunscreen under ₹500 from an Indian
brand," the merchant's catalog has no machine-readable structure the agent can query.

This project bridges that gap: it takes a merchant's raw product data, enriches it with
Claude for semantic discoverability, stores it in a vector database, and exposes it as an
**MCP (Model Context Protocol) tool server** that any AI agent can call — ending with a
real Razorpay UPI payment.

### Demo flow

```
User → AI Agent: "Buy me sunscreen under ₹400 for oily skin"
         ↓ catalog_search_products("sunscreen oily skin", max_price_inr=400)
         ↓ catalog_check_inventory("PRD_001", quantity=1)
         ↓ [Agent confirms price with buyer]
         ↓ catalog_create_order("PRD_001", buyer_id="priya@okaxis", confirmed=true)
         ↓ Razorpay order created → UPI payment link returned
         ↓ catalog_get_audit_trail() → complete explainable record
User ← "Here's your payment link: rzp.io/l/order_xxx — Total: ₹349"
```

**Graceful failure handled:**
If the buyer requests an out-of-stock product, `catalog_check_inventory` returns
alternatives — the agent never proceeds to `catalog_create_order`.

---

## Architecture

```
┌──────────────────────── ONE-TIME SETUP ──────────────────────────┐
│  Raw product data  →  LLM Enrichment (Claude)  →  ChromaDB       │
│  (sample_merchant.py)   (intent phrases,            (semantic     │
│                          attributes, aliases)        vector DB)   │
└───────────────────────────────────────────────────────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │  MCP Tool Server │  ← AI Buyer connects here
                          │  (6 tools)       │
                          └──┬─────────┬─────┘
                             │         │
                   ┌─────────▼─┐  ┌───▼──────────────┐
                   │ ChromaDB  │  │ Razorpay test-mode│
                   │ (search)  │  │ (orders + links)  │
                   └─────────┬─┘  └────────┬──────────┘
                             │             │
                          ┌──▼─────────────▼──┐
                          │   Audit Logger    │
                          │   (SQLite, append │
                          │    -only)         │
                          └───────────────────┘
```

### MCP Tools (7 Protocol Tools)

| Tool | Type | Description |
|---|---|---|
| `catalog_search_products` | read-only | Semantic search by natural language buyer intent |
| `catalog_get_product` | read-only | Full product metadata & stock by SKU/ID |
| `catalog_check_inventory` | read-only | Stock check + automated alternatives if out-of-stock |
| `catalog_create_order` | write | Create Razorpay order + UPI QR / checkout link |
| `catalog_cancel_order` | write | Atomically cancel order, restore stock, refund budget |
| `catalog_payment_status` | read-only | Real-time payment verification & status poll |
| `catalog_get_audit_trail` | read-only | Full explainable session audit log & budget stats |

---

## Tech stack

| Layer | Choice | Reason |
|---|---|---|
| MCP Server | `mcp` (FastMCP) | Anthropic's official Python Model Context Protocol SDK |
| LLM & Agent | Google Gemini (primary) / Claude (fallback) | Multi-turn function calling with Hinglish understanding |
| Vector DB | ChromaDB (persistent) | Zero-infra local vector database with metadata filtering |
| Embeddings | `all-MiniLM-L6-v2` | Sub-10ms local semantic embeddings (zero API cost) |
| Payments | Razorpay Python SDK | Live test-mode Orders API + UPI intent links |
| Audit & CRM | SQLite WAL (append-only) | Immutable event ledger + customer intelligence sheet |
| Web UI | Streamlit 1.37+ | Frosted glass merchant dashboard & live shopping agent |
| Language | Python 3.11+ | Production-grade typing, async IO, and ML tooling |

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/merchant-ai-readability
cd merchant-ai-readability
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Fill in RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, GEMINI_API_KEY (or ANTHROPIC_API_KEY)
```

> **Razorpay test keys**: Dashboard → Settings → API Keys → switch to **Test mode**  
> **Gemini API Key**: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)  
> **Anthropic API Key**: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

### 3. Run Automated Tests

```bash
pytest -v
```

Executes 15 automated integration tests covering semantic search, OOS interception, budget gates, order creation, order cancellation, customer identity resolution, and multi-tenant partitioning.

### 4. Seed the catalog

```bash
python -m src.catalog.ingestion
```

Enriches merchant products with Indian vernacular phrases and indexes them into ChromaDB.

### 5. Run the scripted agentic demo

```bash
python -m src.demo.run_demo
```

Runs three multi-turn scenarios end-to-end with the live agent:
- **Scenario 1**: Happy path — buyer confirms → order placed with live Razorpay UPI link
- **Scenario 2**: Safety gate — buyer declines → ₹0 spent, no order created
- **Scenario 3**: Graceful failure — out-of-stock → alternative accepted & ordered

### 6. Interactive Terminal Chat Mode

```bash
python -m src.demo.chat
```

Free-form chat with the AI shopping agent in English or Hinglish with `/catalog`, `/audit`, `/budget`, and `/cancel` commands.

### 7. Interactive Web Dashboard

```bash
streamlit run src/web/app.py
```

Visual dashboard featuring:
- **AI Shopping Assistant**: Live Gemini multi-turn tool calling
- **Spacious Razorpay Payment Terminal**: Compact 1-click Razorpay checkout overlay + UPI QR
- **Live Catalog Explorer**: 150+ products with instant category filters
- **Immutable Audit Trail & Customer CRM**: Dual-sheet view tracking financial decisions and customer purchase habits with CSV export
- **Universal Ingestion Wizard**: 1-click BigBasket, Flipkart, Shopify CSV import

### 8. Start the MCP Server

```bash
python -m src.mcp_server.server
```

Connect any agent (Claude Desktop, Cursor, MCP Inspector via `npx @modelcontextprotocol/inspector`).

---

## Project structure

```
merchant-ai-readability/
├── README.md                            # Comprehensive project overview & pitch guide
├── requirements.txt                     # Pinned project dependencies
├── .env.example                         # Sanitized credentials template
├── .gitignore                           # Production security & cache exclusions
├── docs/
│   └── ADR.md                           # Architecture Decision Records (7 ADRs)
├── data/
│   ├── audit.db                         # SQLite persistent audit & customer ledger
│   ├── bigbasket_sample.csv             # Curated organic wellness catalog sample
│   ├── sample_import.csv                # Sample CSV for ingestion wizard
│   └── catalog_db/                      # ChromaDB persistent vector database
├── src/
│   ├── config.py                        # Frozen Settings dataclass (.env loader)
│   ├── audit/
│   │   └── logger.py                    # SQLite WAL audit & customer CRM logger
│   ├── catalog/
│   │   ├── enrichment.py                # Gemini/Claude/offline vernacular enrichment
│   │   ├── importer.py                  # Universal CSV/JSON catalog importer CLI
│   │   ├── ingestion.py                 # Catalog seeding orchestrator
│   │   └── vector_store.py              # ChromaDB vector store + stock manager
│   ├── demo/
│   │   ├── chat.py                      # Interactive terminal chat REPL
│   │   ├── run_demo.py                  # Scripted 3-scenario automated demo runner
│   │   └── sample_merchant.py           # Sample merchant catalog fixture
│   ├── mcp_server/
│   │   └── server.py                    # FastMCP Server (7 agentic commerce tools)
│   ├── payment/
│   │   └── razorpay_client.py           # Razorpay SDK client + spending limit gates
│   └── web/
│       └── app.py                       # Streamlit interactive glassmorphism UI
└── tests/
    └── test_gateway.py                  # 15-point automated pytest test suite
```

---

## Razorpay Track-1 Evaluation Rubric Compliance

| Rubric Pillar | Requirement | Implementation |
|---|---|---|
| **Problem Taste** | Solves high-friction business problem | Makes 72% of invisible Indian D2C merchants transactable by autonomous AI agents |
| **Build Quality** | Production modularity & clean testing | Pydantic v2 schemas, SQLite WAL concurrency, 15/15 automated tests, clean packaging |
| **AI Judgment** | Justified AI usage vs deterministic logic | AI for semantic enrichment and buyer intent; deterministic gates for money, stock, and audit |
| **Failure Recovery** | Bounded runtime enforcement & explainability | OOS alternatives, hard NPCI UAP spending limit, cancel/refund rollback, immutable audit log |

---

## 5-Minute Pitch Structure

1. **(0:00–0:30) Problem**: "72% of Indian D2C merchants are invisible to autonomous AI agents like Claude or ChatGPT. When an AI buyer wants to purchase, there's no machine-readable interface."
2. **(0:30–1:30) Demo Scenario A**: Live conversational purchase via AI agent with automatic intent resolution and Razorpay UPI checkout.
3. **(1:30–2:30) Demo Scenario B**: Graceful failure recovery — out-of-stock item intercepted, alternative recommended, order completed safely.
4. **(2:30–3:30) Architecture**: 3-layer architecture: Vernacular LLM Enrichment → FastMCP Protocol Server → Razorpay Payment Rails + SQLite WAL Audit.
5. **(3:30–4:30) Safety & Governance**: NPCI UAP spending limit boundary, human confirmation gate, cancel/refund rollback, and dual-sheet customer CRM.
6. **(4:30–5:00) Scalability**: "Universal CSV/JSON importer onboard any merchant in under 60 seconds."

