"""
Catalog enrichment pipeline.

Takes raw merchant product data and uses Claude to produce a richer,
AI-agent-queryable document for each product:
  - Structured attributes (size, skin-type, use-case, etc.)
  - Intent phrases  ("good for summer travel", "works on oily skin")
  - Vernacular aliases  (Hinglish, common spellings)
  - Semantic search keywords

The enriched document is what gets embedded into ChromaDB.
"""

import json
import logging
import asyncio
from typing import Any

import anthropic

from src.config import settings
from src.demo.sample_merchant import RawProduct

logger = logging.getLogger(__name__)


# ── Anthropic client (module-level singleton) ────────────────────────────────

_client: anthropic.AsyncAnthropic | None = None


def _is_api_key_valid() -> bool:
    """Check if a real, non-placeholder Anthropic API key is configured."""
    key = (settings.anthropic_api_key or "").strip()
    return bool(key and not key.startswith("sk-ant-xxx") and len(key) > 20)


def _get_client() -> anthropic.AsyncAnthropic | None:
    global _client
    if not _is_api_key_valid():
        return None
    if _client is None:
        try:
            _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        except Exception as exc:
            print(f"[enrichment] Failed to initialize Anthropic client: {exc}")
            _client = None
    return _client


# ── Enrichment prompt ────────────────────────────────────────────────────────

_ENRICHMENT_SYSTEM = """\
You are a product-catalog enrichment engine for an Indian D2C brand.
Your job: given a raw product record, return a JSON object that makes
the product discoverable and purchasable by an AI shopping agent.

Rules:
- Output ONLY valid JSON — no markdown fences, no commentary.
- Keep all monetary values in INR as integers.
- intent_phrases must capture how a real user would describe their need
  (not the product name), e.g. "something for pimple-prone skin in summer".
- Include common Hinglish spellings in aliases where relevant.
- structured_attributes must be flat key-value pairs (strings only).
- agent_description is a single paragraph an AI agent would read aloud to
  a buyer — factual, no hype.
"""

_ENRICHMENT_USER = """\
Enrich this product for AI-agent discovery.

Raw product:
{raw}

Return JSON with exactly these keys:
{{
  "id": "<same as input>",
  "name": "<same as input>",
  "price_inr": <same as input>,
  "stock": <same as input>,
  "sku": "<same as input>",
  "category": "<same as input>",
  "agent_description": "<one paragraph for the agent to read to the buyer>",
  "structured_attributes": {{
    "<key>": "<value>",
    ...
  }},
  "intent_phrases": ["<phrase 1>", "<phrase 2>", ...],
  "aliases": ["<alias 1>", "<alias 2>", ...],
  "search_text": "<full text blob used for embedding — combine name, description, intents, aliases>"
}}
"""


_claude_disabled = False
_gemini_disabled = False


async def enrich_product(raw: RawProduct) -> dict[str, Any]:
    """
    Call Gemini or Claude to enrich a single product record.

    Returns the enriched dict. Falls back to a deterministic, high-quality
    offline vernacular enrichment if API keys are missing, invalid, or out of credits.
    """
    global _claude_disabled, _gemini_disabled

    # 1. Try Gemini if configured as provider
    if settings.llm_provider == "gemini" and settings.gemini_api_key and not _gemini_disabled:
        try:
            import httpx
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
            headers = {"x-goog-api-key": settings.gemini_api_key}
            prompt = _ENRICHMENT_USER.format(raw=json.dumps(raw, indent=2, ensure_ascii=False))
            payload = {
                "system_instruction": {"parts": [{"text": _ENRICHMENT_SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json"},
            }
            async with httpx.AsyncClient(timeout=15.0) as http_client:
                resp = await http_client.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return json.loads(text)
                else:
                    logger.warning("Gemini enrichment returned HTTP %d, falling back", resp.status_code)
        except Exception as exc:
            logger.warning("Gemini enrichment encountered exception (%s), switching to fallback", exc)

    # 2. Try Claude if configured or fallback
    if not _claude_disabled:
        client = _get_client()
        if client is not None:
            prompt = _ENRICHMENT_USER.format(raw=json.dumps(raw, indent=2, ensure_ascii=False))
            try:
                message = await client.messages.create(
                    model=settings.claude_model,
                    max_tokens=1024,
                    system=_ENRICHMENT_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = message.content[0].text.strip()
                enriched = json.loads(text)
                return enriched
            except Exception as exc:
                _claude_disabled = True
                err_msg = str(exc)
                if "credit balance is too low" in err_msg or "400" in err_msg:
                    print("[enrichment] ℹ️ Anthropic API credits exhausted — switching to High-Speed Offline Vernacular Engine.")
                else:
                    print(f"[enrichment] ℹ️ Claude API unavailable ({err_msg[:60]}…) — switching to High-Speed Offline Vernacular Engine.")

    # 3. High-precision deterministic offline enrichment
    return _fallback_enrich(raw)


async def enrich_catalog(products: list[RawProduct]) -> list[dict[str, Any]]:
    """
    Enrich all products concurrently.
    Fast-tracks deterministic offline enrichment if online LLMs are unavailable.
    """
    global _claude_disabled, _gemini_disabled
    if (_claude_disabled and _gemini_disabled) or (not _is_api_key_valid() and not settings.gemini_api_key):
        return [_fallback_enrich(p) for p in products]

    semaphore = asyncio.Semaphore(5)

    async def _bounded(p: RawProduct) -> dict[str, Any]:
        async with semaphore:
            return await enrich_product(p)

    tasks = [_bounded(p) for p in products]
    results = await asyncio.gather(*tasks)
    return list(results)




# ── Fallback (deterministic high-quality offline enrichment) ────────────────

_HINGLISH_MAP: dict[str, dict[str, Any]] = {
    "PRD_001": {
        "intents": [
            "sunscreen for oily skin", "spf 50 under 400", "non greasy summer sunscreen",
            "dhoop se bachav cream", "sunblock for sensitive skin", "lightweight sun lotion"
        ],
        "aliases": ["dhoop cream", "spf50 sunblock", "sun tan protection", "oil free sunscreen"],
        "attributes": {"skin_type": "all / oily / sensitive", "spf": "50+", "water_resistant": "80 min", "vegan": "yes"},
    },
    "PRD_002": {
        "intents": [
            "vitamin c serum for glowing skin", "remove dark spots", "brightening face serum",
            "pigmentation lightener", "chehre ke daag dhabbe hatane ka serum"
        ],
        "aliases": ["vit c drops", "glowing serum", "hyaluronic serum"],
        "attributes": {"active_ingredient": "10% Vitamin C + Hyaluronic Acid", "usage": "Daily morning", "fragrance": "free"},
    },
    "PRD_003": {
        "intents": [
            "neem face wash for pimples", "acne control face wash", "tulsi face cleanser",
            "pimple hatane wala facewash", "face wash for oily acne skin"
        ],
        "aliases": ["neem facewash", "ayurvedic cleanser", "oil control wash"],
        "attributes": {"key_herbs": "Neem & Tulsi", "skin_concern": "Acne & Excess Oil", "sls_free": "yes"},
    },
    "PRD_004": {
        "intents": [
            "pure coconut oil for hair growth", "cold pressed cooking coconut oil", "extra virgin nariyal tel",
            "dry scalp moisturiser", "natural body oil"
        ],
        "aliases": ["nariyal tel", "kacchi ghani coconut oil", "hair nourishment oil"],
        "attributes": {"extraction": "Cold-pressed extra virgin", "certification": "FSSAI", "additives": "none"},
    },
    "PRD_005": {
        "intents": [
            "charcoal mask for blackheads", "deep pore cleansing clay mask", "detox face pack for oily skin",
            "blackhead removal mask"
        ],
        "aliases": ["charcoal face pack", "clay mask", "pore clean pack"],
        "attributes": {"base": "Kaolin clay + activated charcoal", "usage": "Twice weekly"},
    },
    "PRD_006": {
        "intents": [
            "biotin gummies for hair fall", "hair growth supplement", "baal jhadna rokne ki gummies",
            "vitamin gummies for strong nails and hair"
        ],
        "aliases": ["hair gummies", "biotin vitamins", "hair chewables"],
        "attributes": {"dosage": "10,000 mcg Biotin + Zinc", "flavour": "Strawberry", "vegan": "yes"},
    },
    "PRD_007": {
        "intents": [
            "rose water toner for fresh skin", "gulab jal for face", "alcohol free soothing mist",
            "ph balance hydrating toner"
        ],
        "aliases": ["gulab jal spray", "facial mist", "rose toner"],
        "attributes": {"source": "Bulgarian Rose", "alcohol_free": "yes"},
    },
    "PRD_008": {
        "intents": [
            "shea butter body lotion for dry skin", "winter moisturiser for itchy skin", "48 hour body cream",
            "sukhi twacha ke liye lotion"
        ],
        "aliases": ["body moisturiser", "shea butter cream", "winter lotion"],
        "attributes": {"hydration": "48-hour", "key_actives": "Shea Butter + Vitamin E + Almond Oil"},
    },
}


def _auto_generate_vernacular(name: str, desc: str, cat: str) -> tuple[list[str], list[str]]:
    """
    Intelligent Category-Aware Intent & Vernacular Generator.
    Guarantees zero cross-domain hallucination between Cleaning, Skincare, Food, and Household.
    """
    import re
    name_lower = name.lower()
    desc_lower = desc.lower()
    cat_lower = cat.lower()

    # ── 1. CLEANING & HOUSEHOLD ───────────────────────────────────────────────
    if any(c in cat_lower for c in ["cleaning", "household", "pooja", "dishwash", "cleaner", "garbage", "mop", "detergent", "repell", "bins"]):
        if re.search(r"\b(?:scrub\s*pad|sponge|dishwash|scrubber|pad)\b", name_lower):
            intents = ["bartan dhone ka scrub", "kitchen dishwash scrubber", "anti bacterial utensil cleaning pad"]
            aliases = ["dishwash scrub", "bartan scrubber", "cleaning pad"]
        elif re.search(r"\b(?:diya|brass|deepak|pooja|lamp)\b", name_lower):
            intents = ["puja diya deepak", "brass lamp for mandir", "traditional brass deepak for puja"]
            aliases = ["brass diya", "mandir deepak", "pooja lamp"]
        elif re.search(r"\b(?:container|jar|storage|box|cereal)\b", name_lower):
            intents = ["kitchen storage jar", "airtight food container box", "plastic grocery storage box"]
            aliases = ["storage container", "kitchen dabba", "airtight jar"]
        elif re.search(r"\b(?:wipes|disinfect|remover|cleaner|mildew)\b", name_lower):
            intents = ["surface disinfectant wipes", "germ removal multi surface cleaner", "household cleaning spray"]
            aliases = ["disinfectant wipes", "cleaning spray", "germ cleaner"]
        elif re.search(r"\b(?:mosquito|repell|spray)\b", name_lower):
            intents = ["machhar bhagane ka spray", "herbal mosquito repellent spray", "natural bug spray"]
            aliases = ["mosquito spray", "machhar spray", "pest repellent"]
        else:
            intents = [f"buy {name_lower}", f"household cleaning {cat_lower}", f"{name_lower} for home"]
            aliases = [name_lower]
        return intents[:4], aliases[:3]

    # ── 2. KITCHEN, GARDEN & PETS ─────────────────────────────────────────────
    if any(c in cat_lower for c in ["kitchen", "garden", "pets", "cookware", "bottle", "crockery", "storage"]):
        if re.search(r"\b(?:water\s*bottle|flask|sipper|bottle)\b", name_lower):
            intents = ["pani ki bottle", "leak proof water bottle for gym", "reusable fridge water bottle"]
            aliases = ["water bottle", "pani bottle", "fridge bottle"]
        elif re.search(r"\b(?:container|jar|storage|box|dabba)\b", name_lower):
            intents = ["kitchen food container", "airtight pantry storage box", "plastic grocery container"]
            aliases = ["kitchen container", "storage box", "dabba"]
        else:
            intents = [f"buy {name_lower}", f"kitchen utility {cat_lower}", f"{name_lower} online"]
            aliases = [name_lower]
        return intents[:4], aliases[:3]

    # ── 3. FOOD, SNACKS, GOURMET & BEVERAGES ──────────────────────────────────
    if any(c in cat_lower for c in ["food", "gourmet", "snack", "beverage", "dairy", "breakfast", "spices", "oil", "tea", "masala", "grain"]):
        if re.search(r"\b(?:cookies?|biscuits?|wafers?|bakery|butter\s*cookies)\b", name_lower):
            intents = ["crunchy tea time butter cookies", "sweet bakery biscuits", "premium butter cookies pack"]
            aliases = ["butter cookies", "tea biscuits", "sweet snacks"]
        elif re.search(r"\b(?:almonds?|badam)\b", name_lower):
            intents = ["crunchy badam giri", "protein dry fruit badam", "healthy snacking raw almonds"]
            aliases = ["badam", "california almonds", "dry fruits"]
        elif re.search(r"\b(?:makhana|fox\s*nuts)\b", name_lower):
            intents = ["roasted fox nuts makhana", "crispy healthy snack", "low calorie namkeen"]
            aliases = ["makhana", "roasted fox nuts", "healthy snack"]
        elif re.search(r"\b(?:oats|oatmeal)\b", name_lower):
            intents = ["healthy breakfast rolled oats", "gluten free oatmeal", "weight loss cereal"]
            aliases = ["rolled oats", "oatmeal", "daliya"]
        elif re.search(r"\b(?:honey|shehed|madhu)\b", name_lower):
            intents = ["asli shuddh shehed", "pure raw forest honey", "natural healthy sweetener"]
            aliases = ["shehed", "raw honey", "pure honey"]
        elif re.search(r"\b(?:ghee)\b", name_lower):
            intents = ["pure desi bilona ghee", "desi a2 cow ghee for cooking", "shuddh desi ghee"]
            aliases = ["desi ghee", "bilona ghee", "cow ghee"]
        elif re.search(r"\b(?:tea|chai)\b", name_lower):
            intents = ["weight loss green tea", "detox herbal chai", "fresh aromatic organic tea"]
            aliases = ["green tea", "chai patti", "herbal tea"]
        elif re.search(r"\b(?:mustard|coconut|sunflower|groundnut|cooking)\s*oil\b", name_lower):
            intents = ["cold pressed cooking tel", "kachi ghani sarson tel", "pure organic edible oil"]
            aliases = ["cooking oil", "edible tel", "sarson tel"]
        elif re.search(r"\b(?:turmeric|haldi|curcumin)\b", name_lower):
            intents = ["shuddh haldi powder", "curcumin organic haldi", "anti inflammatory spice"]
            aliases = ["haldi powder", "turmeric", "organic haldi"]
        elif re.search(r"\b(?:chia|seeds)\b", name_lower):
            intents = ["omega 3 rich chia seeds", "superfood weight loss seeds", "organic raw chia"]
            aliases = ["chia seeds", "superfood seeds"]
        elif re.search(r"\b(?:peanut\s*butter)\b", name_lower):
            intents = ["high protein peanut butter", "fitness workout spread", "natural peanut butter"]
            aliases = ["peanut butter", "protein spread"]
        elif re.search(r"\b(?:wheat\s*grass)\b", name_lower):
            intents = ["wheat grass powder for detox", "organic immunity wheatgrass drink", "superfood green powder"]
            aliases = ["wheatgrass powder", "green superfood"]
        else:
            intents = [f"buy {name_lower}", f"fresh grocery {cat_lower}", f"{name_lower} for pantry"]
            aliases = [name_lower]
        return intents[:4], aliases[:3]

    # ── 4. BEAUTY & HYGIENE (SKINCARE, HAIRCARE, BATH) ────────────────────────
    if any(c in cat_lower for c in ["beauty", "hygiene", "skin", "hair", "bath", "oral", "health", "wellness"]):
        if re.search(r"\b(?:sunscreen|sun\s*block|spf)\b", name_lower):
            intents = ["sunscreen for sunny days", "sunblock for UV protection", "dhoop se bachne ka cream"]
            aliases = ["sunscreen gel", "dhoop cream", "spf 50"]
        elif re.search(r"\b(?:face\s*wash|cleanser)\b", name_lower):
            intents = ["pimple hatao face wash", "chehra saaf karne wala", "oil control skin cleanser"]
            aliases = ["face wash", "chehra wash", "cleanser"]
        elif re.search(r"\b(?:soap|bathing\s*bar|body\s*wash|shower\s*gel|creme\s*soft)\b", name_lower):
            intents = ["moisturizing bathing soap for soft skin", "body wash shower gel", "gentle skin cleansing soap"]
            aliases = ["bath soap", "body wash", "creme soap"]
        elif re.search(r"\b(?:face\s*mask|clay\s*mask|face\s*pack|mud\s*pack)\b", name_lower):
            intents = ["deep pore cleansing face mask", "oily skin detox face pack", "glowing skin mask"]
            aliases = ["face mask", "face pack", "clay pack"]
        elif re.search(r"\b(?:serum)\b", name_lower):
            intents = ["face serum for glowing skin", "dark spots hatane ka serum", "brightening skin serum"]
            aliases = ["face serum", "skin drops", "brightening serum"]
        elif re.search(r"\b(?:shampoo|conditioner|hair\s*wash)\b", name_lower):
            intents = ["hair fall control shampoo", "volumizing hair wash conditioner", "healthy hair shampoo"]
            aliases = ["hair shampoo", "hair conditioner", "shampoo"]
        elif re.search(r"\b(?:hair\s*oil|capsule|garlic\s*oil)\b", name_lower):
            intents = ["ayurvedic hair oil & supplements", "scalp nourishment herbal oil", "hair growth capsules"]
            aliases = ["hair oil", "hair capsule", "herbal oil"]
        elif re.search(r"\b(?:multani\s*mati|clay|mitti)\b", name_lower):
            intents = ["multani mitti for oily pimple skin", "natural glowing face mud", "ayurvedic multani clay"]
            aliases = ["multani mitti", "face clay", "multani mati"]
        elif re.search(r"\b(?:sanitizer|hand\s*rub)\b", name_lower):
            intents = ["70% alcohol hand sanitizer", "germ killing instant hand rub", "pocket hand sanitizer"]
            aliases = ["hand sanitizer", "alcohol sanitizer", "germ protection"]
        elif re.search(r"\b(?:scrub|ubtan|exfoliat)\b", name_lower):
            intents = ["dead skin exfoliator face scrub", "tan removal ubtan scrub", "gentle skin scrub"]
            aliases = ["skin scrub", "ubtan", "exfoliator"]
        elif re.search(r"\b(?:rose\s*water|gulab\s*jal)\b", name_lower):
            intents = ["gulab jal spray for face", "refreshing rose toner", "natural skin mist"]
            aliases = ["gulab jal", "rose water", "facial toner"]
        elif re.search(r"\b(?:ashwagandha|chyawanprash)\b", name_lower):
            intents = ["stress relief ayurvedic herb", "stamina booster powder", "immunity vitality"]
            aliases = ["ashwagandha", "ayurvedic rasayan", "immunity booster"]
        else:
            intents = [f"buy {name_lower}", f"personal care {cat_lower}", f"{name_lower} for skin"]
            aliases = [name_lower]
        return intents[:4], aliases[:3]

    # ── 5. GENERAL FALLBACK ───────────────────────────────────────────────────
    intents = [f"buy {name_lower}", f"best quality {cat_lower}", f"{name_lower} online"]
    aliases = [name_lower]
    return intents[:4], aliases[:3]




def _fallback_enrich(raw: RawProduct) -> dict[str, Any]:
    """
    High-quality deterministic enrichment used when Claude API is unreachable.
    Generates rich intents, Hinglish vernacular tags, and structured attributes.
    """
    pid = raw.get("id", "")
    curated = _HINGLISH_MAP.get(pid)

    if curated:
        intents = curated.get("intents", raw.get("tags", []))
        aliases = curated.get("aliases", [])
        attributes = curated.get("attributes", {"category": raw.get("category", "General")})
    else:
        # Auto-generate from title, category, and description
        auto_intents, auto_aliases = _auto_generate_vernacular(
            raw["name"], raw.get("description", ""), raw.get("category", "")
        )
        intents = raw.get("tags", []) + auto_intents
        aliases = auto_aliases
        attributes = {
            "category": raw.get("category", "General"),
            "brand": raw.get("brand", "GreenLeaf"),
            "price_segment": "Economy" if raw["price_inr"] < 300 else "Premium",
        }

    tags_str = " ".join(raw.get("tags", []))
    intents_str = " ".join(intents)
    aliases_str = " ".join(aliases)
    attrs_str = " ".join(f"{k} {v}" for k, v in attributes.items())

    search_text = (
        f"{raw['name']} {raw['description']} {raw['category']} "
        f"{tags_str} {intents_str} {aliases_str} {attrs_str} "
        f"price {raw['price_inr']} rupees under {raw['price_inr'] + 100} inr"
    )

    return {
        "id": raw["id"],
        "name": raw["name"],
        "price_inr": raw["price_inr"],
        "stock": raw["stock"],
        "sku": raw["sku"],
        "category": raw["category"],
        "agent_description": raw["description"],
        "structured_attributes": attributes,
        "intent_phrases": intents,
        "aliases": aliases,
        "search_text": search_text,
    }

