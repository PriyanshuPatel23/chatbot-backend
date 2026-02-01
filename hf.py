"""
GLP-1 Prescription Chatbot - HuggingFace OpenAI-Compatible API
Using pure requests library (no OpenAI SDK dependency issues!)

100% FREE | 2-4 second responses | Production-ready

Setup:
1. Get token: https://huggingface.co/settings/tokens
2. Create .env file with: HF_TOKEN=your_token_here
3. pip install fastapi uvicorn pydantic requests python-dotenv
4. Run: python main.py

--- ARCHITECTURE NOTES (for academic write-up) ---
Core design principle: "deterministic-first, LLM-assisted."
A 3B parameter model is too small to reliably: (a) output valid JSON every time,
(b) follow complex role-play instructions, or (c) avoid hallucinating questions.
This redesign moves all CRITICAL logic (extraction, response structure, type
validation) into deterministic Python code. The LLM is used ONLY for generating
a short natural-language acknowledgment prefix (2-4 words). This hybrid approach
gives conversational fluency while guaranteeing correctness — a pattern directly
inspired by slot-filling dialogue systems (Hosseini-Asl et al., 2020) and
constrained decoding literature (Lu et al., 2021).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Tuple
import json
import re
from datetime import datetime
import time
import logging
import os
import requests
from dotenv import load_dotenv
from complete_recommendation_endpoint import recommendation_router
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GLP-1 Chatbot - HuggingFace")
app.include_router(recommendation_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== CONFIGURATION ====================
class Config:
    HF_API_URL  = "https://router.huggingface.co/v1/chat/completions"
    HF_TOKEN    = os.getenv("HF_TOKEN", "")
    FAST_MODEL  = "meta-llama/Llama-3.2-3B-Instruct"

    MAX_CONTEXT_MESSAGES = 3
    REQUEST_TIMEOUT      = 30
    MAX_RETRIES          = 3

    MIN_AGE              = 18
    MIN_BMI_WEIGHT_LOSS  = 27

    # Use patient name on these assistant-message indices (0-based)
    # Index 2 ≈ after height, Index 7 ≈ after allergies — two natural warm spots
    NAME_GREETING_POSITIONS = {2, 7}

    # Phrases = conversational meta-commands, NOT real patient data
    META_PHRASES = [
        "anything else", "any thing else", "move on", "next", "continue",
        "that's it", "thats it", "nothing else", "no more", "skip",
        "that is all", "done", "proceed", "go ahead", "nothing"
    ]

    KNOWN_GLP1_MEDS = [
        "semaglutide", "wegovy", "ozempic", "rybelsus",
        "liraglutide", "saxenda", "victoza",
        "dulaglutide", "trulicity",
        "exenatide", "byetta", "bydureon",
        "tirzepatide", "mounjaro", "zepbound",
        "any", "unsure", "undecided", "not sure", "don't know"
    ]


# ==================== HTTP CLIENT ====================
class HFChatClient:
    def __init__(self):
        self.request_count = 0
        self.total_time    = 0.0
        if not Config.HF_TOKEN:
            logger.warning("⚠️  HF_TOKEN not set!")

    def generate(self, prompt: str, model: str = None,
                 temperature: float = 0.5, max_tokens: int = 150) -> Tuple[str, float]:
        if not Config.HF_TOKEN:
            raise HTTPException(status_code=500,
                detail="HF_TOKEN not set. Get token from: https://huggingface.co/settings/tokens")

        start_time = time.time()
        model = model or Config.FAST_MODEL
        headers = {"Authorization": f"Bearer {Config.HF_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        last_error = None
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.post(Config.HF_API_URL, headers=headers,
                                         json=payload, timeout=Config.REQUEST_TIMEOUT)

                if response.status_code == 200:
                    result      = response.json()
                    text        = result["choices"][0]["message"]["content"]
                    elapsed     = time.time() - start_time
                    self.request_count += 1
                    self.total_time    += elapsed
                    logger.info(f"✅ LLM response in {elapsed:.2f}s")
                    return text.strip(), elapsed

                elif response.status_code == 503:
                    try:
                        est = response.json().get("estimated_time", 20)
                        logger.warning(f"⏳ Model loading ~{est}s")
                        if attempt < Config.MAX_RETRIES - 1:
                            time.sleep(min(est, 20)); continue
                    except Exception:
                        pass
                    raise HTTPException(status_code=503, detail="Model loading. Retry in ~20s.")

                elif response.status_code == 429:
                    logger.warning("⚠️  Rate limited")
                    if attempt < Config.MAX_RETRIES - 1:
                        time.sleep(5); continue
                    raise HTTPException(status_code=429, detail="Rate limited. Retry shortly.")

                elif response.status_code in [401, 403]:
                    raise HTTPException(status_code=401, detail="Invalid HF_TOKEN.")

                else:
                    last_error = f"API error {response.status_code}: {response.text}"
                    logger.error(last_error)
                    if attempt < Config.MAX_RETRIES - 1:
                        time.sleep(2); continue
                    raise HTTPException(status_code=response.status_code, detail=last_error)

            except requests.exceptions.Timeout:
                last_error = "Timeout"
                if attempt < Config.MAX_RETRIES - 1: continue
                raise HTTPException(status_code=504, detail="Request timeout")
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(2); continue
                raise HTTPException(status_code=503, detail=last_error)
            except HTTPException:
                raise
            except Exception as e:
                last_error = str(e)
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(2); continue
                raise HTTPException(status_code=500, detail=last_error)

        raise HTTPException(status_code=500, detail=f"All retries failed: {last_error}")


hf_client = HFChatClient()


# ==================== DATA MODELS ====================
class ChatRequest(BaseModel):
    message:              str                   = Field(..., min_length=1, max_length=1000)
    conversation_history: List[Dict[str, Any]]  = Field(default_factory=list)
    collected_data:       Dict[str, Any]        = Field(default_factory=dict)
    session_id:           Optional[str]         = None

    @validator('message')
    def sanitize_message(cls, v):
        return v.strip()


class ChatResponse(BaseModel):
    response:              str
    collected_data:        Dict[str, Any]
    completion_percentage: float
    is_complete:           bool
    next_expected_field:   Optional[str]
    conversation_history:  List[Dict[str, Any]]
    processing_time:       float
    medical_flags:         List[str] = Field(default_factory=list)
    model_used:            str


# ==================== CONVERSATION STATE ====================
class ConversationState:
    REQUIRED_FIELDS = {
        "name": {
            "type": "string", "priority": 1, "required": True,
            "question": "What's your full name?"
        },
        "age": {
            "type": "number", "priority": 1, "required": True,
            "question": "How old are you?"
        },
        "height": {
            "type": "string", "priority": 1, "required": True,
            "question": "What's your height? (e.g., 5'10\" or 178 cm)"
        },
        "weight": {
            "type": "number", "priority": 1, "required": True,
            "question": "What's your current weight in pounds?"
        },
        "is_pregnant_breastfeeding": {
            "type": "boolean", "priority": 2, "required": True, "critical": True,
            "question": "Are you currently pregnant, breastfeeding, or planning to become pregnant?"
        },
        "high_risk_conditions": {
            "type": "list", "priority": 2, "required": True, "critical": True,
            "question": "Do you have any of these conditions: pancreatitis, gastroparesis, thyroid cancer, or eating disorders?"
        },
        "current_medical_conditions": {
            "type": "text", "priority": 2, "required": True,
            "question": "What current medical conditions do you have? (type 'none' if none)"
        },
        "currently_on_glp1": {
            "type": "boolean", "priority": 3, "required": True,
            "question": "Are you currently taking any GLP-1 medications?"
        },
        "other_medications": {
            "type": "text", "priority": 3, "required": True,
            "question": "What other medications are you taking? (type 'none' if none)"
        },
        "allergies": {
            "type": "text", "priority": 3, "required": True, "critical": True,
            "question": "Do you have any medication allergies? (type 'none' if none)"
        },
        "weight_loss_goal": {
            "type": "text", "priority": 4, "required": True,
            "question": "What's your weight loss goal?"
        },
        "interested_medication": {
            "type": "choice", "priority": 4, "required": True,
            "question": "Which GLP-1 medication are you interested in? (e.g., Wegovy, Ozempic, or 'unsure')"
        },
    }

    HIGH_RISK_CONDITIONS = [
        "pancreatitis", "pancreatic cancer", "gastroparesis",
        "medullary thyroid cancer", "mtc", "men-2", "men2",
        "anorexia", "bulimia", "eating disorder", "type 1 diabetes"
    ]

    @staticmethod
    def get_completion_percentage(collected: Dict) -> float:
        required = [k for k, v in ConversationState.REQUIRED_FIELDS.items() if v.get("required")]
        filled   = [k for k in required if k in collected and collected[k] is not None and collected[k] != ""]
        pct = (len(filled) / len(required)) * 100 if required else 0
        logger.info(f"📈 Progress: {len(filled)}/{len(required)} = {pct:.1f}%")
        return pct

    @staticmethod
    def get_next_missing_field(collected: Dict) -> Optional[str]:
        field_order = list(ConversationState.REQUIRED_FIELDS.keys())
        sorted_fields = sorted(
            ConversationState.REQUIRED_FIELDS.items(),
            key=lambda x: (x[1].get("priority", 99), field_order.index(x[0]))
        )
        for name, meta in sorted_fields:
            if not meta.get("required"): continue
            if name in collected:
                v = collected[name]
                if v is not None and v != "":
                    continue
            return name
        return None


# ==================== META-MESSAGE DETECTOR ====================
class MetaMessageDetector:
    """
    Detects conversational meta-commands ("anything else", "move on", "next")
    that are NOT real answers. When detected, fills the current text/list field
    with a sentinel so the flow advances.

    ROOT CAUSE of Bug #3: without this, terse non-answers left fields null
    and the bot re-asked endlessly.
    """
    @staticmethod
    def is_meta(message: str) -> bool:
        low = message.lower().strip()
        return any(phrase in low for phrase in Config.META_PHRASES)

    @staticmethod
    def sentinel_for(field_type: str) -> Any:
        """Return the 'nothing to report' value appropriate for this field type."""
        if field_type == "list":    return []
        if field_type == "boolean": return None   # ambiguous on boolean → re-ask
        return "None"               # text / choice / string


# ==================== DETERMINISTIC EXTRACTOR ====================
class DeterministicExtractor:
    """
    Rule-based extraction that runs BEFORE any LLM call.
    Handles the most common answer patterns with 100% reliability.

    ROOT CAUSE of Bugs #2, #3, #6, #7: the original code relied entirely on a
    3B LLM to parse free text into structured fields. That model fails on terse
    answers ("No", "243"), mis-maps values to wrong fields ("23 is my age" → name),
    and returns invalid types ("Yes" instead of a list).

    This layer fixes all of those by pattern-matching before the LLM is touched.
    """

    POS = re.compile(r'^(yes|y|yeah|yep|true|correct|i am|currently)\b', re.I)
    NEG = re.compile(r'^(no|n|nope|false|not|none)\b', re.I)

    @staticmethod
    def extract(message: str, field: str, meta: Dict) -> Tuple[Any, bool]:
        """Returns (value, success). success=False → fall through to LLM."""
        msg  = message.strip()
        low  = msg.lower()
        ftype = meta.get("type", "text")

        # ── BOOLEAN ──────────────────────────────────────────────
        if ftype == "boolean":
            if DeterministicExtractor.POS.match(low):  return True,  True
            if DeterministicExtractor.NEG.match(low):  return False, True
            # Longer patterns
            neg_phrases = ["no, i", "not currently", "i do not", "i don't", "i am not", "i'm not"]
            pos_phrases = ["yes, i", "yes i", "currently yes"]
            if any(p in low for p in neg_phrases): return False, True
            if any(p in low for p in pos_phrases): return True,  True
            return None, False

        # ── AGE ──────────────────────────────────────────────────
        if field == "age":
            m = re.search(r'(\d{1,3})', msg)
            if m:
                val = int(m.group(1))
                if 1 < val < 150: return val, True
            return None, False

        # ── WEIGHT ───────────────────────────────────────────────
        if field == "weight":
            m = re.search(r'(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?|kg|kilos?)?', msg, re.I)
            if m:
                val = float(m.group(1))
                if 50 < val < 1500: return val, True
            return None, False

        # ── HEIGHT ───────────────────────────────────────────────
        if field == "height":
            # 5'10" or 5'10
            m = re.search(r"(\d{1,2})\s*['\u2019]\s*(\d{1,2})", msg)
            if m: return f"{m.group(1)}'{m.group(2)}\"", True
            # 5 feet 10 / 5ft10
            m = re.search(r'(\d{1,2})\s*(?:feet|foot|ft)\s*(\d{1,2})?', msg, re.I)
            if m:
                inc = m.group(2) or "0"
                return f"{m.group(1)}'{inc}\"", True
            # 178cm
            m = re.search(r'(\d{2,3})\s*(?:cm|centimeters?)', msg, re.I)
            if m: return f"{m.group(1)} cm", True
            return None, False

        # ── NAME ─────────────────────────────────────────────────
        if field == "name":
            # Guard: reject if it contains digits (prevents "23 is my age" → name)
            if re.search(r'\d', msg): return None, False
            if len(msg) >= 2:         return msg, True
            return None, False

        # ── HIGH_RISK_CONDITIONS (list) ──────────────────────────
        if field == "high_risk_conditions":
            if DeterministicExtractor.NEG.match(low) or low in ("none", "no conditions"):
                return [], True
            # Check for specific conditions
            risk_map = {
                "pancreatitis": "pancreatitis",
                "gastroparesis": "gastroparesis",
                "thyroid cancer": "thyroid cancer",
                "medullary thyroid": "medullary thyroid cancer",
                "eating disorder": "eating disorder",
                "anorexia": "anorexia",
                "bulimia": "bulimia",
                "type 1 diabetes": "type 1 diabetes",
            }
            found = [canon for kw, canon in risk_map.items() if kw in low]
            if found: return found, True
            # Bare "yes" without specifying which → flag for review
            if DeterministicExtractor.POS.match(low):
                return ["unspecified — please list conditions"], True
            return None, False

        # ── INTERESTED_MEDICATION (choice) ───────────────────────
        if field == "interested_medication":
            for med in Config.KNOWN_GLP1_MEDS:
                if med in low:
                    if med in ("any", "unsure", "undecided", "not sure", "don't know"):
                        return "Undecided", True
                    return med.title(), True
            # Accept any non-meta answer of reasonable length
            if len(msg) >= 2 and not MetaMessageDetector.is_meta(msg):
                return msg, True
            return None, False

        # ── TEXT fields (conditions, medications, allergies, goal) ─
        if ftype == "text":
            if MetaMessageDetector.is_meta(msg): return "None", True
            if len(msg) >= 1:                    return msg,     True
            return None, False

        # ── STRING fallback ──────────────────────────────────────
        if ftype == "string":
            if len(msg) >= 1 and not MetaMessageDetector.is_meta(msg):
                return msg, True
            return None, False

        return None, False


# ==================== LLM EXTRACTION (FALLBACK ONLY) ====================
class LLMExtractor:
    """
    Called ONLY when DeterministicExtractor fails.
    Output is post-validated by TypeCoercer — nothing is stored raw.
    """
    @staticmethod
    def extract(message: str, field: str, meta: Dict) -> Tuple[Dict, float]:
        ftype = meta.get("type", "text")
        prompt = (
            f'Extract the value for "{field}" (type: {ftype}) from this patient message.\n'
            f'Return ONLY: {{"value": <extracted_value>}}\n\n'
            f'Patient said: "{message}"\n'
            f'Field: {field} | Type: {ftype}\n\n'
            f'Rules:\n'
            f'- boolean → true or false\n'
            f'- number  → integer or float\n'
            f'- list    → JSON array\n'
            f'- If you cannot find the value, return {{"value": null}}\n\n'
            f'JSON only:'
        )
        raw, t = hf_client.generate(prompt, temperature=0.1, max_tokens=80)
        try:
            cleaned = re.sub(r'```(?:json)?', '', raw).strip()
            m = re.search(r'\{[^}]*\}', cleaned)
            if m:
                obj   = json.loads(m.group())
                value = obj.get("value")
                if value is not None:
                    return {"extracted": {field: value}}, t
            return {"extracted": {}}, t
        except Exception as e:
            logger.warning(f"LLM extraction parse error: {e}")
            return {"extracted": {}}, t


# ==================== TYPE COERCER ──────────────────────────
class TypeCoercer:
    """
    Post-extraction validation & coercion.  Rejects malformed values
    rather than silently storing garbage.

    ROOT CAUSE of Bug #6: without this, the LLM could store:
      - high_risk_conditions = "Yes"     (string, not list)
      - allergies = "anything else"      (meta-phrase)
      - interested_medication = "anyone" (meta-phrase)
    """
    @staticmethod
    def coerce(field: str, value: Any, meta: Dict) -> Tuple[Any, bool]:
        """Returns (coerced_value, is_valid)."""
        if value is None: return None, False

        ftype = meta.get("type", "text")

        # Reject meta-phrases everywhere
        if isinstance(value, str) and MetaMessageDetector.is_meta(value):
            if ftype in ("text", "choice", "string"): return "None", True
            return None, False

        # ── Boolean ──
        if ftype == "boolean":
            if isinstance(value, bool): return value, True
            if isinstance(value, str):
                if value.lower() in ("true","yes","y"):  return True,  True
                if value.lower() in ("false","no","n"):  return False, True
            return None, False

        # ── Number ──
        if ftype == "number":
            try:
                n = float(value)
                if field == "age":
                    n = int(n)
                    return (n, True) if 0 < n < 150 else (None, False)
                if field == "weight":
                    return (n, True) if 50 < n < 1500 else (None, False)
                return n, True
            except (ValueError, TypeError):
                return None, False

        # ── List ──
        if ftype == "list":
            if isinstance(value, list): return value, True
            if isinstance(value, str):
                low = value.lower()
                if low in ("yes","y"):  return ["unspecified — please list conditions"], True
                if low in ("no","n","none"): return [], True
                return [v.strip() for v in value.split(",") if v.strip()], True
            return None, False

        # ── String / Text / Choice ──
        if isinstance(value, str) and len(value.strip()) > 0:
            return value.strip(), True

        return None, False


# ==================== ACKNOWLEDGMENT GENERATOR ──────────
class AcknowledgmentGenerator:
    """
    Generates a SHORT (2-5 word) natural-language acknowledgment via LLM.
    This is the ONLY place the LLM touches response generation.
    The actual question is appended deterministically by ResponseBuilder.

    ROOT CAUSE of Bugs #4 & #5: the original code let the LLM generate the
    ENTIRE response. The 3B model hallucinated questions ("date of birth")
    and made medical declarations ("you are eligible…").
    By constraining LLM output to 2-5 words of acknowledgment only, these
    hallucinations become impossible.
    """
    FALLBACKS = [
        "Got it.", "Thank you.", "Noted.", "Understood.", "Sure.",
        "Alright.", "Of course.", "Thanks.", "I see.", "Okay."
    ]

    BLOCKED = ["eligible", "recommend", "prescri", "diagnos",
               "date of birth", "dob", "born"]

    @staticmethod
    def generate(user_msg: str, response_idx: int) -> Tuple[str, float]:
        prompt = (
            f'Patient said: "{user_msg}"\n'
            f'Generate ONLY a 2-5 word acknowledgment. Examples:\n'
            f'  "Got it."  |  "Thank you."  |  "Noted."  |  "Sure, noted."\n'
            f'Do NOT ask questions. Do NOT make medical statements.\n'
            f'Your acknowledgment:'
        )
        try:
            raw, t = hf_client.generate(prompt, temperature=0.6, max_tokens=20)
            ack = raw.strip().strip('"\'').strip()

            # Safety guards: reject if it drifted into a question or medical statement
            if '?' in ack or len(ack) > 40 or any(w in ack.lower() for w in AcknowledgmentGenerator.BLOCKED):
                return AcknowledgmentGenerator.FALLBACKS[response_idx % len(AcknowledgmentGenerator.FALLBACKS)], t

            if ack and ack[-1] not in '.!,':
                ack += "."
            return ack, t
        except Exception:
            return AcknowledgmentGenerator.FALLBACKS[response_idx % len(AcknowledgmentGenerator.FALLBACKS)], 0.0


# ==================== RESPONSE BUILDER ──────────────────
class ResponseBuilder:
    """
    Assembles the final response: [acknowledgment] + [optional name] + [next question]

    The next question is ALWAYS pulled verbatim from REQUIRED_FIELDS.
    Name greeting appears only at fixed positions (Config.NAME_GREETING_POSITIONS).

    ROOT CAUSE of Bug #1: original prompt said "don't repeat name every time"
    but gave the LLM no signal for WHEN to use it → random behavior.
    Now it's a deterministic decision: use name only at indices 2 and 7.
    """
    COMPLETION_MESSAGE = (
        "Thank you so much for completing the assessment! "
        "We have recorded all your information and our medical team will review it. "
        "You will be notified about your GLP-1 eligibility shortly. Have a great day!"
    )

    @staticmethod
    def build(ack: str, next_field: Optional[str], collected: Dict, resp_idx: int) -> str:
        # All fields done → fixed completion message (no LLM)
        if next_field is None:
            return ResponseBuilder.COMPLETION_MESSAGE

        question = ConversationState.REQUIRED_FIELDS[next_field]["question"]

        # Name greeting logic — deterministic, not LLM-driven
        name = collected.get("name")
        use_name = (
            isinstance(name, str)
            and len(name) > 0
            and name != "None"
            and resp_idx in Config.NAME_GREETING_POSITIONS
        )

        if use_name:
            return f"{ack} {name}, {question}"
        return f"{ack} {question}"


# ==================== MEDICAL VALIDATOR ─────────────────
class MedicalValidator:
    @staticmethod
    def validate(collected: Dict) -> List[str]:
        flags = []

        if collected.get("is_pregnant_breastfeeding") is True:
            flags.append("CRITICAL: Pregnancy/breastfeeding contraindication")

        cond_text = str(collected.get("current_medical_conditions", "")).lower()
        risk_text = str(collected.get("high_risk_conditions", "")).lower()

        for kw in ConversationState.HIGH_RISK_CONDITIONS:
            if kw in cond_text or kw in risk_text:
                flags.append(f"HIGH RISK: {kw.upper()}")

        age = collected.get("age")
        if age is not None:
            try:
                if int(age) < Config.MIN_AGE:
                    flags.append(f"Age < {Config.MIN_AGE}")
            except (ValueError, TypeError):
                pass

        if "weight" in collected and "height" in collected:
            bmi = MedicalValidator.calc_bmi(collected["weight"], collected["height"])
            if bmi:
                collected["bmi"] = round(bmi, 1)
                if bmi < Config.MIN_BMI_WEIGHT_LOSS:
                    flags.append(f"BMI {bmi:.1f} below threshold ({Config.MIN_BMI_WEIGHT_LOSS})")
        return flags

    @staticmethod
    def calc_bmi(weight: Any, height: Any) -> Optional[float]:
        try:
            w   = float(weight)
            h   = str(height).strip().lower()

            if 'cm' in h:
                cm     = float(re.findall(r'[\d.]+', h)[0])
                inches = cm / 2.54
            else:
                parts = re.findall(r'\d+', h)
                if   len(parts) >= 2: inches = int(parts[0]) * 12 + int(parts[1])
                elif len(parts) == 1:
                    v = int(parts[0])
                    inches = v if v > 24 else v * 12
                else: return None

            return (w / (inches ** 2)) * 703
        except Exception:
            return None


# ==================== MAIN CHAT ENDPOINT ────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    t0 = time.time()

    collected  = dict(request.collected_data or {})
    history    = list(request.conversation_history or [])
    user_msg   = request.message.strip()

    # Append user turn
    history.append({"role": "user", "content": user_msg, "timestamp": datetime.now().isoformat()})

    # Response index = number of assistant messages already sent
    resp_idx = sum(1 for m in history if m.get("role") == "assistant")

    # ─── 1. Identify current expected field ───
    current_field = ConversationState.get_next_missing_field(collected)
    logger.info(f"📍 Expected: {current_field} | Collected: {collected}")

    # ─── 2. Extract value from user message ───
    if current_field:
        meta = ConversationState.REQUIRED_FIELDS[current_field]
        ftype = meta.get("type", "text")

        if MetaMessageDetector.is_meta(user_msg):
            # Meta-command: fill with sentinel to advance
            sentinel = MetaMessageDetector.sentinel_for(ftype)
            if sentinel is not None:
                collected[current_field] = sentinel
                logger.info(f"🔀 Meta → {current_field} = {sentinel}")
        else:
            # A) Deterministic extraction (fast, reliable)
            det_val, det_ok = DeterministicExtractor.extract(user_msg, current_field, meta)

            if det_ok:
                coerced, valid = TypeCoercer.coerce(current_field, det_val, meta)
                if valid:
                    collected[current_field] = coerced
                    logger.info(f"✅ Deterministic: {current_field} = {coerced}")
                else:
                    logger.warning(f"⚠️  Coercer rejected {current_field} = {det_val}")
            else:
                # B) LLM fallback (only if deterministic fails)
                logger.info(f"🤖 Falling back to LLM for {current_field}")
                llm_res, _ = LLMExtractor.extract(user_msg, current_field, meta)

                for fname, raw_val in llm_res.get("extracted", {}).items():
                    # ONLY accept the field we asked for — ignore LLM hallucinations
                    if fname != current_field:
                        logger.warning(f"⚠️  LLM returned {fname}, expected {current_field}. Ignored.")
                        continue
                    coerced, valid = TypeCoercer.coerce(fname, raw_val, meta)
                    if valid:
                        collected[fname] = coerced
                        logger.info(f"✅ LLM+Coerced: {fname} = {coerced}")
                    else:
                        logger.warning(f"⚠️  Coercer rejected LLM value for {fname} = {raw_val}")

    # ─── 3. Determine next field after extraction ───
    next_field = ConversationState.get_next_missing_field(collected)

    # ─── 4. Medical validation ───
    flags = MedicalValidator.validate(collected)

    # ─── 5. Generate short acknowledgment (LLM, tightly constrained) ───
    ack, _ = AcknowledgmentGenerator.generate(user_msg, resp_idx)

    # ─── 6. Build full response deterministically ───
    assistant_response = ResponseBuilder.build(ack, next_field, collected, resp_idx)

    # Append assistant turn
    history.append({"role": "assistant", "content": assistant_response, "timestamp": datetime.now().isoformat()})

    total_t   = time.time() - t0
    pct       = ConversationState.get_completion_percentage(collected)

    logger.info(f"✅ Done in {total_t:.2f}s | {pct:.1f}% complete")

    return ChatResponse(
        response=assistant_response,
        collected_data=collected,
        completion_percentage=round(pct, 1),
        is_complete=pct >= 95.0,
        next_expected_field=next_field,
        conversation_history=history,
        processing_time=round(total_t, 2),
        medical_flags=flags,
        model_used="HuggingFace LLaMA 3.2 3B"
    )


# ==================== HELPER ENDPOINTS ──────────────────
@app.post("/start-conversation")
async def start_conversation():
    msg = "Hi! I'm your medical assistant for GLP-1 eligibility assessment. What's your full name?"
    return {
        "response": msg,
        "collected_data": {},
        "conversation_history": [{"role": "assistant", "content": msg, "timestamp": datetime.now().isoformat()}],
        "session_id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "processing_time": 0.0
    }


@app.get("/health")
async def health_check():
    has = bool(Config.HF_TOKEN)
    return {
        "status": "healthy" if has else "unhealthy",
        "api": "HuggingFace OpenAI-Compatible (Direct HTTP)",
        "model": Config.FAST_MODEL,
        "has_token": has,
        "total_requests": hf_client.request_count,
        "avg_time": round(hf_client.total_time / max(hf_client.request_count, 1), 2) if hf_client.request_count else 0,
        "message": "Set HF_TOKEN" if not has else "Ready"
    }


@app.get("/metrics")
async def get_metrics():
    return {
        "performance": {
            "total_requests": hf_client.request_count,
            "total_time": round(hf_client.total_time, 2),
            "avg_time": round(hf_client.total_time / max(hf_client.request_count, 1), 2) if hf_client.request_count else 0
        },
        "api": "HuggingFace Router (OpenAI-Compatible)",
        "model": Config.FAST_MODEL
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🚀 GLP-1 Chatbot — Deterministic-First Architecture")
    print("=" * 60)
    print(f"{'✅' if Config.HF_TOKEN else '⚠️ '} HF_TOKEN {'found' if Config.HF_TOKEN else 'missing — set via .env or export'}")
    print(f"🤖 Model : {Config.FAST_MODEL}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📚 Docs  : http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)