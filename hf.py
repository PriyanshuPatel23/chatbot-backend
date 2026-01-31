"""
GLP-1 Prescription Chatbot - HuggingFace OpenAI-Compatible API
Using pure requests library (no OpenAI SDK dependency issues!)

100% FREE | 2-4 second responses | Production-ready

Setup:
1. Get token: https://huggingface.co/settings/tokens
2. Create .env file with: HF_TOKEN=your_token_here
3. pip install fastapi uvicorn pydantic requests python-dotenv
4. Run: python main.py
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

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GLP-1 Chatbot - HuggingFace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== CONFIGURATION ====================
class Config:
    # HuggingFace OpenAI-compatible API
    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    
    # Models
    FAST_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Performance settings
    MAX_CONTEXT_MESSAGES = 3
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # Medical constants
    MIN_AGE = 18
    MIN_BMI_WEIGHT_LOSS = 27


# ==================== HTTP CLIENT FOR HUGGINGFACE ====================
class HFChatClient:
    """Direct HTTP client for HuggingFace OpenAI-compatible API"""
    
    def __init__(self):
        self.request_count = 0
        self.total_time = 0.0
        
        if not Config.HF_TOKEN:
            logger.warning("⚠️  HF_TOKEN not set! Get it from: https://huggingface.co/settings/tokens")
    
    def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.5,
        max_tokens: int = 150
    ) -> Tuple[str, float]:
        """
        Call HuggingFace API using direct HTTP request
        Returns: (response_text, elapsed_time)
        """
        if not Config.HF_TOKEN:
            raise HTTPException(
                status_code=500,
                detail="HF_TOKEN not set. Get token from: https://huggingface.co/settings/tokens"
            )
        
        start_time = time.time()
        model = model or Config.FAST_MODEL
        
        headers = {
            "Authorization": f"Bearer {Config.HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        last_error = None
        
        # Retry logic
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.post(
                    Config.HF_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=Config.REQUEST_TIMEOUT
                )
                
                # Success
                if response.status_code == 200:
                    result = response.json()
                    response_text = result["choices"][0]["message"]["content"]
                    
                    elapsed = time.time() - start_time
                    self.request_count += 1
                    self.total_time += elapsed
                    
                    logger.info(f"✅ Response in {elapsed:.2f}s (attempt {attempt + 1})")
                    return response_text.strip(), elapsed
                
                # Model loading (503)
                elif response.status_code == 503:
                    try:
                        error_data = response.json()
                        estimated_time = error_data.get("estimated_time", 20)
                        logger.warning(f"⏳ Model loading... ~{estimated_time}s (attempt {attempt + 1})")
                        
                        if attempt < Config.MAX_RETRIES - 1:
                            time.sleep(min(estimated_time, 20))
                            continue
                    except:
                        pass
                    
                    raise HTTPException(
                        status_code=503,
                        detail="Model is loading. Please wait ~20s and try again."
                    )
                
                # Rate limit (429)
                elif response.status_code == 429:
                    logger.warning(f"⚠️  Rate limited (attempt {attempt + 1})")
                    if attempt < Config.MAX_RETRIES - 1:
                        time.sleep(5)
                        continue
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded. Please try again in a few seconds."
                    )
                
                # Authentication (401/403)
                elif response.status_code in [401, 403]:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid HF_TOKEN. Get new token from: https://huggingface.co/settings/tokens"
                    )
                
                # Other errors
                else:
                    last_error = f"API error {response.status_code}: {response.text}"
                    logger.error(last_error)
                    
                    if attempt < Config.MAX_RETRIES - 1:
                        time.sleep(2)
                        continue
                    
                    raise HTTPException(status_code=response.status_code, detail=last_error)
            
            except requests.exceptions.Timeout:
                last_error = f"Timeout (attempt {attempt + 1})"
                logger.error(last_error)
                
                if attempt < Config.MAX_RETRIES - 1:
                    continue
                
                raise HTTPException(status_code=504, detail="Request timeout")
            
            except requests.exceptions.RequestException as e:
                last_error = f"Network error: {str(e)}"
                logger.error(last_error)
                
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                
                raise HTTPException(status_code=503, detail=last_error)
            
            except HTTPException:
                raise
            
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(last_error)
                
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                
                raise HTTPException(status_code=500, detail=last_error)
        
        raise HTTPException(status_code=500, detail=f"All retries failed: {last_error}")


hf_client = HFChatClient()


# ==================== DATA MODELS ====================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    collected_data: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    
    @validator('message')
    def sanitize_message(cls, v):
        return v.strip()


class ChatResponse(BaseModel):
    response: str
    collected_data: Dict[str, Any]
    completion_percentage: float
    is_complete: bool
    next_expected_field: Optional[str]
    conversation_history: List[Dict[str, Any]]
    processing_time: float
    medical_flags: List[str] = Field(default_factory=list)
    model_used: str


# ==================== CONVERSATION STATE ====================
class ConversationState:
    """Manages data collection flow"""
    
    REQUIRED_FIELDS = {
        "name": {"type": "string", "question": "What's your full name?", "priority": 1, "required": True},
        "age": {"type": "number", "question": "How old are you?", "priority": 1, "required": True},
        "height": {"type": "string", "question": "What's your height? (e.g., 5'10\" or 178cm)", "priority": 1, "required": True},
        "weight": {"type": "number", "question": "What's your current weight in pounds?", "priority": 1, "required": True},
        "is_pregnant_breastfeeding": {"type": "boolean", "question": "Are you currently pregnant, breastfeeding, or planning to become pregnant?", "priority": 2, "required": True, "critical": True},
        "high_risk_conditions": {"type": "list", "question": "Do you have any of these conditions: pancreatitis, gastroparesis, thyroid cancer, or eating disorders?", "priority": 2, "required": True, "critical": True},
        "current_medical_conditions": {"type": "text", "question": "What current medical conditions do you have?", "priority": 2, "required": True},
        "currently_on_glp1": {"type": "boolean", "question": "Are you currently taking any GLP-1 medications?", "priority": 3, "required": True},
        "other_medications": {"type": "text", "question": "What other medications are you taking?", "priority": 3, "required": True},
        "allergies": {"type": "text", "question": "Do you have any medication allergies?", "priority": 3, "required": True, "critical": True},
        "weight_loss_goal": {"type": "text", "question": "What's your weight loss goal?", "priority": 4, "required": True},
        "interested_medication": {"type": "choice", "question": "Which GLP-1 medication are you interested in?", "priority": 4, "required": True}
    }
    
    HIGH_RISK_CONDITIONS = [
        "pancreatitis", "pancreatic cancer", "gastroparesis",
        "medullary thyroid cancer", "mtc", "men-2",
        "anorexia", "bulimia", "eating disorder",
        "type 1 diabetes"
    ]
    
    @staticmethod
    def get_completion_percentage(collected: Dict) -> float:
        required = [k for k, v in ConversationState.REQUIRED_FIELDS.items() if v.get("required", True)]
        collected_required = [
            k for k in required 
            if k in collected and collected[k] is not None and collected[k] != ""
        ]
        percentage = (len(collected_required) / len(required)) * 100 if required else 0
        logger.info(f"📈 Progress: {len(collected_required)}/{len(required)} fields = {percentage:.1f}%")
        return percentage
    
    @staticmethod
    def get_next_missing_field(collected: Dict) -> Optional[str]:
        sorted_fields = sorted(
            ConversationState.REQUIRED_FIELDS.items(),
            key=lambda x: x[1].get("priority", 99)
        )
        
        for field_name, field_meta in sorted_fields:
            # Skip if field exists and has a non-empty value
            if field_name in collected:
                value = collected[field_name]
                # Check if value is not None and not empty string
                if value is not None and value != "":
                    continue
            
            # Only return required fields
            if not field_meta.get("required", True):
                continue
                
            return field_name
        
        return None


# ==================== ENTITY EXTRACTOR ====================
class EntityExtractor:
    """Extracts structured data from patient responses"""
    
    @staticmethod
    def extract_entities(user_message: str, expected_field: Optional[str], collected_data: Dict) -> Tuple[Dict[str, Any], float]:
        prompt = f"""Extract medical data. Return ONLY JSON.

Expected: {expected_field or "any"}
Patient: "{user_message}"

Format: {{"extracted": {{"field": value}}, "confidence": "high"}}

Examples:
"34 years old" → {{"extracted": {{"age": 34}}, "confidence": "high"}}
"5'10" → {{"extracted": {{"height": "5'10"}}, "confidence": "high"}}
"220 lbs" → {{"extracted": {{"weight": 220}}, "confidence": "high"}}
"No" → {{"extracted": {{"{expected_field}": false}}, "confidence": "high"}}

Extract:"""
        
        raw_response, process_time = hf_client.generate(prompt, temperature=0.2, max_tokens=100)
        
        try:
            cleaned = raw_response.strip()
            for pattern in ['```json', '```', 'json']:
                cleaned = cleaned.replace(pattern, '')
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            return result, process_time
        except json.JSONDecodeError:
            logger.warning(f"Parse failed: {raw_response[:100]}")
            return {"extracted": {}, "confidence": "low"}, process_time


# ==================== RESPONSE GENERATOR ====================
class ConversationalAgent:
    """Generates natural responses"""
    
    @staticmethod
    def generate_response(user_message: str, conversation_history: List[Dict], collected_data: Dict, extraction_result: Dict, next_field: Optional[str]) -> Tuple[str, float]:
        patient_name = collected_data.get('name', 'there')
        field_info = ConversationState.REQUIRED_FIELDS.get(next_field or "", {})
        next_question = field_info.get("question", "")
        
        # Get recent messages and safely build context
        recent = conversation_history[-Config.MAX_CONTEXT_MESSAGES:] if conversation_history else []
        context = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
            for msg in recent 
            if isinstance(msg, dict) and 'content' in msg
        ])
        
        prompt = f"""You are a professional medical assistant collecting patient information for GLP-1 medication eligibility.

Patient name: {patient_name}
Last patient message: "{user_message}"
Next question to ask: "{next_question}"

Rules:
1. You are the ASSISTANT, not the patient
2. Acknowledge their answer briefly
3. Ask the next question: "{next_question}"
4. Keep it SHORT - one sentence only
5. Be professional and direct
6. Do NOT repeat the patient's name in every response

Good examples:
- "Got it. {next_question}"
- "Thank you. {next_question}"
- "Noted. {next_question}"

Generate your response (one sentence only):"""
        
        response, time_taken = hf_client.generate(prompt, temperature=0.4, max_tokens=50)
        
        # Clean response
        response = response.strip()
        while response.startswith(('"', "'")) and response.endswith(('"', "'")):
            response = response[1:-1].strip()
        
        # Remove common AI artifacts
        response = response.replace("I'm John Doe.", "").replace("Hello, I'm", "Hi, I'm the assistant.")
        
        # Fix grammar issues
        if response.startswith("You are currently"):
            response = "Are " + response[0].lower() + response[1:]
        
        response = response.strip()
        
        return response, time_taken


# ==================== MEDICAL VALIDATOR ====================
class MedicalValidator:
    @staticmethod
    def validate_data(collected: Dict) -> List[str]:
        flags = []
        
        if collected.get("is_pregnant_breastfeeding") is True:
            flags.append("CRITICAL: Pregnancy/breastfeeding contraindication")
        
        conditions_text = str(collected.get("current_medical_conditions", "")).lower()
        high_risk_text = str(collected.get("high_risk_conditions", "")).lower()
        
        for keyword in ConversationState.HIGH_RISK_CONDITIONS:
            if keyword in conditions_text or keyword in high_risk_text:
                flags.append(f"HIGH RISK: {keyword.upper()}")
        
        age = collected.get("age")
        if age and age < Config.MIN_AGE:
            flags.append(f"Age < {Config.MIN_AGE}")
        
        if "weight" in collected and "height" in collected:
            bmi = MedicalValidator.calculate_bmi(collected["weight"], collected["height"])
            if bmi:
                collected["bmi"] = round(bmi, 1)
                if bmi < Config.MIN_BMI_WEIGHT_LOSS:
                    flags.append(f"BMI {bmi} below threshold")
        
        return flags
    
    @staticmethod
    def calculate_bmi(weight: float, height: str) -> Optional[float]:
        try:
            if 'cm' in str(height).lower():
                inches = float(re.findall(r'\d+', str(height))[0]) / 2.54
            else:
                parts = re.findall(r'\d+', str(height))
                inches = int(parts[0]) * 12 + int(parts[1]) if len(parts) == 2 else int(parts[0])
            return (float(weight) / (inches ** 2)) * 703
        except:
            return None


# ==================== MAIN CHAT ENDPOINT ====================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    total_start = time.time()
    
    collected_data = request.collected_data or {}
    conversation_history = request.conversation_history or []
    user_message = request.message.strip()
    
    # Safely add user message
    conversation_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().isoformat()
    })
    
    next_field = ConversationState.get_next_missing_field(collected_data)
    
    # Debug logging
    logger.info(f"📊 Collected data: {collected_data}")
    logger.info(f"📍 Next field: {next_field}")
    
    extraction_result, extract_time = EntityExtractor.extract_entities(user_message, next_field, collected_data)
    
    if extraction_result.get("extracted"):
        collected_data.update(extraction_result["extracted"])
        logger.info(f"✅ Extracted: {extraction_result['extracted']}")
    
    # Recalculate next field after extraction
    next_field = ConversationState.get_next_missing_field(collected_data)
    logger.info(f"🔄 Next field after extraction: {next_field}")
    
    assistant_response, response_time = ConversationalAgent.generate_response(
        user_message, conversation_history, collected_data, extraction_result, next_field
    )
    
    medical_flags = MedicalValidator.validate_data(collected_data)
    
    # Safely add assistant response
    conversation_history.append({
        "role": "assistant",
        "content": assistant_response,
        "timestamp": datetime.now().isoformat()
    })
    
    total_time = time.time() - total_start
    completion = ConversationState.get_completion_percentage(collected_data)
    
    logger.info(f"✅ Total: {total_time:.2f}s")
    
    return ChatResponse(
        response=assistant_response,
        collected_data=collected_data,
        completion_percentage=round(completion, 1),
        is_complete=completion >= 95.0,
        next_expected_field=next_field,
        conversation_history=conversation_history,
        processing_time=round(total_time, 2),
        medical_flags=medical_flags,
        model_used="HuggingFace LLaMA 3.2 3B"
    )


# ==================== HELPER ENDPOINTS ====================
@app.post("/start-conversation")
async def start_conversation():
    initial_msg = "Hi! I'm your medical assistant for GLP-1 eligibility assessment. What's your full name?"
    return {
        "response": initial_msg,
        "collected_data": {},
        "conversation_history": [{
            "role": "assistant",
            "content": initial_msg,
            "timestamp": datetime.now().isoformat()
        }],
        "session_id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "processing_time": 0.0
    }


@app.get("/health")
async def health_check():
    has_token = bool(Config.HF_TOKEN)
    return {
        "status": "healthy" if has_token else "unhealthy",
        "api": "HuggingFace OpenAI-Compatible (Direct HTTP)",
        "model": Config.FAST_MODEL,
        "has_token": has_token,
        "total_requests": hf_client.request_count,
        "avg_time": round(hf_client.total_time / max(hf_client.request_count, 1), 2) if hf_client.request_count > 0 else 0,
        "message": "Set HF_TOKEN" if not has_token else "Ready"
    }


@app.get("/metrics")
async def get_metrics():
    return {
        "performance": {
            "total_requests": hf_client.request_count,
            "total_time": round(hf_client.total_time, 2),
            "avg_time": round(hf_client.total_time / max(hf_client.request_count, 1), 2) if hf_client.request_count > 0 else 0
        },
        "api": "HuggingFace Router (OpenAI-Compatible)",
        "model": Config.FAST_MODEL
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 GLP-1 Chatbot - HuggingFace (Pure HTTP)")
    print("="*60)
    
    if Config.HF_TOKEN:
        print("✅ HF_TOKEN found")
    else:
        print("⚠️  Set: export HF_TOKEN='your_token'")
        print("   Get from: https://huggingface.co/settings/tokens")
    
    print(f"🤖 Model: {Config.FAST_MODEL}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)