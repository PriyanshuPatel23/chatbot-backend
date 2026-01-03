"""
ULTRA-FAST VERSION using Groq API
Response time: 2-5 seconds (vs 2-3 minutes!)

Setup:
1. pip install groq python-dotenv
2. Create .env file with: GROQ_API_KEY=your_key_here
3. Get free API key: https://console.groq.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="GLP-1 Conversational Assistant - FAST")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GROQ CLIENT (ULTRA FAST!) ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file")

client = Groq(api_key=GROQ_API_KEY)

# Fast models
EXTRACTION_MODEL = "llama-3.1-8b-instant"
CONVERSATION_MODEL = "llama-3.1-8b-instant"


def ask_llm(prompt: str, temperature: float = 0.3) -> str:
    """Call Groq API - ULTRA FAST (2-5 seconds!)"""
    try:
        response = client.chat.completions.create(
            model=CONVERSATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=300,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")


# ==================== DATA MODELS ====================
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    collected_data: Optional[Dict[str, Any]] = {}
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    collected_data: Dict[str, Any]
    completion_percentage: float
    is_complete: bool
    next_expected_field: Optional[str]
    conversation_history: List[Dict[str, Any]]


# ==================== CONVERSATION STATE ====================
class ConversationState:
    """Manages what data needs to be collected for GLP-1 eligibility"""
    
    REQUIRED_FIELDS = {
        "name": {"type": "string", "category": "basic", "question": "What's your name?"},
        "age": {"type": "number", "category": "basic", "question": "How old are you?"},
        "height": {"type": "string", "category": "basic", "question": "What's your height?"},
        "weight": {"type": "number", "category": "basic", "question": "What's your current weight?"},
        "weight_loss_goal": {"type": "choice", "category": "goals", "question": "How much weight are you looking to lose?"},
        "is_pregnant_breastfeeding": {"type": "boolean", "category": "medical", "question": "Are you currently pregnant or breastfeeding?"},
        "high_risk_conditions": {"type": "list", "category": "medical", "question": "Do you have any of these conditions: diabetes (Type 1), pancreatitis, thyroid cancer, eating disorders, or severe digestive issues?"},
        "current_medical_conditions": {"type": "text", "category": "medical", "question": "What medical conditions do you currently have?"},
        "currently_on_glp1": {"type": "boolean", "category": "medication", "question": "Are you currently taking any GLP-1 medications like Ozempic, Wegovy, or Mounjaro?"},
        "other_medications": {"type": "text", "category": "medication", "question": "What other medications are you currently taking?"},
        "allergies": {"type": "text", "category": "allergies", "question": "Do you have any known drug allergies?"},
    }
    
    # GLP-1 specific contraindications
    HIGH_RISK_CONDITIONS = [
        "type 1 diabetes", "pancreatitis", "pancreatic cancer",
        "medullary thyroid cancer", "MTC", "MEN-2",
        "gastroparesis", "severe digestive problems",
        "anorexia", "bulimia", "eating disorder"
    ]
    
    @staticmethod
    def get_completion_percentage(collected: Dict) -> float:
        total = len(ConversationState.REQUIRED_FIELDS)
        collected_count = sum(1 for k in ConversationState.REQUIRED_FIELDS.keys() 
                            if k in collected and collected[k] is not None)
        return (collected_count / total) * 100
    
    @staticmethod
    def get_next_missing_field(collected: Dict) -> Optional[str]:
        for field in ConversationState.REQUIRED_FIELDS.keys():
            if field not in collected or collected[field] is None:
                return field
        return None
    
    @staticmethod
    def get_question_for_field(field: str) -> str:
        """Get the natural question to ask for a field"""
        field_data = ConversationState.REQUIRED_FIELDS.get(field)
        if field_data and "question" in field_data:
            return field_data["question"]
        return ""  # Return empty string instead of None


# ==================== ENTITY EXTRACTOR ====================
class EntityExtractor:
    @staticmethod
    def extract_entities(user_message: str, expected_field: Optional[str]) -> Dict[str, Any]:
        prompt = f"""You are a medical data extraction system for GLP-1 medication intake.

Patient's message: "{user_message}"
Expected field: {expected_field or "any relevant field"}

EXTRACTION RULES:
1. Names: "my name is X", "I'm X", "this is X" → extract as "name"
2. Age: "X years old", "age X" → extract as number
3. Height: any mention → extract as string with units
4. Weight: any mention → extract as number (pounds assumed)
5. Weight loss goal: "lose X pounds", "want to lose X" → extract amount
6. Medical conditions: diabetes, high blood pressure, etc. → extract as text
7. Medications: any drug names → extract as text
8. Pregnancy: "not pregnant", "pregnant", "male" → extract as boolean
9. Allergies: "allergic to X", "no allergies" → extract as text
10. GLP-1 status: mentions of Ozempic, Wegovy, Mounjaro → extract as boolean

IMPORTANT MEDICAL CONTEXT:
- High blood pressure = hypertension (valid condition)
- Lisinopril, atorvastatin = blood pressure/cholesterol meds
- "None", "N/A", "nothing" for allergies = "none"
- Males should have is_pregnant_breastfeeding = false

Response format (ONLY valid JSON, no markdown):
{{"extracted_fields": {{"field_name": value}}, "confidence": "high"}}

Examples:
"My name is John" → {{"extracted_fields": {{"name": "John"}}, "confidence": "high"}}
"I'm 35, 6 feet, 220 pounds" → {{"extracted_fields": {{"age": 35, "height": "6 feet", "weight": 220}}, "confidence": "high"}}
"Hi, I'm John, 35 years old, 6 feet tall and weigh 220 pounds" → {{"extracted_fields": {{"name": "John", "age": 35, "height": "6 feet", "weight": 220}}, "confidence": "high"}}
"I have high blood pressure and take lisinopril" → {{"extracted_fields": {{"current_medical_conditions": "high blood pressure", "other_medications": "lisinopril"}}, "confidence": "high"}}
"I'm not pregnant" → {{"extracted_fields": {{"is_pregnant_breastfeeding": false}}, "confidence": "high"}}
"No known allergies" → {{"extracted_fields": {{"allergies": "none"}}, "confidence": "high"}}
"I'm not on any GLP-1 medications" → {{"extracted_fields": {{"currently_on_glp1": false}}, "confidence": "high"}}

CRITICAL: Extract ALL relevant information from the message, not just one field!

Extract from the patient's message now:"""
        
        raw_response = ask_llm(prompt, temperature=0.0)  # Zero temp for deterministic extraction
        
        # Clean response - remove markdown, extra text
        raw_response = raw_response.strip()
        
        # Remove markdown code blocks
        if "```json" in raw_response:
            raw_response = raw_response.split("```json")[1].split("```")[0]
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].split("```")[0]
        
        raw_response = raw_response.strip()
        
        # Try to find JSON object
        import re
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            raw_response = json_match.group()
        
        try:
            result = json.loads(raw_response)
            # Ensure proper structure
            if "extracted_fields" not in result:
                print(f"⚠️ No extracted_fields in result: {raw_response[:200]}")
                return {"extracted_fields": {}, "confidence": "low"}
            
            # Debug log
            if result["extracted_fields"]:
                print(f"✅ Extracted: {result['extracted_fields']}")
            
            return result
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            print(f"Raw response: {raw_response[:200]}")
            return {"extracted_fields": {}, "confidence": "low"}


# ==================== CONVERSATIONAL AGENT ====================
class ConversationalAgent:
    @staticmethod
    def generate_response(
        user_message: str,
        conversation_history: List[Dict],
        collected_data: Dict,
        extraction_result: Dict,
        next_field: Optional[str]
    ) -> str:
        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Bot'}: {msg['content']}"
            for msg in conversation_history[-4:]
        ])
        
        completion = ConversationState.get_completion_percentage(collected_data)
        patient_name = collected_data.get('name', 'there')
        next_question = ConversationState.get_question_for_field(next_field) if next_field else ""
        
        # Check what was just extracted
        extracted_fields = extraction_result.get('extracted_fields', {})
        just_collected = list(extracted_fields.keys()) if extracted_fields else []
        
        prompt = f"""You are a professional medical intake assistant for GLP-1 weight loss medications.

Recent conversation:
{history_text}

User just said: "{user_message}"
What you JUST extracted: {list(extracted_fields.keys())}
ALL data collected: {list(collected_data.keys())}
Progress: {completion:.0f}% complete

MANDATORY INSTRUCTIONS:
1. Briefly acknowledge what they just shared (1 sentence)
2. Ask ONLY for the next required field: "{next_field}"
3. Use this exact question: "{next_question}"
4. DO NOT ask follow-up questions about information already collected
5. DO NOT ask about duration, details, or clarifications unless critical
6. Stay focused on collecting the required fields only

Already collected (skip these): {', '.join(collected_data.keys())}
Next REQUIRED field: {next_field or "Complete!"}
Question to ask: "{next_question}"

Patient name: {patient_name}

Generate response (2 sentences max):
- Sentence 1: Thank them for what they shared
- Sentence 2: Ask the next required question directly

Response:"""
        
        response = ask_llm(prompt, temperature=0.7)
        return response.strip().strip('"').strip("'")


# ==================== MAIN CHAT ENDPOINT ====================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    collected_data = request.collected_data or {}
    conversation_history = request.conversation_history or []
    user_message = request.message.strip()
    
    conversation_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().isoformat()
    })
    
    next_field = ConversationState.get_next_missing_field(collected_data)
    
    extraction_result = EntityExtractor.extract_entities(user_message, next_field)
    
    if extraction_result.get("extracted_fields"):
        collected_data.update(extraction_result["extracted_fields"])
    
    assistant_response = ConversationalAgent.generate_response(
        user_message,
        conversation_history,
        collected_data,
        extraction_result,
        next_field
    )
    
    conversation_history.append({
        "role": "assistant",
        "content": assistant_response,
        "timestamp": datetime.now().isoformat()
    })
    
    completion_percentage = ConversationState.get_completion_percentage(collected_data)
    is_complete = completion_percentage >= 95.0
    
    return ChatResponse(
        response=assistant_response,
        collected_data=collected_data,
        completion_percentage=completion_percentage,
        is_complete=is_complete,
        next_expected_field=ConversationState.get_next_missing_field(collected_data),
        conversation_history=conversation_history
    )


@app.post("/start-conversation")
async def start_conversation():
    initial_message = "Hi! 👋 I'm here to help you explore GLP-1 medications for weight loss. To get started, could you tell me your name?"
    
    return {
        "response": initial_message,
        "collected_data": {},
        "conversation_history": [{
            "role": "assistant",
            "content": initial_message,
            "timestamp": datetime.now().isoformat()
        }],
        "session_id": datetime.now().strftime("%Y%m%d%H%M%S")
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "api": "Groq",
        "model": CONVERSATION_MODEL,
        "speed": "2-5 seconds per response"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)