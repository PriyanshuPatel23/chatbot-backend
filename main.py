from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import re
from datetime import datetime
import requests

app = FastAPI(title="GLP-1 Conversational Assistant")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== OLLAMA CLIENT ====================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"

def ask_ollama(prompt: str, temperature: float = 0.3) -> str:
    """Call Ollama API with prompt"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 300,  # Reduced for faster response
                    "num_ctx": 2048  # Context window
                }
            },
            timeout=120  # Increased timeout to 120 seconds
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama model took too long to respond. Try restarting Ollama.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")


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
    conversation_history: List[Dict[str, str]]


# ==================== CONVERSATION STATE ====================
class ConversationState:
    """Manages what data needs to be collected"""
    
    REQUIRED_FIELDS = {
        # Basic Info
        "name": {"type": "string", "category": "basic"},
        "age": {"type": "number", "category": "basic", "validation": "age >= 18"},
        "height": {"type": "string", "category": "basic"},
        "weight": {"type": "number", "category": "basic"},
        
        # Weight Loss Goals
        "weight_loss_goal": {"type": "choice", "category": "goals"},
        "past_weight_loss_attempts": {"type": "list", "category": "goals"},
        
        # Critical Medical History
        "is_pregnant_breastfeeding": {"type": "boolean", "category": "medical", "critical": True},
        "high_risk_conditions": {"type": "list", "category": "medical", "critical": True},
        "current_medical_conditions": {"type": "text", "category": "medical"},
        "past_surgeries": {"type": "text", "category": "medical"},
        
        # Medications
        "currently_on_glp1": {"type": "boolean", "category": "medication"},
        "current_glp1_details": {"type": "text", "category": "medication", "conditional": "currently_on_glp1"},
        "glp1_effectiveness": {"type": "choice", "category": "medication", "conditional": "currently_on_glp1"},
        "interested_medication": {"type": "choice", "category": "medication"},
        "other_medications": {"type": "text", "category": "medication"},
        
        # Allergies
        "allergies": {"type": "text", "category": "allergies"},
    }
    
    HIGH_RISK_CONDITIONS = [
        "gastroparesis", "pancreatic cancer", "pancreatitis", 
        "type 1 diabetes", "medullary thyroid cancer", "MTC",
        "MEN-2", "anorexia", "bulimia", "bipolar disorder", "schizophrenia"
    ]
    
    @staticmethod
    def get_completion_percentage(collected: Dict) -> float:
        """Calculate how much data is collected"""
        total = len(ConversationState.REQUIRED_FIELDS)
        collected_count = sum(1 for k in ConversationState.REQUIRED_FIELDS.keys() if k in collected and collected[k] is not None)
        return (collected_count / total) * 100
    
    @staticmethod
    def get_next_missing_field(collected: Dict) -> Optional[str]:
        """Determine what to ask next based on conversation flow"""
        for field, meta in ConversationState.REQUIRED_FIELDS.items():
            # Skip if already collected
            if field in collected and collected[field] is not None:
                continue
            
            # Check conditional dependencies
            if "conditional" in meta:
                condition_field = meta["conditional"]
                if condition_field not in collected or not collected[condition_field]:
                    continue
            
            return field
        return None


# ==================== ENTITY EXTRACTOR ====================
class EntityExtractor:
    """Extracts structured data from natural language responses"""
    
    @staticmethod
    def extract_entities(user_message: str, expected_field: Optional[str], collected_data: Dict) -> Dict[str, Any]:
        """Use LLM to extract entities from user's message"""
        
        prompt = f"""Extract medical data from patient message. Respond with ONLY valid JSON.

Expected field: {expected_field or "any"}
Patient said: "{user_message}"

Format:
{{"extracted_fields": {{"field": "value"}}, "confidence": "high", "needs_clarification": false}}

Examples:
"I'm 34, 5'10 and 220 lbs" → {{"extracted_fields": {{"age": 34, "height": "5'10", "weight": 220}}, "confidence": "high", "needs_clarification": false}}
"Sarah is my name" → {{"extracted_fields": {{"name": "Sarah"}}, "confidence": "high", "needs_clarification": false}}

Now extract:"""
        
        raw_response = ask_ollama(prompt, temperature=0.2)
        
        # Clean response
        raw_response = raw_response.strip()
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.startswith("```"):
            raw_response = raw_response[3:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        raw_response = raw_response.strip()
        
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            return {
                "extracted_fields": {},
                "confidence": "low",
                "needs_clarification": True,
                "clarification_reason": "Could not parse response"
            }


# ==================== CONVERSATIONAL RESPONSE GENERATOR ====================
class ConversationalAgent:
    """Generates natural, empathetic responses"""
    
    @staticmethod
    def generate_response(
        user_message: str,
        conversation_history: List[Dict],
        collected_data: Dict,
        extraction_result: Dict,
        next_field: Optional[str]
    ) -> str:
        """Generate natural conversational response"""
        
        # Build conversation context
        history_text = "\n".join([
            f"{'Patient' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation_history[-6:]  # Last 3 exchanges
        ])
        
        completion = ConversationState.get_completion_percentage(collected_data)
        
        prompt = f"""You are a friendly medical assistant collecting health info for GLP-1 medications.

Last messages:
{history_text}

Patient said: "{user_message}"
Extracted: {json.dumps(extraction_result.get('extracted_fields', {}), indent=2)}
Next needed: {next_field or "Done"}
Progress: {completion:.0f}%

Instructions:
1. Acknowledge what patient shared warmly
2. Ask next question naturally
3. ONE question only
4. 2-3 sentences max
5. Be conversational, not clinical

Generate response:"""
        
        response = ask_ollama(prompt, temperature=0.7)
        return response.strip()


# ==================== MAIN CHAT ENDPOINT ====================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main conversational endpoint"""
    
    # Initialize
    collected_data = request.collected_data or {}
    conversation_history = request.conversation_history or []
    user_message = request.message.strip()
    
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Determine what field we're expecting
    next_field = ConversationState.get_next_missing_field(collected_data)
    
    # Extract entities from user message
    extraction_result = EntityExtractor.extract_entities(
        user_message, 
        next_field, 
        collected_data
    )
    
    # Update collected data with extracted fields
    if extraction_result.get("extracted_fields"):
        collected_data.update(extraction_result["extracted_fields"])
    
    # Generate conversational response
    assistant_response = ConversationalAgent.generate_response(
        user_message,
        conversation_history,
        collected_data,
        extraction_result,
        next_field
    )
    
    # Add assistant response to history
    conversation_history.append({
        "role": "assistant",
        "content": assistant_response,
        "timestamp": datetime.now().isoformat()
    })
    
    # Calculate completion
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


# ==================== HELPER ENDPOINTS ====================
@app.post("/start-conversation")
async def start_conversation():
    """Initialize a new conversation"""
    initial_message = "Hi there! 👋 I'm here to help you learn about GLP-1 medications for weight loss. To get started, could you tell me your name?"
    
    return {
        "response": initial_message,
        "collected_data": {},
        "conversation_history": [
            {
                "role": "assistant",
                "content": initial_message,
                "timestamp": datetime.now().isoformat()
            }
        ],
        "session_id": datetime.now().strftime("%Y%m%d%H%M%S")
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL}


@app.get("/collected-data/{session_id}")
async def get_collected_data(session_id: str):
    """Retrieve collected data (implement session storage)"""
    # TODO: Implement actual session storage (Redis, Database)
    return {"message": "Session storage not implemented yet"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)