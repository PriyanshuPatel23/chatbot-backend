from fastapi import FastAPI
from pydantic import BaseModel
from ollama_client import ask_ollama
import json

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    state: dict



@app.post("/chat")
def chat(req: ChatRequest):
    prompt = f"""
You are a medical intake assistant.

Already collected information:
{json.dumps(req.state)}

Your task is to decide the NEXT missing field to ask.

Rules:
- Ask ONLY ONE question
- Skip fields already collected
- Respond ONLY in valid JSON
- No explanations

Fields to collect:
- age
- height
- weight
- has_diabetes
- pregnant
- medical_conditions

Response format:
{{
  "field": "<field_name>",
  "question": "<question>",
  "type": "<type>",
  "options": ["..."]
}}

User said:
{req.message}
"""


    raw = ask_ollama(prompt)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "field": "unknown",
            "question": "Could you please clarify?",
            "type": "text"
        }

    return parsed
