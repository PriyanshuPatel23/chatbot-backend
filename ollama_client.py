import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"

def ask_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }
    )
    response.raise_for_status()
    return response.json()["response"]
