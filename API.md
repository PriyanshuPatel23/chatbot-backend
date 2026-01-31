# GLP-1 Chatbot — API Reference 📚

**Location:** `http://localhost:8000` (development)

**Overview**
- Base URL: `http://localhost:8000`
- CORS: enabled for all origins
- Server-side auth: `HF_TOKEN` (HuggingFace token) is used by the backend — no client token is required
- Typical flow: `POST /start-conversation` → repeated `POST /chat` calls until `is_complete` is true

---

## Endpoints 🔧

### POST /start-conversation ✅
- Purpose: Start a new conversation and get the assistant's opening message and `session_id`.
- Request: empty POST
- Response (200):
```json
{
  "response": "Hi! I'm your medical assistant for GLP-1 eligibility assessment. What's your full name?",
  "collected_data": {},
  "conversation_history": [
    { "role":"assistant", "content":"...", "timestamp":"ISO8601" }
  ],
  "session_id": "20250101123456789000",
  "processing_time": 0.0
}
```
- Frontend: display `response`, seed `conversation_history`, save `session_id` locally

---

### POST /chat 🔁
- Purpose: Main conversational endpoint. Send a user message and local state; receive assistant reply, updated `collected_data`, progress, and `next_expected_field`.
- Request body (JSON):
```json
{
  "message": "User message text (string, 1-1000 chars)",
  "conversation_history": [ { "role":"user|assistant", "content":"...", "timestamp":"ISO8601" } ],
  "collected_data": { /* partial patient data */ },
  "session_id": "optional string"
}
```
- Response (200) — ChatResponse:
```json
{
  "response": "Short assistant reply (one-sentence)",
  "collected_data": { "name": "John Doe", "age": 34, "height": "5'10\"", "weight": 220 },
  "completion_percentage": 42.9,
  "is_complete": false,
  "next_expected_field": "age",
  "conversation_history": [ /* messages */ ],
  "processing_time": 2.3,
  "medical_flags": ["HIGH RISK: PANCREATITIS"],
  "model_used": "HuggingFace LLaMA 3.2 3B"
}
```

Frontend integration notes:
- Optimistically append the user's message to local `conversation_history` and send the POST request.
- Merge `collected_data` from the response into local state.
- Use `next_expected_field` to guide UX (auto-focus, suggest input type).
- If `medical_flags` includes critical flags, display a blocking warning or require clinician review.
- Use `completion_percentage` to show a progress bar; `is_complete` (or >=95%) signals a finished flow.

---

### GET /health ⚕️
- Purpose: health and token status
- Response (200):
```json
{
  "status": "healthy",
  "api": "HuggingFace OpenAI-Compatible (Direct HTTP)",
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "has_token": true,
  "total_requests": 123,
  "avg_time": 2.1,
  "message": "Ready"
}
```

---

### GET /metrics 📊
- Purpose: performance metrics
- Response (200):
```json
{
  "performance": { "total_requests": 123, "total_time": 256.3, "avg_time": 2.1 },
  "api": "HuggingFace Router (OpenAI-Compatible)",
  "model": "meta-llama/Llama-3.2-3B-Instruct"
}
```

---

## `collected_data` — Fields to collect (frontend guidance) 📋
The backend expects and manages these required fields (ask these to users or let the agent collect them):

- **name** (string) — "What's your full name?" (required)
- **age** (number) — "How old are you?" (required)
- **height** (string) — "What's your height? (e.g., 5'10\" or 178cm)" (required)
- **weight** (number) — "What's your current weight in pounds?" (required)
- **is_pregnant_breastfeeding** (boolean) — Pregnancy/breastfeeding/planning pregnancy? (critical)
- **high_risk_conditions** (list) — pancreatitis, gastroparesis, thyroid cancer, eating disorders (critical)
- **current_medical_conditions** (text)
- **currently_on_glp1** (boolean)
- **other_medications** (text)
- **allergies** (text) (critical)
- **weight_loss_goal** (text)
- **interested_medication** (choice)

Frontend tips:
- Validate types locally (e.g., `age` numeric). Convert `height` to a parseable format to estimate BMI if desired.
- Send the current `collected_data` with each `POST /chat` so the backend can extract/validate entities.

---

## Medical flags & Validation ⚠️
The server runs validations and returns `medical_flags`:
- "CRITICAL: Pregnancy/breastfeeding contraindication"
- "HIGH RISK: <CONDITION>" (e.g., PANCREATITIS)
- "Age < 18"
- "BMI < 27" (if weight & height parse)

Use flags to show warnings or block progress as needed.

---

## Errors & Status Codes ❗
- 200: success
- 401/403: invalid HF token (server-side); contact server admin
- 429: rate limited — retry with backoff
- 503: service or model loading — retry
- 504: timeout
- 500: server error

Frontend behavior:
- Show friendly messages and retry options.
- For 429/503, prompt to retry after delay (e.g., 5–20s).

---

## Examples & Snippets 💡

cURL (start):
```bash
curl -X POST http://localhost:8000/start-conversation
```

fetch() (send message):
```js
const payload = {
  message: "I'm 34 years old and 5'10\"",
  conversation_history: localConversationHistory,
  collected_data: localCollectedData,
  session_id: localSessionId
};

const res = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
});
const data = await res.json();
// Merge data.collected_data and append data.response to UI
```

Suggested TypeScript interfaces:
```ts
interface ConversationEntry { role: 'user'|'assistant'; content: string; timestamp: string }

interface ChatRequest {
  message: string;
  conversation_history?: ConversationEntry[];
  collected_data?: Record<string, any>;
  session_id?: string;
}

interface ChatResponse {
  response: string;
  collected_data: Record<string, any>;
  completion_percentage: number;
  is_complete: boolean;
  next_expected_field?: string;
  conversation_history: ConversationEntry[];
  processing_time: number;
  medical_flags: string[];
  model_used: string;
}
```

---

## UX Recommendations ✅
- Start with `POST /start-conversation` to seed the chat.
- Use a single text input for the user; let the assistant drive the flow.
- Keep `conversation_history` synced with the server on each `POST /chat`.
- Show a progress bar from `completion_percentage` and a final review screen once `is_complete`.
- Immediately surface `medical_flags` and escalate if critical.

---

## Checklist for frontend implementers ✔️
- [ ] Call `POST /start-conversation` to begin
- [ ] Send user messages to `POST /chat`
- [ ] Merge `collected_data` from responses
- [ ] Use `next_expected_field` for UX guidance
- [ ] Show `completion_percentage` and `medical_flags`
- [ ] Handle errors and retry logic

---

## Next steps
If you'd like, I can also:
- Add a small TypeScript client wrapper (`startConversation()` / `sendMessage()`)
- Create an example React chat component that integrates with this API

---

> Notes: This document matches the current `hf.py` implementation. If you modify `ConversationState.REQUIRED_FIELDS` or response model, please update this doc accordingly.
