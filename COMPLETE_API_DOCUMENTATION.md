# GLP-1 Conversational Assistant - Complete API Documentation 📚

**Last Updated:** February 1, 2026  
**API Version:** 1.0.1  
**Base URL:** `http://localhost:8000`

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication & CORS](#authentication--cors)
3. [Endpoints](#endpoints)
   - [Conversation Management](#conversation-management)
   - [Eligibility Assessment](#eligibility-assessment)
   - [Medication Recommendation](#medication-recommendation)
   - [Health & Utility](#health--utility)
4. [Data Models](#data-models)
5. [Frontend Integration Guide](#frontend-integration-guide)
6. [Error Handling](#error-handling)
7. [Complete Flow Examples](#complete-flow-examples)

---

## Overview

This API provides a **three-stage GLP-1 prescription recommendation system**:

| Stage | Component | Purpose | Method |
|-------|-----------|---------|--------|
| 1 | **Eligibility Assessment** | Check contraindications & calculate eligibility score | Rule-based CDSS |
| 2 | **Medication Selection** | Rank medications based on patient profile | MCDM (Multi-Criteria Decision Making) |
| 3 | **Prescription Generation** | Generate personalized prescription | CBR (Case-Based Reasoning) |

### Key Features

- ✅ **Conversational AI** - Natural language data collection
- ✅ **Clinical Decision Support** - Evidence-based eligibility assessment
- ✅ **Medication Optimization** - Multi-criteria ranking of 8 GLP-1 drugs
- ✅ **Personalized Prescriptions** - Case-based reasoning with patient adaptation
- ✅ **Session Management** - Track conversation state across requests
- ✅ **CORS Enabled** - Works with all frontend frameworks

---

## Authentication & CORS

### CORS Configuration

```
- Allow Origins: * (all)
- Allow Methods: GET, POST, PUT, DELETE, OPTIONS
- Allow Headers: * (all)
- Credentials: true
```

### No Client Authentication Required

The API uses a backend-stored HuggingFace token (`HF_TOKEN` environment variable). **No token is needed in frontend requests.**

---

## Endpoints

---

### CONVERSATION MANAGEMENT

#### 1. Start Conversation

**Initialize a new conversation session**

```http
POST /start-conversation
```

**Request:**
```json
// Empty body - no parameters required
```

**Response (200 OK):**
```json
{
  "response": "Hi there! 👋 I'm here to help you learn about GLP-1 medications for weight loss. To get started, could you tell me your name?",
  "collected_data": {},
  "conversation_history": [
    {
      "role": "assistant",
      "content": "Hi there! 👋 I'm here to help you learn about GLP-1 medications for weight loss. To get started, could you tell me your name?",
      "timestamp": "2026-02-01T10:30:45.123456"
    }
  ],
  "session_id": "20260201103045",
  "processing_time": 0.05
}
```

**Frontend Integration:**
```javascript
async function startConversation() {
  const response = await fetch('http://localhost:8000/start-conversation', {
    method: 'POST'
  });
  
  const data = await response.json();
  
  // Store session ID in localStorage or state
  localStorage.setItem('sessionId', data.session_id);
  
  // Display initial message
  displayMessage(data.response, 'assistant');
  
  // Initialize conversation state
  window.conversationState = {
    sessionId: data.session_id,
    conversationHistory: data.conversation_history,
    collectedData: data.collected_data
  };
}
```

---

#### 2. Send Chat Message

**Send user message and get AI response with data extraction**

```http
POST /chat
Content-Type: application/json
```

**Request Body:**
```json
{
  "message": "My name is John Doe and I'm 34 years old",
  "conversation_history": [
    {
      "role": "assistant",
      "content": "Hi there! 👋 I'm here to help you...",
      "timestamp": "2026-02-01T10:30:45.123456"
    }
  ],
  "collected_data": {},
  "session_id": "20260201103045"
}
```

**Response (200 OK):**
```json
{
  "response": "Great to meet you, John! 34 is a good age to start thinking about weight management. What's your current height and weight?",
  "collected_data": {
    "name": "John Doe",
    "age": 34
  },
  "completion_percentage": 15.7,
  "is_complete": false,
  "next_expected_field": "height",
  "conversation_history": [
    {
      "role": "assistant",
      "content": "Hi there! 👋 I'm here to help you...",
      "timestamp": "2026-02-01T10:30:45.123456"
    },
    {
      "role": "user",
      "content": "My name is John Doe and I'm 34 years old",
      "timestamp": "2026-02-01T10:31:02.456789"
    },
    {
      "role": "assistant",
      "content": "Great to meet you, John! 34 is a good age...",
      "timestamp": "2026-02-01T10:31:05.789012"
    }
  ]
}
```

**Frontend Integration - Complete Chat Loop:**

```javascript
class ChatManager {
  constructor() {
    this.state = {
      sessionId: null,
      conversationHistory: [],
      collectedData: {}
    };
  }

  async sendMessage(userMessage) {
    try {
      // Show user message immediately in UI
      this.displayMessage(userMessage, 'user');

      // Send to backend
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: userMessage,
          conversation_history: this.state.conversationHistory,
          collected_data: this.state.collectedData,
          session_id: this.state.sessionId
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();

      // Update local state
      this.state.conversationHistory = data.conversation_history;
      this.state.collectedData = data.collected_data;

      // Display assistant response
      this.displayMessage(data.response, 'assistant');

      // Show progress
      this.updateProgressBar(data.completion_percentage);

      // Check if conversation is complete
      if (data.is_complete) {
        this.handleConversationComplete();
      }

      return data;
    } catch (error) {
      console.error('Chat error:', error);
      this.displayMessage('Sorry, there was an error. Please try again.', 'error');
    }
  }

  displayMessage(text, role) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  updateProgressBar(percentage) {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = percentage + '%';
    document.getElementById('progress-text').textContent = Math.round(percentage) + '%';
  }

  async handleConversationComplete() {
    document.getElementById('chat-input').disabled = true;
    document.getElementById('send-button').disabled = true;
    
    // After 2 seconds, prompt to get recommendation
    setTimeout(() => {
      this.showGetRecommendationButton();
    }, 2000);
  }

  showGetRecommendationButton() {
    const button = document.createElement('button');
    button.textContent = 'Get Recommendation';
    button.className = 'btn btn-primary';
    button.onclick = () => this.getRecommendation();
    document.getElementById('action-container').appendChild(button);
  }

  async getRecommendation() {
    await window.recommendationManager.getCompleteRecommendation(
      this.state.sessionId,
      this.state.collectedData
    );
  }
}

// Usage in HTML
document.getElementById('send-button').addEventListener('click', () => {
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if (message) {
    chatManager.sendMessage(message);
    input.value = '';
  }
});
```

---

### ELIGIBILITY ASSESSMENT

#### 1. Evaluate Eligibility

**Complete eligibility assessment with scoring**

```http
POST /eligibility/evaluate
Content-Type: application/json
```

**Request Body:**
```json
{
  "collected_data": {
    "name": "Jane Smith",
    "age": 42,
    "height": "5'6\"",
    "weight": 185,
    "is_pregnant_breastfeeding": false,
    "high_risk_conditions": [],
    "current_medical_conditions": "Type 2 diabetes",
    "currently_on_glp1": false,
    "other_medications": "Metformin, Lisinopril",
    "allergies": "None",
    "weight_loss_goal": "Lose 25 pounds",
    "interested_medication": "Any"
  },
  "session_id": "20260201103045"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "timestamp": "2026-02-01T10:35:20.123456",
  "session_id": "20260201103045",
  "eligibility_status": "ELIGIBLE",
  "eligibility_score": 78.5,
  "risk_level": "LOW",
  "clinical_assessment": {
    "bmi": {
      "value": 29.9,
      "category": "Overweight",
      "meets_criteria": true,
      "score": 38
    },
    "diabetes_status": {
      "has_type2_diabetes": true,
      "controlled": true,
      "score": 25
    },
    "comorbidities": {
      "present": ["hypertension"],
      "cardiovascular_risk": "moderate",
      "score": 18
    },
    "weight_loss_goal": {
      "realistic": true,
      "score": 12
    },
    "contraindications": {
      "has_contraindications": false,
      "violations": []
    }
  },
  "decision_support": {
    "recommendation": "Patient meets FDA criteria for GLP-1 therapy",
    "clinical_reasoning": [
      "BMI 29.9 with comorbid hypertension meets weight loss indication",
      "Type 2 diabetes diagnosis provides strong indication",
      "No absolute contraindications detected",
      "Age 42 is appropriate for GLP-1 therapy",
      "Medication compliance expected (currently on 2 medications)"
    ],
    "key_considerations": [
      "Monitor for GI side effects (most common)",
      "Baseline renal function adequate",
      "Review medication interactions",
      "Assess weight loss goals (realistic and achievable)"
    ]
  },
  "physician_review": {
    "review_required": true,
    "review_level": "Standard",
    "focus_areas": [
      "Confirm diabetes diagnosis and control status",
      "Review renal function (eGFR)",
      "Assess GI tolerance risk",
      "Discuss weight loss expectations"
    ]
  },
  "constraints": {
    "hard_constraints_passed": true,
    "soft_constraints": [
      {
        "constraint": "Renal function",
        "status": "UNKNOWN",
        "action_required": "Obtain eGFR before prescribing"
      }
    ]
  },
  "processing_time": 0.82,
  "engine_version": "1.0.0"
}
```

**Frontend Integration:**

```javascript
class RecommendationManager {
  async getCompleteRecommendation(sessionId, collectedData) {
    try {
      // Show loading state
      this.showLoadingSpinner();

      // Step 1: Get eligibility
      const eligibilityResponse = await fetch(
        'http://localhost:8000/eligibility/evaluate',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            collected_data: collectedData,
            session_id: sessionId
          })
        }
      );

      const eligibility = await eligibilityResponse.json();

      // Handle eligibility result
      this.displayEligibilityResult(eligibility);

      // Check if patient is eligible to proceed
      if (!['ELIGIBLE', 'CONDITIONALLY_ELIGIBLE'].includes(
        eligibility.eligibility_status
      )) {
        this.handleIneligiblePatient(eligibility);
        return;
      }

      // If eligible, get full recommendation (includes medication + prescription)
      await this.getFullRecommendation(sessionId, collectedData);
    } catch (error) {
      console.error('Recommendation error:', error);
      this.displayError('Failed to get recommendation');
    }
  }

  displayEligibilityResult(eligibility) {
    const resultsDiv = document.getElementById('results-container');
    resultsDiv.innerHTML = `
      <div class="eligibility-card ${eligibility.eligibility_status.toLowerCase()}">
        <h2>Eligibility Assessment</h2>
        <div class="score-display">
          <div class="score-value">${eligibility.eligibility_score}</div>
          <div class="score-max">/100</div>
        </div>
        <p class="status ${eligibility.eligibility_status.toLowerCase()}">
          Status: ${eligibility.eligibility_status}
        </p>
        <p class="risk-level">Risk Level: ${eligibility.risk_level}</p>
        
        <h3>Clinical Assessment</h3>
        <ul class="clinical-details">
          ${Object.entries(eligibility.clinical_assessment)
            .map(([key, value]) => 
              `<li><strong>${this.formatKey(key)}:</strong> ${JSON.stringify(value).substring(0, 100)}</li>`
            )
            .join('')}
        </ul>

        <h3>Physician Recommendations</h3>
        <ul class="recommendations">
          ${eligibility.physician_review.focus_areas
            .map(area => `<li>• ${area}</li>`)
            .join('')}
        </ul>
      </div>
    `;
  }

  handleIneligiblePatient(eligibility) {
    const resultsDiv = document.getElementById('results-container');
    resultsDiv.innerHTML = `
      <div class="alert alert-danger">
        <h2>⚠️ ${eligibility.eligibility_status}</h2>
        <p>${eligibility.decision_support.recommendation}</p>
        <p>Please consult with your physician for alternative treatment options.</p>
      </div>
    `;
  }

  formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  showLoadingSpinner() {
    document.getElementById('results-container').innerHTML = 
      '<div class="spinner"></div><p>Analyzing your data...</p>';
  }

  displayError(message) {
    document.getElementById('results-container').innerHTML = 
      `<div class="alert alert-danger">${message}</div>`;
  }
}
```

---

#### 2. Check Contraindications (Quick Check)

**Fast contraindication screening without full scoring**

```http
POST /eligibility/check-contraindications
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 42,
  "is_pregnant_breastfeeding": false,
  "high_risk_conditions": [],
  "current_medical_conditions": "Type 2 diabetes"
}
```

**Response (200 OK):**
```json
{
  "has_contraindications": false,
  "contraindications": [],
  "safe_to_proceed": true,
  "recommendation": "Patient can proceed with full eligibility assessment."
}
```

**Error Case - With Contraindications:**
```json
{
  "has_contraindications": true,
  "contraindications": [
    "Patient is pregnant (absolute contraindication)",
    "History of medullary thyroid cancer (absolute contraindication)"
  ],
  "safe_to_proceed": false,
  "recommendation": "STOP: Absolute contraindications detected. Physician consultation required."
}
```

---

#### 3. Get Eligibility Criteria

**Retrieve eligibility criteria and scoring methodology**

```http
GET /eligibility/criteria
```

**Response (200 OK):**
```json
{
  "hard_constraints": {
    "description": "Absolute contraindications that prevent GLP-1 use",
    "criteria": [
      "Age ≥ 18 years",
      "Not pregnant or breastfeeding",
      "No history of medullary thyroid cancer or MEN-2",
      "No active pancreatitis or pancreatic cancer",
      "No severe gastroparesis",
      "No active eating disorders",
      "Not Type 1 diabetes (for weight loss indication)"
    ]
  },
  "scoring_criteria": {
    "total_points": 100,
    "breakdown": {
      "bmi_and_weight": {
        "points": 40,
        "description": "BMI ≥30 (obesity) or BMI ≥27 with comorbidities"
      },
      "diabetes_status": {
        "points": 25,
        "description": "Type 2 diabetes diagnosis (primary indication)"
      },
      "comorbidities": {
        "points": 20,
        "description": "Hypertension, dyslipidemia, PCOS, NAFLD, cardiovascular disease"
      },
      "weight_loss_goal": {
        "points": 15,
        "description": "Realistic, sustainable weight loss goals"
      }
    }
  },
  "thresholds": {
    "highly_eligible": "≥75 points",
    "eligible": "60-74 points",
    "conditionally_eligible": "40-59 points",
    "requires_review": "20-39 points",
    "not_eligible": "<20 points"
  },
  "clinical_guidelines": [
    "FDA Prescribing Information - Wegovy, Ozempic, Mounjaro",
    "ADA Standards of Medical Care in Diabetes 2024",
    "AACE Obesity Clinical Practice Guidelines 2023"
  ]
}
```

---

### MEDICATION RECOMMENDATION

#### 1. Get Complete Recommendation

**Generate complete recommendation with eligibility + medication ranking + prescription**

```http
POST /recommendation/complete
Content-Type: application/json
```

**Request Body:**
```json
{
  "collected_data": {
    "name": "Jane Smith",
    "age": 45,
    "height": "5'6\"",
    "weight": 185,
    "is_pregnant_breastfeeding": false,
    "high_risk_conditions": [],
    "current_medical_conditions": "Type 2 diabetes, hypertension",
    "currently_on_glp1": false,
    "other_medications": "Metformin, Lisinopril",
    "allergies": "None",
    "weight_loss_goal": "Lose 25 pounds",
    "interested_medication": "any"
  },
  "session_id": "20260201103045"
}
```

**Response (200 OK) - Complete Recommendation:**
```json
{
  "success": true,
  "timestamp": "2026-02-01T10:40:15.654321",
  "session_id": "20260201103045",
  "processing_time": 3.45,
  "eligibility": {
    "success": true,
    "eligibility_status": "ELIGIBLE",
    "eligibility_score": 82.5,
    "risk_level": "LOW",
    "clinical_assessment": {
      "bmi": {
        "value": 29.9,
        "category": "Overweight",
        "meets_criteria": true,
        "score": 40
      },
      "diabetes_status": {
        "has_type2_diabetes": true,
        "controlled": true,
        "score": 25
      },
      "comorbidities": {
        "present": ["hypertension"],
        "score": 20
      },
      "weight_loss_goal": {
        "realistic": true,
        "score": 15
      },
      "contraindications": {
        "has_contraindications": false,
        "violations": []
      }
    },
    "decision_support": {
      "recommendation": "Highly suitable candidate for GLP-1 therapy",
      "clinical_reasoning": [
        "BMI 29.9 with comorbid hypertension and diabetes",
        "Well-controlled Type 2 diabetes",
        "No absolute contraindications",
        "Realistic weight loss expectations (25 lbs)",
        "Stable on current medications"
      ]
    }
  },
  "recommended_medication": {
    "rank": 1,
    "medication": "Mounjaro (tirzepatide)",
    "total_score": 92.5,
    "efficacy_score": 95,
    "safety_score": 88,
    "convenience_score": 90,
    "cost_score": 85,
    "suitability_score": 96,
    "strengths": [
      "Dual GIP/GLP-1 agonist - superior weight loss vs GLP-1 alone",
      "Weekly injection - excellent compliance",
      "Cardiovascular benefits documented",
      "More effective for diabetes control than competitors",
      "Well-tolerated in comorbid conditions"
    ],
    "weaknesses": [
      "Similar GI side effects as other GLP-1s",
      "Requires refrigeration (manageable)"
    ],
    "rationale": "For patients with Type 2 diabetes AND obesity, Mounjaro's dual mechanism provides superior outcomes. Weekly dosing improves compliance, and clinical data shows better A1C reduction than semaglutide. Ideal for this patient."
  },
  "alternative_medications": [
    {
      "rank": 2,
      "medication": "Ozempic (semaglutide)",
      "total_score": 89.0,
      "efficacy_score": 88,
      "safety_score": 92,
      "convenience_score": 90,
      "cost_score": 82,
      "suitability_score": 87,
      "strengths": [
        "Most clinical evidence and long-term safety data",
        "Well-established dosing for diabetes",
        "Excellent cardiovascular outcomes (SUSTAIN-6 trial)",
        "Reasonable cost with insurance coverage"
      ],
      "weaknesses": [
        "GLP-1 monotherapy - less effective than dual agonists",
        "Slightly less weight loss vs Mounjaro"
      ],
      "rationale": "Excellent alternative if Mounjaro unavailable or cost-prohibitive. Proven track record for diabetes + weight loss with strong cardiovascular data."
    },
    {
      "rank": 3,
      "medication": "Zepbound (tirzepatide)",
      "total_score": 88.5,
      "efficacy_score": 95,
      "safety_score": 89,
      "convenience_score": 85,
      "cost_score": 88,
      "suitability_score": 89,
      "strengths": [
        "Same active drug as Mounjaro (same efficacy)",
        "FDA approved specifically for weight loss",
        "May have better insurance coverage under obesity indication",
        "Identical clinical outcomes to Mounjaro"
      ],
      "weaknesses": [
        "Newer medication - less long-term data than Ozempic",
        "Requires careful storage"
      ],
      "rationale": "Identical efficacy to Mounjaro but may offer better insurance coverage under obesity code. Consider if Mounjaro denied by insurance."
    },
    {
      "rank": 4,
      "medication": "Wegovy (semaglutide)",
      "total_score": 87.0,
      "efficacy_score": 85,
      "safety_score": 91,
      "convenience_score": 88,
      "cost_score": 79,
      "suitability_score": 84,
      "strengths": [
        "Higher dose titration available (up to 2.4 mg)",
        "FDA approved specifically for chronic weight management",
        "Excellent long-term weight loss (STEP trials)",
        "Safe in pre-diabetic and diabetic patients"
      ],
      "weaknesses": [
        "More expensive than Ozempic",
        "GLP-1 monotherapy (less effective than dual agonists)",
        "Longer titration schedule (16 weeks)"
      ],
      "rationale": "Good option if dual agonists unavailable. Proven weight loss efficacy but efficacy less than Mounjaro for diabetes+obesity combination."
    }
  ],
  "prescription": {
    "patient_name": "Jane Smith",
    "date": "2026-02-01",
    "medication_name": "Mounjaro (tirzepatide)",
    "starting_dose": "2.5 mg",
    "target_dose": "5 mg to 15 mg weekly",
    "titration_schedule": [
      {
        "weeks": "1-4",
        "dose": "2.5 mg subcutaneously once weekly"
      },
      {
        "weeks": "5+",
        "dose": "5 mg subcutaneously once weekly"
      },
      {
        "weeks": "If needed after ≥4 weeks",
        "dose": "7.5 mg subcutaneously once weekly"
      },
      {
        "weeks": "If needed after ≥4 weeks",
        "dose": "10 mg subcutaneously once weekly"
      },
      {
        "weeks": "If needed after ≥4 weeks",
        "dose": "12.5 mg or 15 mg once weekly (max)"
      }
    ],
    "route": "Subcutaneous injection",
    "frequency": "Once weekly (same day each week)",
    "indication": "Type 2 diabetes with obesity and cardiovascular risk factors",
    "dosing_instructions": "Inject subcutaneously into abdomen, thigh, or upper arm. Rotate injection sites. Use new needle each time. Can be taken any time of day with or without food.",
    "administration_technique": [
      "Allow pen to reach room temperature (15-30 min before injection)",
      "Visually inspect - solution should be clear and colorless",
      "Wash hands with soap and water",
      "Clean injection site with alcohol swab",
      "Pinch skin fold, insert needle at 90° angle",
      "Inject slowly over 5-6 seconds",
      "Remove needle and apply pressure with cotton ball"
    ],
    "baseline_labs": [
      "Fasting glucose",
      "HbA1c",
      "Comprehensive metabolic panel (BUN, creatinine, eGFR)",
      "Liver function tests (AST, ALT, alkaline phosphatase)",
      "Lipid panel",
      "Thyroid function (TSH, free T4)",
      "Baseline weight and BMI"
    ],
    "follow_up_visits": [
      {
        "timepoint": "Week 2",
        "purpose": "Assess tolerance and side effects",
        "tests": []
      },
      {
        "timepoint": "Week 4 (dose increase)",
        "purpose": "Evaluate initial response, side effects, dose escalation readiness",
        "tests": ["Blood glucose if available"]
      },
      {
        "timepoint": "Week 8",
        "purpose": "Assess tolerance to 5 mg dose",
        "tests": []
      },
      {
        "timepoint": "Week 12",
        "purpose": "Check weight loss progress, GI adaptation",
        "tests": ["Weight, vital signs"]
      },
      {
        "timepoint": "Month 3",
        "purpose": "Full assessment including labs",
        "tests": [
          "Fasting glucose",
          "HbA1c",
          "Comprehensive metabolic panel",
          "Weight, BMI, blood pressure"
        ]
      },
      {
        "timepoint": "Month 6",
        "purpose": "Maintenance assessment",
        "tests": [
          "HbA1c",
          "Renal function",
          "Weight loss progress",
          "Medication compliance"
        ]
      }
    ],
    "monitoring_parameters": [
      "Blood glucose (home monitoring if diabetic)",
      "Weight and BMI weekly",
      "Blood pressure",
      "Gastrointestinal symptoms (nausea, vomiting, diarrhea, constipation)",
      "Appetite changes",
      "Injection site reactions",
      "Signs of pancreatitis (severe abdominal pain)",
      "HbA1c every 3 months until stabilized",
      "Renal function annually"
    ],
    "common_side_effects": [
      {
        "symptom": "Nausea",
        "onset": "Days 1-7",
        "severity": "Usually mild-moderate",
        "management": "Take with food, ginger, small frequent meals, resolve within 1-2 weeks"
      },
      {
        "symptom": "Vomiting",
        "onset": "Days 1-7",
        "severity": "Mild-moderate",
        "management": "Usually resolves; contact MD if persistent"
      },
      {
        "symptom": "Diarrhea or constipation",
        "onset": "Days 1-14",
        "severity": "Mild-moderate",
        "management": "Increase fiber intake, hydration, stool softener if needed"
      },
      {
        "symptom": "Decreased appetite",
        "onset": "Days 1-7",
        "severity": "Expected",
        "management": "Ensure adequate protein and nutrient intake despite reduced appetite"
      },
      {
        "symptom": "Fatigue",
        "onset": "Days 1-7",
        "severity": "Mild",
        "management": "Usually improves in 1-2 weeks; ensure adequate sleep and hydration"
      }
    ],
    "serious_side_effects": [
      {
        "symptom": "Severe abdominal pain",
        "indicates": "Possible pancreatitis",
        "action": "STOP medication and seek immediate medical attention"
      },
      {
        "symptom": "Severe vomiting or dehydration",
        "indicates": "Dehydration, acute kidney injury risk",
        "action": "Seek medical attention; may need IV fluids"
      },
      {
        "symptom": "Severe allergic reaction (rash, swelling, breathing difficulty)",
        "indicates": "Possible allergy to tirzepatide",
        "action": "Seek emergency care; discontinue medication"
      },
      {
        "symptom": "Rapid heartbeat or chest pain",
        "indicates": "Cardiac event",
        "action": "Seek immediate medical attention"
      },
      {
        "symptom": "Vision changes",
        "indicates": "Retinopathy worsening",
        "action": "Contact ophthalmologist and endocrinologist"
      },
      {
        "symptom": "Signs of thyroid tumor (lump in neck, hoarseness, difficulty swallowing)",
        "indicates": "Possible thyroid condition",
        "action": "Contact physician immediately"
      }
    ],
    "drug_interactions": [
      {
        "drug": "Metformin",
        "interaction": "No significant interaction",
        "recommendation": "Continue; may improve glycemic control"
      },
      {
        "drug": "Lisinopril (ACE inhibitor)",
        "interaction": "Additive hypotensive effect",
        "recommendation": "Monitor blood pressure; may need dose adjustment"
      },
      {
        "drug": "Sulfonylureas or insulin",
        "interaction": "Increased hypoglycemia risk",
        "recommendation": "May need dose reduction of other agents"
      },
      {
        "drug": "Oral contraceptives",
        "interaction": "Delayed gastric emptying may reduce absorption",
        "recommendation": "Consider alternative contraception or monitoring"
      }
    ],
    "lifestyle_modifications": [
      "Increase protein intake to preserve muscle during weight loss (1.2-1.5 g/kg/day)",
      "Reduce refined carbohydrates and sugary foods",
      "Increase physical activity gradually (150 min moderate aerobic activity/week)",
      "Strength training 2x/week to preserve lean mass",
      "Improve sleep quality (7-9 hours/night)",
      "Manage stress through meditation or counseling",
      "Eliminate sugary beverages",
      "Practice mindful eating and portion control"
    ],
    "dietary_recommendations": [
      "High protein meals: fish, chicken, tofu, legumes, Greek yogurt",
      "Non-starchy vegetables: leafy greens, broccoli, bell peppers, zucchini",
      "Healthy fats: olive oil, nuts, avocado, fatty fish",
      "Complex carbohydrates: whole grains, oats, brown rice (in moderation)",
      "AVOID: Fast food, fried foods, sugary snacks, high-fat processed foods",
      "Smaller portions: appetite suppression requires conscious portion awareness",
      "Frequent small meals if tolerated"
    ],
    "expected_outcomes": {
      "weight_loss": "5-10% body weight at 12 weeks; 15-22% at 52 weeks (average)",
      "glycemic_control": "HbA1c reduction of 1.5-2.0% (if diabetic)",
      "blood_pressure": "Modest reduction of 3-5 mmHg",
      "metabolic": "Improved lipid panel, reduced liver fat",
      "timeline": "Maximum benefit at 6-9 months; some weight loss plateau after 12 months"
    },
    "when_to_contact_physician": [
      "Severe or persistent nausea/vomiting (>3 days)",
      "Severe abdominal pain (possible pancreatitis)",
      "Severe diarrhea or constipation lasting >7 days",
      "Signs of severe allergic reaction",
      "Chest pain or shortness of breath",
      "Vision changes",
      "Signs of hypoglycemia (shakiness, sweating, confusion) if on insulin/sulfonylureas",
      "Dehydration symptoms (extreme thirst, dark urine, dizziness)",
      "Any concerning symptoms not listed here"
    ],
    "cbr_metadata": {
      "case_based_rationale": "Generated from case of 46-year-old female with Type 2 diabetes, BMI 30, on metformin + antihypertensive",
      "similarity_score": 92,
      "adaptations_made": [
        "Adjusted baseline labs for patient's hypertension status",
        "Modified follow-up schedule based on patient comorbidities",
        "Emphasized cardiovascular monitoring given hypertension",
        "Added specific guidance on ACE inhibitor interaction"
      ],
      "selected_by": "MCDM (Multi-Criteria Decision Making)"
    }
  },
  "next_steps": [
    "✅ Recommended medication: Mounjaro (tirzepatide)",
    "Review complete prescription details with physician",
    "Schedule baseline labs: Fasting glucose, HbA1c, Comprehensive metabolic panel",
    "Obtain prior authorization from insurance (if required)",
    "Schedule follow-up appointment in 2 weeks to assess tolerance",
    "Physician will review and sign prescription if appropriate"
  ],
  "physician_review_required": true
}
```

**Frontend Integration:**

```javascript
async function displayCompleteRecommendation(recommendation) {
  const container = document.getElementById('recommendation-container');
  
  // Build recommendation tabs
  const tabs = `
    <div class="recommendation-tabs">
      <button class="tab-btn active" data-tab="eligibility">Eligibility</button>
      <button class="tab-btn" data-tab="medication">Medication</button>
      <button class="tab-btn" data-tab="prescription">Prescription</button>
      <button class="tab-btn" data-tab="next-steps">Next Steps</button>
    </div>
  `;
  
  // Eligibility Tab
  const eligibilityTab = `
    <div class="tab-content" id="eligibility-tab">
      <h2>Eligibility Assessment</h2>
      <div class="status-badge ${recommendation.eligibility.eligibility_status.toLowerCase()}">
        ${recommendation.eligibility.eligibility_status}
      </div>
      <p class="score">Score: ${recommendation.eligibility.eligibility_score}/100</p>
      <p class="risk">Risk Level: <span class="risk-${recommendation.eligibility.risk_level.toLowerCase()}">${recommendation.eligibility.risk_level}</span></p>
      
      <h3>Clinical Assessment</h3>
      <div class="assessment-grid">
        ${Object.entries(recommendation.eligibility.clinical_assessment)
          .map(([key, val]) => `
            <div class="assessment-card">
              <h4>${this.formatKey(key)}</h4>
              <p>${JSON.stringify(val).substring(0, 50)}...</p>
            </div>
          `).join('')}
      </div>
      
      <h3>Clinical Reasoning</h3>
      <ul>
        ${recommendation.eligibility.decision_support.clinical_reasoning
          .map(reason => `<li>✓ ${reason}</li>`).join('')}
      </ul>
    </div>
  `;
  
  // Medication Tab
  const medicationTab = `
    <div class="tab-content" id="medication-tab">
      <h2>Medication Recommendation</h2>
      
      <div class="recommended-card primary">
        <h3>🏆 #1 Recommended: ${recommendation.recommended_medication.medication}</h3>
        <div class="score-breakdown">
          <div class="score-item">
            <label>Total Score:</label>
            <span class="score-value">${recommendation.recommended_medication.total_score}/100</span>
          </div>
          <div class="score-item">
            <label>Efficacy:</label>
            <span>${recommendation.recommended_medication.efficacy_score}/100</span>
          </div>
          <div class="score-item">
            <label>Safety:</label>
            <span>${recommendation.recommended_medication.safety_score}/100</span>
          </div>
          <div class="score-item">
            <label>Convenience:</label>
            <span>${recommendation.recommended_medication.convenience_score}/100</span>
          </div>
          <div class="score-item">
            <label>Cost:</label>
            <span>${recommendation.recommended_medication.cost_score}/100</span>
          </div>
        </div>
        
        <h4>Why This Medication?</h4>
        <p>${recommendation.recommended_medication.rationale}</p>
        
        <h4>Strengths</h4>
        <ul>
          ${recommendation.recommended_medication.strengths
            .map(s => `<li>✓ ${s}</li>`).join('')}
        </ul>
        
        <h4>Considerations</h4>
        <ul>
          ${recommendation.recommended_medication.weaknesses
            .map(w => `<li>⚠️ ${w}</li>`).join('')}
        </ul>
      </div>
      
      <h3>Alternative Options</h3>
      ${recommendation.alternative_medications.map((med, idx) => `
        <div class="alternative-card">
          <h4>#${med.rank} ${med.medication}</h4>
          <p class="score">Score: ${med.total_score}/100</p>
          <p>${med.rationale}</p>
          <button class="btn-secondary" onclick="showDetails('${med.medication}')">
            View Details
          </button>
        </div>
      `).join('')}
    </div>
  `;
  
  // Prescription Tab
  const prescriptionTab = `
    <div class="tab-content" id="prescription-tab">
      <h2>Prescription Details</h2>
      
      <div class="prescription-header">
        <p><strong>Patient:</strong> ${recommendation.prescription.patient_name}</p>
        <p><strong>Date:</strong> ${recommendation.prescription.date}</p>
        <p><strong>Medication:</strong> ${recommendation.prescription.medication_name}</p>
      </div>
      
      <h3>Dosing Schedule</h3>
      <div class="dosing-table">
        <table>
          <thead>
            <tr>
              <th>Period</th>
              <th>Dose</th>
            </tr>
          </thead>
          <tbody>
            ${recommendation.prescription.titration_schedule.map(schedule => `
              <tr>
                <td>${Object.keys(schedule)[0]}: ${Object.values(schedule)[0]}</td>
                <td>${Object.values(schedule)[1]}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
      
      <h3>Administration</h3>
      <p><strong>Route:</strong> ${recommendation.prescription.route}</p>
      <p><strong>Frequency:</strong> ${recommendation.prescription.frequency}</p>
      <ol>
        ${recommendation.prescription.administration_technique
          .map(step => `<li>${step}</li>`).join('')}
      </ol>
      
      <h3>Monitoring Requirements</h3>
      <h4>Baseline Labs Needed:</h4>
      <ul>
        ${recommendation.prescription.baseline_labs
          .map(lab => `<li>✓ ${lab}</li>`).join('')}
      </ul>
      
      <h4>Follow-up Schedule:</h4>
      ${recommendation.prescription.follow_up_visits.map(visit => `
        <div class="follow-up-card">
          <strong>${visit.timepoint}:</strong> ${visit.purpose}
          ${visit.tests.length > 0 ? `<br/>Tests: ${visit.tests.join(', ')}` : ''}
        </div>
      `).join('')}
      
      <h3>Side Effects</h3>
      <h4>Common (Usually Mild):</h4>
      <ul>
        ${recommendation.prescription.common_side_effects
          .map(se => `
            <li>
              <strong>${se.symptom}</strong> (${se.onset}): ${se.management}
            </li>
          `).join('')}
      </ul>
      
      <h4>⚠️ Serious (Contact MD Immediately):</h4>
      <ul>
        ${recommendation.prescription.serious_side_effects
          .map(se => `
            <li>
              <strong>${se.symptom}</strong> → ${se.action}
            </li>
          `).join('')}
      </ul>
      
      <h3>Dietary & Lifestyle</h3>
      <h4>Recommended Foods:</h4>
      <ul>
        ${recommendation.prescription.dietary_recommendations.slice(0, 5)
          .map(item => `<li>${item}</li>`).join('')}
      </ul>
      
      <h4>Lifestyle Changes:</h4>
      <ul>
        ${recommendation.prescription.lifestyle_modifications
          .map(mod => `<li>${mod}</li>`).join('')}
      </ul>
      
      <h3>Drug Interactions</h3>
      <table>
        <thead>
          <tr>
            <th>Drug</th>
            <th>Interaction</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          ${recommendation.prescription.drug_interactions.map(interaction => `
            <tr>
              <td>${interaction.drug}</td>
              <td>${interaction.interaction}</td>
              <td>${interaction.recommendation}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
      
      <h3>Expected Outcomes</h3>
      <ul>
        <li><strong>Weight Loss:</strong> ${recommendation.prescription.expected_outcomes.weight_loss}</li>
        <li><strong>Blood Sugar Control:</strong> ${recommendation.prescription.expected_outcomes.glycemic_control}</li>
        <li><strong>Blood Pressure:</strong> ${recommendation.prescription.expected_outcomes.blood_pressure}</li>
        <li><strong>Timeline:</strong> ${recommendation.prescription.expected_outcomes.timeline}</li>
      </ul>
    </div>
  `;
  
  // Next Steps Tab
  const nextStepsTab = `
    <div class="tab-content" id="next-steps-tab">
      <h2>Action Plan</h2>
      ${recommendation.physician_review_required ? 
        '<div class="alert alert-info">⚠️ Physician Review Required</div>' : ''}
      
      <ol class="next-steps-list">
        ${recommendation.next_steps
          .map(step => `<li>${step}</li>`).join('')}
      </ol>
      
      <div class="action-buttons">
        <button class="btn-primary" onclick="downloadPrescription()">
          📥 Download Prescription PDF
        </button>
        <button class="btn-secondary" onclick="sendToPractitioner()">
          📧 Send to My Doctor
        </button>
        <button class="btn-secondary" onclick="printRecommendation()">
          🖨️ Print Report
        </button>
      </div>
    </div>
  `;
  
  // Assemble and display
  container.innerHTML = tabs + eligibilityTab + medicationTab + prescriptionTab + nextStepsTab;
  
  // Add tab switching functionality
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      
      const tabId = e.target.dataset.tab + '-tab';
      document.getElementById(tabId).style.display = 'block';
      e.target.classList.add('active');
    });
  });
  
  // Show first tab by default
  document.getElementById('eligibility-tab').style.display = 'block';
}
```

---

#### 2. Get Pipeline Info

**Retrieve information about the recommendation pipeline**

```http
GET /recommendation/pipeline-info
```

**Response (200 OK):**
```json
{
  "pipeline": "GLP-1 Prescription Recommendation System",
  "version": "1.0.1",
  "update": "CBR now uses MCDM-selected medication for consistency",
  "stages": [
    {
      "stage": 1,
      "name": "Eligibility Determination",
      "method": "Rule-based Clinical Decision Support System (CDSS)",
      "output": "Eligibility status + score (0-100)"
    },
    {
      "stage": 2,
      "name": "Medication Selection",
      "method": "Multi-Criteria Decision Making (MCDM)",
      "output": "Ranked list of 8 GLP-1 medications"
    },
    {
      "stage": 3,
      "name": "Prescription Generation",
      "method": "Case-Based Reasoning (CBR) with MCDM-selected medication",
      "output": "Personalized prescription document"
    }
  ],
  "safety_features": [
    "Contraindication checking",
    "Drug interaction detection",
    "Age-based dose adjustments",
    "Comorbidity-specific monitoring",
    "Mandatory physician review",
    "Consistent MCDM-CBR medication selection"
  ]
}
```

---

### HEALTH & UTILITY

#### 1. Health Check

**Check API health status**

```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model": "llama3:8b"
}
```

---

#### 2. Get Collected Data

**Retrieve previously collected data for a session** *(Not yet implemented)*

```http
GET /collected-data/{session_id}
```

**Response (200 OK):**
```json
{
  "message": "Session storage not implemented yet"
}
```

---

## Data Models

### ChatRequest
```json
{
  "message": "string (required, 1-1000 chars)",
  "conversation_history": [
    {
      "role": "user | assistant",
      "content": "string",
      "timestamp": "ISO8601 datetime"
    }
  ],
  "collected_data": { "field": "value" },
  "session_id": "string (optional)"
}
```

### ChatResponse
```json
{
  "response": "string",
  "collected_data": { "field": "value" },
  "completion_percentage": "float (0-100)",
  "is_complete": "boolean",
  "next_expected_field": "string | null",
  "conversation_history": [{ "role", "content", "timestamp" }]
}
```

### EligibilityRequest
```json
{
  "collected_data": {
    "name": "string",
    "age": "number",
    "height": "string",
    "weight": "number",
    "is_pregnant_breastfeeding": "boolean",
    "high_risk_conditions": ["string"],
    "current_medical_conditions": "string",
    "currently_on_glp1": "boolean",
    "other_medications": "string",
    "allergies": "string",
    "weight_loss_goal": "string",
    "interested_medication": "string"
  },
  "session_id": "string (optional)"
}
```

### CompleteRecommendationRequest
```json
{
  "collected_data": { /* same as EligibilityRequest */ },
  "session_id": "string (optional)"
}
```

---

## Frontend Integration Guide

### React Example (Complete Implementation)

```jsx
import React, { useState, useRef, useEffect } from 'react';
import './ChatInterface.css';

const API_BASE_URL = 'http://localhost:8000';

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [collectedData, setCollectedData] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [recommendation, setRecommendation] = useState(null);
  const messagesEndRef = useRef(null);

  // Initialize conversation
  useEffect(() => {
    startConversation();
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const startConversation = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/start-conversation`, {
        method: 'POST'
      });
      const data = await response.json();
      
      setSessionId(data.session_id);
      setMessages([{
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString()
      }]);
    } catch (error) {
      console.error('Error starting conversation:', error);
      setMessages([{
        role: 'error',
        content: 'Failed to start conversation. Please refresh the page.',
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setIsLoading(true);

    // Add user message to UI
    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    }]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          conversation_history: messages,
          collected_data: collectedData,
          session_id: sessionId
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      // Update state
      setMessages(data.conversation_history);
      setCollectedData(data.collected_data);
      setProgress(data.completion_percentage);
      setIsComplete(data.is_complete);

    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        role: 'error',
        content: 'Failed to get response. Please try again.',
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const getRecommendation = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/recommendation/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          collected_data: collectedData,
          session_id: sessionId
        })
      });

      const data = await response.json();
      setRecommendation(data);
    } catch (error) {
      console.error('Recommendation error:', error);
      alert('Failed to get recommendation');
    } finally {
      setIsLoading(false);
    }
  };

  if (recommendation) {
    return <RecommendationDisplay recommendation={recommendation} />;
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>GLP-1 Assessment Assistant</h1>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }}></div>
        </div>
        <p className="progress-text">{Math.round(progress)}% Complete</p>
      </div>

      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
            <small className="timestamp">{new Date(msg.timestamp).toLocaleTimeString()}</small>
          </div>
        ))}
        {isLoading && <div className="message loading">Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
          placeholder="Type your response..."
          disabled={isLoading || isComplete}
        />
        <button 
          onClick={sendMessage} 
          disabled={isLoading || !inputValue.trim()}
          className="send-btn"
        >
          Send
        </button>
      </div>

      {isComplete && (
        <div className="action-area">
          <button 
            onClick={getRecommendation}
            disabled={isLoading}
            className="btn-primary"
          >
            Get My Recommendation
          </button>
        </div>
      )}
    </div>
  );
}

function RecommendationDisplay({ recommendation }) {
  return (
    <div className="recommendation-container">
      <div className="rec-tabs">
        <RecommendationTabs recommendation={recommendation} />
      </div>
      <button className="btn-secondary" onClick={() => window.location.reload()}>
        Start New Assessment
      </button>
    </div>
  );
}
```

### Vue.js Example

```vue
<template>
  <div class="chat-interface">
    <div class="chat-header">
      <h1>GLP-1 Assessment</h1>
      <div class="progress">
        <div class="progress-bar" :style="{ width: progress + '%' }"></div>
        <span>{{ Math.round(progress) }}%</span>
      </div>
    </div>

    <div class="messages" v-if="!showRecommendation">
      <div v-for="(msg, idx) in messages" :key="idx" :class="['message', msg.role]">
        {{ msg.content }}
      </div>
    </div>

    <div class="input-area" v-if="!showRecommendation && !isComplete">
      <input 
        v-model="inputValue"
        @keyup.enter="sendMessage"
        placeholder="Type here..."
        :disabled="isLoading"
      />
      <button @click="sendMessage" :disabled="isLoading || !inputValue">
        Send
      </button>
    </div>

    <button 
      v-if="isComplete && !showRecommendation"
      @click="getRecommendation"
      :disabled="isLoading"
      class="btn-primary"
    >
      Get My Recommendation
    </button>

    <div v-if="showRecommendation" class="recommendation">
      <RecommendationComponent :data="recommendation" />
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      messages: [],
      inputValue: '',
      sessionId: null,
      collectedData: {},
      progress: 0,
      isComplete: false,
      isLoading: false,
      showRecommendation: false,
      recommendation: null
    }
  },
  mounted() {
    this.startConversation();
  },
  methods: {
    async startConversation() {
      const res = await fetch('http://localhost:8000/start-conversation', {
        method: 'POST'
      });
      const data = await res.json();
      this.sessionId = data.session_id;
      this.messages = [{ role: 'assistant', content: data.response }];
    },
    async sendMessage() {
      if (!this.inputValue.trim()) return;
      
      this.messages.push({ role: 'user', content: this.inputValue });
      const msg = this.inputValue;
      this.inputValue = '';
      this.isLoading = true;

      try {
        const res = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: msg,
            conversation_history: this.messages,
            collected_data: this.collectedData,
            session_id: this.sessionId
          })
        });
        const data = await res.json();
        this.messages = data.conversation_history;
        this.collectedData = data.collected_data;
        this.progress = data.completion_percentage;
        this.isComplete = data.is_complete;
      } catch (e) {
        console.error(e);
      } finally {
        this.isLoading = false;
      }
    },
    async getRecommendation() {
      this.isLoading = true;
      try {
        const res = await fetch('http://localhost:8000/recommendation/complete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            collected_data: this.collectedData,
            session_id: this.sessionId
          })
        });
        this.recommendation = await res.json();
        this.showRecommendation = true;
      } finally {
        this.isLoading = false;
      }
    }
  }
}
</script>
```

---

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Status Codes

| Code | Meaning | Solution |
|------|---------|----------|
| `200` | Success | No action needed |
| `400` | Bad Request | Check request body syntax and required fields |
| `404` | Not Found | Check endpoint URL spelling |
| `500` | Server Error | Check backend logs, restart server if needed |
| `504` | Gateway Timeout | Ollama model taking too long; restart Ollama |

### Error Examples

**Missing Required Field:**
```json
{
  "detail": "Missing required fields for eligibility evaluation: age, weight"
}
```

**Ollama Timeout:**
```json
{
  "detail": "Ollama model took too long to respond. Try restarting Ollama."
}
```

**Ollama Connection Error:**
```json
{
  "detail": "Ollama error: Connection refused"
}
```

---

## Complete Flow Examples

### Example 1: Conversation → Eligibility → Recommendation

```javascript
// Step 1: Start conversation
POST /start-conversation
← session_id: "20260201103045"

// Step 2: Send messages until is_complete = true
POST /chat
{
  "message": "My name is John",
  "session_id": "20260201103045",
  ...
}
← completion_percentage: 95%
← is_complete: true

// Step 3: Get complete recommendation
POST /recommendation/complete
{
  "collected_data": { /* all collected data */ },
  "session_id": "20260201103045"
}
← {
    "eligibility": { ... },
    "recommended_medication": { ... },
    "prescription": { ... },
    "next_steps": [ ... ]
  }
```

### Example 2: Direct Eligibility Check (Skip Chat)

If you already have collected data from another source:

```javascript
// Skip chat, go straight to eligibility
POST /eligibility/evaluate
{
  "collected_data": {
    "name": "John Doe",
    "age": 45,
    "height": "5'10\"",
    "weight": 220,
    ...
  }
}
← EligibilityResponse

// If eligible, get full recommendation
POST /recommendation/complete
{
  "collected_data": { ... }
}
← CompleteRecommendationResponse
```

---

## Troubleshooting

### API Not Responding

1. **Check Backend is Running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check Ollama is Running:**
   ```bash
   curl http://localhost:11434/api/generate
   ```

3. **Restart Backend:**
   ```bash
   python main.py
   ```

### Timeout Errors

- Ollama model is slow on first request
- Solution: Increase timeout in frontend or restart Ollama
- Restart Ollama: `ollama serve llama3:8b`

### CORS Errors

- Make sure frontend is making requests to `http://localhost:8000`
- CORS is enabled for all origins on the backend

### Missing Dependencies

```bash
pip install -r requirements.txt
```

---

## Contact & Support

For issues or questions:
- Check logs: `python main.py` output
- Verify backend running: `curl http://localhost:8000/health`
- Ensure Ollama running: `ollama serve`

---

**Version:** 1.0.1  
**Last Updated:** February 1, 2026
