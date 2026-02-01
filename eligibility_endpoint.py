"""
FastAPI Endpoint for GLP-1 Eligibility Evaluation

Integrates the research-grade eligibility engine into the existing backend.
Provides RESTful API for eligibility determination.

This endpoint should be added to your main.py FastAPI application.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Import the eligibility engine
from eligibility_engine import (
    GLP1EligibilityEngine,
    format_eligibility_response,
    EligibilityStatus
)

logger = logging.getLogger(__name__)

# Create router (can be included in main app)
eligibility_router = APIRouter(prefix="/eligibility", tags=["eligibility"])


# ==================== REQUEST/RESPONSE MODELS ====================

class EligibilityRequest(BaseModel):
    """Request model for eligibility evaluation"""
    collected_data: Dict[str, Any] = Field(..., description="Patient data from chat endpoint")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "collected_data": {
                    "name": "Jane Smith",
                    "age": 42,
                    "height": "5'6\"",
                    "weight": 185,
                    "is_pregnant_breastfeeding": False,
                    "high_risk_conditions": [],
                    "current_medical_conditions": "Type 2 diabetes",
                    "currently_on_glp1": False,
                    "other_medications": "Metformin",
                    "allergies": "None",
                    "weight_loss_goal": "Lose 25 pounds",
                    "interested_medication": "Ozempic"
                },
                "session_id": "20260131123456"
            }
        }


class EligibilityResponse(BaseModel):
    """Response model for eligibility evaluation"""
    success: bool
    timestamp: str
    session_id: Optional[str]
    
    # Core results
    eligibility_status: str
    eligibility_score: float
    risk_level: str
    
    # Detailed assessment
    clinical_assessment: Dict[str, Any]
    decision_support: Dict[str, Any]
    physician_review: Dict[str, Any]
    constraints: Dict[str, Any]
    
    # Metadata
    processing_time: float
    engine_version: str = "1.0.0"


# ==================== ENDPOINTS ====================

@eligibility_router.post("/evaluate", response_model=EligibilityResponse)
async def evaluate_eligibility(request: EligibilityRequest):
    """
    Evaluate patient eligibility for GLP-1 prescription.
    
    This endpoint performs a comprehensive clinical decision support analysis
    based on collected patient data. The evaluation includes:
    
    1. Hard constraint validation (contraindications)
    2. Multi-criteria scoring (BMI, diabetes, comorbidities)
    3. Risk stratification
    4. Clinical reasoning and recommendation
    
    **Academic Foundation:**
    - Rule-based CDSS approach (Kawamoto et al., 2005)
    - Explainable medical AI (Rudin, 2019)
    - FDA prescribing guidelines compliance
    
    **Returns:**
    - Eligibility determination with detailed rationale
    - Clinical decision support information
    - Physician review recommendations
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"🔍 Evaluating eligibility for session: {request.session_id}")
        
        # Validate required fields
        required_fields = [
            "age", "height", "weight", "is_pregnant_breastfeeding",
            "high_risk_conditions", "current_medical_conditions"
        ]
        
        missing_fields = [f for f in required_fields if f not in request.collected_data]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields for eligibility evaluation: {', '.join(missing_fields)}"
            )
        
        # Run eligibility engine
        result = GLP1EligibilityEngine.evaluate(request.collected_data)
        
        # Format response
        formatted_result = format_eligibility_response(result)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = EligibilityResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            session_id=request.session_id,
            eligibility_status=formatted_result["eligibility_status"],
            eligibility_score=formatted_result["eligibility_score"],
            risk_level=formatted_result["risk_level"],
            clinical_assessment=formatted_result["clinical_assessment"],
            decision_support=formatted_result["decision_support"],
            physician_review=formatted_result["physician_review"],
            constraints=formatted_result["constraints"],
            processing_time=round(processing_time, 3)
        )
        
        logger.info(f"✅ Eligibility evaluation completed: {result.status.value} (score: {result.score})")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Eligibility evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Eligibility evaluation error: {str(e)}"
        )


@eligibility_router.post("/check-contraindications")
async def check_contraindications(collected_data: Dict[str, Any]):
    """
    Quick endpoint to check only for absolute contraindications.
    
    Useful for early screening during data collection phase.
    Runs STAGE 1 only (hard constraints) without full scoring.
    
    **Returns:**
    - Boolean: has_contraindications
    - List of detected contraindications
    - Recommendation: proceed or stop
    """
    from eligibility_engine import HardConstraintValidator
    
    try:
        passed, violations = HardConstraintValidator.validate(collected_data)
        
        return {
            "has_contraindications": not passed,
            "contraindications": violations,
            "safe_to_proceed": passed,
            "recommendation": (
                "Patient can proceed with full eligibility assessment."
                if passed else
                "STOP: Absolute contraindications detected. Physician consultation required."
            )
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@eligibility_router.get("/criteria")
async def get_eligibility_criteria():
    """
    Return the eligibility criteria and scoring methodology.
    
    Useful for transparency and patient education.
    """
    return {
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


# ==================== INTEGRATION HELPER ====================

def add_eligibility_routes_to_app(app):
    """
    Helper function to integrate eligibility routes into main FastAPI app.
    
    Usage in main.py:
        from eligibility_endpoint import add_eligibility_routes_to_app
        add_eligibility_routes_to_app(app)
    """
    app.include_router(eligibility_router)
    logger.info("✅ Eligibility routes registered")


# ==================== STANDALONE TESTING ====================

if __name__ == "__main__":
    """
    Standalone test server for eligibility endpoint.
    Run: python eligibility_endpoint.py
    """
    import uvicorn
    from fastapi import FastAPI
    
    test_app = FastAPI(title="GLP-1 Eligibility Engine - Test Server")
    test_app.include_router(eligibility_router)
    
    @test_app.get("/")
    async def root():
        return {
            "service": "GLP-1 Eligibility Evaluation Engine",
            "version": "1.0.0",
            "status": "active",
            "endpoints": {
                "evaluate": "/eligibility/evaluate",
                "check_contraindications": "/eligibility/check-contraindications",
                "criteria": "/eligibility/criteria"
            }
        }
    
    print("\n" + "=" * 70)
    print("🏥 GLP-1 Eligibility Engine - Test Server")
    print("=" * 70)
    print("📍 Server: http://localhost:8001")
    print("📚 Docs:   http://localhost:8001/docs")
    print("=" * 70 + "\n")
    
    uvicorn.run(test_app, host="0.0.0.0", port=8001)