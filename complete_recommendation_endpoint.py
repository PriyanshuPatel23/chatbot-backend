"""
Complete GLP-1 Recommendation Pipeline - Integration Endpoint
FRONTEND-COMPATIBLE VERSION

This version matches the exact data structure expected by the React frontend.

This module integrates all three components:
1. Eligibility Determination (Rule-based CDSS)
2. Medication Selection (MCDM)
3. Prescription Generation (CBR)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Import all three engines
from eligibility_engine import (
    GLP1EligibilityEngine,
    EligibilityStatus,
    RiskLevel
)
from medication_selection_mcdm import (
    select_optimal_medication,
    MedicationScore,
    GLP1Medication
)
from prescription_generation_cbr import (
    CBREngine,
    PrescriptionOutput,
    CaseLibrary
)

logger = logging.getLogger(__name__)

# Create router
recommendation_router = APIRouter(prefix="/recommendation", tags=["recommendation"])


# ==================== REQUEST/RESPONSE MODELS ====================

class CompleteRecommendationRequest(BaseModel):
    """Request for complete GLP-1 recommendation pipeline"""
    collected_data: Dict[str, Any] = Field(..., description="Patient data from chat endpoint")
    session_id: Optional[str] = Field(None, description="Session identifier")


class MedicationRecommendation(BaseModel):
    """Single medication recommendation with scoring"""
    rank: int
    medication: str
    total_score: float
    efficacy_score: float
    safety_score: float
    convenience_score: float
    cost_score: float
    suitability_score: float
    strengths: List[str]
    weaknesses: List[str]
    rationale: str


class CompleteRecommendationResponse(BaseModel):
    """Complete recommendation response with all components"""
    success: bool
    timestamp: str
    session_id: Optional[str]
    processing_time: float
    
    # Component 1: Eligibility
    eligibility: Dict[str, Any]
    
    # Component 2: Medication Selection
    recommended_medication: Optional[MedicationRecommendation] = None
    alternative_medications: List[MedicationRecommendation] = []
    
    # Component 3: Prescription
    prescription: Optional[Dict[str, Any]] = None
    
    # Summary
    next_steps: List[str]
    physician_review_required: bool


# ==================== MAIN ENDPOINT ====================

@recommendation_router.post("/complete", response_model=CompleteRecommendationResponse)
async def generate_complete_recommendation(request: CompleteRecommendationRequest):
    """
    Generate complete GLP-1 recommendation: Eligibility → Medication → Prescription
    """
    start_time = datetime.now()
    
    try:
        logger.info("=" * 70)
        logger.info("🚀 STARTING COMPLETE RECOMMENDATION PIPELINE")
        logger.info("=" * 70)
        
        # ────────────────────────────────────────────────────────────
        # STAGE 1: ELIGIBILITY DETERMINATION
        # ────────────────────────────────────────────────────────────
        
        logger.info("\n📋 STAGE 1: Eligibility Determination")
        
        eligibility_result = GLP1EligibilityEngine.evaluate(request.collected_data)
        
        # Format for frontend (matches expected structure)
        eligibility_formatted = format_eligibility_for_frontend(eligibility_result, request.collected_data)
        
        logger.info(f"✅ Eligibility: {eligibility_result.status.value} (score: {eligibility_result.score})")
        
        # Check if patient is eligible enough to proceed
        can_prescribe = eligibility_result.status not in [
            EligibilityStatus.CONTRAINDICATED,
            EligibilityStatus.NOT_ELIGIBLE
        ]
        
        # ────────────────────────────────────────────────────────────
        # STAGE 2: MEDICATION SELECTION (only if eligible)
        # ────────────────────────────────────────────────────────────
        
        medication_rankings = []
        recommended_med = None
        alternatives = []
        
        if can_prescribe:
            logger.info("\n💊 STAGE 2: Medication Selection (MCDM)")
            
            medication_rankings = select_optimal_medication(
                request.collected_data,
                eligibility_result
            )
            
            # Top recommendation
            recommended_med = medication_rankings[0]
            alternatives = medication_rankings[1:4]  # Next 3 alternatives
            
            logger.info(f"✅ Recommended: {recommended_med.medication.value} (score: {recommended_med.total_score})")
            
        else:
            logger.info("\n⚠️  STAGE 2: SKIPPED - Patient not eligible for GLP-1 therapy")
        
        # ────────────────────────────────────────────────────────────
        # STAGE 3: PRESCRIPTION GENERATION (only if eligible)
        # ────────────────────────────────────────────────────────────
        
        prescription_data = None
        
        if can_prescribe and recommended_med:
            logger.info("\n📄 STAGE 3: Prescription Generation (CBR)")
            logger.info(f"   Using MCDM-selected medication: {recommended_med.medication.value}")
            
            prescription_output = generate_prescription_for_medication(
                request.collected_data,
                recommended_med.medication.value,
                eligibility_result
            )
            
            # Convert to dict
            prescription_data = {
                "patient_name": prescription_output.patient_name,
                "date": prescription_output.date,
                "medication_name": prescription_output.medication_name,
                "starting_dose": prescription_output.starting_dose,
                "target_dose": prescription_output.target_maintenance_dose,
                "titration_schedule": prescription_output.titration_schedule,
                "route": prescription_output.route_of_administration,
                "frequency": prescription_output.frequency,
                "indication": prescription_output.indication,
                "dosing_instructions": prescription_output.dosing_instructions,
                "administration_technique": prescription_output.administration_technique,
                "baseline_labs": prescription_output.baseline_labs,
                "follow_up_visits": prescription_output.follow_up_visits,
                "monitoring_parameters": prescription_output.monitoring_parameters,
                "common_side_effects": prescription_output.common_side_effects,
                "serious_side_effects": prescription_output.serious_side_effects,
                "drug_interactions": prescription_output.drug_interactions,
                "lifestyle_modifications": prescription_output.lifestyle_modifications,
                "dietary_recommendations": prescription_output.dietary_recommendations,
                "expected_outcomes": prescription_output.expected_outcomes,
                "when_to_contact_physician": prescription_output.when_to_contact_physician,
                "cbr_metadata": {
                    "case_based_rationale": prescription_output.case_based_rationale,
                    "similarity_score": prescription_output.similarity_score,
                    "adaptations_made": prescription_output.adaptations_made,
                    "selected_by": "MCDM (Multi-Criteria Decision Making)"
                }
            }
            
            logger.info(f"✅ Prescription generated for {recommended_med.medication.value}")
        
        else:
            logger.info("\n⚠️  STAGE 3: SKIPPED - No prescription generated")
        
        # ────────────────────────────────────────────────────────────
        # GENERATE NEXT STEPS
        # ────────────────────────────────────────────────────────────
        
        next_steps = generate_next_steps(
            eligibility_result,
            recommended_med,
            prescription_data
        )
        
        # ────────────────────────────────────────────────────────────
        # BUILD RESPONSE
        # ────────────────────────────────────────────────────────────
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Format medication recommendations
        recommended_med_formatted = None
        if recommended_med:
            recommended_med_formatted = MedicationRecommendation(
                rank=recommended_med.rank,
                medication=recommended_med.medication.value,
                total_score=recommended_med.total_score,
                efficacy_score=recommended_med.efficacy_score,
                safety_score=recommended_med.safety_score,
                convenience_score=recommended_med.convenience_score,
                cost_score=recommended_med.cost_score,
                suitability_score=recommended_med.suitability_score,
                strengths=recommended_med.strengths,
                weaknesses=recommended_med.weaknesses,
                rationale=recommended_med.recommendation_rationale
            )
        
        alternatives_formatted = [
            MedicationRecommendation(
                rank=alt.rank,
                medication=alt.medication.value,
                total_score=alt.total_score,
                efficacy_score=alt.efficacy_score,
                safety_score=alt.safety_score,
                convenience_score=alt.convenience_score,
                cost_score=alt.cost_score,
                suitability_score=alt.suitability_score,
                strengths=alt.strengths,
                weaknesses=alt.weaknesses,
                rationale=alt.recommendation_rationale
            )
            for alt in alternatives
        ]
        
        response = CompleteRecommendationResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            session_id=request.session_id,
            processing_time=round(processing_time, 3),
            eligibility=eligibility_formatted,
            recommended_medication=recommended_med_formatted,
            alternative_medications=alternatives_formatted,
            prescription=prescription_data,
            next_steps=next_steps,
            physician_review_required=True
        )
        
        logger.info("=" * 70)
        logger.info(f"✅ COMPLETE PIPELINE FINISHED in {processing_time:.2f}s")
        logger.info("=" * 70)
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recommendation pipeline error: {str(e)}")


# ==================== FRONTEND FORMATTING FUNCTION ====================

def format_eligibility_for_frontend(result, collected_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format eligibility result to match frontend's expected structure.
    
    This ensures compatibility with the React EligibilityDisplay component.
    """
    
    # Get BMI category
    bmi = result.bmi
    bmi_category = "Normal"
    bmi_meets_criteria = False
    
    if bmi:
        if bmi >= 30:
            bmi_category = "Obese"
            bmi_meets_criteria = True
        elif bmi >= 27:
            bmi_category = "Overweight"
            bmi_meets_criteria = True  # If has comorbidities
        elif bmi >= 25:
            bmi_category = "Overweight"
        else:
            bmi_category = "Normal"
    
    # Get diabetes status
    has_diabetes = result.diabetes_status == "Type 2 Diabetes"
    diabetes_controlled = has_diabetes  # Assume controlled if mentioned
    
    # Extract comorbidities
    conditions_text = str(collected_data.get("current_medical_conditions", "")).lower()
    present_comorbidities = []
    cv_risk = "Low"
    
    if "hypertension" in conditions_text or "high blood pressure" in conditions_text:
        present_comorbidities.append("Hypertension")
    if "cholesterol" in conditions_text or "dyslipidemia" in conditions_text:
        present_comorbidities.append("Dyslipidemia")
    if "cardiovascular" in conditions_text or "heart" in conditions_text:
        present_comorbidities.append("Cardiovascular Disease")
        cv_risk = "High"
    if "pcos" in conditions_text:
        present_comorbidities.append("PCOS")
    if "sleep apnea" in conditions_text:
        present_comorbidities.append("Sleep Apnea")
    
    # Get scoring breakdown
    scoring = result.scoring_breakdown
    
    # Build frontend-compatible structure
    return {
        "eligibility_status": result.status.value.upper(),
        "eligibility_score": result.score,
        "risk_level": result.risk_level.value.upper(),
        
        "clinical_assessment": {
            "bmi": {
                "value": bmi,
                "category": bmi_category,
                "meets_criteria": bmi_meets_criteria,
                "score": scoring.get("bmi", 0)
            },
            "diabetes_status": {
                "has_type2_diabetes": has_diabetes,
                "controlled": diabetes_controlled,
                "score": scoring.get("diabetes", 0)
            },
            "comorbidities": {
                "present": present_comorbidities,
                "cardiovascular_risk": cv_risk,
                "score": scoring.get("comorbidities", 0)
            },
            "weight_loss_goal": {
                "realistic": True,  # Assume realistic if provided
                "score": scoring.get("weight_goal", 0)
            },
            "contraindications": {
                "has_contraindications": len(result.contraindications) > 0,
                "violations": result.contraindications if result.contraindications else []
            }
        },
        
        "decision_support": {
            "recommendation": result.decision_rationale,
            "clinical_reasoning": [
                f"BMI {bmi}: {bmi_category}",
                f"Diabetes Status: {result.diabetes_status}",
                f"Eligibility Score: {result.score}/100"
            ] if bmi else ["Assessment complete"],
            "key_considerations": [
                guideline for guideline in result.clinical_guidelines_applied
            ] if result.clinical_guidelines_applied else []
        },
        
        "physician_review": {
            "review_level": "Standard" if result.risk_level == RiskLevel.LOW else "Enhanced",
            "focus_areas": result.physician_review_reasons if result.physician_review_reasons else [
                "Standard physician approval required for GLP-1 prescription"
            ]
        },
        
        "constraints": {
            "hard_constraints_passed": result.hard_constraints_passed,
            "violations": result.constraint_violations if result.constraint_violations else [],
            "soft_constraints": []  # Can add if needed
        }
    }


# ==================== PRESCRIPTION GENERATION ====================

def generate_prescription_for_medication(
    patient_data: Dict[str, Any],
    selected_medication: str,
    eligibility_result: Any
) -> PrescriptionOutput:
    """Generate prescription for the MCDM-selected medication."""
    
    logger.info(f"🔍 Finding best case for medication: {selected_medication}")
    
    all_cases = CaseLibrary.get_all_cases()
    
    # Filter cases by medication
    medication_cases = [
        case for case in all_cases 
        if selected_medication.lower() in case.medication.lower()
    ]
    
    if medication_cases:
        logger.info(f"   Found {len(medication_cases)} cases for {selected_medication}")
        
        scored_cases = []
        for case in medication_cases:
            similarity = CBREngine.calculate_similarity(patient_data, case)
            scored_cases.append((case, similarity))
        
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        best_case, best_similarity = scored_cases[0]
        
        logger.info(f"   Best match: {best_case.case_id} (similarity: {best_similarity}%)")
    
    else:
        logger.info(f"   No direct cases for {selected_medication}, using most similar patient profile")
        
        similar_cases = CBREngine.retrieve_most_similar_cases(patient_data, top_k=1)
        best_case, best_similarity = similar_cases[0]
        
        logger.info(f"   Adapting from: {best_case.case_id} ({best_case.medication})")
    
    # Generate prescription
    prescription_draft = CBREngine.reuse_prescription(best_case, patient_data)
    prescription_draft["medication"] = selected_medication
    prescription_draft = adapt_dosing_for_medication(prescription_draft, selected_medication)
    
    final_prescription, adaptations = CBREngine.revise_prescription(prescription_draft, patient_data)
    
    adaptations.insert(0, f"Medication selected by Multi-Criteria Decision Making (MCDM) analysis")
    
    prescription_output = CBREngine.format_prescription_output(
        patient_data,
        final_prescription,
        selected_medication,
        best_case,
        best_similarity,
        adaptations
    )
    
    return prescription_output


def adapt_dosing_for_medication(prescription: Dict[str, Any], medication: str) -> Dict[str, Any]:
    """Adapt dosing schedule based on selected medication."""
    
    med_lower = medication.lower()
    
    if "ozempic" in med_lower:
        prescription["starting_dose"] = "0.25 mg"
        prescription["target_dose"] = "1.0 mg or 2.0 mg weekly"
        prescription["titration_schedule"] = [
            {"weeks": "1-4", "dose": "0.25 mg subcutaneously once weekly (not therapeutic)"},
            {"weeks": "5+", "dose": "0.5 mg subcutaneously once weekly"},
            {"weeks": "If needed after ≥4 weeks", "dose": "Increase to 1.0 mg"},
            {"weeks": "If needed after ≥4 weeks", "dose": "Increase to 2.0 mg (maximum)"}
        ]
    
    elif "wegovy" in med_lower:
        prescription["starting_dose"] = "0.25 mg"
        prescription["target_dose"] = "2.4 mg weekly"
        prescription["titration_schedule"] = [
            {"weeks": "1-4", "dose": "0.25 mg subcutaneously once weekly"},
            {"weeks": "5-8", "dose": "0.5 mg subcutaneously once weekly"},
            {"weeks": "9-12", "dose": "1.0 mg subcutaneously once weekly"},
            {"weeks": "13-16", "dose": "1.7 mg subcutaneously once weekly"},
            {"weeks": "17+", "dose": "2.4 mg subcutaneously once weekly (maintenance)"}
        ]
    
    elif "rybelsus" in med_lower:
        prescription["starting_dose"] = "3 mg"
        prescription["target_dose"] = "14 mg daily"
        prescription["titration_schedule"] = [
            {"days": "1-30", "dose": "3 mg orally once daily"},
            {"days": "31-60", "dose": "7 mg orally once daily"},
            {"days": "61+", "dose": "14 mg orally once daily (maintenance)"}
        ]
    
    elif "mounjaro" in med_lower:
        prescription["starting_dose"] = "2.5 mg"
        prescription["target_dose"] = "5 mg to 15 mg weekly"
        prescription["titration_schedule"] = [
            {"weeks": "1-4", "dose": "2.5 mg subcutaneously once weekly"},
            {"weeks": "5+", "dose": "5 mg subcutaneously once weekly"},
            {"weeks": "If needed after ≥4 weeks", "dose": "7.5 mg subcutaneously once weekly"},
            {"weeks": "If needed after ≥4 weeks", "dose": "10 mg subcutaneously once weekly"},
            {"weeks": "If needed after ≥4 weeks", "dose": "12.5 mg or 15 mg once weekly (max)"}
        ]
    
    elif "zepbound" in med_lower:
        prescription["starting_dose"] = "2.5 mg"
        prescription["target_dose"] = "5 mg to 15 mg weekly"
        prescription["titration_schedule"] = [
            {"weeks": "1-4", "dose": "2.5 mg subcutaneously once weekly"},
            {"weeks": "5-8", "dose": "5 mg subcutaneously once weekly"},
            {"weeks": "9-12", "dose": "7.5 mg subcutaneously once weekly"},
            {"weeks": "13-16", "dose": "10 mg subcutaneously once weekly"},
            {"weeks": "17-20", "dose": "12.5 mg subcutaneously once weekly"},
            {"weeks": "21+", "dose": "15 mg subcutaneously once weekly (maintenance)"}
        ]
    
    elif "victoza" in med_lower:
        prescription["starting_dose"] = "0.6 mg"
        prescription["target_dose"] = "1.2 mg or 1.8 mg daily"
        prescription["titration_schedule"] = [
            {"week": "1", "dose": "0.6 mg subcutaneously once daily"},
            {"week": "2+", "dose": "1.2 mg subcutaneously once daily"},
            {"week": "If needed", "dose": "Increase to 1.8 mg once daily (maximum)"}
        ]
    
    elif "saxenda" in med_lower:
        prescription["starting_dose"] = "0.6 mg"
        prescription["target_dose"] = "3.0 mg daily"
        prescription["titration_schedule"] = [
            {"week": "1", "dose": "0.6 mg subcutaneously once daily"},
            {"week": "2", "dose": "1.2 mg subcutaneously once daily"},
            {"week": "3", "dose": "1.8 mg subcutaneously once daily"},
            {"week": "4", "dose": "2.4 mg subcutaneously once daily"},
            {"week": "5+", "dose": "3.0 mg subcutaneously once daily (maintenance)"}
        ]
    
    elif "trulicity" in med_lower:
        prescription["starting_dose"] = "0.75 mg"
        prescription["target_dose"] = "1.5 mg weekly"
        prescription["titration_schedule"] = [
            {"weeks": "1-4", "dose": "0.75 mg subcutaneously once weekly"},
            {"weeks": "5+", "dose": "1.5 mg subcutaneously once weekly (if tolerated)"}
        ]
    
    return prescription


# ==================== HELPER FUNCTIONS ====================

def generate_next_steps(eligibility_result, recommended_med, prescription) -> List[str]:
    """Generate actionable next steps"""
    steps = []
    
    if eligibility_result.status == EligibilityStatus.CONTRAINDICATED:
        steps.append("⚠️ CRITICAL: Absolute contraindications identified - GLP-1 therapy not safe")
        steps.append("Schedule consultation with physician to discuss alternative treatments")
        steps.append("Review contraindications with medical team")
        return steps
    
    if eligibility_result.status == EligibilityStatus.NOT_ELIGIBLE:
        steps.append("Patient does not meet FDA criteria for GLP-1 therapy at this time")
        steps.append("Consider lifestyle modifications and alternative weight management approaches")
        steps.append("Re-evaluate eligibility after 3-6 months of lifestyle intervention")
        return steps
    
    if eligibility_result.status == EligibilityStatus.REQUIRES_REVIEW:
        steps.append("⚠️ Borderline eligibility - detailed physician evaluation required")
        steps.append("Schedule comprehensive medical assessment")
        steps.append("Discuss risks and benefits with healthcare provider")
        return steps
    
    if recommended_med and prescription:
        steps.append(f"✅ Recommended medication: {recommended_med.medication.value}")
        steps.append("Review complete prescription details with physician")
        steps.append(f"Schedule baseline labs: {', '.join(prescription['baseline_labs'][:3])}")
        steps.append("Obtain prior authorization from insurance (if required)")
        steps.append("Schedule follow-up appointment in 2 weeks to assess tolerance")
        steps.append("Physician will review and sign prescription if appropriate")
    
    return steps


# ==================== UTILITY ENDPOINTS ====================

@recommendation_router.get("/pipeline-info")
async def get_pipeline_info():
    """Return information about the recommendation pipeline"""
    return {
        "pipeline": "GLP-1 Prescription Recommendation System",
        "version": "1.0.2",
        "update": "Frontend-compatible data structure",
        "stages": [
            {
                "stage": 1,
                "name": "Eligibility Determination",
                "method": "Rule-based Clinical Decision Support System (CDSS)",
                "output": "Eligibility status + score (0-100)",
                "reference": "Kawamoto et al. (2005) - BMJ"
            },
            {
                "stage": 2,
                "name": "Medication Selection",
                "method": "Multi-Criteria Decision Making (MCDM)",
                "output": "Ranked list of 8 GLP-1 medications",
                "reference": "Saaty (1980) - Analytic Hierarchy Process"
            },
            {
                "stage": 3,
                "name": "Prescription Generation",
                "method": "Case-Based Reasoning (CBR) with MCDM-selected medication",
                "output": "Personalized prescription document",
                "reference": "Bichindaritz & Marling (2006)"
            }
        ]
    }


def add_recommendation_routes_to_app(app):
    """Helper to integrate routes into main app"""
    app.include_router(recommendation_router)
    logger.info("✅ Recommendation routes registered (v1.0.2 - Frontend compatible)")


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    test_app = FastAPI(title="GLP-1 Complete Recommendation Pipeline")
    test_app.include_router(recommendation_router)
    
    @test_app.get("/")
    async def root():
        return {
            "service": "GLP-1 Complete Recommendation System",
            "version": "1.0.2",
            "status": "active",
            "endpoints": {
                "complete": "/recommendation/complete",
                "info": "/recommendation/pipeline-info"
            }
        }
    
    print("\n" + "=" * 70)
    print("🏥 GLP-1 Recommendation Pipeline v1.0.2 - FRONTEND COMPATIBLE")
    print("=" * 70)
    print("📍 Server: http://localhost:8002")
    print("📚 Docs:   http://localhost:8002/docs")
    print("=" * 70 + "\n")
    
    uvicorn.run(test_app, host="0.0.0.0", port=8002)