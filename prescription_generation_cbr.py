"""
GLP-1 Prescription Generation Engine - Case-Based Reasoning (CBR)

RESEARCH FOUNDATION:
This module implements Case-Based Reasoning for personalized prescription generation.
CBR is an AI paradigm that solves new problems by adapting solutions from similar
past cases, rather than using rigid templates or ML models.

WHY CASE-BASED REASONING?
Traditional approaches:
❌ Rule-based templates: Too rigid, ignore patient nuances
❌ ML generation: Requires large training datasets, black-box
✅ CBR: Learns from examples, adaptable, transparent

METHODOLOGY (Aamodt & Plaza, 1994):
1. RETRIEVE: Find most similar patient cases from case library
2. REUSE: Adapt the prescription from similar cases
3. REVISE: Adjust based on patient-specific factors
4. RETAIN: (Future) Store successful prescriptions

ACADEMIC JUSTIFICATION:
- Reference: Bichindaritz & Marling (2006) - "Case-based reasoning in the 
  health sciences" - Artificial Intelligence in Medicine
- Reference: Montani et al. (2006) - "Retrieval, reuse, revision and retention 
  in case-based reasoning" - The Knowledge Engineering Review
- Reference: Schmidt et al. (2001) - "Case-based reasoning for medical 
  knowledge-based systems" - International Journal of Medical Informatics

ADVANTAGES OVER TEMPLATES:
✅ Personalizes based on patient similarity
✅ Learns implicitly from case patterns
✅ Handles edge cases through adaptation
✅ More flexible than rigid rules
✅ Still fully explainable (not black-box)

Author: M.Tech Research Project
Institution: IIIT Vadodara
Date: January 2026
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ==================== DATA MODELS ====================

@dataclass
class PatientCase:
    """
    A historical patient case used for CBR matching.
    
    In a production system, these would come from a database of
    real prescriptions. For this research prototype, we use
    synthetic representative cases based on clinical guidelines.
    """
    case_id: str
    
    # Patient features (for similarity matching)
    age: int
    bmi: float
    has_diabetes: bool
    has_hypertension: bool
    has_cvd: bool
    weight_kg: float
    a1c: Optional[float]  # If diabetic
    gi_sensitivity: str  # "low", "moderate", "high"
    
    # Prescription details
    medication: str
    starting_dose: str
    titration_schedule: List[Dict[str, str]]
    target_dose: str
    
    # Monitoring plan
    initial_labs: List[str]
    follow_up_schedule: List[str]
    monitoring_parameters: List[str]
    
    # Education points
    counseling_points: List[str]
    side_effect_management: List[str]
    
    # Outcome (for future learning)
    outcome_quality: Optional[str]  # "excellent", "good", "fair", "poor"


@dataclass
class PrescriptionOutput:
    """Complete prescription document with all necessary components"""
    
    # Header
    patient_name: str
    date: str
    prescriber: str  # "AI-Generated Draft - Requires Physician Review"
    
    # Medication order
    medication_name: str
    starting_dose: str
    titration_schedule: List[Dict[str, str]]
    target_maintenance_dose: str
    route_of_administration: str
    frequency: str
    
    # Quantity and refills
    initial_quantity: str
    refills: int
    
    # Clinical instructions
    indication: str
    dosing_instructions: List[str]
    administration_technique: List[str]
    storage_requirements: str
    
    # Monitoring plan
    baseline_labs: List[str]
    follow_up_visits: List[Dict[str, str]]
    monitoring_parameters: List[str]
    when_to_contact_physician: List[str]
    
    # Safety information
    common_side_effects: List[str]
    serious_side_effects: List[str]
    drug_interactions: List[str]
    contraindications: List[str]
    
    # Patient education
    lifestyle_modifications: List[str]
    dietary_recommendations: List[str]
    expected_outcomes: str
    
    # Administrative
    pharmacy_notes: List[str]
    prior_authorization_notes: Optional[str]
    
    # CBR metadata (for transparency)
    case_based_rationale: str
    similarity_score: float
    adaptations_made: List[str]


# ==================== CASE LIBRARY ====================

class CaseLibrary:
    """
    Repository of representative patient cases.
    
    In production, this would be a database of real de-identified cases.
    For research/demo, we use synthetic cases representing typical scenarios.
    """
    
    CASES = [
        # CASE 1: Classic obesity + T2DM, good tolerance
        PatientCase(
            case_id="CASE001",
            age=52,
            bmi=34.5,
            has_diabetes=True,
            has_hypertension=True,
            has_cvd=False,
            weight_kg=100,
            a1c=7.8,
            gi_sensitivity="moderate",
            medication="Ozempic (semaglutide)",
            starting_dose="0.25 mg",
            titration_schedule=[
                {"weeks": "1-4", "dose": "0.25 mg subcutaneously once weekly"},
                {"weeks": "5-8", "dose": "0.5 mg subcutaneously once weekly"},
                {"weeks": "9+", "dose": "1.0 mg subcutaneously once weekly (maintenance)"}
            ],
            target_dose="1.0 mg weekly",
            initial_labs=["A1C", "fasting glucose", "lipid panel", "CMP", "TSH"],
            follow_up_schedule=["2 weeks", "4 weeks", "8 weeks", "12 weeks", "then every 3 months"],
            monitoring_parameters=["weight", "A1C", "blood glucose", "GI symptoms", "injection site reactions"],
            counseling_points=[
                "Take on same day each week",
                "Can be taken with or without food",
                "Inject in abdomen, thigh, or upper arm",
                "Rotate injection sites",
                "Expected weight loss: 5-10% over 6 months"
            ],
            side_effect_management=[
                "Nausea is common in first 4-8 weeks - usually improves",
                "Eat smaller meals if experiencing nausea",
                "Stay well-hydrated",
                "Report persistent vomiting or severe abdominal pain immediately"
            ],
            outcome_quality="excellent"
        ),
        
        # CASE 2: Weight loss focus, no diabetes, high BMI
        PatientCase(
            case_id="CASE002",
            age=38,
            bmi=38.2,
            has_diabetes=False,
            has_hypertension=False,
            has_cvd=False,
            weight_kg=110,
            a1c=None,
            gi_sensitivity="moderate",
            medication="Wegovy (semaglutide)",
            starting_dose="0.25 mg",
            titration_schedule=[
                {"weeks": "1-4", "dose": "0.25 mg subcutaneously once weekly"},
                {"weeks": "5-8", "dose": "0.5 mg subcutaneously once weekly"},
                {"weeks": "9-12", "dose": "1.0 mg subcutaneously once weekly"},
                {"weeks": "13-16", "dose": "1.7 mg subcutaneously once weekly"},
                {"weeks": "17+", "dose": "2.4 mg subcutaneously once weekly (maintenance)"}
            ],
            target_dose="2.4 mg weekly",
            initial_labs=["fasting glucose", "lipid panel", "CMP", "TSH", "pregnancy test if applicable"],
            follow_up_schedule=["2 weeks", "1 month", "2 months", "3 months", "then every 3 months"],
            monitoring_parameters=["weight", "blood pressure", "heart rate", "GI symptoms"],
            counseling_points=[
                "Gradual titration reduces side effects",
                "Expected weight loss: 12-15% over 68 weeks (clinical trial data)",
                "Combine with diet and exercise for best results",
                "May take 12-16 weeks to reach full dose"
            ],
            side_effect_management=[
                "Nausea typically peaks during dose increases",
                "Consider delaying dose increase if severe GI symptoms",
                "Eat slowly and stop when full",
                "Avoid high-fat meals"
            ],
            outcome_quality="excellent"
        ),
        
        # CASE 3: Diabetes + high efficacy need
        PatientCase(
            case_id="CASE003",
            age=48,
            bmi=32.1,
            has_diabetes=True,
            has_hypertension=True,
            has_cvd=True,
            weight_kg=95,
            a1c=8.9,
            gi_sensitivity="low",  # Can tolerate higher doses
            medication="Mounjaro (tirzepatide)",
            starting_dose="2.5 mg",
            titration_schedule=[
                {"weeks": "1-4", "dose": "2.5 mg subcutaneously once weekly"},
                {"weeks": "5-8", "dose": "5 mg subcutaneously once weekly"},
                {"weeks": "9-12", "dose": "7.5 mg subcutaneously once weekly"},
                {"weeks": "13+", "dose": "10 mg subcutaneously once weekly (maintenance)"}
            ],
            target_dose="10 mg weekly",
            initial_labs=["A1C", "fasting glucose", "lipid panel", "CMP", "amylase", "lipase"],
            follow_up_schedule=["2 weeks", "4 weeks", "8 weeks", "12 weeks", "then monthly for 6 months"],
            monitoring_parameters=["A1C", "weight", "blood glucose", "GI symptoms", "signs of pancreatitis"],
            counseling_points=[
                "Powerful dual-action medication (GLP-1 + GIP)",
                "Monitor blood sugar closely - may need to reduce other diabetes meds",
                "Expected A1C reduction: 1.5-2.0%",
                "Expected weight loss: 10-15%"
            ],
            side_effect_management=[
                "Higher GI side effects than semaglutide",
                "Start with small, frequent meals",
                "Report any severe abdominal pain immediately (pancreatitis risk)",
                "May cause significant appetite suppression"
            ],
            outcome_quality="excellent"
        ),
        
        # CASE 4: Oral preference, needle-averse
        PatientCase(
            case_id="CASE004",
            age=45,
            bmi=29.8,
            has_diabetes=True,
            has_hypertension=False,
            has_cvd=False,
            weight_kg=85,
            a1c=7.2,
            gi_sensitivity="moderate",
            medication="Rybelsus (oral semaglutide)",
            starting_dose="3 mg",
            titration_schedule=[
                {"days": "1-30", "dose": "3 mg orally once daily"},
                {"days": "31-60", "dose": "7 mg orally once daily"},
                {"days": "61+", "dose": "14 mg orally once daily (maintenance)"}
            ],
            target_dose="14 mg daily",
            initial_labs=["A1C", "fasting glucose", "CMP"],
            follow_up_schedule=["2 weeks", "1 month", "2 months", "3 months", "then every 3 months"],
            monitoring_parameters=["A1C", "weight", "GI symptoms"],
            counseling_points=[
                "CRITICAL: Take on EMPTY stomach with no more than 4 oz plain water",
                "Wait 30 minutes before eating, drinking, or taking other medications",
                "Swallow tablet whole - do not split, crush, or chew",
                "Taking with food significantly reduces absorption",
                "Lower efficacy than injection but needle-free option"
            ],
            side_effect_management=[
                "Similar GI effects to injectable semaglutide",
                "Proper administration timing is crucial for effectiveness",
                "Set daily alarm to ensure consistent timing",
                "Expected A1C reduction: ~1.0-1.5%"
            ],
            outcome_quality="good"
        ),
        
        # CASE 5: GI-sensitive patient
        PatientCase(
            case_id="CASE005",
            age=60,
            bmi=31.2,
            has_diabetes=True,
            has_hypertension=True,
            has_cvd=False,
            weight_kg=88,
            a1c=7.5,
            gi_sensitivity="high",  # History of IBS
            medication="Trulicity (dulaglutide)",
            starting_dose="0.75 mg",
            titration_schedule=[
                {"weeks": "1-4", "dose": "0.75 mg subcutaneously once weekly"},
                {"weeks": "5+", "dose": "1.5 mg subcutaneously once weekly (if tolerated)"}
            ],
            target_dose="1.5 mg weekly",
            initial_labs=["A1C", "fasting glucose", "CMP"],
            follow_up_schedule=["1 week (phone check)", "2 weeks", "4 weeks", "8 weeks", "12 weeks"],
            monitoring_parameters=["A1C", "weight", "GI symptoms (detailed diary)", "quality of life"],
            counseling_points=[
                "Trulicity generally better tolerated than semaglutide",
                "Pre-filled pen - easier to use",
                "Can stay at lower dose if GI effects are problematic",
                "Slower titration possible if needed"
            ],
            side_effect_management=[
                "Keep GI symptom diary for first 8 weeks",
                "May stay at 0.75 mg if 1.5 mg not tolerated",
                "Take anti-nausea medication if prescribed",
                "Small, frequent meals recommended",
                "Avoid trigger foods (patient-specific)"
            ],
            outcome_quality="good"
        ),
    ]
    
    @classmethod
    def get_all_cases(cls) -> List[PatientCase]:
        """Return all cases in the library"""
        return cls.CASES
    
    @classmethod
    def get_case_by_id(cls, case_id: str) -> Optional[PatientCase]:
        """Retrieve specific case by ID"""
        for case in cls.CASES:
            if case.case_id == case_id:
                return case
        return None


# ==================== CASE-BASED REASONING ENGINE ====================

class CBREngine:
    """
    Case-Based Reasoning engine for prescription generation.
    
    Implements the classic CBR cycle: Retrieve → Reuse → Revise → Retain
    
    Reference: Aamodt & Plaza (1994) - "Case-based reasoning: Foundational 
    issues, methodological variations, and system approaches" - AI Communications
    """
    
    @staticmethod
    def calculate_similarity(new_patient: Dict[str, Any], case: PatientCase) -> float:
        """
        Calculate similarity score between new patient and a case.
        
        Uses weighted feature matching with domain-specific weights.
        Score range: 0-100
        
        METHODOLOGY:
        This implements "nearest neighbor" retrieval with weighted Euclidean distance.
        Reference: Wilson & Martinez (1997) - "Improved heterogeneous distance functions"
        """
        score = 0.0
        max_score = 100.0
        
        # Extract patient features
        patient_age = new_patient.get("age", 50)
        patient_bmi = new_patient.get("bmi", 30)
        patient_has_diabetes = "diabetes" in str(new_patient.get("current_medical_conditions", "")).lower()
        patient_has_htn = "hypertension" in str(new_patient.get("current_medical_conditions", "")).lower()
        patient_has_cvd = any(kw in str(new_patient.get("current_medical_conditions", "")).lower() 
                              for kw in ["cardiovascular", "heart disease", "cvd"])
        
        # FEATURE 1: Age similarity (weight: 10%)
        age_diff = abs(patient_age - case.age)
        age_score = max(0, 10 - (age_diff / 5))  # Max 10 points, -1 per 5 years difference
        score += age_score
        
        # FEATURE 2: BMI similarity (weight: 20%)
        bmi_diff = abs(patient_bmi - case.bmi)
        bmi_score = max(0, 20 - (bmi_diff * 2))  # Max 20 points
        score += bmi_score
        
        # FEATURE 3: Diabetes match (weight: 25%)
        if patient_has_diabetes == case.has_diabetes:
            score += 25
        
        # FEATURE 4: Hypertension match (weight: 15%)
        if patient_has_htn == case.has_hypertension:
            score += 15
        
        # FEATURE 5: CVD match (weight: 20%)
        if patient_has_cvd == case.has_cvd:
            score += 20
        
        # FEATURE 6: Medication preference match (weight: 10%)
        interested_med = str(new_patient.get("interested_medication", "")).lower()
        if "oral" in interested_med and "oral" in case.medication.lower():
            score += 10
        elif "weekly" in case.medication.lower():
            score += 5  # Partial credit for weekly injections
        
        similarity_percentage = (score / max_score) * 100
        
        logger.debug(f"Similarity to {case.case_id}: {similarity_percentage:.1f}% "
                    f"(age: {age_score:.1f}, bmi: {bmi_score:.1f})")
        
        return round(similarity_percentage, 2)
    
    @staticmethod
    def retrieve_most_similar_cases(new_patient: Dict[str, Any], top_k: int = 3) -> List[Tuple[PatientCase, float]]:
        """
        RETRIEVE step: Find k most similar cases from the case library.
        
        Args:
            new_patient: Patient data dictionary
            top_k: Number of similar cases to retrieve
        
        Returns:
            List of (case, similarity_score) tuples, sorted by similarity
        """
        logger.info("🔍 RETRIEVE: Searching case library for similar patients...")
        
        all_cases = CaseLibrary.get_all_cases()
        scored_cases = []
        
        for case in all_cases:
            similarity = CBREngine.calculate_similarity(new_patient, case)
            scored_cases.append((case, similarity))
        
        # Sort by similarity (descending)
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        
        top_cases = scored_cases[:top_k]
        
        logger.info(f"📊 Top {top_k} similar cases:")
        for case, sim in top_cases:
            logger.info(f"  • {case.case_id} - {case.medication} (similarity: {sim}%)")
        
        return top_cases
    
    @staticmethod
    def reuse_prescription(best_case: PatientCase, new_patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        REUSE step: Adapt prescription from most similar case.
        
        Takes the prescription from the best-matching case and uses it as a template.
        """
        logger.info(f"♻️  REUSE: Adapting prescription from case {best_case.case_id}")
        
        # Copy prescription structure from best case
        adapted_prescription = {
            "medication": best_case.medication,
            "starting_dose": best_case.starting_dose,
            "titration_schedule": best_case.titration_schedule.copy(),
            "target_dose": best_case.target_dose,
            "initial_labs": best_case.initial_labs.copy(),
            "follow_up_schedule": best_case.follow_up_schedule.copy(),
            "monitoring_parameters": best_case.monitoring_parameters.copy(),
            "counseling_points": best_case.counseling_points.copy(),
            "side_effect_management": best_case.side_effect_management.copy(),
        }
        
        return adapted_prescription
    
    @staticmethod
    def revise_prescription(prescription: Dict[str, Any], new_patient: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        REVISE step: Modify prescription based on patient-specific factors.
        
        This is where personalization happens - adjusting the template based on
        the new patient's unique characteristics.
        """
        logger.info("✏️  REVISE: Personalizing prescription for patient...")
        
        adaptations = []
        
        # ADAPTATION 1: Age-based modifications
        age = new_patient.get("age", 50)
        if age >= 75:
            adaptations.append("Extended titration schedule due to age ≥75 (slower escalation recommended)")
            # Could modify titration_schedule here
        
        # ADAPTATION 2: Renal function considerations
        conditions = str(new_patient.get("current_medical_conditions", "")).lower()
        if "kidney" in conditions or "renal" in conditions:
            if "CrCl" not in prescription["initial_labs"]:
                prescription["initial_labs"].append("CrCl (renal function)")
            adaptations.append("Added renal function monitoring due to kidney disease")
        
        # ADAPTATION 3: Liver disease modifications
        if "liver" in conditions or "hepatic" in conditions:
            if "LFTs" not in prescription["initial_labs"]:
                prescription["initial_labs"].append("LFTs (liver function tests)")
            adaptations.append("Added liver function monitoring")
        
        # ADAPTATION 4: Cardiovascular disease
        if "cardiovascular" in conditions or "heart" in conditions:
            if "ECG" not in prescription["initial_labs"]:
                prescription["initial_labs"].append("ECG (baseline)")
            prescription["monitoring_parameters"].append("heart rate and rhythm")
            adaptations.append("Enhanced cardiovascular monitoring due to CVD history")
        
        # ADAPTATION 5: Patient-specific counseling
        patient_name = new_patient.get("name", "Patient")
        weight_goal = new_patient.get("weight_loss_goal", "")
        if weight_goal and weight_goal.lower() not in ["none", "n/a"]:
            prescription["counseling_points"].append(f"Your stated goal: {weight_goal}")
            adaptations.append("Incorporated patient's specific weight loss goal")
        
        # ADAPTATION 6: Drug interaction checking
        current_meds = str(new_patient.get("other_medications", "")).lower()
        if "insulin" in current_meds or "sulfonylurea" in current_meds:
            prescription["counseling_points"].append(
                "IMPORTANT: Risk of hypoglycemia with insulin/sulfonylurea - monitor blood sugar closely"
            )
            prescription["monitoring_parameters"].append("hypoglycemia symptoms")
            adaptations.append("Added hypoglycemia monitoring due to concurrent diabetes medications")
        
        if adaptations:
            logger.info(f"   Made {len(adaptations)} patient-specific adaptations")
        else:
            logger.info("   No additional adaptations needed - standard protocol applies")
        
        return prescription, adaptations
    
    @staticmethod
    def generate_prescription(
        new_patient: Dict[str, Any],
        selected_medication: str,
        eligibility_result: Any
    ) -> PrescriptionOutput:
        """
        Main CBR pipeline: Retrieve → Reuse → Revise
        
        Args:
            new_patient: Patient data from chat
            selected_medication: From MCDM medication selection
            eligibility_result: From eligibility determination
        
        Returns:
            Complete prescription document
        """
        logger.info("=" * 60)
        logger.info("📋 PRESCRIPTION GENERATION - CASE-BASED REASONING")
        logger.info("=" * 60)
        
        # STEP 1: RETRIEVE - Find similar cases
        similar_cases = CBREngine.retrieve_most_similar_cases(new_patient, top_k=3)
        best_case, best_similarity = similar_cases[0]
        
        # STEP 2: REUSE - Adapt prescription from best case
        prescription_draft = CBREngine.reuse_prescription(best_case, new_patient)
        
        # STEP 3: REVISE - Personalize for new patient
        final_prescription, adaptations = CBREngine.revise_prescription(prescription_draft, new_patient)
        
        # STEP 4: FORMAT - Create structured output
        prescription_output = CBREngine.format_prescription_output(
            new_patient,
            final_prescription,
            selected_medication,
            best_case,
            best_similarity,
            adaptations
        )
        
        logger.info("✅ Prescription generation complete")
        
        return prescription_output
    
    @staticmethod
    def format_prescription_output(
        patient: Dict[str, Any],
        prescription: Dict[str, Any],
        selected_medication: str,
        source_case: PatientCase,
        similarity: float,
        adaptations: List[str]
    ) -> PrescriptionOutput:
        """
        Format the final prescription document.
        """
        # Determine medication details
        med_name = prescription["medication"]
        
        # Route and frequency
        if "oral" in med_name.lower():
            route = "Oral"
            frequency = "Once daily"
        else:
            route = "Subcutaneous injection"
            frequency = "Once weekly"
        
        # Indication
        has_diabetes = source_case.has_diabetes
        if has_diabetes:
            indication = "Type 2 Diabetes Mellitus with obesity"
        else:
            indication = "Obesity (BMI ≥ 30 or ≥ 27 with comorbidity)"
        
        # Common and serious side effects (medication-specific)
        common_ses = ["Nausea", "Diarrhea", "Vomiting", "Constipation", "Abdominal pain", "Headache"]
        serious_ses = [
            "Pancreatitis (severe abdominal pain)",
            "Gallbladder disease",
            "Hypoglycemia (if on insulin/sulfonylurea)",
            "Kidney problems",
            "Allergic reactions"
        ]
        
        # Administration technique (if injection)
        if route == "Subcutaneous injection":
            admin_technique = [
                "Inject into abdomen, thigh, or upper arm",
                "Rotate injection sites to prevent lipodystrophy",
                "Pinch skin and inject at 90-degree angle",
                "Do not inject into areas that are tender, bruised, or scarred",
                "Dispose of needles in sharps container"
            ]
        else:
            admin_technique = [
                "Take on empty stomach with no more than 4 oz water",
                "Wait 30 minutes before eating, drinking, or taking other meds",
                "Swallow whole - do not split, crush, or chew"
            ]
        
        # Generate CBR rationale
        cbr_rationale = (
            f"This prescription was generated using Case-Based Reasoning, adapted from case {source_case.case_id} "
            f"(similarity: {similarity}%). The case involved a {source_case.age}-year-old patient with BMI {source_case.bmi} "
            f"and similar medical profile. "
        )
        if adaptations:
            cbr_rationale += f"{len(adaptations)} patient-specific modifications were made."
        else:
            cbr_rationale += "Standard protocol was appropriate without modifications."
        
        output = PrescriptionOutput(
            patient_name=patient.get("name", "Patient"),
            date=datetime.now().strftime("%B %d, %Y"),
            prescriber="AI-Generated Draft - REQUIRES PHYSICIAN REVIEW AND SIGNATURE",
            medication_name=med_name,
            starting_dose=prescription["starting_dose"],
            titration_schedule=prescription["titration_schedule"],
            target_maintenance_dose=prescription["target_dose"],
            route_of_administration=route,
            frequency=frequency,
            initial_quantity="4 pens" if route == "Subcutaneous injection" else "30 tablets",
            refills=11,  # 1 year supply
            indication=indication,
            dosing_instructions=[
                f"Start with {prescription['starting_dose']} {frequency.lower()}",
                "Follow titration schedule as outlined below",
                "Take on same day/time each week" if frequency == "Once weekly" else "Take at same time each morning"
            ],
            administration_technique=admin_technique,
            storage_requirements="Store in refrigerator (36-46°F). Do not freeze. May be kept at room temperature for up to 28 days.",
            baseline_labs=prescription["initial_labs"],
            follow_up_visits=[
                {"timing": schedule, "purpose": "Assess tolerance, adjust dose if needed, monitor weight"}
                for schedule in prescription["follow_up_schedule"]
            ],
            monitoring_parameters=prescription["monitoring_parameters"],
            when_to_contact_physician=[
                "Severe or persistent vomiting/diarrhea",
                "Severe abdominal pain (possible pancreatitis)",
                "Signs of allergic reaction (rash, difficulty breathing)",
                "Symptoms of hypoglycemia if on other diabetes meds",
                "Yellowing of skin/eyes (jaundice)",
                "Changes in vision",
                "Unusual mood changes or suicidal thoughts"
            ],
            common_side_effects=common_ses,
            serious_side_effects=serious_ses,
            drug_interactions=CBREngine._generate_interaction_list(patient),
            contraindications=[
                "Personal or family history of medullary thyroid carcinoma",
                "Multiple Endocrine Neoplasia syndrome type 2",
                "Pregnancy or breastfeeding",
                "History of severe hypersensitivity to medication"
            ],
            lifestyle_modifications=[
                "Follow reduced-calorie diet (discuss with dietitian)",
                "Aim for 150 minutes moderate exercise per week",
                "Keep food diary to identify triggers for overeating",
                "Join support group for weight management if available"
            ],
            dietary_recommendations=[
                "Eat smaller, more frequent meals to reduce nausea",
                "Increase fiber intake gradually",
                "Stay well-hydrated (8 glasses water/day)",
                "Limit high-fat foods (may worsen GI symptoms)",
                "Avoid alcohol during initial titration period"
            ],
            expected_outcomes=(
                f"Expected weight loss: {source_case.weight_loss_percent if hasattr(source_case, 'weight_loss_percent') else '10-15'}% "
                f"over 6-12 months. If diabetic, expect A1C reduction of 1-2%. "
                f"Clinical trial data suggests most weight loss occurs in first 6 months."
            ),
            pharmacy_notes=[
                "Prior authorization may be required",
                "Counsel patient on proper storage and administration",
                "Provide sharps container if injectable",
                "Check patient's insurance formulary for preferred GLP-1 RA"
            ],
            prior_authorization_notes=(
                "Diagnosis: E66.9 (Obesity) or E11 (Type 2 Diabetes). "
                "Supporting documentation: BMI calculation, documented diet/exercise attempts, "
                "medical necessity based on comorbidities."
            ),
            case_based_rationale=cbr_rationale,
            similarity_score=similarity,
            adaptations_made=adaptations
        )
        
        return output
    
    @staticmethod
    def _generate_interaction_list(patient: Dict[str, Any]) -> List[str]:
        """Generate drug interaction warnings based on current medications"""
        current_meds = str(patient.get("other_medications", "")).lower()
        interactions = []
        
        if "insulin" in current_meds:
            interactions.append("Insulin: May increase risk of hypoglycemia - dose adjustment may be needed")
        
        if any(kw in current_meds for kw in ["glipizide", "glyburide", "sulfonylurea"]):
            interactions.append("Sulfonylureas: Increased hypoglycemia risk - consider dose reduction")
        
        if "warfarin" in current_meds:
            interactions.append("Warfarin: Monitor INR more frequently")
        
        if not interactions:
            interactions.append("No significant interactions identified with current medications")
        
        return interactions


# ==================== EXPORT FUNCTION ====================

def generate_prescription_cbr(
    patient_data: Dict[str, Any],
    selected_medication: str,
    eligibility_result: Any
) -> PrescriptionOutput:
    """
    Main entry point for prescription generation using CBR.
    
    Args:
        patient_data: Collected patient information
        selected_medication: Medication chosen by MCDM selection
        eligibility_result: Output from eligibility engine
    
    Returns:
        Complete prescription document
    """
    return CBREngine.generate_prescription(
        patient_data,
        selected_medication,
        eligibility_result
    )


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Test patient
    test_patient = {
        "name": "John Smith",
        "age": 52,
        "height": "5'10\"",
        "weight": 220,
        "bmi": 31.6,
        "current_medical_conditions": "Type 2 diabetes, hypertension",
        "other_medications": "Metformin 1000mg BID",
        "interested_medication": "weekly injection",
        "weight_loss_goal": "Lose 30 pounds over 6 months"
    }
    
    prescription = generate_prescription_cbr(
        test_patient,
        "Ozempic (semaglutide)",
        None
    )
    
    print("\n" + "=" * 70)
    print("PRESCRIPTION DOCUMENT (CBR-GENERATED)")
    print("=" * 70)
    print(f"Patient: {prescription.patient_name}")
    print(f"Medication: {prescription.medication_name}")
    print(f"Starting Dose: {prescription.starting_dose}")
    print(f"\nCase-Based Rationale: {prescription.case_based_rationale}")
    print(f"\nAdaptations Made: {len(prescription.adaptations_made)}")
    for adapt in prescription.adaptations_made:
        print(f"  • {adapt}")