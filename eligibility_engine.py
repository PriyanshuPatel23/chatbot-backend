"""
GLP-1 Eligibility Determination Engine - Research-Grade CDSS Implementation

THEORETICAL FOUNDATION:
This module implements a Clinical Decision Support System (CDSS) based on
established medical informatics research rather than black-box LLM inference.

CORE DESIGN PRINCIPLES (from literature):
1. Rule-based transparency (Kawamoto et al., 2005) - BMJ
2. Constraint satisfaction approach (Tsang, 1993)
3. Explainable medical AI (Rudin, 2019) - Nature MI
4. Multi-criteria decision analysis for clinical scoring

CLINICAL GUIDELINES ENCODED:
- FDA prescribing information for Wegovy/Ozempic/Mounjaro
- ADA Standards of Medical Care in Diabetes (2024)
- AACE Obesity Management Guidelines (2023)

ACADEMIC JUSTIFICATION:
Unlike generic LLM approaches, this deterministic system provides:
- Traceable decision pathways
- Clinical audit trails
- Regulatory compliance
- Zero hallucination risk
- Reproducible outcomes

Author: M.Tech Research Project
Institution: IIIT Vadodara
Date: January 2026
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ==================== DATA MODELS ====================

class EligibilityStatus(Enum):
    """Final eligibility determination categories"""
    HIGHLY_ELIGIBLE = "highly_eligible"
    ELIGIBLE = "eligible"
    CONDITIONALLY_ELIGIBLE = "conditionally_eligible"
    REQUIRES_REVIEW = "requires_review"
    NOT_ELIGIBLE = "not_eligible"
    CONTRAINDICATED = "contraindicated"


class RiskLevel(Enum):
    """Patient risk stratification for GLP-1 therapy"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EligibilityResult:
    """Structured output from eligibility determination"""
    status: EligibilityStatus
    score: float  # 0-100 scale
    risk_level: RiskLevel
    
    # Detailed reasoning
    hard_constraints_passed: bool
    constraint_violations: List[str]
    scoring_breakdown: Dict[str, float]
    
    # Clinical decision support
    primary_indication: str
    contraindications: List[str]
    warnings: List[str]
    physician_review_required: bool
    physician_review_reasons: List[str]
    
    # Evidence and transparency
    decision_rationale: str
    clinical_guidelines_applied: List[str]
    recommendation_confidence: str  # "high", "moderate", "low"
    
    # Metadata
    bmi: Optional[float]
    diabetes_status: Optional[str]


# ==================== STAGE 1: HARD CONSTRAINTS ====================

class HardConstraintValidator:
    """
    Implements absolute contraindications and legal requirements.
    
    Based on FDA prescribing information and medical safety standards.
    Any violation results in immediate CONTRAINDICATED status.
    
    Reference: FDA Label - Wegovy (semaglutide) - Section 4: CONTRAINDICATIONS
    """
    
    # Absolute contraindications from FDA guidelines
    ABSOLUTE_CONTRAINDICATIONS = {
        "pregnancy": "Pregnancy (GLP-1 RAs contraindicated - fetal risk)",
        "breastfeeding": "Breastfeeding (insufficient safety data)",
        "medullary thyroid cancer": "Personal/family history of medullary thyroid carcinoma",
        "mtc": "Personal/family history of medullary thyroid carcinoma",
        "men-2": "Multiple Endocrine Neoplasia syndrome type 2",
        "men2": "Multiple Endocrine Neoplasia syndrome type 2",
        "pancreatitis": "Active or history of pancreatitis (high recurrence risk)",
        "pancreatic cancer": "Pancreatic malignancy",
        "gastroparesis": "Severe gastroparesis (delayed gastric emptying contraindication)",
        "type 1 diabetes": "Type 1 diabetes (not indicated for GLP-1 RA monotherapy)",
        "anorexia": "Active eating disorder (contraindicated for weight loss use)",
        "bulimia": "Active eating disorder (contraindicated for weight loss use)"
    }
    
    MINIMUM_AGE = 18  # FDA approval threshold
    
    @staticmethod
    def validate(collected_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Returns: (passed: bool, violations: List[str])
        """
        violations = []
        
        # ── Age requirement ──
        age = collected_data.get("age")
        if age is not None:
            try:
                age_int = int(age)
                if age_int < HardConstraintValidator.MINIMUM_AGE:
                    violations.append(f"Age {age_int} below FDA minimum ({HardConstraintValidator.MINIMUM_AGE})")
            except (ValueError, TypeError):
                violations.append("Invalid age data")
        else:
            violations.append("Age not provided")
        
        # ── Pregnancy/breastfeeding (critical safety) ──
        if collected_data.get("is_pregnant_breastfeeding") is True:
            violations.append(HardConstraintValidator.ABSOLUTE_CONTRAINDICATIONS["pregnancy"])
        
        # ── High-risk conditions ──
        high_risk = collected_data.get("high_risk_conditions", [])
        if isinstance(high_risk, list):
            for condition in high_risk:
                condition_lower = str(condition).lower()
                for keyword, description in HardConstraintValidator.ABSOLUTE_CONTRAINDICATIONS.items():
                    if keyword in condition_lower:
                        violations.append(description)
        
        # ── Check current medical conditions text ──
        current_conditions = str(collected_data.get("current_medical_conditions", "")).lower()
        for keyword, description in HardConstraintValidator.ABSOLUTE_CONTRAINDICATIONS.items():
            if keyword in current_conditions and description not in violations:
                violations.append(description)
        
        passed = len(violations) == 0
        
        if not passed:
            logger.warning(f"❌ Hard constraints FAILED: {violations}")
        else:
            logger.info("✅ Hard constraints PASSED")
        
        return passed, violations


# ==================== STAGE 2: WEIGHTED SCORING ====================

class EligibilityScorer:
    """
    Multi-criteria scoring system for GLP-1 eligibility.
    
    Inspired by clinical risk calculators (e.g., ASCVD Risk Estimator, FRAX).
    Each criterion contributes to a 0-100 score representing eligibility strength.
    
    SCORING CRITERIA (weighted):
    1. BMI & Weight Status (40 points) - Primary FDA indication
    2. Diabetes Status (25 points) - Major indication for GLP-1
    3. Metabolic Comorbidities (20 points) - Risk modifiers
    4. Weight Loss Goal Alignment (15 points) - Patient motivation
    
    Reference: ADA Standards of Care 2024 - Section 9: Pharmacologic Approaches
    Reference: AACE Obesity CPG 2023 - Algorithm for GLP-1 RA use
    """
    
    # BMI thresholds from FDA guidelines
    BMI_OBESITY = 30.0          # Obesity threshold (FDA approved)
    BMI_OVERWEIGHT_WITH_COMORB = 27.0  # Overweight + comorbidity (FDA approved)
    
    @staticmethod
    def calculate_bmi(weight_lbs: float, height_str: str) -> Optional[float]:
        """
        Calculate BMI from weight (lbs) and height (various formats).
        Returns None if calculation fails.
        """
        import re
        
        try:
            # Parse height
            h = height_str.strip().lower()
            
            if 'cm' in h:
                cm = float(re.findall(r'[\d.]+', h)[0])
                inches = cm / 2.54
            else:
                parts = re.findall(r'\d+', h)
                if len(parts) >= 2:
                    inches = int(parts[0]) * 12 + int(parts[1])
                elif len(parts) == 1:
                    val = int(parts[0])
                    inches = val if val > 24 else val * 12
                else:
                    return None
            
            # BMI formula: (weight_lbs / height_inches²) × 703
            bmi = (weight_lbs / (inches ** 2)) * 703
            return round(bmi, 1)
        
        except Exception as e:
            logger.error(f"BMI calculation error: {e}")
            return None
    
    @staticmethod
    def score_bmi_criterion(bmi: float, has_comorbidities: bool) -> Tuple[float, str]:
        """
        Score BMI-based eligibility (0-40 points).
        
        FDA Criteria:
        - BMI ≥ 30: Approved for weight management
        - BMI ≥ 27 + comorbidity: Approved for weight management
        - BMI < 27: Not indicated unless diabetes
        """
        if bmi >= EligibilityScorer.BMI_OBESITY:
            return 40.0, f"BMI {bmi} ≥ 30 (obesity - strong indication)"
        
        elif bmi >= EligibilityScorer.BMI_OVERWEIGHT_WITH_COMORB:
            if has_comorbidities:
                return 35.0, f"BMI {bmi} ≥ 27 with comorbidities (approved indication)"
            else:
                return 20.0, f"BMI {bmi} ≥ 27 but no documented comorbidities (borderline)"
        
        elif bmi >= 25.0:
            return 10.0, f"BMI {bmi} overweight but below FDA threshold (requires diabetes)"
        
        else:
            return 0.0, f"BMI {bmi} normal weight (not indicated for weight loss)"
    
    @staticmethod
    def score_diabetes_criterion(collected_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Score diabetes status (0-25 points).
        
        GLP-1 RAs are FDA-approved for T2DM regardless of BMI.
        Strong indication even with lower BMI.
        """
        conditions_text = str(collected_data.get("current_medical_conditions", "")).lower()
        
        diabetes_keywords = [
            "type 2 diabetes", "t2dm", "diabetes mellitus", "diabetic",
            "high blood sugar", "hyperglycemia", "a1c"
        ]
        
        has_t2dm = any(kw in conditions_text for kw in diabetes_keywords)
        
        # Exclude type 1 (already caught in hard constraints)
        if "type 1" in conditions_text:
            return 0.0, "Type 1 diabetes detected (contraindicated)"
        
        if has_t2dm:
            return 25.0, "Type 2 diabetes present (primary indication for GLP-1)"
        else:
            return 0.0, "No diabetes documented (weight management only)"
    
    @staticmethod
    def score_comorbidity_criterion(collected_data: Dict[str, Any]) -> Tuple[float, str, bool]:
        """
        Score metabolic comorbidities (0-20 points).
        
        Comorbidities that strengthen GLP-1 indication:
        - Hypertension
        - Dyslipidemia / high cholesterol
        - PCOS
        - NAFLD / fatty liver
        - Sleep apnea
        - Cardiovascular disease history
        
        Returns: (score, rationale, has_comorbidities_flag)
        """
        conditions_text = str(collected_data.get("current_medical_conditions", "")).lower()
        
        comorbidity_map = {
            "hypertension": 5,
            "high blood pressure": 5,
            "cholesterol": 5,
            "dyslipidemia": 5,
            "hyperlipidemia": 5,
            "pcos": 4,
            "polycystic": 4,
            "fatty liver": 4,
            "nafld": 4,
            "sleep apnea": 4,
            "cardiovascular": 6,
            "heart disease": 6,
            "stroke": 6,
            "prediabetes": 3,
        }
        
        score = 0.0
        found_conditions = []
        
        for condition, points in comorbidity_map.items():
            if condition in conditions_text:
                score += points
                found_conditions.append(condition)
        
        # Cap at 20 points
        score = min(score, 20.0)
        
        has_comorbidities = score > 0
        
        if found_conditions:
            rationale = f"Comorbidities detected: {', '.join(found_conditions)} (+{score} pts)"
        else:
            rationale = "No documented comorbidities"
        
        return score, rationale, has_comorbidities
    
    @staticmethod
    def score_weight_goal_criterion(collected_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Score weight loss goal alignment (0-15 points).
        
        Assesses patient motivation and realistic expectations.
        """
        goal_text = str(collected_data.get("weight_loss_goal", "")).lower()
        
        # Realistic goal indicators
        realistic_keywords = [
            "5", "10", "15", "20", "25", "30", "percent", "%",
            "healthy", "sustainable", "gradual", "long-term"
        ]
        
        # Unrealistic/concerning indicators
        concerning_keywords = [
            "100", "200", "fast", "quick", "rapid", "extreme"
        ]
        
        if not goal_text or goal_text in ["none", "n/a"]:
            return 5.0, "No specific goal stated"
        
        if any(kw in goal_text for kw in concerning_keywords):
            return 0.0, "Unrealistic expectations (requires counseling)"
        
        if any(kw in goal_text for kw in realistic_keywords):
            return 15.0, "Realistic weight loss goal stated"
        
        # Default for any stated goal
        return 10.0, "Weight loss goal stated"
    
    @staticmethod
    def calculate_total_score(collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive eligibility score (0-100).
        
        Returns detailed scoring breakdown for transparency.
        """
        breakdown = {}
        total_score = 0.0
        
        # Calculate BMI
        weight = collected_data.get("weight")
        height = collected_data.get("height")
        bmi = None
        
        if weight and height:
            bmi = EligibilityScorer.calculate_bmi(float(weight), str(height))
        
        # 1. BMI Scoring (need comorbidity flag first)
        comorbidity_score, comorbidity_rationale, has_comorbidities = \
            EligibilityScorer.score_comorbidity_criterion(collected_data)
        
        if bmi:
            bmi_score, bmi_rationale = EligibilityScorer.score_bmi_criterion(bmi, has_comorbidities)
            breakdown["bmi"] = {"score": bmi_score, "rationale": bmi_rationale}
            total_score += bmi_score
        else:
            breakdown["bmi"] = {"score": 0.0, "rationale": "BMI not calculable"}
        
        # 2. Diabetes Scoring
        diabetes_score, diabetes_rationale = EligibilityScorer.score_diabetes_criterion(collected_data)
        breakdown["diabetes"] = {"score": diabetes_score, "rationale": diabetes_rationale}
        total_score += diabetes_score
        
        # 3. Comorbidity Scoring
        breakdown["comorbidities"] = {"score": comorbidity_score, "rationale": comorbidity_rationale}
        total_score += comorbidity_score
        
        # 4. Weight Goal Scoring
        goal_score, goal_rationale = EligibilityScorer.score_weight_goal_criterion(collected_data)
        breakdown["weight_goal"] = {"score": goal_score, "rationale": goal_rationale}
        total_score += goal_score
        
        return {
            "total_score": round(total_score, 1),
            "breakdown": breakdown,
            "bmi": bmi,
            "has_diabetes": breakdown["diabetes"]["score"] > 0,
            "has_comorbidities": has_comorbidities
        }


# ==================== STAGE 3: CLINICAL REASONING ENGINE ====================

class ClinicalReasoningEngine:
    """
    Maps eligibility scores to clinical recommendations.
    
    Provides structured decision support with evidence-based rationale.
    Implements safety layers and physician escalation triggers.
    
    Reference: Sutton et al. (2020) - "Clinical decision support systems" - npj Digital Medicine
    """
    
    @staticmethod
    def determine_status(score: float, violations: List[str], warnings: List[str]) -> EligibilityStatus:
        """
        Map score to eligibility status with safety overrides.
        """
        # Safety override: contraindications trump score
        if violations:
            return EligibilityStatus.CONTRAINDICATED
        
        # High-risk warnings require review regardless of score
        critical_warnings = [w for w in warnings if "CRITICAL" in w or "HIGH RISK" in w]
        if critical_warnings:
            return EligibilityStatus.REQUIRES_REVIEW
        
        # Score-based determination
        if score >= 75:
            return EligibilityStatus.HIGHLY_ELIGIBLE
        elif score >= 60:
            return EligibilityStatus.ELIGIBLE
        elif score >= 40:
            return EligibilityStatus.CONDITIONALLY_ELIGIBLE
        elif score >= 20:
            return EligibilityStatus.REQUIRES_REVIEW
        else:
            return EligibilityStatus.NOT_ELIGIBLE
    
    @staticmethod
    def determine_risk_level(collected_data: Dict[str, Any], score: float) -> RiskLevel:
        """
        Stratify patient risk for GLP-1 therapy.
        """
        # Critical safety factors
        age = collected_data.get("age")
        if age and int(age) >= 75:
            return RiskLevel.HIGH
        
        conditions_text = str(collected_data.get("current_medical_conditions", "")).lower()
        high_risk_keywords = [
            "kidney disease", "renal", "liver disease", "hepatic",
            "heart failure", "retinopathy", "neuropathy"
        ]
        
        if any(kw in conditions_text for kw in high_risk_keywords):
            return RiskLevel.HIGH
        
        # Score-based risk
        if score >= 70:
            return RiskLevel.LOW
        elif score >= 50:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH
    
    @staticmethod
    def generate_rationale(
        status: EligibilityStatus,
        score_data: Dict[str, Any],
        violations: List[str],
        warnings: List[str]
    ) -> str:
        """
        Generate human-readable clinical decision rationale.
        """
        if violations:
            return (
                f"Patient has absolute contraindications to GLP-1 therapy: "
                f"{'; '.join(violations)}. GLP-1 receptor agonists are not safe "
                f"for this patient per FDA prescribing guidelines."
            )
        
        score = score_data["total_score"]
        breakdown = score_data["breakdown"]
        
        rationale_parts = [f"Eligibility score: {score}/100."]
        
        # BMI analysis
        if "bmi" in breakdown:
            rationale_parts.append(breakdown["bmi"]["rationale"] + ".")
        
        # Diabetes
        if breakdown["diabetes"]["score"] > 0:
            rationale_parts.append(breakdown["diabetes"]["rationale"] + ".")
        
        # Comorbidities
        if breakdown["comorbidities"]["score"] > 0:
            rationale_parts.append(breakdown["comorbidities"]["rationale"] + ".")
        
        # Final assessment
        if status == EligibilityStatus.HIGHLY_ELIGIBLE:
            rationale_parts.append(
                "Patient meets FDA criteria for GLP-1 therapy with strong clinical indication. "
                "Recommend physician evaluation for prescription initiation."
            )
        elif status == EligibilityStatus.ELIGIBLE:
            rationale_parts.append(
                "Patient meets FDA criteria for GLP-1 therapy. "
                "Physician evaluation recommended to confirm appropriateness."
            )
        elif status == EligibilityStatus.CONDITIONALLY_ELIGIBLE:
            rationale_parts.append(
                "Patient has partial indication for GLP-1 therapy. "
                "Requires detailed physician assessment of risk-benefit profile."
            )
        elif status == EligibilityStatus.REQUIRES_REVIEW:
            rationale_parts.append(
                "Eligibility uncertain. Mandatory physician review required before proceeding."
            )
        else:
            rationale_parts.append(
                "Patient does not meet standard FDA criteria for GLP-1 therapy at this time."
            )
        
        return " ".join(rationale_parts)
    
    @staticmethod
    def determine_primary_indication(score_data: Dict[str, Any]) -> str:
        """
        Identify the primary clinical indication for GLP-1.
        """
        if score_data.get("has_diabetes"):
            return "Type 2 diabetes mellitus (glycemic control + weight management)"
        elif score_data.get("bmi") and score_data["bmi"] >= 30:
            return "Obesity (BMI ≥ 30) - weight management"
        elif score_data.get("bmi") and score_data["bmi"] >= 27 and score_data.get("has_comorbidities"):
            return "Overweight (BMI ≥ 27) with weight-related comorbidities"
        else:
            return "No clear primary indication identified"
    
    @staticmethod
    def check_physician_review_needed(
        status: EligibilityStatus,
        risk_level: RiskLevel,
        warnings: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Determine if physician review is mandatory.
        Returns: (required: bool, reasons: List[str])
        """
        reasons = []
        
        # Always require review for certain statuses
        if status in [EligibilityStatus.REQUIRES_REVIEW, EligibilityStatus.CONTRAINDICATED]:
            reasons.append("Eligibility status requires physician evaluation")
        
        # High risk always needs review
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            reasons.append("High-risk patient profile")
        
        # Any warnings trigger review
        if warnings:
            reasons.append("Medical warnings detected")
        
        # Default: all GLP-1 prescriptions require physician approval
        if not reasons:
            reasons.append("Standard physician approval required for GLP-1 prescription")
        
        return True, reasons  # Always require review in this system


# ==================== MAIN ELIGIBILITY ENGINE ====================

class GLP1EligibilityEngine:
    """
    Main orchestrator for GLP-1 eligibility determination.
    
    Integrates all stages into a single, transparent decision pipeline.
    """
    
    @staticmethod
    def evaluate(collected_data: Dict[str, Any]) -> EligibilityResult:
        """
        Complete eligibility evaluation pipeline.
        
        Args:
            collected_data: Patient data dictionary from chat endpoint
        
        Returns:
            EligibilityResult with comprehensive decision support
        """
        logger.info("=" * 60)
        logger.info("🏥 STARTING GLP-1 ELIGIBILITY EVALUATION")
        logger.info("=" * 60)
        
        # ─── STAGE 1: Hard Constraints ───
        hard_passed, violations = HardConstraintValidator.validate(collected_data)
        
        # ─── STAGE 2: Scoring (only if hard constraints passed) ───
        if hard_passed:
            score_data = EligibilityScorer.calculate_total_score(collected_data)
            score = score_data["total_score"]
            bmi = score_data.get("bmi")
            
            logger.info(f"📊 Eligibility Score: {score}/100")
            for criterion, data in score_data["breakdown"].items():
                logger.info(f"  • {criterion}: {data['score']} - {data['rationale']}")
        else:
            # Contraindicated - no scoring needed
            score_data = {"total_score": 0.0, "breakdown": {}, "bmi": None}
            score = 0.0
            bmi = None
        
        # ─── STAGE 3: Clinical Reasoning ───
        
        # Extract warnings from data (from original medical validator)
        warnings = []
        if collected_data.get("is_pregnant_breastfeeding") is True:
            warnings.append("CRITICAL: Pregnancy/breastfeeding")
        
        # Determine final status
        status = ClinicalReasoningEngine.determine_status(score, violations, warnings)
        risk_level = ClinicalReasoningEngine.determine_risk_level(collected_data, score)
        rationale = ClinicalReasoningEngine.generate_rationale(status, score_data, violations, warnings)
        primary_indication = ClinicalReasoningEngine.determine_primary_indication(score_data)
        
        physician_required, review_reasons = ClinicalReasoningEngine.check_physician_review_needed(
            status, risk_level, warnings
        )
        
        # Determine confidence
        if status == EligibilityStatus.CONTRAINDICATED:
            confidence = "high"
        elif status in [EligibilityStatus.HIGHLY_ELIGIBLE, EligibilityStatus.ELIGIBLE]:
            confidence = "high" if score >= 70 else "moderate"
        else:
            confidence = "low"
        
        # Clinical guidelines applied
        guidelines = [
            "FDA Prescribing Information - Wegovy (semaglutide)",
            "ADA Standards of Medical Care in Diabetes 2024",
            "AACE Obesity Clinical Practice Guidelines 2023"
        ]
        
        # Diabetes status
        diabetes_status = "Type 2 Diabetes" if score_data.get("has_diabetes") else "No diabetes documented"
        
        result = EligibilityResult(
            status=status,
            score=score,
            risk_level=risk_level,
            hard_constraints_passed=hard_passed,
            constraint_violations=violations,
            scoring_breakdown={k: v["score"] for k, v in score_data["breakdown"].items()},
            primary_indication=primary_indication,
            contraindications=violations,
            warnings=warnings,
            physician_review_required=physician_required,
            physician_review_reasons=review_reasons,
            decision_rationale=rationale,
            clinical_guidelines_applied=guidelines,
            recommendation_confidence=confidence,
            bmi=bmi,
            diabetes_status=diabetes_status
        )
        
        logger.info("=" * 60)
        logger.info(f"🎯 FINAL STATUS: {status.value.upper()}")
        logger.info(f"📈 Score: {score}/100 | Risk: {risk_level.value.upper()}")
        logger.info(f"🔍 Confidence: {confidence.upper()}")
        logger.info("=" * 60)
        
        return result


# ==================== UTILITY FUNCTIONS ====================

def format_eligibility_response(result: EligibilityResult) -> Dict[str, Any]:
    """
    Convert EligibilityResult to JSON-serializable dictionary for API response.
    """
    return {
        "eligibility_status": result.status.value,
        "eligibility_score": result.score,
        "risk_level": result.risk_level.value,
        
        "clinical_assessment": {
            "primary_indication": result.primary_indication,
            "bmi": result.bmi,
            "diabetes_status": result.diabetes_status,
            "contraindications": result.contraindications,
            "warnings": result.warnings,
        },
        
        "decision_support": {
            "rationale": result.decision_rationale,
            "confidence": result.recommendation_confidence,
            "guidelines_applied": result.clinical_guidelines_applied,
            "scoring_breakdown": result.scoring_breakdown,
        },
        
        "physician_review": {
            "required": result.physician_review_required,
            "reasons": result.physician_review_reasons,
        },
        
        "constraints": {
            "hard_constraints_passed": result.hard_constraints_passed,
            "violations": result.constraint_violations,
        }
    }


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example test case
    test_patient = {
        "name": "John Doe",
        "age": 45,
        "height": "5'10\"",
        "weight": 220,
        "is_pregnant_breastfeeding": False,
        "high_risk_conditions": [],
        "current_medical_conditions": "Type 2 diabetes, hypertension, high cholesterol",
        "currently_on_glp1": False,
        "other_medications": "Metformin, Lisinopril",
        "allergies": "None",
        "weight_loss_goal": "Lose 30 pounds gradually",
        "interested_medication": "Wegovy"
    }
    
    # Run evaluation
    result = GLP1EligibilityEngine.evaluate(test_patient)
    
    # Format response
    response = format_eligibility_response(result)
    
    import json
    print("\n" + "=" * 60)
    print("ELIGIBILITY EVALUATION RESULT")
    print("=" * 60)
    print(json.dumps(response, indent=2))