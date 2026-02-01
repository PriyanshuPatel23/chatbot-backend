"""
Comprehensive Test Suite for GLP-1 Eligibility Engine

Tests all critical pathways, edge cases, and clinical scenarios.
Validates correctness of the research-grade CDSS implementation.

Run: python test_eligibility_engine.py
"""

import sys
from typing import Dict, Any
from eligibility_engine import (
    GLP1EligibilityEngine,
    EligibilityStatus,
    RiskLevel,
    format_eligibility_response
)


# ==================== TEST CASES ====================

class TestCase:
    """Test case container"""
    def __init__(self, name: str, patient_data: Dict[str, Any], 
                 expected_status: EligibilityStatus, 
                 expected_score_range: tuple = None,
                 description: str = ""):
        self.name = name
        self.patient_data = patient_data
        self.expected_status = expected_status
        self.expected_score_range = expected_score_range
        self.description = description


# ──────────────────────────────────────────────────────────────
# CATEGORY 1: CONTRAINDICATED CASES (Should FAIL hard constraints)
# ──────────────────────────────────────────────────────────────

TEST_CONTRAINDICATED = [
    TestCase(
        name="Pregnancy - Absolute Contraindication",
        patient_data={
            "age": 32,
            "height": "5'5\"",
            "weight": 180,
            "is_pregnant_breastfeeding": True,  # CONTRAINDICATION
            "high_risk_conditions": [],
            "current_medical_conditions": "None",
            "other_medications": "Prenatal vitamins",
            "allergies": "None",
            "weight_loss_goal": "Postpartum weight loss",
            "interested_medication": "Wegovy"
        },
        expected_status=EligibilityStatus.CONTRAINDICATED,
        description="Pregnancy is an absolute contraindication per FDA guidelines"
    ),
    
    TestCase(
        name="Medullary Thyroid Cancer - Absolute Contraindication",
        patient_data={
            "age": 55,
            "height": "5'10\"",
            "weight": 240,
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": ["medullary thyroid cancer"],  # CONTRAINDICATION
            "current_medical_conditions": "Type 2 diabetes",
            "other_medications": "Metformin",
            "allergies": "None",
            "weight_loss_goal": "Lose 40 pounds",
            "interested_medication": "Ozempic"
        },
        expected_status=EligibilityStatus.CONTRAINDICATED,
        description="MTC history contraindicated for GLP-1 RAs"
    ),
    
    TestCase(
        name="Active Pancreatitis - Absolute Contraindication",
        patient_data={
            "age": 45,
            "height": "6'0\"",
            "weight": 260,
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": ["pancreatitis"],  # CONTRAINDICATION
            "current_medical_conditions": "Hypertension",
            "other_medications": "Lisinopril",
            "allergies": "None",
            "weight_loss_goal": "Weight management",
            "interested_medication": "Mounjaro"
        },
        expected_status=EligibilityStatus.CONTRAINDICATED,
        description="Active or history of pancreatitis is contraindicated"
    ),
    
    TestCase(
        name="Type 1 Diabetes - Contraindication for Weight Loss",
        patient_data={
            "age": 28,
            "height": "5'7\"",
            "weight": 190,
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": [],
            "current_medical_conditions": "Type 1 diabetes",  # CONTRAINDICATION
            "other_medications": "Insulin",
            "allergies": "None",
            "weight_loss_goal": "Lose 20 pounds",
            "interested_medication": "Wegovy"
        },
        expected_status=EligibilityStatus.CONTRAINDICATED,
        description="T1DM not indicated for GLP-1 monotherapy for weight loss"
    ),
    
    TestCase(
        name="Age Below 18 - Legal Contraindication",
        patient_data={
            "age": 17,  # BELOW MINIMUM
            "height": "5'6\"",
            "weight": 200,
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": [],
            "current_medical_conditions": "None",
            "other_medications": "None",
            "allergies": "None",
            "weight_loss_goal": "Lose weight",
            "interested_medication": "Wegovy"
        },
        expected_status=EligibilityStatus.CONTRAINDICATED,
        description="FDA approval only for adults ≥18"
    ),
]


# ──────────────────────────────────────────────────────────────
# CATEGORY 2: HIGHLY ELIGIBLE CASES (Score ≥75)
# ──────────────────────────────────────────────────────────────

TEST_HIGHLY_ELIGIBLE = [
    TestCase(
        name="Obesity + Type 2 Diabetes + Comorbidities - Ideal Candidate",
        patient_data={
            "age": 52,
            "height": "5'8\"",
            "weight": 230,  # BMI ~35 (obesity)
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": [],
            "current_medical_conditions": "Type 2 diabetes, hypertension, high cholesterol",
            "other_medications": "Metformin, Lisinopril, Atorvastatin",
            "allergies": "None",
            "weight_loss_goal": "Lose 30 pounds for better diabetes control",
            "interested_medication": "Ozempic"
        },
        expected_status=EligibilityStatus.HIGHLY_ELIGIBLE,
        expected_score_range=(75, 100),
        description="Strong indication: obesity + diabetes + multiple comorbidities"
    ),
    
    TestCase(
        name="Severe Obesity - Strong Weight Loss Indication",
        patient_data={
            "age": 38,
            "height": "5'4\"",
            "weight": 220,  # BMI ~38 (class II obesity)
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": [],
            "current_medical_conditions": "Hypertension, sleep apnea, PCOS",
            "other_medications": "Lisinopril",
            "allergies": "None",
            "weight_loss_goal": "Lose 50 pounds to improve health",
            "interested_medication": "Wegovy"
        },
        expected_status=EligibilityStatus.HIGHLY_ELIGIBLE,
        expected_score_range=(75, 100),
        description="High BMI with significant comorbidities"
    ),
]


# ──────────────────────────────────────────────────────────────
# CATEGORY 3: ELIGIBLE CASES (Score 60-74)
# ──────────────────────────────────────────────────────────────

TEST_ELIGIBLE = [
    TestCase(
        name="Type 2 Diabetes + Overweight",
        patient_data={
            "age": 48,
            "height": "5'9\"",
            "weight": 200,  # BMI ~29.5 (overweight, near obesity)
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": [],
            "current_medical_conditions": "Type 2 diabetes",
            "other_medications": "Metformin",
            "allergies": "None",
            "weight_loss_goal": "Lose 20 pounds to improve A1C",
            "interested_medication": "Ozempic"
        },
        expected_status=EligibilityStatus.ELIGIBLE,
        expected_score_range=(60, 74),
        description="Diabetes provides primary indication even with lower BMI"
    ),
    
    TestCase(
        name="Obesity without Comorbidities",
        patient_data={
            "age": 35,
            "height": "5'10\"",
            "weight": 220,  # BMI ~31.5 (class I obesity)
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": [],
            "current_medical_conditions": "None",
            "other_medications": "None",
            "allergies": "None",
            "weight_loss_goal": "Lose 25 pounds",
            "interested_medication": "Wegovy"
        },
        expected_status=EligibilityStatus.ELIGIBLE,
        expected_score_range=(60, 74),
        description="BMI ≥30 meets FDA criteria for weight management"
    ),
]


# ──────────────────────────────────────────────────────────────
# CATEGORY 4: CONDITIONALLY ELIGIBLE (Score 40-59)
# ──────────────────────────────────────────────────────────────

TEST_CONDITIONAL = [
    TestCase(
        name="Overweight (BMI 27-30) with One Comorbidity",
        patient_data={
            "age": 42,
            "height": "5'6\"",
            "weight": 170,  # BMI ~27.4 (overweight)
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": [],
            "current_medical_conditions": "Hypertension",
            "other_medications": "Lisinopril",
            "allergies": "None",
            "weight_loss_goal": "Lose 15 pounds",
            "interested_medication": "Wegovy"
        },
        expected_status=EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        expected_score_range=(40, 59),
        description="BMI ≥27 with comorbidity - borderline indication"
    ),
    
    TestCase(
        name="Prediabetes + Overweight",
        patient_data={
            "age": 50,
            "height": "5'8\"",
            "weight": 195,  # BMI ~29.6
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": [],
            "current_medical_conditions": "Prediabetes, borderline high cholesterol",
            "other_medications": "None",
            "allergies": "None",
            "weight_loss_goal": "Prevent diabetes progression",
            "interested_medication": "Wegovy"
        },
        expected_status=EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        expected_score_range=(40, 59),
        description="Preventive indication - requires careful evaluation"
    ),
]


# ──────────────────────────────────────────────────────────────
# CATEGORY 5: NOT ELIGIBLE (Score <20)
# ──────────────────────────────────────────────────────────────

TEST_NOT_ELIGIBLE = [
    TestCase(
        name="Normal Weight - No Medical Indication",
        patient_data={
            "age": 30,
            "height": "5'7\"",
            "weight": 140,  # BMI ~22 (normal)
            "is_pregnant_breastfeeding": False,
            "high_risk_conditions": [],
            "current_medical_conditions": "None",
            "other_medications": "None",
            "allergies": "None",
            "weight_loss_goal": "Lose 10 pounds for cosmetic reasons",
            "interested_medication": "Wegovy"
        },
        expected_status=EligibilityStatus.NOT_ELIGIBLE,
        expected_score_range=(0, 19),
        description="BMI in normal range - not indicated for GLP-1"
    ),
]


# ==================== TEST RUNNER ====================

def run_test_case(test: TestCase) -> bool:
    """Run a single test case and return pass/fail"""
    print(f"\n{'='*70}")
    print(f"TEST: {test.name}")
    print(f"{'='*70}")
    print(f"Description: {test.description}")
    
    try:
        # Run evaluation
        result = GLP1EligibilityEngine.evaluate(test.patient_data)
        
        # Check status
        status_match = result.status == test.expected_status
        
        # Check score range if specified
        score_match = True
        if test.expected_score_range:
            min_score, max_score = test.expected_score_range
            score_match = min_score <= result.score <= max_score
        
        # Print results
        print(f"\n📊 RESULTS:")
        print(f"  Status: {result.status.value} (expected: {test.expected_status.value})")
        print(f"  Score: {result.score}/100")
        if test.expected_score_range:
            print(f"  Expected Range: {test.expected_score_range[0]}-{test.expected_score_range[1]}")
        print(f"  Risk Level: {result.risk_level.value}")
        print(f"  BMI: {result.bmi}")
        
        print(f"\n📋 RATIONALE:")
        print(f"  {result.decision_rationale}")
        
        if result.contraindications:
            print(f"\n⚠️  CONTRAINDICATIONS:")
            for c in result.contraindications:
                print(f"  • {c}")
        
        # Verdict
        test_passed = status_match and score_match
        
        if test_passed:
            print(f"\n✅ TEST PASSED")
        else:
            print(f"\n❌ TEST FAILED")
            if not status_match:
                print(f"   Status mismatch: got {result.status.value}, expected {test.expected_status.value}")
            if not score_match:
                print(f"   Score {result.score} outside expected range {test.expected_score_range}")
        
        return test_passed
    
    except Exception as e:
        print(f"\n❌ TEST ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("GLP-1 ELIGIBILITY ENGINE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    all_tests = [
        ("CONTRAINDICATED CASES", TEST_CONTRAINDICATED),
        ("HIGHLY ELIGIBLE CASES", TEST_HIGHLY_ELIGIBLE),
        ("ELIGIBLE CASES", TEST_ELIGIBLE),
        ("CONDITIONALLY ELIGIBLE CASES", TEST_CONDITIONAL),
        ("NOT ELIGIBLE CASES", TEST_NOT_ELIGIBLE),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category_name, test_list in all_tests:
        print(f"\n\n{'#'*70}")
        print(f"# CATEGORY: {category_name}")
        print(f"{'#'*70}")
        
        category_passed = 0
        for test in test_list:
            total_tests += 1
            if run_test_case(test):
                passed_tests += 1
                category_passed += 1
        
        print(f"\n{category_name} Results: {category_passed}/{len(test_list)} passed")
    
    # Final summary
    print("\n\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print("="*70)
    
    return passed_tests == total_tests


# ==================== MAIN ====================

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)