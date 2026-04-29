# -*- coding: utf-8 -*-
"""Central configuration for the top-journal consensus pipeline.

Design target:
- accident-level split;
- train-only preprocessing and feature selection;
- consensus-guided, mechanism-constrained feature composition;
- manifest-based rule mining and blind accident-level replay;
- minimal but sufficient outputs for manuscript and reviewer audit.
"""
from __future__ import annotations

RANDOM_STATE = 42
TEST_SIZE = 0.20

INPUT_EXCEL = "data.xlsx"
CLEANED_DATA = "Cleaned_Data_Base.csv"
TRAIN_MATRIX = "Ready_Matrix_Train.csv"
TEST_MATRIX = "Ready_Matrix_Test.csv"
# Aliases resolve to the primary matrices to avoid duplicate same-content outputs.
TRAIN_MATRIX_ALIAS = TRAIN_MATRIX
TEST_MATRIX_ALIAS = TEST_MATRIX

SENTINEL_UNKNOWN_AUDIT = "Sentinel_Unknown_Audit.csv"
SENTINEL_FLAGS_TRAIN = "Sentinel_Flags_Train.csv"
SENTINEL_FLAGS_TEST = "Sentinel_Flags_Test.csv"

TARGET_SOURCE_COL = "TARGET_MAIS_Merged"
LABEL_COL = "Label_Severe_Injury"
GROUP_CANDIDATES = (
    "事故编号(FALL)", "事故编号_FALL", "KEY_ACC", "FALL", "Accident_ID", "ACCIDENT_ID",
)

# Robust preprocessing strategy.
# Bounded variables use train-median imputation + fixed physical clipping.
# This avoids unstable MICE extrapolation for engineering-bounded variables.
PHYSICAL_BOUNDS = {
    "V0": (0.0, 250.0),
    "VK": (0.0, 250.0),
    "MUE": (0.0, 1.2),
    "GROESP": (30.0, 250.0),
    "GEWP": (2.0, 200.0),
    # CIDAS vehicle dimensions are treated as millimetres in the ready matrix.
    "LAENGE": (300.0, 25000.0),
    "LÄNGE": (300.0, 25000.0),
    "BREITE": (500.0, 3500.0),
    "HOEHE": (500.0, 4500.0),
    "ALTER1": (0.0, 120.0),
    "FEATURE_AGE_YEARS": (0.0, 120.0),
    "TEMP": (-50.0, 60.0),
}

# Variables with these patterns are treated as continuous engineering variables when present.
CONTINUOUS_PATTERNS = (
    "V0", "VK", "MUE", "GROESP", "GEWP", "ALTER1", "FEATURE_AGE_YEARS", "TEMP",
    "LAENGE", "LÄNGE", "BREITE", "HOEHE", "车长", "车宽", "车高",
    "STOSSP", "STOSS", "STOSSPX", "STOSSPY", "STOSSPZ", "车宽", "车高", "车长", "碰撞角度",
)

# Consensus feature-composition settings.
SELECTED_FEATURE_N = 25
SCREENER_TOPK = 60
STABILITY_SPLITS = 5
STABILITY_TOPK = 50
PERMUTATION_CANDIDATE_TOPK = 80
PERMUTATION_REPEATS = 2
GWO_CANDIDATE_TOPK = 50
GWO_WOLVES = 8
GWO_ITERATIONS = 8
GWO_REPEATS = 2
GWO_SIZE_PENALTY = 0.010

# Family minimums are methodological constraints: they align the compact rule-input
# space with the paper's multi-source, mechanism-informed evidence-chain objective.
MANDATORY_FAMILY_MIN = {
    "speed_v0": 1,
    "speed_vk": 1,
    "bio": 1,
    "age": 1,
    "light": 1,
    "road_env": 1,
    "road_type": 1,
    "road_surface": 1,
    "vehicle_size": 1,
    "safety": 1,
    "crash_type": 1,
    "lane": 1,
    "weather": 1,
}

FAMILY_CAPS = {
    "speed_v0": 2,
    "speed_vk": 2,
    "age": 2,
    "bio": 3,
    "light": 2,
    "road_env": 3,
    "road_type": 3,
    "road_surface": 2,
    "vehicle_size": 3,
    "safety": 2,
    "crash_type": 3,
    "lane": 3,
    "weather": 2,
    "vehicle_state": 2,
    "engineering": 3,
    "other": 2,
}

FAMILY_PRIORITY = {
    "speed_v0": 1.00,
    "speed_vk": 1.00,
    "light": 0.85,
    "road_env": 0.85,
    "road_type": 0.80,
    "road_surface": 0.80,
    "vehicle_size": 0.80,
    "safety": 0.80,
    "bio": 0.75,
    "age": 0.70,
    "crash_type": 0.70,
    "lane": 0.65,
    "weather": 0.55,
    "vehicle_state": 0.55,
    "engineering": 0.50,
    "other": 0.25,
}

# Rule mining.
RULE_MIN_SUPPORT = 0.015
RULE_MIN_LIFT = 1.10
RULE_MIN_LEN = 2
RULE_MAX_LEN = 4
RULE_MAX_FINAL = 20
TARGET_ITEM = "__TARGET_SEVERE_INJURY__"

# Train-only rule diversity constraints. These caps shape manuscript-facing
# mechanism coverage without changing support/lift/evidence thresholds.
RULE_FINAL_DISPLAY_MAX = 15
RULE_AXIS_CAPS = {
    "speed_energy": 3,
    "vulnerability_speed": 3,
    "vulnerability_body": 2,
    "road_speed_environment": 3,
    "vehicle_geometry_interaction": 2,
    "lighting_visibility": 2,
    "surface_weather_friction": 2,
    "safety_assistance": 2,
    "crash_configuration": 2,
    "mixed_multimechanism": 2,
}
RULE_AXIS_MIN_SOFT = {
    "speed_energy": 1,
    "road_speed_environment": 1,
    "vulnerability_speed": 1,
    "vehicle_geometry_interaction": 1,
    "lighting_visibility": 0,
    "surface_weather_friction": 0,
    "safety_assistance": 0,
    "crash_configuration": 0,
}

# Supplementary triplet rule extension. These are governance-scenario extensions
# and must not modify the primary rule universe or evidence tiers.
TRIPLET_MIN_TRAIN_SUPPORT = max(RULE_MIN_SUPPORT * 0.50, 0.015)
TRIPLET_MIN_TRAIN_HIT_N = 40
TRIPLET_MIN_TRAIN_LIFT = 1.05
TRIPLET_MAX_FINAL = 20
TRIPLET_AXIS_CAP = 3

# Feature-space size sensitivity. The primary analysis remains SELECTED_FEATURE_N.
FEATURE_SIZE_SENSITIVITY_N = [15, 20, 25, 30, 35]
SIZE_SENS_RULE_TOPK = 15

# Evidence grading.
CORE_Q_THRESHOLD = 0.05
CORE_BOOTSTRAP_THRESHOLD = 0.95
CORE_RR_THRESHOLD = 1.00
CORE_THRESHOLD_MIN_LIFT = 1.00
CORE_TEST_HIT_MIN = 5
REPLAY_TEST_HIT_MIN = 1
BOOTSTRAP_ITERATIONS = 200
THRESHOLD_DELTAS = (-0.20, -0.10, 0.0, 0.10, 0.20)

# Outputs retained for manuscript and audit. All other intermediate files are intentionally suppressed.
OUTPUTS = {
    "split_audit": "Train_Test_Split_Audit.csv",
    "preprocess_audit": "Preprocess_Plausibility_Audit.csv",
    "postprocess_check": "PostProcess_Matrix_Plausibility_Check.csv",
    "baseline_table": "Train_Test_Baseline_Characteristics.csv",
    "selected_features": "Final_Consensus_Mechanism_Features.csv",
    "screener_rankings": "Feature_Screener_Rankings.csv",
    "consensus_scores": "Feature_Consensus_Scores.csv",
    "feature_stability": "Feature_Selection_Stability.csv",
    "feature_method_comparison": "Feature_Selection_Baseline_Comparison.csv",
    "family_audit": "Feature_Family_Coverage_Audit.csv",
    "rule_manifest": "04_Rule_Threshold_Manifest.csv",
    "rule_universe": "Final_Rules_Universe_B_MICE_Final_Mechanism_Machine.csv",
    "rules": "Final_Rules_Mechanism_Evidence.csv",
    "blind_replay": "Final_Blind_Test_Validation_Report.csv",
    "rule_bootstrap": "06C_BootstrapRuleStability.csv",
    "threshold_sensitivity": "06D_ThresholdSensitivity.csv",
    "mechanism_axis_summary": "Rule_Mechanism_Axis_Summary.csv",
    "governance_scene_map": "Rule_Governance_Scene_Map.csv",
    "main_text_rule_table": "Main_Text_Rule_Table.csv",
    "rule_universe_mechanism_audit": "Rule_Universe_Mechanism_Audit.csv",
    "governance_diversity_summary": "Governance_Diversity_Summary.csv",
    "main_vs_governance_interpretation_check": "Main_vs_Governance_Interpretation_Check.csv",
    "triplet_extension_report": "Triplet_Rule_Extension_Report.csv",
    "triplet_item_audit": "Triplet_Rule_Item_Replay_Audit.csv",
    "triplet_bootstrap": "Triplet_BootstrapRuleStability.csv",
    "triplet_threshold_sensitivity": "Triplet_ThresholdSensitivity.csv",
    "triplet_evidence_tiers": "Triplet_Rule_Evidence_Tiers.csv",
    "main_text_triplet_rule_table": "Main_Text_Triplet_Rule_Table.csv",
    "triplet_governance_scene_map": "Triplet_Governance_Scene_Map.csv",
    "triplet_interpretation_check": "Triplet_Interpretation_Check.csv",
    "feature_size_sensitivity": "Feature_Space_Size_Sensitivity.csv",
    "feature_size_selected_features_long": "Feature_Space_Size_Selected_Features_Long.csv",
    "feature_size_rule_complexity": "Feature_Space_Size_Rule_Complexity.csv",
    "evidence_tiers": "06F_Rule_Evidence_Tiers.csv",
    "model_fixed": "09A_FixedTest_Model_Comparison.csv",
    "model_repeated": "09B_RepeatedGroupedHoldout_Summary.csv",
    "logistic_coef": "09C_Selected_Logistic_Coefficients.csv",
    "sentinel_unknown_audit": SENTINEL_UNKNOWN_AUDIT,
    "sentinel_flags_train": SENTINEL_FLAGS_TRAIN,
    "sentinel_flags_test": SENTINEL_FLAGS_TEST,
}
