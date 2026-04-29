# -*- coding: utf-8 -*-
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable


KEEP_FINAL_OUTPUTS_IN_ROOT = False
FINAL_ZIP_NAME = "new_test.zip"
MANUSCRIPT_PACKAGE_DIR = "manuscript_package"
CLEANUP_AUDIT_FILE = "File_Cleanup_Audit.csv"

GENERATED_OUTPUT_DIR_PATTERNS = [
    "__pycache__",
    ".pycache*",
    ".py_compile_cache*",
    "result",
]

FINAL_PACKAGE_FILES = [
    "Train_Test_Split_Audit.csv",
    "Preprocess_Plausibility_Audit.csv",
    "Derived_Feature_Audit.csv",
    "PostProcess_Matrix_Plausibility_Check.csv",
    "Matrix_Column_Alignment_Audit.csv",
    "Matrix_Test_Zero_Collapse_Audit.csv",
    "Sentinel_Unknown_Audit.csv",
    "Sentinel_Usage_Audit.csv",
    "Engineering_Outlier_Audit.csv",
    "Feature_Semantic_Redundancy_Audit.csv",
    "Categorical_Missing_Level_Audit.csv",
    "Sentinel_Flags_Train.csv",
    "Sentinel_Flags_Test.csv",
    "Train_Test_Baseline_Characteristics.csv",
    "Feature_Screener_Rankings.csv",
    "Feature_Consensus_Scores.csv",
    "Final_Consensus_Mechanism_Features.csv",
    "Feature_Family_Coverage_Audit.csv",
    "04_Rule_Threshold_Manifest.csv",
    "Final_Rules_Universe_B_MICE_Final_Mechanism_Machine.csv",
    "Rule_Semantic_Redundancy_Audit.csv",
    "Final_Rules_Mechanism_Evidence.csv",
    "Final_Blind_Test_Validation_Report.csv",
    "Rule_Item_Replay_Audit.csv",
    "06C_BootstrapRuleStability.csv",
    "06D_ThresholdSensitivity.csv",
    "06F_Rule_Evidence_Tiers.csv",
    "Rule_Mechanism_Axis_Summary.csv",
    "Rule_Governance_Scene_Map.csv",
    "Main_Text_Rule_Table.csv",
    "Rule_Universe_Mechanism_Audit.csv",
    "Governance_Diversity_Summary.csv",
    "Main_vs_Governance_Interpretation_Check.csv",
    "09A_FixedTest_Model_Comparison.csv",
    "09B_RepeatedGroupedHoldout_Summary.csv",
    "09C_Selected_Logistic_Coefficients.csv",
    "Leakage_Control_Audit.csv",
    "Test_Sanctity_Verification_Summary.csv",
    "Final_Verification_Summary.csv",
    "Triplet_Rule_Extension_Report.csv",
    "Triplet_Semantic_Redundancy_Audit.csv",
    "Triplet_Rule_Item_Replay_Audit.csv",
    "Triplet_BootstrapRuleStability.csv",
    "Triplet_ThresholdSensitivity.csv",
    "Triplet_Rule_Evidence_Tiers.csv",
    "Main_Text_Triplet_Rule_Table.csv",
    "Triplet_Governance_Scene_Map.csv",
    "Triplet_Interpretation_Check.csv",
    "Feature_Space_Size_Sensitivity.csv",
    "Feature_Space_Size_Selected_Features_Long.csv",
    "Feature_Space_Size_Rule_Complexity.csv",
    "02_Run_Manifest.json",
    "03_Run_Manifest.json",
    "04_Run_Manifest.json",
    "05_Run_Manifest.json",
    "06_Blind_Replay_Manifest.json",
    "06_Run_Manifest.json",
    "07_Run_Manifest.json",
    "09_Run_Manifest.json",
    "10_Triplet_Extension_Manifest.json",
    "11_Feature_Size_Sensitivity_Manifest.json",
]

TRIPLET_MANUSCRIPT_FILES = [
    "Triplet_BootstrapRuleStability.csv",
    "Triplet_ThresholdSensitivity.csv",
    "Triplet_Rule_Evidence_Tiers.csv",
    "Main_Text_Triplet_Rule_Table.csv",
    "Triplet_Governance_Scene_Map.csv",
    "Triplet_Interpretation_Check.csv",
]

OBSOLETE_OUTPUT_FILES = [
    "Triplet_Rule_Item_Audit.csv",
    "Final_Blind_Test_Unresolved_Rules.csv",
    "result.zip",
    "new_test_old.zip",
    "Cleaned_Data_Base.csv",
    "Ready_Matrix_Train.csv",
    "Ready_Matrix_Test.csv",
    "Feature_Selection_Stability.csv",
    "Feature_Selection_Baseline_Comparison.csv",
    "01_Base_Duplicate_Summary.csv",
    "01_Cohort_Flow_Audit.csv",
    "01_Feature_Missingness_Audit.csv",
    "01_Key_Feature_NonNull_Audit.csv",
    "01_Merge_Audit.csv",
    "01_Sheet_Detection_Audit.csv",
    "01_Counterparty_Mapping_Audit.csv",
    CLEANUP_AUDIT_FILE,
]

CLEAN_OUTPUT_PATTERNS = [
    "result*.zip",
    "new_test_old.zip",
    "new_test_*.zip",
    "*_Manifest.json",
    "01_*_Audit.csv",
    "Feature_Selection_*.csv",
    "Ready_Matrix_*.csv",
]

SOURCE_KEEP_PATTERNS = [
    "*.py",
    "*.md",
    "*.docx",
    "*.pdf",
    "*.xlsx",
    "README*",
]

FINAL_PACKAGE_FILE_SET = set(FINAL_PACKAGE_FILES)
OBSOLETE_OUTPUT_FILE_SET = set(OBSOLETE_OUTPUT_FILES)


def _name(path: str | Path) -> str:
    return Path(path).name


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def is_source_or_input_file(path: str | Path) -> bool:
    p = Path(path)
    name = p.name
    if name in {FINAL_ZIP_NAME, CLEANUP_AUDIT_FILE}:
        return False
    return matches_any(name, SOURCE_KEEP_PATTERNS)


def is_generated_output_file(path: str | Path) -> bool:
    p = Path(path)
    name = p.name
    if is_source_or_input_file(p):
        return False
    if name in FINAL_PACKAGE_FILE_SET or name in OBSOLETE_OUTPUT_FILE_SET:
        return True
    if name == FINAL_ZIP_NAME:
        return True
    if name.endswith(".zip") and name != FINAL_ZIP_NAME:
        return True
    return matches_any(name, CLEAN_OUTPUT_PATTERNS)


def cleanup_reason(path: str | Path) -> str:
    name = _name(path)
    if name in OBSOLETE_OUTPUT_FILE_SET:
        return "obsolete_output"
    if name in FINAL_PACKAGE_FILE_SET:
        return "previous_final_package_file"
    if name.endswith(".zip") and name != FINAL_ZIP_NAME:
        return "old_zip"
    if name == FINAL_ZIP_NAME:
        return "previous_final_zip"
    if name.endswith("_Manifest.json"):
        return "stale_manifest"
    if matches_any(name, CLEAN_OUTPUT_PATTERNS):
        return "stale_generated_output"
    return "unknown"


def is_generated_output_dir(path: str | Path) -> bool:
    return Path(path).is_dir() and matches_any(Path(path).name, GENERATED_OUTPUT_DIR_PATTERNS)
