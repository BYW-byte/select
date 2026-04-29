# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _config import *
from _utils import infer_feature_semantic_group, infer_feature_source_group, is_raw_age_column, missing_category_type, read_csv_smart, require_columns, sanitize_column_name, write_csv, write_json
from file_output_registry import (
    CLEANUP_AUDIT_FILE,
    FINAL_PACKAGE_FILES,
    FINAL_PACKAGE_FILE_SET,
    FINAL_ZIP_NAME,
    KEEP_FINAL_OUTPUTS_IN_ROOT,
    MANUSCRIPT_PACKAGE_DIR,
    OBSOLETE_OUTPUT_FILE_SET,
    TRIPLET_MANUSCRIPT_FILES,
    cleanup_reason,
    is_generated_output_dir,
    is_generated_output_file,
    is_source_or_input_file,
)


AUDIT_COLUMNS = [
    "Stage",
    "Script",
    "Reads_Train",
    "Reads_Test",
    "Test_Used_For_Fit",
    "Test_Used_For_Transform_Fit",
    "Test_Used_For_Feature_Selection",
    "Test_Used_For_Threshold_Derivation",
    "Test_Used_For_Rule_Mining",
    "Test_Used_For_Rule_Filtering",
    "Test_Used_For_Model_Training",
    "Test_Used_For_Hyperparameter_Selection",
    "Test_Used_For_Selection",
    "Test_Used_For_Evaluation_Only",
    "Status",
    "Evidence_File",
]


def load_json(path: str) -> Dict:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def triplet_extension_status() -> str:
    m10 = load_json("10_Triplet_Extension_Manifest.json")
    if m10.get("triplet_manuscript_ready") is True:
        return "PASS_manuscript_ready_triplet_train_only_generation_test_only_replay"
    return "PASS_supplementary_triplet_train_only_generation_test_only_replay"


def audit_rows() -> pd.DataFrame:
    rows = [
        {
            "Stage": "02_preprocess_bounded",
            "Script": "02_preprocess_bounded.py",
            "Reads_Train": True,
            "Reads_Test": True,
            "Test_Used_For_Fit": False,
            "Test_Used_For_Transform_Fit": False,
            "Test_Used_For_Feature_Selection": False,
            "Test_Used_For_Threshold_Derivation": False,
            "Test_Used_For_Rule_Mining": False,
            "Test_Used_For_Rule_Filtering": False,
            "Test_Used_For_Model_Training": False,
            "Test_Used_For_Hyperparameter_Selection": False,
            "Test_Used_For_Selection": False,
            "Test_Used_For_Evaluation_Only": False,
            "Status": "PASS_transform_only_after_train_fit",
            "Evidence_File": "02_Run_Manifest.json",
        },
        {
            "Stage": "03_check_preprocess_outputs",
            "Script": "03_check_preprocess_outputs.py",
            "Reads_Train": True,
            "Reads_Test": True,
            "Test_Used_For_Fit": False,
            "Test_Used_For_Transform_Fit": False,
            "Test_Used_For_Feature_Selection": False,
            "Test_Used_For_Threshold_Derivation": False,
            "Test_Used_For_Rule_Mining": False,
            "Test_Used_For_Rule_Filtering": False,
            "Test_Used_For_Model_Training": False,
            "Test_Used_For_Hyperparameter_Selection": False,
            "Test_Used_For_Selection": False,
            "Test_Used_For_Evaluation_Only": False,
            "Status": "PASS_audit_only_hard_stop",
            "Evidence_File": "03_Run_Manifest.json",
        },
        {
            "Stage": "04_feature_selection_consensus",
            "Script": "04_feature_selection_consensus.py",
            "Reads_Train": True,
            "Reads_Test": False,
            "Test_Used_For_Fit": False,
            "Test_Used_For_Transform_Fit": False,
            "Test_Used_For_Feature_Selection": False,
            "Test_Used_For_Threshold_Derivation": False,
            "Test_Used_For_Rule_Mining": False,
            "Test_Used_For_Rule_Filtering": False,
            "Test_Used_For_Model_Training": False,
            "Test_Used_For_Hyperparameter_Selection": False,
            "Test_Used_For_Selection": False,
            "Test_Used_For_Evaluation_Only": False,
            "Status": "PASS_train_only_selection",
            "Evidence_File": "04_Run_Manifest.json",
        },
        {
            "Stage": "05_rule_mining",
            "Script": "05_rule_mining.py",
            "Reads_Train": True,
            "Reads_Test": False,
            "Test_Used_For_Fit": False,
            "Test_Used_For_Transform_Fit": False,
            "Test_Used_For_Feature_Selection": False,
            "Test_Used_For_Threshold_Derivation": False,
            "Test_Used_For_Rule_Mining": False,
            "Test_Used_For_Rule_Filtering": False,
            "Test_Used_For_Model_Training": False,
            "Test_Used_For_Hyperparameter_Selection": False,
            "Test_Used_For_Selection": False,
            "Test_Used_For_Evaluation_Only": False,
            "Status": "PASS_train_only_rule_mining_and_filtering",
            "Evidence_File": "05_Run_Manifest.json",
        },
        {
            "Stage": "06_blind_replay",
            "Script": "06_blind_replay.py",
            "Reads_Train": True,
            "Reads_Test": True,
            "Test_Used_For_Fit": False,
            "Test_Used_For_Transform_Fit": False,
            "Test_Used_For_Feature_Selection": False,
            "Test_Used_For_Threshold_Derivation": False,
            "Test_Used_For_Rule_Mining": False,
            "Test_Used_For_Rule_Filtering": False,
            "Test_Used_For_Model_Training": False,
            "Test_Used_For_Hyperparameter_Selection": False,
            "Test_Used_For_Selection": False,
            "Test_Used_For_Evaluation_Only": True,
            "Status": "PASS_evaluation_only_blind_replay",
            "Evidence_File": "06_Blind_Replay_Manifest.json",
        },
        {
            "Stage": "07_evidence_grading",
            "Script": "07_evidence_grading.py",
            "Reads_Train": True,
            "Reads_Test": True,
            "Test_Used_For_Fit": False,
            "Test_Used_For_Transform_Fit": False,
            "Test_Used_For_Feature_Selection": False,
            "Test_Used_For_Threshold_Derivation": False,
            "Test_Used_For_Rule_Mining": False,
            "Test_Used_For_Rule_Filtering": False,
            "Test_Used_For_Model_Training": False,
            "Test_Used_For_Hyperparameter_Selection": False,
            "Test_Used_For_Selection": False,
            "Test_Used_For_Evaluation_Only": True,
            "Status": "PASS_train_bootstrap_and_fixed_threshold_test_evaluation",
            "Evidence_File": "07_Run_Manifest.json",
        },
        {
            "Stage": "08_model_baseline",
            "Script": "08_model_baseline.py",
            "Reads_Train": True,
            "Reads_Test": True,
            "Test_Used_For_Fit": False,
            "Test_Used_For_Transform_Fit": False,
            "Test_Used_For_Feature_Selection": False,
            "Test_Used_For_Threshold_Derivation": False,
            "Test_Used_For_Rule_Mining": False,
            "Test_Used_For_Rule_Filtering": False,
            "Test_Used_For_Model_Training": False,
            "Test_Used_For_Hyperparameter_Selection": False,
            "Test_Used_For_Selection": False,
            "Test_Used_For_Evaluation_Only": True,
            "Status": "PASS_fixed_test_eval_and_train_only_internal_resampling",
            "Evidence_File": "09_Run_Manifest.json",
        },
        {
            "Stage": "10_triplet_rule_extension",
            "Script": "10_triplet_rule_extension.py",
            "Reads_Train": True,
            "Reads_Test": True,
            "Test_Used_For_Fit": False,
            "Test_Used_For_Transform_Fit": False,
            "Test_Used_For_Feature_Selection": False,
            "Test_Used_For_Threshold_Derivation": False,
            "Test_Used_For_Rule_Mining": False,
            "Test_Used_For_Rule_Filtering": False,
            "Test_Used_For_Model_Training": False,
            "Test_Used_For_Hyperparameter_Selection": False,
            "Test_Used_For_Selection": False,
            "Test_Used_For_Evaluation_Only": True,
            "Status": triplet_extension_status(),
            "Evidence_File": "10_Triplet_Extension_Manifest.json",
        },
        {
            "Stage": "11_feature_space_size_sensitivity",
            "Script": "11_feature_space_size_sensitivity.py",
            "Reads_Train": True,
            "Reads_Test": True,
            "Test_Used_For_Fit": False,
            "Test_Used_For_Transform_Fit": False,
            "Test_Used_For_Feature_Selection": False,
            "Test_Used_For_Threshold_Derivation": False,
            "Test_Used_For_Rule_Mining": False,
            "Test_Used_For_Rule_Filtering": False,
            "Test_Used_For_Model_Training": False,
            "Test_Used_For_Hyperparameter_Selection": False,
            "Test_Used_For_Selection": False,
            "Test_Used_For_Evaluation_Only": True,
            "Status": "PASS_train_only_size_sensitivity_test_evaluation_only",
            "Evidence_File": "11_Feature_Size_Sensitivity_Manifest.json",
        },
    ]
    out = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    bad = out[(out["Test_Used_For_Fit"] == True) | (out["Test_Used_For_Selection"] == True)]
    if not bad.empty:
        raise RuntimeError("Leakage-control hard stop: test used for fit/selection in " + "||".join(bad["Stage"].astype(str)))
    return out


def status_row(check: str, ok: bool, evidence: str, required: str) -> Dict[str, str]:
    return {
        "Check": check,
        "Status": "PASS" if ok else "FAIL",
        "Evidence": evidence,
        "Evidence_File": evidence,
        "Key_Value": "",
        "Required_Condition": required,
    }


def set_key_value(row: Dict[str, str], value: object) -> Dict[str, str]:
    row["Key_Value"] = str(value)
    return row


def is_age_related_feature(feature: str) -> bool:
    text = str(feature)
    upper = text.upper()
    compact = __import__("re").sub(r"[^A-Z0-9]+", "_", upper)
    tokens = set(x for x in compact.split("_") if x)
    if "VEHUSAGE" in compact or "USAGE" in tokens:
        return False
    return bool(
        "AGE" in tokens
        or "AGE_YEARS" in compact
        or "FEATURE_AGE" in compact
        or "YEARS_OLD" in compact
        or "ALTER1" in tokens
        or "ALTERG" in tokens
        or "年龄" in text
        or "老年" in text
        or "青壮年" in text
        or "未成年" in text
    )


def duplicate_semantic_rule_count(rules: pd.DataFrame, manifest: pd.DataFrame) -> int:
    if rules.empty or manifest.empty or "Semantic_Group" not in manifest.columns:
        return -1
    group_map = dict(zip(manifest["item"].astype(str), manifest["Semantic_Group"].astype(str)))
    duplicate_n = 0
    for _, rule in rules.iterrows():
        groups = [
            group_map.get(str(item), "")
            for item in str(rule.get("Antecedent_Items", "")).split("||")
            if str(item)
        ]
        groups = [g for g in groups if g]
        if len(groups) != len(set(groups)):
            duplicate_n += 1
    return duplicate_n


def gewges_manifest_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty or "source_feature" not in manifest.columns:
        return pd.DataFrame()
    return manifest[
        manifest["source_feature"].astype(str).str.contains(
            "GEWGES|碰撞时的总重|纰版挒鏃剁殑鎬婚噸",
            regex=True,
            na=False,
        )
    ].copy()


def gewges_manifest_violations(manifest: pd.DataFrame) -> pd.DataFrame:
    gewges_rows = gewges_manifest_rows(manifest)
    if gewges_rows.empty:
        return gewges_rows
    required_cols = {"family", "Semantic_Group", "Mechanism_Axis"}
    if not required_cols.issubset(set(gewges_rows.columns)):
        out = gewges_rows.copy()
        out["GEWGES_Manifest_Violation"] = "missing_required_manifest_columns"
        return out
    bad = gewges_rows[
        ~(
            gewges_rows["family"].astype(str).eq("vehicle_mass")
            & gewges_rows["Semantic_Group"].astype(str).eq("vehicle_mass")
            & ~gewges_rows["Mechanism_Axis"].astype(str).isin(
                ["crash_configuration", "vehicle_geometry_interaction"]
            )
        )
    ].copy()
    if not bad.empty:
        reasons = []
        for _, r in bad.iterrows():
            row_reasons = []
            if str(r.get("family", "")) != "vehicle_mass":
                row_reasons.append("family!=vehicle_mass")
            if str(r.get("Semantic_Group", "")) != "vehicle_mass":
                row_reasons.append("Semantic_Group!=vehicle_mass")
            if str(r.get("Mechanism_Axis", "")) in ["crash_configuration", "vehicle_geometry_interaction"]:
                row_reasons.append("Mechanism_Axis_forbidden")
            reasons.append("|".join(row_reasons))
        bad["GEWGES_Manifest_Violation"] = reasons
    return bad


def generate_sentinel_usage_audit() -> pd.DataFrame:
    sent = read_csv_smart(OUTPUTS["sentinel_unknown_audit"]) if Path(OUTPUTS["sentinel_unknown_audit"]).exists() else pd.DataFrame()
    selected = read_csv_smart(OUTPUTS["selected_features"]) if Path(OUTPUTS["selected_features"]).exists() else pd.DataFrame()
    manifest = read_csv_smart(OUTPUTS["rule_manifest"]) if Path(OUTPUTS["rule_manifest"]).exists() else pd.DataFrame()
    rules = read_csv_smart(OUTPUTS["rules"]) if Path(OUTPUTS["rules"]).exists() else pd.DataFrame()
    tiers = read_csv_smart(OUTPUTS["evidence_tiers"]) if Path(OUTPUTS["evidence_tiers"]).exists() else pd.DataFrame()
    if sent.empty:
        out = pd.DataFrame(columns=[
            "Feature", "Family", "Semantic_Group", "Source_Group", "Train_Sentinel_N", "Test_Sentinel_N",
            "Used_In_Selected25", "Used_In_Rule_Manifest", "Used_In_Final_Rules",
            "Used_In_Core_Or_Binary_Rules", "Used_In_Physical_Core_Or_Binary_Rules",
        ])
        write_csv(out, "Sentinel_Usage_Audit.csv")
        return out
    selected_features = set(selected.get("Selected_Features", selected.get("Feature", pd.Series(dtype=str))).astype(str)) if not selected.empty else set()
    manifest_sources = set(manifest.get("source_feature", pd.Series(dtype=str)).astype(str)) if not manifest.empty else set()
    item_to_source = dict(zip(manifest.get("item", pd.Series(dtype=str)).astype(str), manifest.get("source_feature", pd.Series(dtype=str)).astype(str))) if not manifest.empty else {}
    final_items = set()
    for value in rules.get("Antecedent_Items", pd.Series(dtype=str)).astype(str):
        final_items.update([x for x in value.split("||") if x])
    final_sources = {item_to_source.get(i, i) for i in final_items}
    core_items = set()
    physical_core_items = set()
    if not tiers.empty:
        core_like = tiers[tiers.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).isin(["core-confirmed", "binary-stable confirmed"])]
        physical_core_like = core_like[pd.to_numeric(core_like.get("Physical_Mechanism_Valid_Flag", pd.Series(dtype=float)), errors="coerce").fillna(1).astype(int).eq(1)]
        for value in core_like.get("Antecedent_Items", pd.Series(dtype=str)).astype(str):
            core_items.update([x for x in value.split("||") if x])
        for value in physical_core_like.get("Antecedent_Items", pd.Series(dtype=str)).astype(str):
            physical_core_items.update([x for x in value.split("||") if x])
    core_sources = {item_to_source.get(i, i) for i in core_items}
    physical_core_sources = {item_to_source.get(i, i) for i in physical_core_items}
    rows = []
    for feature, g in sent.groupby("Feature", dropna=False):
        feature = str(feature)
        train_n = int(pd.to_numeric(g.loc[g.get("Split", pd.Series(dtype=str)).astype(str).eq("Train"), "Raw_Sentinel_N"], errors="coerce").fillna(0).sum())
        test_n = int(pd.to_numeric(g.loc[g.get("Split", pd.Series(dtype=str)).astype(str).eq("Test"), "Raw_Sentinel_N"], errors="coerce").fillna(0).sum())
        fam = str(g.get("Family", pd.Series([""])).dropna().astype(str).iloc[0]) if "Family" in g.columns and not g["Family"].dropna().empty else ""
        rows.append({
            "Feature": feature,
            "Family": fam,
            "Semantic_Group": infer_feature_semantic_group(feature),
            "Source_Group": infer_feature_source_group(feature),
            "Train_Sentinel_N": train_n,
            "Test_Sentinel_N": test_n,
            "Used_In_Selected25": int(feature in selected_features),
            "Used_In_Rule_Manifest": int(feature in manifest_sources),
            "Used_In_Final_Rules": int(feature in final_sources),
            "Used_In_Core_Or_Binary_Rules": int(feature in core_sources),
            "Used_In_Physical_Core_Or_Binary_Rules": int(feature in physical_core_sources),
        })
    audited = {str(r["Feature"]) for r in rows}
    for feature in sorted((selected_features | manifest_sources | final_sources | core_sources | physical_core_sources) - audited):
        if feature != "FEATURE_Age_Over60":
            continue
        rows.append({
            "Feature": feature,
            "Family": infer_family(feature) if "infer_family" in globals() else "age",
            "Semantic_Group": infer_feature_semantic_group(feature),
            "Source_Group": infer_feature_source_group(feature),
            "Train_Sentinel_N": 0,
            "Test_Sentinel_N": 0,
            "Used_In_Selected25": int(feature in selected_features),
            "Used_In_Rule_Manifest": int(feature in manifest_sources),
            "Used_In_Final_Rules": int(feature in final_sources),
            "Used_In_Core_Or_Binary_Rules": int(feature in core_sources),
            "Used_In_Physical_Core_Or_Binary_Rules": int(feature in physical_core_sources),
        })
    out = pd.DataFrame(rows)
    write_csv(out, "Sentinel_Usage_Audit.csv")
    return out


def generate_engineering_outlier_audit() -> pd.DataFrame:
    prep = read_csv_smart(OUTPUTS["preprocess_audit"]) if Path(OUTPUTS["preprocess_audit"]).exists() else pd.DataFrame()
    selected = read_csv_smart(OUTPUTS["selected_features"]) if Path(OUTPUTS["selected_features"]).exists() else pd.DataFrame()
    rules = read_csv_smart(OUTPUTS["rules"]) if Path(OUTPUTS["rules"]).exists() else pd.DataFrame()
    manifest = read_csv_smart(OUTPUTS["rule_manifest"]) if Path(OUTPUTS["rule_manifest"]).exists() else pd.DataFrame()
    selected_features = set(selected.get("Selected_Features", selected.get("Feature", pd.Series(dtype=str))).astype(str)) if not selected.empty else set()
    item_to_source = dict(zip(manifest.get("item", pd.Series(dtype=str)).astype(str), manifest.get("source_feature", pd.Series(dtype=str)).astype(str))) if not manifest.empty else {}
    rule_manifest_sources = set(manifest.get("source_feature", pd.Series(dtype=str)).astype(str)) if not manifest.empty else set()
    final_items = set()
    for value in rules.get("Antecedent_Items", pd.Series(dtype=str)).astype(str):
        final_items.update([x for x in value.split("||") if x])
    final_sources = {item_to_source.get(i, i) for i in final_items}
    item_audit = read_csv_smart("Rule_Item_Replay_Audit.csv") if Path("Rule_Item_Replay_Audit.csv").exists() else pd.DataFrame()
    core_sources = set()
    if not item_audit.empty and {"source_feature", "Used_In_Physical_Core_Or_Binary_Rules"}.issubset(item_audit.columns):
        core_sources = set(item_audit.loc[pd.to_numeric(item_audit["Used_In_Physical_Core_Or_Binary_Rules"], errors="coerce").fillna(0).astype(int).eq(1), "source_feature"].astype(str))
    rows = []
    if not prep.empty and {"Feature", "Lower_Bound", "Upper_Bound", "Split"}.issubset(prep.columns):
        for _, r in prep.dropna(subset=["Lower_Bound", "Upper_Bound"], how="all").iterrows():
            lo = pd.to_numeric(pd.Series([r.get("Lower_Bound")]), errors="coerce").iloc[0]
            hi = pd.to_numeric(pd.Series([r.get("Upper_Bound")]), errors="coerce").iloc[0]
            min_before = pd.to_numeric(pd.Series([r.get("Min_Before_Clip")]), errors="coerce").iloc[0]
            max_before = pd.to_numeric(pd.Series([r.get("Max_Before_Clip")]), errors="coerce").iloc[0]
            feature = str(r.get("Feature", ""))
            sanitized = sanitize_column_name(feature)
            below = int(r.get("Below_Bound_Before", 0) or 0)
            above = int(r.get("Above_Bound_Before", 0) or 0)
            extreme_below = int(pd.notna(lo) and lo > 0 and pd.notna(min_before) and min_before < lo * 0.5)
            extreme_above = int(pd.notna(hi) and pd.notna(max_before) and max_before > hi * 2)
            rows.append({
                "Feature": feature,
                "Sanitized_Feature": sanitized,
                "Split": r.get("Split", ""),
                "Lower_Bound": lo,
                "Upper_Bound": hi,
                "Below_Bound_N": below,
                "Above_Bound_N": above,
                "Extreme_Below_N": extreme_below,
                "Extreme_Above_N": extreme_above,
                "Min_Before_Clip": min_before,
                "Max_Before_Clip": max_before,
                "Used_In_Selected25": int(sanitized in selected_features or feature in selected_features),
                "Used_In_Rule_Manifest": int(sanitized in rule_manifest_sources or feature in rule_manifest_sources),
                "Used_In_Final_Rules": int(sanitized in final_sources or feature in final_sources),
                "Used_In_Core_Or_Binary_Rules": int(sanitized in core_sources or feature in core_sources),
                "Warning": "extreme_value_in_final_rule_source" if ((sanitized in final_sources or feature in final_sources) and (extreme_below or extreme_above)) else "",
            })
    out = pd.DataFrame(rows, columns=[
        "Feature", "Sanitized_Feature", "Split", "Lower_Bound", "Upper_Bound", "Below_Bound_N", "Above_Bound_N",
        "Extreme_Below_N", "Extreme_Above_N", "Min_Before_Clip", "Max_Before_Clip",
        "Used_In_Selected25", "Used_In_Rule_Manifest", "Used_In_Final_Rules", "Used_In_Core_Or_Binary_Rules", "Warning",
    ])
    write_csv(out, "Engineering_Outlier_Audit.csv")
    return out


def contains_age_alias_self_combination(table: pd.DataFrame) -> bool:
    if table.empty or "Antecedent_Items" not in table.columns:
        return True
    for value in table["Antecedent_Items"].astype(str):
        has_age_group = "FEATURE_Age_Group_老年_60" in value
        has_alter_over60 = "年龄年数记录_ALTER1___GT_60_0" in value or ("ALTER1" in value and "GT_60" in value)
        if has_age_group and has_alter_over60:
            return True
    return False


def verification_rows() -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    split = read_csv_smart(OUTPUTS["split_audit"]) if Path(OUTPUTS["split_audit"]).exists() else pd.DataFrame()
    overlap = int(split["Overlap_Accidents"].iloc[0]) if not split.empty and "Overlap_Accidents" in split.columns else -1
    rows.append(set_key_value(status_row("Accident overlap is zero.", overlap == 0, OUTPUTS["split_audit"], "Overlap_Accidents == 0"), overlap))

    m02 = load_json("02_Run_Manifest.json")
    rows.append(status_row("Test not used for imputation fit.", m02.get("test_used_for_imputation_fit") is False, "02_Run_Manifest.json", "test_used_for_imputation_fit == false"))
    rows.append(status_row("Test not used for categorical level fit.", m02.get("test_used_for_category_level_fit") is False, "02_Run_Manifest.json", "test_used_for_category_level_fit == false"))
    rows.append(status_row("02_Run_Manifest confirms sentinel cleaning before imputation", m02.get("sentinel_cleaning_before_imputation") is True, "02_Run_Manifest.json", "sentinel_cleaning_before_imputation == true"))
    rows.append(status_row("02_Run_Manifest confirms sentinel cleaning before physical clipping", m02.get("sentinel_cleaning_before_physical_clipping") is True, "02_Run_Manifest.json", "sentinel_cleaning_before_physical_clipping == true"))

    m04 = load_json("04_Run_Manifest.json")
    rows.append(status_row("Test not used for feature selection.", m04.get("test_used_for_feature_selection") is False, "04_Run_Manifest.json", "test_used_for_feature_selection == false"))
    rows.append(status_row("Test not used for GWO candidate generation.", m04.get("test_used_for_gwo_candidate_generation") is False, "04_Run_Manifest.json", "test_used_for_gwo_candidate_generation == false"))
    selected = read_csv_smart(OUTPUTS["selected_features"]) if Path(OUTPUTS["selected_features"]).exists() else pd.DataFrame()
    rows.append(status_row("Final_Consensus_Mechanism_Features has Missing_Category_Type.", (not selected.empty) and "Missing_Category_Type" in selected.columns, OUTPUTS["selected_features"], "Missing_Category_Type column exists"))
    rows.append(status_row("Final_Consensus_Mechanism_Features has Semantic_Group and Source_Group.", (not selected.empty) and {"Semantic_Group", "Source_Group"}.issubset(selected.columns), OUTPUTS["selected_features"], "Semantic_Group and Source_Group columns exist"))
    if Path(TRAIN_MATRIX).exists() and Path(TEST_MATRIX).exists():
        train_cols_for_age = set(read_csv_smart(TRAIN_MATRIX).columns)
        test_cols_for_age = set(read_csv_smart(TEST_MATRIX).columns)
    else:
        train_cols_for_age = test_cols_for_age = set()
    rows.append(status_row("FEATURE_Age_Over60 exists in Ready_Matrix_Train/Test.", "FEATURE_Age_Over60" in train_cols_for_age and "FEATURE_Age_Over60" in test_cols_for_age, "Ready_Matrix_Train.csv;Ready_Matrix_Test.csv", "FEATURE_Age_Over60 column exists in both matrices"))
    rows.append(status_row("Final_Consensus_Mechanism_Features contains FEATURE_Age_Over60.", (not selected.empty) and selected.get("Selected_Features", pd.Series(dtype=str)).astype(str).eq("FEATURE_Age_Over60").any(), OUTPUTS["selected_features"], "FEATURE_Age_Over60 selected"))
    selected_features_for_age = selected.get("Selected_Features", selected.get("Feature", pd.Series(dtype=str))).astype(str).tolist() if not selected.empty else []
    selected_raw_age = [f for f in selected_features_for_age if is_raw_age_column(f) or infer_feature_semantic_group(f) == "raw_age_continuous"]
    rows.append(set_key_value(status_row("Final_Consensus_Mechanism_Features does not contain raw age columns.", not selected_raw_age, OUTPUTS["selected_features"], "no raw age continuous feature selected"), selected_raw_age or "ok"))
    selected_pure = selected[selected.get("Missing_Category_Type", pd.Series(dtype=str)).astype(str).eq("pure_missing_unknown")] if not selected.empty else pd.DataFrame()
    rows.append(set_key_value(status_row("Final selected 25 contains no pure_missing_unknown features.", selected_pure.empty and len(selected) == SELECTED_FEATURE_N, OUTPUTS["selected_features"], "25 rows and no Missing_Category_Type == pure_missing_unknown"), {"rows": len(selected), "pure_missing": len(selected_pure)}))
    age_dup = int((selected.get("Semantic_Group", pd.Series(dtype=str)).astype(str) == "age_over60").sum()) if not selected.empty else -1
    rows.append(set_key_value(status_row("No age_over60 duplicate in selected 25.", age_dup <= 1, OUTPUTS["selected_features"], "age_over60 selected count <= 1"), age_dup))
    selected_age_over60_exists = "FEATURE_Age_Over60" in selected_features_for_age
    selected_alterg_age = [
        f for f in selected_features_for_age
        if infer_feature_source_group(f) == "ALTERG"
        or infer_feature_semantic_group(f) in {"age_group", "age_elderly_group"}
        or "ALTERG" in str(f).upper()
    ]
    rows.append(set_key_value(
        status_row(
            "Selected 25 contains no ALTERG age-group aliases when FEATURE_Age_Over60 exists.",
            (not selected_age_over60_exists) or not selected_alterg_age,
            OUTPUTS["selected_features"],
            "FEATURE_Age_Over60 selected implies no ALTERG/年龄段 age-group selected",
        ),
        selected_alterg_age or "ok",
    ))
    source_counts = selected.get("Source_Group", pd.Series(dtype=str)).astype(str).value_counts() if not selected.empty and "Source_Group" in selected.columns else pd.Series(dtype=int)
    duplicate_sources = source_counts[source_counts > 1].to_dict()
    relaxed_manifested = bool(m04.get("relaxed_source_group_cap_used") is True and m04.get("relaxed_source_group_cap_features"))
    rows.append(set_key_value(status_row("No duplicate one-hot Source_Group in selected 25 unless manifest-documented.", (not duplicate_sources) or relaxed_manifested, OUTPUTS["selected_features"] + ";04_Run_Manifest.json", "duplicate Source_Group absent or relaxed_source_group_cap_used documented"), duplicate_sources or "ok"))
    crash_count = int((selected.get("Semantic_Group", pd.Series(dtype=str)).astype(str) == "crash_contact_geometry").sum()) if not selected.empty else -1
    rows.append(set_key_value(status_row("crash_contact_geometry selected count <= 2.", crash_count <= 2, OUTPUTS["selected_features"], "crash_contact_geometry count <= 2"), crash_count))
    rows.append(status_row("Feature_Semantic_Redundancy_Audit.csv exists.", Path("Feature_Semantic_Redundancy_Audit.csv").exists(), "Feature_Semantic_Redundancy_Audit.csv", "feature semantic redundancy audit generated"))
    rows.append(status_row("Feature semantic deduplication is train-only.", m04.get("test_used_for_semantic_deduplication") is False and m04.get("feature_semantic_deduplication_enabled") is True, "04_Run_Manifest.json", "test_used_for_semantic_deduplication == false"))

    m05 = load_json("05_Run_Manifest.json")
    rows.append(status_row("Test not used for threshold derivation.", m05.get("test_used_for_threshold_derivation") is False, "05_Run_Manifest.json", "test_used_for_threshold_derivation == false"))
    rows.append(status_row("Test not used for FPGrowth rule mining.", m05.get("test_used_for_rule_mining") is False, "05_Run_Manifest.json", "test_used_for_rule_mining == false"))
    rows.append(status_row("Test not used for final rule filtering.", m05.get("test_used_for_rule_filtering") is False, "05_Run_Manifest.json", "test_used_for_rule_filtering == false"))
    rows.append(status_row("Rule manifest includes Semantic_Group and Source_Group.", Path(OUTPUTS["rule_manifest"]).exists() and {"Semantic_Group", "Source_Group"}.issubset(read_csv_smart(OUTPUTS["rule_manifest"]).columns), OUTPUTS["rule_manifest"], "Semantic_Group and Source_Group columns exist"))
    rule_manifest_for_age = read_csv_smart(OUTPUTS["rule_manifest"]) if Path(OUTPUTS["rule_manifest"]).exists() else pd.DataFrame()
    raw_age_manifest = rule_manifest_for_age[
        rule_manifest_for_age.get("source_feature", pd.Series(dtype=str)).astype(str).map(is_raw_age_column)
        | rule_manifest_for_age.get("item", pd.Series(dtype=str)).astype(str).str.contains("ALTER1___GT_60_0|年龄年数记录_ALTER1", regex=True, na=False)
    ] if not rule_manifest_for_age.empty else pd.DataFrame()
    rows.append(set_key_value(status_row("04_Rule_Threshold_Manifest has no raw age primary threshold item.", raw_age_manifest.empty, OUTPUTS["rule_manifest"], "no ALTER1 age threshold item"), len(raw_age_manifest)))

    m07 = load_json("07_Run_Manifest.json")
    rows.append(status_row("Test not used for bootstrap stability.", m07.get("bootstrap_source") == "train_only_accident_level_group_bootstrap_with_replacement", "07_Run_Manifest.json", "bootstrap_source == train_only_accident_level_group_bootstrap_with_replacement"))
    rows.append(status_row("Evidence grading does not modify primary artifacts.", m07.get("modifies_rules") is False and m07.get("modifies_manifest") is False and m07.get("modifies_features") is False and m07.get("modifies_selected_features") is False, "07_Run_Manifest.json", "modifies_rules/manifest/features/selected_features == false"))
    rows.append(status_row("Governance diversity outputs are reporting-only.", m07.get("governance_diversity_reporting_only") is True, "07_Run_Manifest.json", "governance_diversity_reporting_only == true"))
    rows.append(status_row("Governance diversity outputs do not modify final rules.", m07.get("governance_diversity_does_not_modify_rules") is True and m07.get("modifies_rules") is False, "07_Run_Manifest.json", "governance_diversity_does_not_modify_rules == true and modifies_rules == false"))
    rows.append(status_row("Governance diversity outputs do not modify evidence tiers.", m07.get("governance_diversity_does_not_modify_evidence_tiers") is True, "07_Run_Manifest.json", "governance_diversity_does_not_modify_evidence_tiers == true"))
    rows.append(status_row("Governance diversity outputs do not use test set for rule selection.", m07.get("governance_diversity_does_not_use_test_for_selection") is True, "07_Run_Manifest.json", "governance_diversity_does_not_use_test_for_selection == true"))
    rows.append(status_row("No rule-mining threshold was changed for diversity reporting.", m07.get("primary_result_not_rerun_for_diversity") is True and m07.get("no_threshold_or_parameter_tuning_for_diversity") is True, "07_Run_Manifest.json", "primary_result_not_rerun_for_diversity == true and no_threshold_or_parameter_tuning_for_diversity == true"))
    rows.append(status_row("No evidence-tier threshold was changed for diversity reporting.", m07.get("governance_diversity_does_not_modify_evidence_tiers") is True and m07.get("no_threshold_or_parameter_tuning_for_diversity") is True, "07_Run_Manifest.json", "governance_diversity_does_not_modify_evidence_tiers == true and no_threshold_or_parameter_tuning_for_diversity == true"))

    m09 = load_json("09_Run_Manifest.json")
    rows.append(status_row("Test not used for repeated grouped holdout training.", m09.get("test_used_for_repeated_holdout") is False, "09_Run_Manifest.json", "test_used_for_repeated_holdout == false"))

    m10 = load_json("10_Triplet_Extension_Manifest.json")
    rows.append(status_row("Triplet rules generated train-only.", m10.get("rule_generation_source") == "train_only" and m10.get("test_used_for_triplet_generation") is False, "10_Triplet_Extension_Manifest.json", "rule_generation_source == train_only and test_used_for_triplet_generation == false"))
    rows.append(status_row("Triplet rules not used to modify primary rules.", m10.get("modifies_primary_rules") is False and m10.get("modifies_manifest") is False and m10.get("modifies_selected_features") is False, "10_Triplet_Extension_Manifest.json", "modifies_primary_rules/manifest/selected_features == false"))
    rows.append(status_row("Manuscript-ready triplets use sentinel-aware replay and audits.", (m10.get("triplet_manuscript_ready") is not True) or (
        m10.get("sentinel_aware_triplet_replay") is True
        and m10.get("triplet_bootstrap_stability_done") is True
        and m10.get("triplet_threshold_sensitivity_done") is True
        and m10.get("triplet_physical_validity_audit_done") is True
    ), "10_Triplet_Extension_Manifest.json", "ready triplets require sentinel replay/bootstrap/sensitivity/physical audit"))
    rows.append(status_row("Triplet manuscript-ready path keeps test evaluation-only.", (m10.get("triplet_manuscript_ready") is not True) or (
        m10.get("test_used_for_triplet_generation") is False
        and m10.get("test_used_for_triplet_filtering") is False
        and m10.get("test_used_for_threshold_derivation") is False
        and m10.get("test_used_for_triplet_selection") is False
        and m10.get("test_used_for_evaluation_only") is True
    ), "10_Triplet_Extension_Manifest.json", "ready triplets require all test-used-for-selection flags false and evaluation-only true"))
    triplet_tiers = read_csv_smart("Triplet_Rule_Evidence_Tiers.csv") if Path("Triplet_Rule_Evidence_Tiers.csv").exists() else pd.DataFrame()
    triplet_main = read_csv_smart("Main_Text_Triplet_Rule_Table.csv") if Path("Main_Text_Triplet_Rule_Table.csv").exists() else pd.DataFrame()
    triplet_tiers_has_threshold = "Threshold_Stable_Flag" in triplet_tiers.columns
    rows.append(status_row(
        "Triplet_Rule_Evidence_Tiers.csv contains Threshold_Stable_Flag.",
        triplet_tiers_has_threshold,
        "Triplet_Rule_Evidence_Tiers.csv",
        "Threshold_Stable_Flag column exists",
    ))
    main_triplet_threshold_ok = (
        Path("Main_Text_Triplet_Rule_Table.csv").exists()
        and (triplet_main.empty or (
            "Threshold_Stable_Flag" in triplet_main.columns
            and pd.to_numeric(triplet_main["Threshold_Stable_Flag"], errors="coerce").fillna(0).astype(int).eq(1).all()
        ))
    )
    rows.append(set_key_value(status_row(
        "Every main-text triplet has Threshold_Stable_Flag == 1.",
        main_triplet_threshold_ok,
        "Main_Text_Triplet_Rule_Table.csv",
        "all main-text triplet Threshold_Stable_Flag values equal 1",
    ), int(len(triplet_main)) if Path("Main_Text_Triplet_Rule_Table.csv").exists() else "missing"))
    main_triplet_tier_ok = (
        Path("Main_Text_Triplet_Rule_Table.csv").exists()
        and (triplet_main.empty or (
            "Evidence_Tier" in triplet_main.columns
            and triplet_main["Evidence_Tier"].astype(str).isin(["core-confirmed", "binary-stable confirmed"]).all()
        ))
    )
    rows.append(set_key_value(status_row(
        "Every main-text triplet has a core-confirmed or binary-stable confirmed Evidence_Tier.",
        main_triplet_tier_ok,
        "Main_Text_Triplet_Rule_Table.csv",
        "Evidence_Tier in core-confirmed/binary-stable confirmed",
    ), triplet_main.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if Path("Main_Text_Triplet_Rule_Table.csv").exists() else "missing"))
    main_triplet_physical_ok = (
        Path("Main_Text_Triplet_Rule_Table.csv").exists()
        and (triplet_main.empty or (
            "Physical_Mechanism_Valid_Flag" in triplet_main.columns
            and pd.to_numeric(triplet_main["Physical_Mechanism_Valid_Flag"], errors="coerce").fillna(0).astype(int).eq(1).all()
        ))
    )
    rows.append(set_key_value(status_row(
        "Every main-text triplet has Physical_Mechanism_Valid_Flag == 1.",
        main_triplet_physical_ok,
        "Main_Text_Triplet_Rule_Table.csv",
        "all main-text triplet Physical_Mechanism_Valid_Flag values equal 1",
    ), int(len(triplet_main)) if Path("Main_Text_Triplet_Rule_Table.csv").exists() else "missing"))
    rows.append(status_row(
        "Triplet core-confirmed tier requires threshold stability.",
        m10.get("triplet_core_confirmed_requires_threshold_stable") is True,
        "10_Triplet_Extension_Manifest.json",
        "triplet_core_confirmed_requires_threshold_stable == true",
    ))
    rows.append(status_row(
        "Triplet main-text eligibility requires threshold stability.",
        m10.get("triplet_main_text_requires_threshold_stable") is True,
        "10_Triplet_Extension_Manifest.json",
        "triplet_main_text_requires_threshold_stable == true",
    ))

    m11 = load_json("11_Feature_Size_Sensitivity_Manifest.json")
    rows.append(status_row("Feature size sensitivity does not modify primary selected_n.", m11.get("modifies_primary_selected_features") is False and m11.get("primary_selected_n") == SELECTED_FEATURE_N, "11_Feature_Size_Sensitivity_Manifest.json", "modifies_primary_selected_features == false and primary_selected_n == SELECTED_FEATURE_N"))
    rows.append(status_row("Test not used for feature-space size selection.", m11.get("test_used_for_size_selection") is False and m11.get("test_usage") == "evaluation_only", "11_Feature_Size_Sensitivity_Manifest.json", "test_used_for_size_selection == false and test_usage == evaluation_only"))
    rows.append(status_row("Feature size sensitivity is evaluation-only and does not modify primary rules.", m11.get("sensitivity_role") == "evaluation-only sensitivity; does not participate in selected features or primary rule set" and m11.get("modifies_primary_rules") is False and m11.get("modifies_primary_rule_set") is False, "11_Feature_Size_Sensitivity_Manifest.json", "evaluation-only sensitivity and modifies_primary_rules/rule_set == false"))

    m06 = load_json("06_Blind_Replay_Manifest.json")
    eval_only = (
        m06.get("test_usage") == "evaluation_only_blind_replay"
        and m09.get("fixed_test_training_source") == "train_only"
        and m09.get("fixed_test_evaluation_source") == "test_only"
    )
    rows.append(status_row("Test used only for fixed evaluation and blind replay.", eval_only, "06_Blind_Replay_Manifest.json;09_Run_Manifest.json", "test appears only in evaluation-only stages"))
    rows.append(status_row("Blind replay uses observed source constraints", m06.get("blind_replay_uses_observed_source_value_constraint") is True, "06_Blind_Replay_Manifest.json", "blind_replay_uses_observed_source_value_constraint == true"))

    if Path(TRAIN_MATRIX).exists() and Path(TEST_MATRIX).exists():
        train = read_csv_smart(TRAIN_MATRIX)
        test = read_csv_smart(TEST_MATRIX)
        aligned = list(train.columns) == list(test.columns)
    else:
        train = test = pd.DataFrame()
        aligned = False
    rows.append(set_key_value(status_row("Ready train/test columns aligned.", aligned, "Ready_Matrix_Train.csv;Ready_Matrix_Test.csv", "identical column order"), aligned))

    sentinel_paths = [OUTPUTS["sentinel_unknown_audit"], OUTPUTS["sentinel_flags_train"], OUTPUTS["sentinel_flags_test"]]
    for p in sentinel_paths:
        rows.append(status_row(f"{p} exists", Path(p).exists(), p, "file exists"))
    sentinel_audit = read_csv_smart(OUTPUTS["sentinel_unknown_audit"]) if Path(OUTPUTS["sentinel_unknown_audit"]).exists() else pd.DataFrame()
    if not sentinel_audit.empty:
        require_columns(sentinel_audit, OUTPUTS["sentinel_unknown_audit"], ["Feature", "Family", "Split", "Raw_Sentinel_N", "Raw_Sentinel_Rate", "Converted_To_NaN_N", "Missing_After_Sentinel_N"])
    if Path(OUTPUTS["sentinel_flags_train"]).exists() and Path(OUTPUTS["sentinel_flags_test"]).exists() and not train.empty and not test.empty:
        sf_train = read_csv_smart(OUTPUTS["sentinel_flags_train"])
        sf_test = read_csv_smart(OUTPUTS["sentinel_flags_test"])
        flags_shape_ok = len(sf_train) == len(train) and len(sf_test) == len(test)
        ready_flag_like_cols = [
            c for c in list(train.columns) + list(test.columns)
            if "sentinel" in str(c).lower() or "unknown_audit" in str(c).lower()
        ]
        flags_audit_only = flags_shape_ok and not ready_flag_like_cols
    else:
        flags_audit_only = False
    rows.append(status_row("Sentinel flags are audit-only and not included in Ready_Matrix_Train/Test feature columns", flags_audit_only, OUTPUTS["sentinel_flags_train"] + ";" + OUTPUTS["sentinel_flags_test"], "sentinel flag files exist separately and no flag columns are appended to ready matrices"))

    zero = read_csv_smart("Matrix_Test_Zero_Collapse_Audit.csv") if Path("Matrix_Test_Zero_Collapse_Audit.csv").exists() else pd.DataFrame()
    zero_ok = (not zero.empty) and (zero["Status"].astype(str).eq("ok").all())
    rows.append(set_key_value(status_row("Zero-collapse audit passed.", zero_ok, "Matrix_Test_Zero_Collapse_Audit.csv", "all Status == ok"), zero["Status"].value_counts().to_dict() if not zero.empty and "Status" in zero.columns else "missing"))

    phys = read_csv_smart(OUTPUTS["postprocess_check"]) if Path(OUTPUTS["postprocess_check"]).exists() else pd.DataFrame()
    phys_ok = (not phys.empty) and int(phys.get("Violation_Count", pd.Series(dtype=float)).fillna(0).sum()) == 0
    phys_ok = phys_ok and not phys.get("Status", pd.Series(dtype=str)).astype(str).isin(["physical_positive_lower_bound_all_zero", "suspicious_distribution_collapse"]).any()
    rows.append(set_key_value(status_row("Physical plausibility audit passed.", phys_ok, OUTPUTS["postprocess_check"], "no physical violations or hard-stop collapse status"), int(phys.get("Violation_Count", pd.Series(dtype=float)).fillna(0).sum()) if not phys.empty else "missing"))

    vehicle_dim = phys[phys.get("Is_Vehicle_Dimension", pd.Series(dtype=float)).fillna(0).astype(int).eq(1)] if not phys.empty and "Is_Vehicle_Dimension" in phys.columns else pd.DataFrame()
    vehicle_dim_bound_ok = (
        not vehicle_dim.empty
        and vehicle_dim.get("Vehicle_Dimension_Bound_Applied", pd.Series(dtype=float)).fillna(0).astype(int).eq(1).any()
        and int(vehicle_dim.get("Violation_Count", pd.Series(dtype=float)).fillna(0).sum()) == 0
        and vehicle_dim["Status"].astype(str).eq("physical_bound_check").any()
    )
    rows.append(set_key_value(status_row("Vehicle dimension bounds applied.", vehicle_dim_bound_ok, OUTPUTS["postprocess_check"], "vehicle dimension rows have applied bounds and zero postprocess violations"), vehicle_dim[["Feature", "Status"]].drop_duplicates().head(10).to_dict("records") if not vehicle_dim.empty and "Feature" in vehicle_dim.columns else "missing"))

    family_files = []
    for p in [OUTPUTS["selected_features"], OUTPUTS["family_audit"], OUTPUTS["rule_manifest"], OUTPUTS["triplet_item_audit"]]:
        if Path(p).exists():
            dfp = read_csv_smart(p)
            feature_cols = [c for c in ("Feature", "Selected_Features", "source_feature", "item") if c in dfp.columns]
            if "Family" in dfp.columns:
                fam_col = "Family"
            elif "family" in dfp.columns:
                fam_col = "family"
            else:
                fam_col = None
            if feature_cols and fam_col:
                tmp = dfp[[feature_cols[0], fam_col]].copy()
                tmp.columns = ["Feature", "Family"]
                tmp["Evidence_File"] = p
                family_files.append(tmp)
    fam_all = pd.concat(family_files, ignore_index=True) if family_files else pd.DataFrame(columns=["Feature", "Family", "Evidence_File"])
    vehusage_age = fam_all[fam_all["Feature"].astype(str).str.contains("VEHUSAGE|车辆用途|USAGE", case=False, regex=True, na=False) & fam_all["Family"].astype(str).eq("age")]
    rows.append(set_key_value(status_row("VEHUSAGE not assigned to age.", vehusage_age.empty, "Final_Consensus_Mechanism_Features.csv;Feature_Family_Coverage_Audit.csv;04_Rule_Threshold_Manifest.csv;Triplet_Rule_Item_Replay_Audit.csv", "no VEHUSAGE/USAGE feature has Family == age"), len(vehusage_age)))
    age_rows = fam_all[fam_all["Family"].astype(str).eq("age")].copy()
    bad_age = age_rows[~age_rows["Feature"].map(is_age_related_feature)] if not age_rows.empty else pd.DataFrame()
    rows.append(set_key_value(status_row("Age family contains only age-related features.", bad_age.empty, "Final_Consensus_Mechanism_Features.csv;Feature_Family_Coverage_Audit.csv;04_Rule_Threshold_Manifest.csv;Triplet_Rule_Item_Replay_Audit.csv", "all age-family features match explicit age tokens/labels"), bad_age["Feature"].drop_duplicates().head(10).tolist() if not bad_age.empty else "ok"))

    manifest = read_csv_smart(OUTPUTS["rule_manifest"]) if Path(OUTPUTS["rule_manifest"]).exists() else pd.DataFrame()
    if not manifest.empty:
        require_columns(manifest, OUTPUTS["rule_manifest"], ["requires_observed_source_value", "sentinel_flag_available", "missing_or_sentinel_values_can_trigger", "Missing_Category_Type", "Semantic_Group", "Source_Group", "source_feature", "family", "Mechanism_Axis"])
    manifest_pure = manifest[manifest.get("Missing_Category_Type", pd.Series(dtype=str)).astype(str).eq("pure_missing_unknown")] if not manifest.empty else pd.DataFrame()
    rows.append(set_key_value(status_row("04_Rule_Threshold_Manifest contains no pure_missing_unknown items.", manifest_pure.empty and not manifest.empty, OUTPUTS["rule_manifest"], "no Missing_Category_Type == pure_missing_unknown"), len(manifest_pure)))
    rows.append(status_row("04_Rule_Threshold_Manifest has requires_observed_source_value", (not manifest.empty) and "requires_observed_source_value" in manifest.columns, OUTPUTS["rule_manifest"], "requires_observed_source_value column exists"))
    threshold_manifest = manifest[manifest.get("transform_type", pd.Series(dtype=str)).astype(str).eq("threshold")] if not manifest.empty else pd.DataFrame()
    threshold_requires = (not threshold_manifest.empty) and pd.to_numeric(threshold_manifest["requires_observed_source_value"], errors="coerce").fillna(0).eq(1).all()
    threshold_cannot = (not threshold_manifest.empty) and pd.to_numeric(threshold_manifest["missing_or_sentinel_values_can_trigger"], errors="coerce").fillna(1).eq(0).all()
    threshold_flags_available = (not threshold_manifest.empty) and pd.to_numeric(threshold_manifest["sentinel_flag_available"], errors="coerce").fillna(0).eq(1).all()
    rows.append(set_key_value(status_row("All threshold items require observed source value", threshold_requires, OUTPUTS["rule_manifest"], "all threshold requires_observed_source_value == 1"), int(pd.to_numeric(threshold_manifest.get("requires_observed_source_value", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1).sum()) if not threshold_manifest.empty else 0))
    rows.append(status_row("Threshold items cannot be triggered by missing or sentinel values", threshold_cannot, OUTPUTS["rule_manifest"], "all threshold missing_or_sentinel_values_can_trigger == 0"))
    rows.append(set_key_value(status_row("All threshold items have sentinel flags available", threshold_flags_available, OUTPUTS["rule_manifest"] + ";05_Run_Manifest.json", "all threshold sentinel_flag_available == 1"), int((~pd.to_numeric(threshold_manifest.get("sentinel_flag_available", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1)).sum()) if not threshold_manifest.empty else "missing"))
    gewges_manifest_bad = gewges_manifest_violations(manifest)
    gewges_manifest_total = len(gewges_manifest_rows(manifest))
    rows.append(set_key_value(
        status_row(
            "GEWGES manifest rows use vehicle_mass semantics and are not crash/geometry axes.",
            (not manifest.empty) and gewges_manifest_total > 0 and gewges_manifest_bad.empty,
            OUTPUTS["rule_manifest"],
            "GEWGES/碰撞时的总重 source_feature rows have family == vehicle_mass, Semantic_Group == vehicle_mass, and Mechanism_Axis not in {crash_configuration, vehicle_geometry_interaction}",
        ),
        gewges_manifest_bad[["item", "source_feature", "family", "Semantic_Group", "Mechanism_Axis", "GEWGES_Manifest_Violation"]].to_dict("records") if not gewges_manifest_bad.empty else {"GEWGES_manifest_rows": gewges_manifest_total},
    ))
    if not train.empty and not test.empty and not manifest.empty and "source_feature" in manifest.columns:
        src = set(manifest["source_feature"].astype(str))
        exists_ok = src.issubset(set(map(str, train.columns))) and src.issubset(set(map(str, test.columns)))
    else:
        exists_ok = False
    rows.append(set_key_value(status_row("Rule manifest source features exist in train and test.", exists_ok, OUTPUTS["rule_manifest"], "all source_feature values exist in both matrices"), exists_ok))

    item_audit = read_csv_smart("Rule_Item_Replay_Audit.csv") if Path("Rule_Item_Replay_Audit.csv").exists() else pd.DataFrame()
    if not item_audit.empty:
        require_columns(item_audit, "Rule_Item_Replay_Audit.csv", ["test_item_hit_sentinel_rate", "known_value_test_hit_n", "unknown_dominated_flag", "Missing_Category_Type", "Semantic_Group", "Source_Group", "Physical_Mechanism_Eligible"])
    rows.append(status_row("Rule_Item_Replay_Audit has sentinel hit-rate columns", (not item_audit.empty) and {"test_item_hit_sentinel_rate", "known_value_test_hit_n", "unknown_dominated_flag"}.issubset(set(item_audit.columns)), "Rule_Item_Replay_Audit.csv", "sentinel replay audit columns exist"))
    rows.append(status_row("Rule_Item_Replay_Audit has sentinel and missing-category fields.", (not item_audit.empty) and {"test_item_hit_sentinel_rate", "Missing_Category_Type", "Semantic_Group", "Source_Group", "Physical_Mechanism_Eligible"}.issubset(set(item_audit.columns)), "Rule_Item_Replay_Audit.csv", "sentinel and missing-category replay audit columns exist"))
    if not item_audit.empty and "status" in item_audit.columns:
        allowed_status = {"ok", "matrix_or_distribution_warning", "missing_source_feature", "item_not_in_manifest"}
        item_ok = set(item_audit["status"].astype(str).unique()).issubset(allowed_status)
        item_value = item_audit["status"].value_counts().to_dict()
    else:
        item_ok = False
        item_value = "missing"
    rows.append(set_key_value(status_row("Rule item replay audit all ok or warnings explicitly flagged.", item_ok, "Rule_Item_Replay_Audit.csv", "status values are ok or explicit audit warnings"), item_value))

    rules = read_csv_smart(OUTPUTS["rules"]) if Path(OUTPUTS["rules"]).exists() else pd.DataFrame()
    rule_items = set()
    for value in rules.get("Antecedent_Items", pd.Series(dtype=str)).astype(str):
        rule_items.update([x for x in value.split("||") if x])
    pure_items = set(manifest_pure.get("item", pd.Series(dtype=str)).astype(str)) if not manifest_pure.empty else set()
    rows.append(set_key_value(status_row("Final_Rules_Mechanism_Evidence contains no pure_missing_unknown items.", not (rule_items & pure_items), OUTPUTS["rules"] + ";" + OUTPUTS["rule_manifest"], "no final rule antecedent item has Missing_Category_Type == pure_missing_unknown"), sorted(rule_items & pure_items)))
    raw_age_rule_items = [i for i in rule_items if "ALTER1___GT_60_0" in i or "年龄年数记录_ALTER1" in i]
    rows.append(set_key_value(status_row("Final_Rules_Mechanism_Evidence contains no raw age item.", not raw_age_rule_items, OUTPUTS["rules"], "no raw age antecedent item"), raw_age_rule_items or "ok"))
    tiers = read_csv_smart(OUTPUTS["evidence_tiers"]) if Path(OUTPUTS["evidence_tiers"]).exists() else pd.DataFrame()
    gewges_bad_axis = pd.DataFrame()
    if not tiers.empty and {"Antecedent_Items", "Mechanism_Axis"}.issubset(tiers.columns):
        gewges_mask = tiers["Antecedent_Items"].astype(str).str.contains("GEWGES|碰撞时的总重", regex=True, na=False)
        gewges_bad_axis = tiers[gewges_mask & tiers["Mechanism_Axis"].astype(str).isin(["crash_configuration", "vehicle_geometry_interaction"])]
    rows.append(set_key_value(
        status_row(
            "GEWGES rules are not labeled crash_configuration or vehicle_geometry_interaction.",
            gewges_bad_axis.empty,
            OUTPUTS["evidence_tiers"],
            "GEWGES/碰撞时的总重 Mechanism_Axis != crash_configuration",
        ),
        gewges_bad_axis.get("Rule_ID", pd.Series(dtype=str)).astype(str).tolist(),
    ))
    age_sex_body = pd.DataFrame()
    if not tiers.empty and {"Antecedent_Items", "Mechanism_Axis"}.issubset(tiers.columns):
        age_sex_mask = tiers["Antecedent_Items"].astype(str).str.contains("FEATURE_Age_Over60", na=False) & tiers["Antecedent_Items"].astype(str).str.contains("GESCHL|SEX|性别", regex=True, na=False)
        age_sex_body = tiers[age_sex_mask & tiers["Mechanism_Axis"].astype(str).eq("vulnerability_body")]
    rows.append(set_key_value(status_row("Age + sex rules are not labeled vulnerability_body.", age_sex_body.empty, OUTPUTS["evidence_tiers"], "age+sex Mechanism_Axis != vulnerability_body"), age_sex_body.get("Rule_ID", pd.Series(dtype=str)).astype(str).tolist()))
    physical_cols = [
        "Rule_Sentinel_Dominated_Flag", "Max_Item_Test_Hit_Sentinel_Rate",
        "Rule_Missing_Category_Flag", "Rule_Pure_Missing_Unknown_Flag",
        "Rule_Partial_Unknown_Detail_Flag", "Rule_Semantic_Duplicate_Flag",
        "Physical_Mechanism_Valid_Flag", "Physical_Evidence_Tier",
    ]
    if not tiers.empty:
        require_columns(tiers, OUTPUTS["evidence_tiers"], physical_cols)
    rows.append(status_row("06F_Rule_Evidence_Tiers has Physical_Evidence_Tier", (not tiers.empty) and "Physical_Evidence_Tier" in tiers.columns, OUTPUTS["evidence_tiers"], "Physical_Evidence_Tier column exists"))
    if not rules.empty and not tiers.empty and "Rule_ID" in rules.columns and "Rule_ID" in tiers.columns:
        tier_ok = set(tiers["Rule_ID"].astype(str)) == set(rules["Rule_ID"].astype(str))
    else:
        tier_ok = False
    tier_ok = tier_ok and m07.get("modifies_rules") is False
    rows.append(set_key_value(status_row("Evidence tier does not modify rules.", tier_ok, OUTPUTS["evidence_tiers"] + ";07_Run_Manifest.json", "evidence Rule_ID set equals final rules and modifies_rules == false"), tier_ok))

    primary_dup_n = duplicate_semantic_rule_count(rules, manifest)
    rows.append(set_key_value(status_row("No primary rule contains duplicate Semantic_Group", primary_dup_n == 0, OUTPUTS["rules"] + ";" + OUTPUTS["rule_manifest"], "duplicate Semantic_Group count == 0"), primary_dup_n))

    triplets = read_csv_smart(OUTPUTS["triplet_extension_report"]) if Path(OUTPUTS["triplet_extension_report"]).exists() else pd.DataFrame()
    triplet_dup_n = duplicate_semantic_rule_count(triplets, manifest)
    rows.append(set_key_value(status_row("No triplet rule contains duplicate Semantic_Group", triplet_dup_n == 0, OUTPUTS["triplet_extension_report"] + ";" + OUTPUTS["rule_manifest"], "duplicate Semantic_Group count == 0"), triplet_dup_n))

    main_text = read_csv_smart(OUTPUTS["main_text_rule_table"]) if Path(OUTPUTS["main_text_rule_table"]).exists() else pd.DataFrame()
    if not main_text.empty:
        require_columns(main_text, OUTPUTS["main_text_rule_table"], physical_cols)
    rows.append(status_row("Main_Text_Rule_Table has Physical_Evidence_Tier", (not main_text.empty) and "Physical_Evidence_Tier" in main_text.columns, OUTPUTS["main_text_rule_table"], "Physical_Evidence_Tier column exists"))
    main_text_blob = " ".join(main_text.astype(str).fillna("").agg(" ".join, axis=1).tolist()) if not main_text.empty else ""
    rows.append(status_row("Main_Text_Rule_Table has semantic display labels.", (not main_text.empty) and {"Antecedent_Display_CN", "Antecedent_Display_EN"}.issubset(main_text.columns), OUTPUTS["main_text_rule_table"], "display label columns exist"))
    rows.append(status_row("Main_Text_Rule_Table does not display raw age column names.", "年龄年数记录_ALTER1" not in main_text_blob and "年龄年数记录_ALTER1___GT_60_0" not in main_text_blob, OUTPUTS["main_text_rule_table"], "raw age names absent from main table"))
    display_blob = ""
    if not main_text.empty:
        display_cols = [c for c in ["Antecedent_Display_CN", "Antecedent_Display_EN"] if c in main_text.columns]
        if display_cols:
            display_blob = " ".join(main_text[display_cols].astype(str).fillna("").agg(" ".join, axis=1).tolist())
    raw_gewges_display = (
        "VEH_碰撞时的总重_GEWGES___GT_1497_5" in display_blob
        or "GEWGES___GT_" in display_blob
    )
    rows.append(status_row(
        "Main_Text_Rule_Table does not display raw GEWGES item names.",
        not raw_gewges_display,
        OUTPUTS["main_text_rule_table"],
        "GEWGES rules use vehicle total weight display labels",
    ))
    age_display_ok = True
    age_rows_display = main_text[main_text.get("Antecedent_Items", pd.Series(dtype=str)).astype(str).str.contains("FEATURE_Age_Over60", na=False)] if not main_text.empty else pd.DataFrame()
    if not age_rows_display.empty:
        age_display_ok = age_rows_display.get("Antecedent_Display_CN", pd.Series(dtype=str)).astype(str).str.contains("年龄 > 60岁", regex=False, na=False).all() and age_rows_display.get("Antecedent_Display_EN", pd.Series(dtype=str)).astype(str).str.contains("Age > 60 years", regex=False, na=False).all()
    rows.append(status_row("Age rules use semantic display labels.", age_display_ok, OUTPUTS["main_text_rule_table"], "年龄 > 60岁 / Age > 60 years"))
    main_items = set()
    for value in main_text.get("Antecedent_Items", pd.Series(dtype=str)).astype(str):
        main_items.update([x for x in value.split("||") if x])
    rows.append(set_key_value(status_row("Main_Text_Rule_Table contains no pure_missing_unknown items.", not (main_items & pure_items), OUTPUTS["main_text_rule_table"] + ";" + OUTPUTS["rule_manifest"], "no main-text antecedent item has Missing_Category_Type == pure_missing_unknown"), sorted(main_items & pure_items)))
    governance = read_csv_smart(OUTPUTS["governance_diversity_summary"]) if Path(OUTPUTS["governance_diversity_summary"]).exists() else pd.DataFrame()
    bad_gov = governance[
        governance.get("Evidence_Strength_Label", pd.Series(dtype=str)).astype(str).str.contains("priority governance target", case=False, na=False)
        & governance.get("Interpretation_Boundary", pd.Series(dtype=str)).astype(str).str.contains("missing-category|semantic-redundancy|data-availability|redundancy", case=False, regex=True, na=False)
    ] if not governance.empty else pd.DataFrame()
    rows.append(set_key_value(status_row("Governance_Diversity_Summary does not label missing-category-only axes as priority governance target.", bad_gov.empty, OUTPUTS["governance_diversity_summary"], "data-availability/redundancy axes are not priority governance target"), len(bad_gov)))
    if not tiers.empty and set(["Rule_Sentinel_Dominated_Flag", "Evidence_Tier", "Physical_Evidence_Tier"]).issubset(tiers.columns):
        dominated_core = tiers[
            pd.to_numeric(tiers["Rule_Sentinel_Dominated_Flag"], errors="coerce").fillna(0).astype(int).eq(1)
            & tiers["Evidence_Tier"].astype(str).isin(["core-confirmed", "binary-stable confirmed"])
            & tiers["Physical_Evidence_Tier"].astype(str).isin(["core-confirmed", "binary-stable confirmed"])
        ]
        no_dominated_physical_core = dominated_core.empty
    else:
        no_dominated_physical_core = False
    rows.append(set_key_value(status_row("No sentinel-dominated rule is interpreted as physical core-confirmed", no_dominated_physical_core, OUTPUTS["evidence_tiers"], "sentinel-dominated confirmed rules have Physical_Evidence_Tier downgraded"), len(dominated_core) if "dominated_core" in locals() else "missing"))
    if not tiers.empty and {"Physical_Mechanism_Valid_Flag", "Evidence_Tier", "Physical_Evidence_Tier"}.issubset(tiers.columns):
        invalid_core = tiers[
            pd.to_numeric(tiers["Physical_Mechanism_Valid_Flag"], errors="coerce").fillna(1).astype(int).eq(0)
            & tiers["Evidence_Tier"].astype(str).isin(["core-confirmed", "binary-stable confirmed"])
            & tiers["Physical_Evidence_Tier"].astype(str).isin(["core-confirmed", "binary-stable confirmed"])
        ]
        no_invalid_physical_core = invalid_core.empty
    else:
        no_invalid_physical_core = False
    rows.append(set_key_value(status_row("No Physical_Mechanism_Valid_Flag == 0 rule is interpreted as physical core-confirmed.", no_invalid_physical_core, OUTPUTS["evidence_tiers"], "invalid physical mechanism rules have downgraded Physical_Evidence_Tier"), len(invalid_core) if "invalid_core" in locals() else "missing"))
    if not tiers.empty and {"Has_Continuous_Threshold", "Test_Hit_N_Min", "Test_Lift_Min", "Threshold_Stable_Flag"}.issubset(tiers.columns):
        has_cont = pd.to_numeric(tiers["Has_Continuous_Threshold"], errors="coerce").fillna(0).astype(int).eq(1)
        expected_threshold_stable = ((~has_cont) | ((pd.to_numeric(tiers["Test_Hit_N_Min"], errors="coerce") >= CORE_TEST_HIT_MIN) & (pd.to_numeric(tiers["Test_Lift_Min"], errors="coerce") >= CORE_THRESHOLD_MIN_LIFT))).astype(int)
        threshold_stable_ok = expected_threshold_stable.reset_index(drop=True).equals(pd.to_numeric(tiers["Threshold_Stable_Flag"], errors="coerce").fillna(-1).astype(int).reset_index(drop=True))
    else:
        threshold_stable_ok = False
    rows.append(status_row("Threshold_Stable_Flag requires both hit stability and lift stability", threshold_stable_ok and m07.get("threshold_stable_flag_requires_hit_and_lift") is True, OUTPUTS["evidence_tiers"] + ";07_Run_Manifest.json", "continuous thresholds require Test_Hit_N_Min and Test_Lift_Min criteria"))
    primary_tiers = {"core-confirmed", "binary-stable confirmed", "replayable"}
    if not tiers.empty and not main_text.empty and "Evidence_Tier" in tiers.columns and "Rule_ID" in tiers.columns and "Rule_ID" in main_text.columns:
        expected_main_ids = set(tiers.loc[tiers["Evidence_Tier"].isin(primary_tiers), "Rule_ID"].astype(str))
        actual_main_ids = set(main_text["Rule_ID"].astype(str))
        missing_main_ids = sorted(expected_main_ids - actual_main_ids)
        excess_main_ids = sorted(actual_main_ids - expected_main_ids)
        main_complete = not missing_main_ids and not excess_main_ids
    else:
        missing_main_ids = ["unable_to_evaluate"]
        excess_main_ids = []
        main_complete = False
    if not main_complete:
        raise RuntimeError(
            "Main_Text_Rule_Table Rule_ID mismatch; "
            f"missing={missing_main_ids}; excess={excess_main_ids}"
        )
    rows.append(set_key_value(status_row("Main text table includes every primary manuscript rule.", main_complete, OUTPUTS["main_text_rule_table"] + ";" + OUTPUTS["evidence_tiers"], "Rule_ID set equals core-confirmed/binary-stable confirmed/replayable evidence tiers"), {"missing": missing_main_ids, "excess": excess_main_ids}))

    fss = read_csv_smart(OUTPUTS["feature_size_sensitivity"]) if Path(OUTPUTS["feature_size_sensitivity"]).exists() else pd.DataFrame()
    fss_cols = set(fss.columns) if not fss.empty else set()
    fss_field_ok = (
        "Core_Confirmed_N_if_evaluated" not in fss_cols
        and "Confirmed_or_BinaryStable_Criterion_N_if_evaluated" in fss_cols
        and "Criterion_Note" in fss_cols
    )
    rows.append(set_key_value(status_row("Feature size sensitivity criterion is clearly labeled.", fss_field_ok, OUTPUTS["feature_size_sensitivity"], "no Core_Confirmed_N_if_evaluated column; criterion count and note present"), sorted(fss_cols)))

    age_alias_present = contains_age_alias_self_combination(main_text)
    rows.append(set_key_value(status_row("Age alias self-combination excluded from Main_Text_Rule_Table", not age_alias_present, OUTPUTS["main_text_rule_table"], "ALTER1>60 and Age_Group=old do not co-occur in one displayed rule"), age_alias_present))

    rule_sem_audit = read_csv_smart("Rule_Semantic_Redundancy_Audit.csv") if Path("Rule_Semantic_Redundancy_Audit.csv").exists() else pd.DataFrame()
    rule_audit_ok = Path("Rule_Semantic_Redundancy_Audit.csv").exists() and {"Rule_ID", "Antecedent_Items", "Reason"}.issubset(set(rule_sem_audit.columns))
    rows.append(set_key_value(status_row("Rule_Semantic_Redundancy_Audit.csv generated", rule_audit_ok, "Rule_Semantic_Redundancy_Audit.csv", "file exists with semantic redundancy audit schema"), len(rule_sem_audit) if Path("Rule_Semantic_Redundancy_Audit.csv").exists() else "missing"))

    triplet_sem_audit = read_csv_smart("Triplet_Semantic_Redundancy_Audit.csv") if Path("Triplet_Semantic_Redundancy_Audit.csv").exists() else pd.DataFrame()
    triplet_audit_ok = Path("Triplet_Semantic_Redundancy_Audit.csv").exists() and {"Antecedent_Items", "Reason"}.issubset(set(triplet_sem_audit.columns))
    rows.append(set_key_value(status_row("Triplet_Semantic_Redundancy_Audit.csv generated", triplet_audit_ok, "Triplet_Semantic_Redundancy_Audit.csv", "file exists with semantic redundancy audit schema"), len(triplet_sem_audit) if Path("Triplet_Semantic_Redundancy_Audit.csv").exists() else "missing"))

    rows.append(set_key_value(status_row("Triplet rules remain an extension layer.", m10.get("modifies_primary_rules") is False and m10.get("modifies_manifest") is False and m10.get("modifies_selected_features") is False, "10_Triplet_Extension_Manifest.json", "triplets do not modify primary rules, manifest, or selected features"), m10.get("triplet_rule_n", "missing")))
    triplet_ready_or_excluded = bool(m10.get("triplet_manuscript_ready") is True or (
        m10.get("triplet_extension_not_used_for_manuscript") is True
        and m10.get("triplet_extension_requires_sentinel_aware_update") in (True, False)
        and m10.get("triplet_extension_excluded_from_final_manuscript_package") is True
    ))
    rows.append(status_row("Triplet extension is manuscript-ready or explicitly excluded from manuscript supplementary.", triplet_ready_or_excluded, "10_Triplet_Extension_Manifest.json", "triplet manuscript-ready OR excluded flags true"))
    rows.append(set_key_value(status_row("Feature size sensitivity does not modify selected_n.", m11.get("modifies_primary_selected_features") is False and m11.get("primary_selected_n") == SELECTED_FEATURE_N, "11_Feature_Size_Sensitivity_Manifest.json", "primary_selected_n == 25 and modifies_primary_selected_features == false"), m11.get("primary_selected_n", "missing")))

    sentinel_usage = read_csv_smart("Sentinel_Usage_Audit.csv") if Path("Sentinel_Usage_Audit.csv").exists() else pd.DataFrame()
    rows.append(status_row("Sentinel_Usage_Audit.csv exists.", not sentinel_usage.empty or Path("Sentinel_Usage_Audit.csv").exists(), "Sentinel_Usage_Audit.csv", "usage audit generated after 04/05/07"))
    rows.append(status_row("Sentinel_Usage_Audit.csv generated after 04/05/07.", Path("Sentinel_Usage_Audit.csv").exists() and Path(OUTPUTS["evidence_tiers"]).exists() and Path("Sentinel_Usage_Audit.csv").stat().st_mtime >= Path(OUTPUTS["evidence_tiers"]).stat().st_mtime, "Sentinel_Usage_Audit.csv;06F_Rule_Evidence_Tiers.csv", "usage audit timestamp is after evidence tiers"))
    rows.append(status_row("Engineering_Outlier_Audit.csv exists.", Path("Engineering_Outlier_Audit.csv").exists(), "Engineering_Outlier_Audit.csv", "engineering outlier audit generated"))

    referenced = set()
    for row in rows:
        for part in str(row.get("Evidence_File", "")).split(";"):
            if part.endswith(".json"):
                referenced.add(part)
    referenced |= {
        "02_Run_Manifest.json", "03_Run_Manifest.json", "04_Run_Manifest.json", "05_Run_Manifest.json",
        "06_Blind_Replay_Manifest.json", "06_Run_Manifest.json", "07_Run_Manifest.json", "09_Run_Manifest.json",
        "10_Triplet_Extension_Manifest.json", "11_Feature_Size_Sensitivity_Manifest.json",
    }
    missing_manifests = sorted([p for p in referenced if not Path(p).exists()])
    rows.append(set_key_value(status_row("All referenced JSON manifests exist.", not missing_manifests, "Leakage_Control_Audit.csv;Test_Sanctity_Verification_Summary.csv", "all referenced JSON manifest files exist"), missing_manifests or "ok"))

    core_manifest_paths = [
        "02_Run_Manifest.json", "04_Run_Manifest.json", "05_Run_Manifest.json",
        "06_Run_Manifest.json", "07_Run_Manifest.json", "09_Run_Manifest.json",
    ]
    core_run_ids = {p: load_json(p).get("run_id") for p in core_manifest_paths if Path(p).exists()}
    run_id_consistent = len(core_run_ids) == len(core_manifest_paths) and len(set(core_run_ids.values())) == 1 and all(core_run_ids.values())
    rows.append(set_key_value(status_row("Core manifest run_id values are consistent.", run_id_consistent, "02/04/05/06/07/09_Run_Manifest.json", "all core manifests have identical run_id"), core_run_ids))

    out = pd.DataFrame(rows)
    failed = out[out["Status"].eq("FAIL")]
    if not failed.empty:
        raise RuntimeError("Test sanctity verification failed: " + "||".join(failed["Check"].astype(str)))
    return out


CLEANUP_AUDIT_ROWS: List[Dict[str, object]] = []


def add_cleanup_audit(file_path: str | Path, action: str, reason: str, status: str = "ok") -> None:
    p = Path(file_path)
    try:
        size = int(p.stat().st_size) if p.exists() and p.is_file() else 0
    except Exception:
        size = 0
    CLEANUP_AUDIT_ROWS.append({
        "File": str(p),
        "Action": action,
        "Reason": reason,
        "Size_Bytes": size,
        "Status": status,
    })


def write_cleanup_audit() -> None:
    cols = ["File", "Action", "Reason", "Size_Bytes", "Status"]
    df = pd.DataFrame(CLEANUP_AUDIT_ROWS)
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    write_csv(df[cols], CLEANUP_AUDIT_FILE)


def final_package_required_files() -> List[str]:
    m10 = load_json("10_Triplet_Extension_Manifest.json")
    excluded = m10.get("triplet_extension_excluded_from_final_manuscript_package") is True
    if excluded and m10.get("triplet_manuscript_ready") is not True:
        return [
            p for p in FINAL_PACKAGE_FILES
            if p not in TRIPLET_MANUSCRIPT_FILES or Path(p).exists()
        ]
    return list(FINAL_PACKAGE_FILES)


def validate_triplet_package_condition() -> List[Dict[str, object]]:
    rows = []
    m10 = load_json("10_Triplet_Extension_Manifest.json")
    ready = m10.get("triplet_manuscript_ready") is True
    excluded = m10.get("triplet_extension_excluded_from_final_manuscript_package") is True

    if ready:
        missing = [p for p in TRIPLET_MANUSCRIPT_FILES if not Path(p).exists()]
        ok = not missing
        rows.append(status_row(
            "Triplet manuscript-ready package files exist.",
            ok,
            "10_Triplet_Extension_Manifest.json;" + ";".join(TRIPLET_MANUSCRIPT_FILES),
            "triplet_manuscript_ready true requires all triplet manuscript files",
        ))
        rows[-1] = set_key_value(rows[-1], missing or "ok")
        if missing:
            raise RuntimeError("Triplet manuscript-ready package files missing: " + "||".join(missing))
        return rows

    main_path = Path("Main_Text_Triplet_Rule_Table.csv")
    if main_path.exists():
        main = read_csv_smart(str(main_path))
        ok = main.empty and len(main.columns) > 0
        rows.append(status_row(
            "Triplet main text table is empty with schema when triplet extension is not manuscript-ready.",
            ok,
            "Main_Text_Triplet_Rule_Table.csv;10_Triplet_Extension_Manifest.json",
            "not manuscript-ready requires empty schema table unless excluded",
        ))
        rows[-1] = set_key_value(rows[-1], {"rows": int(len(main)), "columns": list(main.columns)})
        if not ok:
            raise RuntimeError("Triplet extension is not manuscript-ready, but Main_Text_Triplet_Rule_Table.csv is not an empty schema table.")
    else:
        rows.append(status_row(
            "Triplet main text table absence is explicitly manifest-excluded.",
            excluded,
            "10_Triplet_Extension_Manifest.json",
            "absent Main_Text_Triplet_Rule_Table.csv requires triplet_extension_excluded_from_final_manuscript_package true",
        ))
        if not excluded:
            raise RuntimeError("Main_Text_Triplet_Rule_Table.csv is absent without explicit triplet package exclusion.")
    return rows


def manifest_outputs(manifest_path: str) -> List[str]:
    data = load_json(manifest_path)
    outputs = data.get("outputs", [])
    if isinstance(outputs, str):
        outputs = [outputs]
    if not isinstance(outputs, list):
        return []
    return [Path(str(p)).name for p in outputs if str(p).strip()]


def write_final_package(path: str = FINAL_ZIP_NAME) -> int:
    required_files = final_package_required_files()
    missing = [p for p in required_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Final package missing required files: " + "||".join(missing))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in required_files:
            zf.write(p, arcname=Path(p).name)
            add_cleanup_audit(Path(p).name, "included_in_zip", path)
    return len(required_files)


def verify_final_package(path: str = FINAL_ZIP_NAME) -> List[Dict[str, object]]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
    checks: List[Tuple[str, List[str], str]] = [
        ("Every FINAL_PACKAGE_FILES entry is included in new_test.zip.", final_package_required_files(), "FINAL_PACKAGE_FILES"),
        ("Every 10_Triplet_Extension_Manifest output is included in new_test.zip.", manifest_outputs("10_Triplet_Extension_Manifest.json"), "10_Triplet_Extension_Manifest.json outputs"),
        ("Every 07_Run_Manifest output is included in new_test.zip.", manifest_outputs("07_Run_Manifest.json"), "07_Run_Manifest.json outputs"),
    ]
    rows = []
    missing_all: List[str] = []
    for check, expected, condition in checks:
        expected_names = [Path(p).name for p in expected]
        missing = sorted([p for p in expected_names if p not in names])
        missing_all.extend(missing)
        row = status_row(check, not missing, path, condition)
        rows.append(set_key_value(row, {"missing": missing or "ok", "expected_n": len(expected_names), "zip_n": len(names)}))
    if missing_all:
        raise RuntimeError("Final zip verification failed; missing files: " + "||".join(sorted(set(missing_all))))
    return rows


def package_category(file_name: str) -> str:
    if file_name.startswith(("Train_Test", "Preprocess", "Derived", "PostProcess", "Matrix_", "Sentinel_", "Engineering_", "Categorical_")):
        return "core_cohort_preprocessing"
    if file_name.startswith(("Feature_Screener", "Feature_Consensus", "Final_Consensus", "Feature_Family", "Feature_Semantic")):
        return "feature_selection"
    if file_name.startswith(("04_Rule", "Final_Rules", "Rule_", "06C_", "06D_", "06F_", "Governance_", "Main_Text_Rule", "Main_vs_")) or file_name == "Final_Blind_Test_Validation_Report.csv":
        return "primary_rule_mining_replay"
    if file_name.startswith("09"):
        return "model_baseline"
    if file_name.startswith(("Leakage", "Test_Sanctity", "Final_Verification")):
        return "leakage_verification"
    if file_name.startswith("Triplet") or file_name.startswith("Main_Text_Triplet"):
        return "triplet_extension"
    if file_name.startswith("Feature_Space_Size"):
        return "feature_space_sensitivity"
    if file_name.endswith("_Manifest.json"):
        return "manifest"
    return "other"


def file_modified_time(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def write_manuscript_package_dir(package_dir: str = MANUSCRIPT_PACKAGE_DIR) -> int:
    dst = Path(package_dir)
    if dst.exists():
        resolved = dst.resolve()
        if resolved.parent != Path(".").resolve():
            raise RuntimeError(f"Refusing to remove package directory outside workspace: {resolved}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(FINAL_ZIP_NAME, "r") as zf:
        zip_names = set(zf.namelist())

    rows = []
    copied = 0
    for name in final_package_required_files():
        src = Path(name)
        dest = dst / src.name
        exists = src.exists()
        if exists:
            shutil.copy2(src, dest)
            copied += 1
            add_cleanup_audit(dest, "copied_to_manuscript_package", "final_evidence_file")
        rows.append({
            "File": src.name,
            "Category": package_category(src.name),
            "Required": True,
            "Exists": bool(exists),
            "Size_Bytes": int(src.stat().st_size) if exists else 0,
            "Modified_Time": file_modified_time(src),
            "Included_In_Zip": src.name in zip_names,
            "Notes": "" if exists else "missing",
        })

    index = pd.DataFrame(rows, columns=[
        "File", "Category", "Required", "Exists", "Size_Bytes", "Modified_Time", "Included_In_Zip", "Notes",
    ])
    write_csv(index, str(dst / "00_PACKAGE_INDEX.csv"))
    return copied


def remove_file(path: Path) -> None:
    try:
        path.chmod(0o666)
    except Exception:
        pass
    path.unlink()


def dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except Exception:
                pass
    return total


def remove_dir_if_possible(path: Path) -> bool:
    try:
        shutil.rmtree(path)
        return True
    except PermissionError:
        return False
    except OSError:
        return False


def prune_workspace_after_success() -> int:
    pruned = 0
    for p in sorted(Path(".").iterdir(), key=lambda x: x.name.lower()):
        if p.is_dir():
            if p.name == MANUSCRIPT_PACKAGE_DIR:
                continue
            if is_generated_output_dir(p):
                resolved = p.resolve()
                if resolved.parent != Path(".").resolve():
                    continue
                size = dir_size_bytes(p)
                if remove_dir_if_possible(p):
                    status = "deleted"
                    action = "deleted_stale_generated"
                    pruned += 1
                else:
                    status = "skipped_permission_denied"
                    action = "skipped_unknown_user_file"
                CLEANUP_AUDIT_ROWS.append({
                    "File": p.name,
                    "Action": action,
                    "Reason": "generated_cache_or_result_directory",
                    "Size_Bytes": size,
                    "Status": status,
                })
                continue
            continue
        name = p.name
        if name in {FINAL_ZIP_NAME, CLEANUP_AUDIT_FILE}:
            action = "kept_final_package_file" if name == FINAL_ZIP_NAME else "kept_final_package_file"
            add_cleanup_audit(name, action, "required_final_artifact")
            continue
        if is_source_or_input_file(p):
            if p.suffix.lower() == ".py":
                add_cleanup_audit(name, "kept_source", "source_file")
            elif p.name.lower() == "data.xlsx" or p.suffix.lower() == ".xlsx":
                add_cleanup_audit(name, "kept_input", "input_or_workbook_file")
            continue
        if name in FINAL_PACKAGE_FILE_SET and KEEP_FINAL_OUTPUTS_IN_ROOT:
            add_cleanup_audit(name, "kept_final_package_file", "KEEP_FINAL_OUTPUTS_IN_ROOT=True")
            continue
        if name in FINAL_PACKAGE_FILE_SET and not KEEP_FINAL_OUTPUTS_IN_ROOT:
            size = int(p.stat().st_size)
            remove_file(p)
            pruned += 1
            CLEANUP_AUDIT_ROWS.append({
                "File": name,
                "Action": "deleted_stale_generated",
                "Reason": "final_file_copied_to_manuscript_package",
                "Size_Bytes": size,
                "Status": "deleted",
            })
            continue
        if is_generated_output_file(p):
            size = int(p.stat().st_size)
            reason = cleanup_reason(p)
            remove_file(p)
            pruned += 1
            action = "deleted_obsolete" if name in OBSOLETE_OUTPUT_FILE_SET or reason in {"obsolete_output", "old_zip"} else "deleted_stale_generated"
            CLEANUP_AUDIT_ROWS.append({
                "File": name,
                "Action": action,
                "Reason": reason,
                "Size_Bytes": size,
                "Status": "deleted",
            })
            continue
        if p.suffix.lower() in {".csv", ".json", ".zip"}:
            add_cleanup_audit(name, "skipped_unknown_user_file", "not_registry_generated_output")
    return pruned


def main() -> None:
    write_csv(audit_rows(), "Leakage_Control_Audit.csv")
    generate_sentinel_usage_audit()
    generate_engineering_outlier_audit()
    verification = verification_rows()
    triplet_rows = validate_triplet_package_condition()
    if triplet_rows:
        verification = pd.concat([verification, pd.DataFrame(triplet_rows)], ignore_index=True)
    write_csv(verification, "Test_Sanctity_Verification_Summary.csv")
    final_cols = ["Check", "Status", "Evidence_File", "Key_Value", "Required_Condition"]
    final = verification.copy()
    for col in final_cols:
        if col not in final.columns:
            final[col] = ""
    write_csv(final[final_cols], "Final_Verification_Summary.csv")
    if not final["Check"].astype(str).str.contains("Sentinel|sentinel|Physical_Evidence_Tier|Threshold_Stable_Flag", regex=True).any():
        raise RuntimeError("Final_Verification_Summary.csv missing sentinel verification checks.")
    if not final["Status"].astype(str).eq("PASS").all():
        failed = final.loc[~final["Status"].astype(str).eq("PASS"), "Check"].astype(str).tolist()
        raise RuntimeError("Final verification failed; zip package will not be generated: " + "||".join(failed))
    prior_m09 = load_json("09_Run_Manifest.json")
    prior_m09.update({
        "stage": prior_m09.get("stage", "09_leakage_control_audit"),
        "final_leakage_audit_completed": True,
        "final_zip_generated_after_all_checks_pass": True,
        "sentinel_final_checks_enforced": True,
    })
    write_json(prior_m09, "09_Run_Manifest.json")
    files_in_zip = write_final_package(FINAL_ZIP_NAME)
    zip_rows = verify_final_package(FINAL_ZIP_NAME)
    final = pd.concat([final[final_cols], pd.DataFrame(zip_rows)[final_cols]], ignore_index=True)
    write_csv(final[final_cols], "Final_Verification_Summary.csv")
    files_in_zip = write_final_package(FINAL_ZIP_NAME)
    verify_final_package(FINAL_ZIP_NAME)
    copied = write_manuscript_package_dir(MANUSCRIPT_PACKAGE_DIR)
    pruned = prune_workspace_after_success()
    write_cleanup_audit()
    print("Leakage control audit, final verification summary, and final zip package written.")
    print(f"Final files copied to {MANUSCRIPT_PACKAGE_DIR}/: {copied}")
    print(f"Files included in {FINAL_ZIP_NAME}: {files_in_zip}")
    print(f"Stale files pruned after success: {pruned}")
    print(f"Final zip path: {Path(FINAL_ZIP_NAME).resolve()}")
    print(f"Package index path: {(Path(MANUSCRIPT_PACKAGE_DIR) / '00_PACKAGE_INDEX.csv').resolve()}")


if __name__ == "__main__":
    main()
