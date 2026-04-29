# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _config import *
from _utils import benjamini_hochberg, enrichment_pvalue, infer_feature_semantic_group, infer_feature_source_group, missing_category_type, read_csv_smart, require_columns, safe_numeric, wilson_ci, write_csv, write_json


def display_antecedent(items: List[str], manifest_map: Dict[str, dict], lang: str) -> str:
    col = "Display_Item_Label_CN" if lang == "CN" else "Display_Item_Label_EN"
    return " + ".join(str(manifest_map.get(item, {}).get(col) or item) for item in items)


def load_sentinel_flags(path: str, index) -> pd.DataFrame:
    if not Path(path).exists():
        raise RuntimeError(f"Required sentinel flag file missing: {path}")
    flags = read_csv_smart(path)
    if len(flags) != len(index):
        raise RuntimeError(f"Sentinel flag row count mismatch for {path}: {len(flags)} != {len(index)}")
    flags = flags.reset_index(drop=True)
    flags.index = index
    return flags


def resolve_item(df: pd.DataFrame, manifest_map: Dict[str, dict], item: str, sentinel_flags: pd.DataFrame = None):
    item = str(item).strip()
    if item not in manifest_map:
        return False, f"item_not_in_manifest:{item}", pd.Series(False, index=df.index)
    spec = manifest_map[item]
    source = spec["source_feature"]
    if source not in df.columns:
        return False, f"missing_source_feature:{source}", pd.Series(False, index=df.index)
    mtype = str(spec.get("Missing_Category_Type", missing_category_type(source)))
    if mtype == "pure_missing_unknown":
        return False, "invalid_missing_category_item_in_replay", pd.Series(False, index=df.index)
    if spec["transform_type"] == "exact_binary_column":
        mask = pd.to_numeric(df[source], errors="coerce").fillna(0) == 1
        return True, "exact_binary_column", mask.astype(bool)
    thr = float(spec["threshold"])
    op = str(spec["operator"])
    s = safe_numeric(df[source])
    if op == ">":
        threshold_mask = s > thr
    elif op == "<=":
        threshold_mask = s <= thr
    elif op == "==":
        threshold_mask = s == thr
    else:
        return False, f"unknown_operator:{op}", pd.Series(False, index=df.index)
    requires_observed = int(spec.get("requires_observed_source_value", 0) or 0) == 1
    if requires_observed and sentinel_flags is not None and source in sentinel_flags.columns:
        observed_source_value = pd.to_numeric(sentinel_flags[source], errors="coerce").fillna(1).eq(0)
        mask = observed_source_value & threshold_mask
        reason = f"threshold:{op}{thr}:observed_source_value_required"
    else:
        mask = threshold_mask
        reason = f"threshold:{op}{thr}:observed_source_value_default_observed" if requires_observed else f"threshold:{op}{thr}"
    return True, reason, mask.fillna(False).astype(bool)


def replay_rules(df: pd.DataFrame, rules: pd.DataFrame, manifest: pd.DataFrame):
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)
    base_rate = float(y.mean())
    manifest_map = {str(r["item"]): r.to_dict() for _, r in manifest.iterrows()}
    sentinel_flags = load_sentinel_flags(OUTPUTS["sentinel_flags_test"], df.index)
    rows = []
    unresolved = []
    for _, r in rules.iterrows():
        items = [x for x in str(r["Antecedent_Items"]).split("||") if x]
        mask = pd.Series(True, index=df.index)
        reasons = []
        resolved_all = True
        has_cont = False
        families = []
        for item in items:
            ok, reason, m = resolve_item(df, manifest_map, item, sentinel_flags)
            resolved_all = resolved_all and ok
            reasons.append(f"{item}:{reason}")
            mask &= m
            if item in manifest_map:
                has_cont = has_cont or bool(int(manifest_map[item].get("has_continuous_threshold", 0)))
                families.append(str(manifest_map[item].get("family", "other")))
        hit_n = int(mask.sum())
        severe_n = int(y[mask].sum()) if hit_n else 0
        conf = severe_n / hit_n if hit_n else np.nan
        lo, hi = wilson_ci(severe_n, hit_n)
        rr = conf / base_rate if hit_n and base_rate > 0 else np.nan
        p = enrichment_pvalue(severe_n, hit_n, base_rate)
        row = r.to_dict()
        row.update({
            "Antecedent_Display_CN": display_antecedent(items, manifest_map, "CN"),
            "Antecedent_Display_EN": display_antecedent(items, manifest_map, "EN"),
            "Resolved": bool(resolved_all),
            "Resolve_Trace": " | ".join(reasons),
            "Test_N": int(len(df)),
            "Test_Base_Rate": base_rate,
            "Test_Hit_N": hit_n,
            "Test_Support": hit_n / len(df),
            "Test_Severe_N": severe_n,
            "Test_Confidence": conf,
            "Wilson_Lower": lo,
            "Wilson_Upper": hi,
            "Risk_Ratio_vs_BaseRate": rr,
            "p_value_enrichment": p,
            "Has_Continuous_Threshold": int(has_cont),
            "Families_Replayed": "|".join(sorted(set(families))),
        })
        rows.append(row)
        if not resolved_all:
            unresolved.append(row)
    out = pd.DataFrame(rows)
    out["q_value_BH"] = benjamini_hochberg(out["p_value_enrichment"].tolist()) if len(out) else []
    return out, pd.DataFrame(unresolved)


def rule_item_replay_audit(train: pd.DataFrame, test: pd.DataFrame, rules: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    manifest_map = {str(r["item"]): r.to_dict() for _, r in manifest.iterrows()}
    train_flags = load_sentinel_flags(OUTPUTS["sentinel_flags_train"], train.index)
    test_flags = load_sentinel_flags(OUTPUTS["sentinel_flags_test"], test.index)
    final_items = set()
    for _, rule in rules.iterrows():
        for item in [x for x in str(rule["Antecedent_Items"]).split("||") if x]:
            final_items.add(str(item))
    core_items = set()
    items = [str(x) for x in manifest["item"].dropna().astype(str).tolist()]

    rows = []
    for item in items:
        spec = manifest_map.get(str(item))
        usage = {
            "Used_In_Final_Rules": int(str(item) in final_items),
            "Used_In_Core_Confirmed_Rules": int(str(item) in core_items),
            "Usage_Fields_Role": "audit_only_not_rule_selection_input",
        }
        if spec is None:
            rows.append({
                "item": item,
                "source_feature": None,
                "transform_type": None,
                "operator": None,
                "threshold": None,
                "family": None,
                "train_item_hit_n": 0,
                "train_item_hit_rate": 0.0,
                "test_item_hit_n": 0,
                "test_item_hit_rate": 0.0,
                "train_source_min": np.nan,
                "train_source_max": np.nan,
                "test_source_min": np.nan,
                "test_source_max": np.nan,
                **usage,
                "status": "item_not_in_manifest",
            })
            continue
        source = str(spec.get("source_feature", ""))
        base = {
            "item": item,
            "source_feature": source,
            "Display_Item_Label_CN": spec.get("Display_Item_Label_CN", item),
            "Display_Item_Label_EN": spec.get("Display_Item_Label_EN", item),
            "transform_type": spec.get("transform_type"),
            "operator": spec.get("operator"),
            "threshold": spec.get("threshold"),
            "family": spec.get("family"),
            "Missing_Category_Type": spec.get("Missing_Category_Type", missing_category_type(source)),
            "Semantic_Group": spec.get("Semantic_Group", infer_feature_semantic_group(source)),
            "Source_Group": spec.get("Source_Group", infer_feature_source_group(source)),
            "Physical_Mechanism_Eligible": int(spec.get("Physical_Mechanism_Eligible", 1) or 0),
        }
        if str(base["Missing_Category_Type"]) == "pure_missing_unknown":
            rows.append({
                **base,
                "train_item_hit_n": 0,
                "train_item_hit_rate": 0.0,
                "test_item_hit_n": 0,
                "test_item_hit_rate": 0.0,
                "train_source_min": np.nan,
                "train_source_max": np.nan,
                "test_source_min": np.nan,
                "test_source_max": np.nan,
                **usage,
                "status": "invalid_missing_category_item_in_replay",
            })
            continue
        if source not in train.columns or source not in test.columns:
            rows.append({
                **base,
                "train_item_hit_n": 0,
                "train_item_hit_rate": 0.0,
                "test_item_hit_n": 0,
                "test_item_hit_rate": 0.0,
                "train_source_min": np.nan,
                "train_source_max": np.nan,
                "test_source_min": np.nan,
                "test_source_max": np.nan,
                **usage,
                "status": "missing_source_feature",
            })
            continue
        ok_tr, _, tr_mask = resolve_item(train, manifest_map, item, train_flags)
        ok_te, _, te_mask = resolve_item(test, manifest_map, item, test_flags)
        tr_source = safe_numeric(train[source])
        te_source = safe_numeric(test[source])
        tr_sentinel = pd.to_numeric(train_flags[source], errors="coerce").fillna(0).eq(1) if source in train_flags.columns else pd.Series(False, index=train.index)
        te_sentinel = pd.to_numeric(test_flags[source], errors="coerce").fillna(0).eq(1) if source in test_flags.columns else pd.Series(False, index=test.index)
        tr_hit = int(tr_mask.sum()) if ok_tr else 0
        te_hit = int(te_mask.sum()) if ok_te else 0
        tr_rate = tr_hit / max(1, len(train))
        te_rate = te_hit / max(1, len(test))
        tr_hit_sentinel = int((tr_mask & tr_sentinel).sum()) if ok_tr else 0
        te_hit_sentinel = int((te_mask & te_sentinel).sum()) if ok_te else 0
        tr_hit_sentinel_rate = tr_hit_sentinel / max(1, tr_hit)
        te_hit_sentinel_rate = te_hit_sentinel / max(1, te_hit)
        status = "ok"
        if tr_rate >= 0.05 and te_hit == 0:
            status = "matrix_or_distribution_warning"
        rows.append({
            **base,
            "train_item_hit_n": tr_hit,
            "train_item_hit_rate": tr_rate,
            "test_item_hit_n": te_hit,
            "test_item_hit_rate": te_rate,
            "train_source_min": float(tr_source.min()) if tr_source.notna().any() else np.nan,
            "train_source_max": float(tr_source.max()) if tr_source.notna().any() else np.nan,
            "test_source_min": float(te_source.min()) if te_source.notna().any() else np.nan,
            "test_source_max": float(te_source.max()) if te_source.notna().any() else np.nan,
            "train_sentinel_n": int(tr_sentinel.sum()),
            "test_sentinel_n": int(te_sentinel.sum()),
            "train_sentinel_rate": float(tr_sentinel.mean()) if len(tr_sentinel) else 0.0,
            "test_sentinel_rate": float(te_sentinel.mean()) if len(te_sentinel) else 0.0,
            "train_item_hit_sentinel_n": tr_hit_sentinel,
            "test_item_hit_sentinel_n": te_hit_sentinel,
            "train_item_hit_sentinel_rate": tr_hit_sentinel_rate,
            "test_item_hit_sentinel_rate": te_hit_sentinel_rate,
            "known_value_train_hit_n": int((tr_mask & ~tr_sentinel).sum()) if ok_tr else 0,
            "known_value_test_hit_n": int((te_mask & ~te_sentinel).sum()) if ok_te else 0,
            "Sentinel_Hit_Rate_GE_20": int(max(tr_hit_sentinel_rate, te_hit_sentinel_rate) >= 0.20),
            "Sentinel_Hit_Rate_GE_50": int(max(tr_hit_sentinel_rate, te_hit_sentinel_rate) >= 0.50),
            "unknown_dominated_flag": int(te_hit_sentinel_rate >= 0.50),
            **usage,
            "status": status,
        })
    return pd.DataFrame(rows)


def validate_replay_audit_schema(item_audit: pd.DataFrame) -> None:
    required = [
        "train_sentinel_n", "test_sentinel_n", "train_sentinel_rate", "test_sentinel_rate",
        "train_item_hit_sentinel_n", "test_item_hit_sentinel_n",
        "train_item_hit_sentinel_rate", "test_item_hit_sentinel_rate",
        "known_value_train_hit_n", "known_value_test_hit_n",
        "Sentinel_Hit_Rate_GE_20", "Sentinel_Hit_Rate_GE_50", "unknown_dominated_flag",
    ]
    require_columns(item_audit, "Rule_Item_Replay_Audit.csv", required)
    required2 = ["Missing_Category_Type", "Semantic_Group", "Source_Group", "Physical_Mechanism_Eligible"]
    require_columns(item_audit, "Rule_Item_Replay_Audit.csv", required2)
    bad = item_audit[item_audit["Missing_Category_Type"].astype(str).eq("pure_missing_unknown")]
    if not bad.empty:
        raise RuntimeError("Pure missing/unknown items are forbidden in blind replay: " + "||".join(bad["item"].astype(str).head(20).tolist()))


def main():
    train = read_csv_smart(TRAIN_MATRIX_ALIAS if Path(TRAIN_MATRIX_ALIAS).exists() else TRAIN_MATRIX)
    test = read_csv_smart(TEST_MATRIX_ALIAS if Path(TEST_MATRIX_ALIAS).exists() else TEST_MATRIX)
    rules = read_csv_smart(OUTPUTS["rules"])
    manifest = read_csv_smart(OUTPUTS["rule_manifest"])
    report, unresolved = replay_rules(test, rules, manifest)
    item_audit = rule_item_replay_audit(train, test, rules, manifest)
    validate_replay_audit_schema(item_audit)
    write_csv(report, OUTPUTS["blind_replay"])
    write_csv(item_audit, "Rule_Item_Replay_Audit.csv")
    outputs = [OUTPUTS["blind_replay"], "Rule_Item_Replay_Audit.csv"]
    if not unresolved.empty:
        write_csv(unresolved, "Final_Blind_Test_Unresolved_Rules.csv")
        outputs.append("Final_Blind_Test_Unresolved_Rules.csv")
    manifest_obj = {
        "stage": "06_blind_replay",
        "test_usage": "evaluation_only_blind_replay",
        "fit_source": "none",
        "selection_source": "none",
        "evaluation_source": "test",
        "rule_manifest_source": "frozen_train_derived_manifest",
        "modifies_rules": False,
        "modifies_manifest": False,
        "modifies_features": False,
        "unresolved_rules_role": "audit_only_no_rule_remining_trigger",
        "rule_item_replay_audit_scope": "all_manifest_items",
        "blind_replay_uses_observed_source_value_constraint": True,
        "sentinel_flags_used_for_blind_replay": True,
        "missing_category_items_checked_in_blind_replay": True,
        "pure_missing_unknown_forbidden_in_frozen_rules": True,
        "semantic_group_written_to_rule_item_audit": True,
        "test_used_for_rule_selection": False,
        "rule_n": int(len(rules)),
        "unresolved_n": int(len(unresolved)),
        "outputs": outputs,
    }
    write_json(manifest_obj, "06_Blind_Replay_Manifest.json")
    write_json(manifest_obj, "06_Run_Manifest.json")
    print(f"✅ 05 blind replay finished. Replayable: {int(report['Resolved'].sum())}/{len(report)}")


if __name__ == "__main__":
    main()
