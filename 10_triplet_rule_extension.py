# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _config import *
from _utils import (
    assert_test_not_used_for_fit,
    benjamini_hochberg,
    enrichment_pvalue,
    find_group_col,
    missing_category_type,
    normalize_group_ids,
    read_csv_smart,
    safe_numeric,
    wilson_ci,
    write_csv,
    write_json,
)

rule05 = importlib.import_module("05_rule_mining")
blind06 = importlib.import_module("06_blind_replay")
evidence07 = importlib.import_module("07_evidence_grading")

TRIPLET_ITEM_REPLAY_AUDIT = "Triplet_Rule_Item_Replay_Audit.csv"
TRIPLET_BOOTSTRAP_STABILITY = "Triplet_BootstrapRuleStability.csv"
TRIPLET_THRESHOLD_SENSITIVITY = "Triplet_ThresholdSensitivity.csv"
TRIPLET_EVIDENCE_TIERS = "Triplet_Rule_Evidence_Tiers.csv"
MAIN_TEXT_TRIPLET_RULE_TABLE = "Main_Text_Triplet_Rule_Table.csv"
TRIPLET_GOVERNANCE_SCENE_MAP = "Triplet_Governance_Scene_Map.csv"
TRIPLET_INTERPRETATION_CHECK = "Triplet_Interpretation_Check.csv"
TRIPLET_SEMANTIC_REDUNDANCY_AUDIT = "Triplet_Semantic_Redundancy_Audit.csv"
TRIPLET_MANIFEST = "10_Triplet_Extension_Manifest.json"

CORE_FAMILIES = {
    "speed_v0", "speed_vk", "age", "bio", "road_env", "road_type", "lane",
    "vehicle_size", "vehicle_state", "light", "road_surface", "weather",
    "safety", "crash_type",
}

CONFIRMED_TIERS = {"core-confirmed", "binary-stable confirmed", "statistically confirmed"}
MAIN_TEXT_TIERS = {"core-confirmed", "binary-stable confirmed"}


def manifest_lookup(manifest: pd.DataFrame) -> Dict[str, dict]:
    return {str(r["item"]): r.to_dict() for _, r in manifest.iterrows()}


def adjusted_manifest_map(manifest_map: Dict[str, dict], items: List[str], delta: float) -> Dict[str, dict]:
    if float(delta) == 0.0:
        return manifest_map
    out = dict(manifest_map)
    for item in items:
        spec = manifest_map.get(str(item))
        if spec is None or str(spec.get("transform_type")) == "exact_binary_column":
            continue
        new_spec = dict(spec)
        try:
            new_spec["threshold"] = float(spec.get("threshold")) * (1.0 + float(delta))
        except Exception:
            pass
        out[str(item)] = new_spec
    return out


def sentinel_aware_combined_mask(
    df: pd.DataFrame,
    manifest_map: Dict[str, dict],
    items: Tuple[str, ...],
    sentinel_flags: pd.DataFrame,
    delta: float = 0.0,
) -> Tuple[bool, str, pd.Series, bool]:
    items_list = [str(item) for item in items if str(item)]
    use_map = adjusted_manifest_map(manifest_map, items_list, delta)
    mask = pd.Series(True, index=df.index)
    traces = []
    resolved_all = True
    has_continuous = False
    for item in items_list:
        ok, reason, item_mask = blind06.resolve_item(df, use_map, item, sentinel_flags)
        traces.append(f"{item}:{reason}")
        resolved_all = resolved_all and bool(ok)
        mask &= item_mask
        spec = manifest_map.get(item)
        if spec is not None:
            has_continuous = has_continuous or bool(int(spec.get("has_continuous_threshold", 0) or 0))
    return bool(resolved_all), " | ".join(traces), mask.fillna(False).astype(bool), bool(has_continuous)


def rule_stats(
    df: pd.DataFrame,
    y: pd.Series,
    base_rate: float,
    manifest_map: Dict[str, dict],
    items: Tuple[str, ...],
    sentinel_flags: pd.DataFrame,
) -> Dict[str, float]:
    resolved, trace, mask, _ = sentinel_aware_combined_mask(df, manifest_map, items, sentinel_flags)
    hit_n = int(mask.sum()) if resolved else 0
    severe_n = int(y[mask].sum()) if hit_n else 0
    conf = severe_n / hit_n if hit_n else np.nan
    support = hit_n / max(1, len(df))
    lift = conf / max(base_rate, 1e-12) if hit_n else np.nan
    return {
        "Resolved": bool(resolved),
        "Resolve_Trace": trace,
        "Hit_N": hit_n,
        "Support": support,
        "Severe_N": severe_n,
        "Confidence": conf,
        "Lift": lift,
    }


def valid_triplet(items: Tuple[str, ...], manifest_map: Dict[str, dict]) -> bool:
    specs = [manifest_map.get(str(item)) for item in items]
    if any(spec is None for spec in specs):
        return False
    if any(str(spec.get("Missing_Category_Type", missing_category_type(spec.get("source_feature", "")))) == "pure_missing_unknown" for spec in specs):
        return False
    if rule05.semantic_duplicate_details(items, manifest_map):
        return False
    families = [str(s.get("family", "other")) for s in specs]
    if len(set(families)) < 2:
        return False
    if not (set(families) & CORE_FAMILIES):
        return False
    sources = [str(s.get("source_feature", "")) for s in specs]
    if len(set(sources)) < len(sources):
        return False
    return True


def semantic_redundancy_audit_row(items: Tuple[str, ...], manifest_map: Dict[str, dict]) -> List[Dict[str, str]]:
    rows = []
    for detail in rule05.semantic_duplicate_details(items, manifest_map):
        rows.append({
            "Triplet_Rule_ID": "",
            "Antecedent_Items": "||".join(items),
            "Duplicate_Semantic_Group": detail["Duplicate_Semantic_Group"],
            "Duplicate_Items": detail["Duplicate_Items"],
            "Action": "excluded_from_triplet_extension",
            "Reason": "same semantic construct repeated in one rule",
        })
    return rows


def parent_lifts(
    train: pd.DataFrame,
    y: pd.Series,
    base_rate: float,
    manifest_map: Dict[str, dict],
    items: Tuple[str, ...],
    train_flags: pd.DataFrame,
) -> Dict[str, object]:
    rows = []
    for pair in itertools.combinations(items, 2):
        stats = rule_stats(train, y, base_rate, manifest_map, tuple(pair), train_flags)
        rows.append((tuple(pair), stats["Lift"]))
    lifts = [float(x[1]) for x in rows if not pd.isna(x[1])]
    best = max(lifts) if lifts else np.nan
    out = {}
    for i, (pair, lift) in enumerate(rows, start=1):
        out[f"Parent_{i}_Items"] = "||".join(pair)
        out[f"Parent_{i}_Train_Lift"] = lift
    out["Best_Parent_2Item_Lift"] = best
    return out


def build_train_candidates(train: pd.DataFrame, manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.to_numeric(train[LABEL_COL], errors="coerce").fillna(0).astype(int)
    base_rate = float(y.mean())
    manifest_map = manifest_lookup(manifest)
    train_flags = blind06.load_sentinel_flags(OUTPUTS["sentinel_flags_train"], train.index)
    items = [str(x) for x in manifest["item"].dropna().astype(str).tolist()]
    rows = []
    audit_rows = []
    for combo in itertools.combinations(items, 3):
        duplicate_rows = semantic_redundancy_audit_row(combo, manifest_map)
        if duplicate_rows:
            audit_rows.extend(duplicate_rows)
            continue
        if not valid_triplet(combo, manifest_map):
            continue
        stats = rule_stats(train, y, base_rate, manifest_map, combo, train_flags)
        if not stats["Resolved"]:
            continue
        if stats["Hit_N"] < TRIPLET_MIN_TRAIN_HIT_N:
            continue
        if stats["Support"] < TRIPLET_MIN_TRAIN_SUPPORT:
            continue
        if pd.isna(stats["Lift"]) or stats["Lift"] < TRIPLET_MIN_TRAIN_LIFT:
            continue
        specs = [manifest_map[item] for item in combo]
        families = sorted(set(str(s.get("family", "other")) for s in specs))
        sources = [str(s.get("source_feature", "")) for s in specs]
        parent = parent_lifts(train, y, base_rate, manifest_map, combo, train_flags)
        best_parent = parent["Best_Parent_2Item_Lift"]
        incr = stats["Lift"] - best_parent if not pd.isna(best_parent) else np.nan
        axis = rule05.infer_mechanism_axis(families, sources, combo)
        rows.append({
            "Antecedent_Items": "||".join(combo),
            "Source_Features": "||".join(sources),
            "Mechanism_Axis": axis,
            "Mechanism_Families": "|".join(families),
            "Mechanism_Family_N": len(families),
            "Train_Hit_N": stats["Hit_N"],
            "Train_Support": stats["Support"],
            "Train_Severe_N": stats["Severe_N"],
            "Train_Confidence": stats["Confidence"],
            "Train_Lift": stats["Lift"],
            **parent,
            "Incremental_Lift_Over_Best_Parent": incr,
            "Incremental_Value_Flag": "incremental_lift" if not pd.isna(incr) and incr > 0 else "scene_context_only",
            "Governance_Scene": rule05.governance_scene(axis),
            "Governance_Interpretation_Template": rule05.governance_template(axis),
            "Interpretation_Boundary": rule05.interpretation_boundary(axis),
        })
    audit = pd.DataFrame(audit_rows, columns=[
        "Triplet_Rule_ID", "Antecedent_Items", "Duplicate_Semantic_Group",
        "Duplicate_Items", "Action", "Reason",
    ])
    if not rows:
        return pd.DataFrame(), audit
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["Train_Lift", "Train_Confidence", "Train_Hit_N", "Mechanism_Family_N", "Incremental_Lift_Over_Best_Parent"],
        ascending=[False, False, False, False, False],
    )
    selected = []
    axis_counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        axis = str(row["Mechanism_Axis"])
        if axis_counts.get(axis, 0) >= TRIPLET_AXIS_CAP:
            continue
        selected.append(row)
        axis_counts[axis] = axis_counts.get(axis, 0) + 1
        if len(selected) >= TRIPLET_MAX_FINAL:
            break
    out = pd.DataFrame(selected).reset_index(drop=True)
    out.insert(0, "Triplet_Rule_ID", [f"T{i+1:02d}" for i in range(len(out))])
    return out, audit


def replay_triplets_sentinel_aware(test: pd.DataFrame, manifest: pd.DataFrame, triplets: pd.DataFrame) -> pd.DataFrame:
    if triplets.empty:
        return triplets.copy()
    y = pd.to_numeric(test[LABEL_COL], errors="coerce").fillna(0).astype(int)
    base_rate = float(y.mean())
    manifest_map = manifest_lookup(manifest)
    test_flags = blind06.load_sentinel_flags(OUTPUTS["sentinel_flags_test"], test.index)
    rows = []
    for _, row in triplets.iterrows():
        items = tuple(x for x in str(row["Antecedent_Items"]).split("||") if x)
        resolved, trace, mask, has_cont = sentinel_aware_combined_mask(test, manifest_map, items, test_flags)
        hit_n = int(mask.sum()) if resolved else 0
        severe_n = int(y[mask].sum()) if hit_n else 0
        conf = severe_n / hit_n if hit_n else np.nan
        lo, hi = wilson_ci(severe_n, hit_n)
        rr = conf / max(base_rate, 1e-12) if hit_n else np.nan
        p = enrichment_pvalue(severe_n, hit_n, base_rate)
        rec = row.to_dict()
        rec.update({
            "Antecedent_Display_CN": blind06.display_antecedent(list(items), manifest_map, "CN"),
            "Antecedent_Display_EN": blind06.display_antecedent(list(items), manifest_map, "EN"),
            "Resolved": bool(resolved),
            "Resolve_Trace": trace,
            "Test_N": int(len(test)),
            "Test_Base_Rate": base_rate,
            "Test_Hit_N": hit_n,
            "Test_Support": hit_n / max(1, len(test)),
            "Test_Severe_N": severe_n,
            "Test_Confidence": conf,
            "Wilson_Lower": lo,
            "Wilson_Upper": hi,
            "Risk_Ratio_vs_BaseRate": rr,
            "p_value_enrichment": p,
            "Has_Continuous_Threshold": int(has_cont),
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["q_value_BH"] = benjamini_hochberg(out["p_value_enrichment"].tolist()) if len(out) else []
    out["Does_Not_Modify_Primary_Rules"] = 1
    return out


def triplet_rule_item_replay_audit(train: pd.DataFrame, test: pd.DataFrame, triplets: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    audit = blind06.rule_item_replay_audit(train, test, triplets, manifest).copy()
    usage: Dict[str, int] = {}
    for _, row in triplets.iterrows():
        for item in [x for x in str(row.get("Antecedent_Items", "")).split("||") if x]:
            usage[str(item)] = usage.get(str(item), 0) + 1
    audit["Used_In_Triplet_Rules"] = audit["item"].astype(str).map(lambda x: int(x in usage))
    audit["Triplet_Usage_N"] = audit["item"].astype(str).map(lambda x: int(usage.get(x, 0)))
    for col in [
        "train_sentinel_n", "test_sentinel_n", "train_sentinel_rate", "test_sentinel_rate",
        "train_item_hit_sentinel_n", "test_item_hit_sentinel_n",
        "train_item_hit_sentinel_rate", "test_item_hit_sentinel_rate",
        "known_value_train_hit_n", "known_value_test_hit_n",
        "Sentinel_Hit_Rate_GE_20", "Sentinel_Hit_Rate_GE_50", "unknown_dominated_flag",
    ]:
        if col not in audit.columns:
            audit[col] = 0
    keep = [
        "item", "source_feature", "transform_type", "operator", "threshold", "family",
        "Missing_Category_Type", "Semantic_Group", "Source_Group", "Physical_Mechanism_Eligible",
        "Used_In_Triplet_Rules", "Triplet_Usage_N",
        "train_item_hit_n", "train_item_hit_rate", "test_item_hit_n", "test_item_hit_rate",
        "train_sentinel_n", "test_sentinel_n", "train_sentinel_rate", "test_sentinel_rate",
        "train_item_hit_sentinel_n", "test_item_hit_sentinel_n",
        "train_item_hit_sentinel_rate", "test_item_hit_sentinel_rate",
        "known_value_train_hit_n", "known_value_test_hit_n",
        "Sentinel_Hit_Rate_GE_20", "Sentinel_Hit_Rate_GE_50", "unknown_dominated_flag",
        "status",
    ]
    for col in keep:
        if col not in audit.columns:
            audit[col] = np.nan
    return audit[keep]


def triplet_bootstrap_stability(train: pd.DataFrame, triplets: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if triplets.empty:
        return pd.DataFrame(columns=[
            "Triplet_Rule_ID", "Antecedent_Items", "Bootstrap_Iterations",
            "Bootstrap_Recurrent_N", "Bootstrap_Recurrence",
        ])
    group_col = find_group_col(train.columns, GROUP_CANDIDATES)
    groups = normalize_group_ids(train[group_col])
    uniq = groups.drop_duplicates().to_numpy()
    group_to_indices = {group: train.index[groups.eq(group)].tolist() for group in uniq}
    manifest_map = manifest_lookup(manifest)
    train_flags = blind06.load_sentinel_flags(OUTPUTS["sentinel_flags_train"], train.index)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for _, rule in triplets.iterrows():
        recurrent = 0
        valid_iter = 0
        items = tuple(x for x in str(rule["Antecedent_Items"]).split("||") if x)
        for _b in range(BOOTSTRAP_ITERATIONS):
            sampled_groups = rng.choice(uniq, size=len(uniq), replace=True)
            sampled_indices = []
            for group in sampled_groups:
                sampled_indices.extend(group_to_indices.get(group, []))
            if len(sampled_indices) < 20:
                continue
            df_b = train.loc[sampled_indices].reset_index(drop=True)
            flags_b = train_flags.loc[sampled_indices].reset_index(drop=True)
            yb = pd.to_numeric(df_b[LABEL_COL], errors="coerce").fillna(0).astype(int)
            resolved, _, mask, _ = sentinel_aware_combined_mask(df_b, manifest_map, items, flags_b)
            hit = int(mask.sum()) if resolved else 0
            if hit <= 0:
                valid_iter += 1
                continue
            conf = float(yb[mask].mean())
            lift = conf / max(float(yb.mean()), 1e-12)
            support = hit / len(df_b)
            if support >= RULE_MIN_SUPPORT * 0.80 and lift >= 1.0:
                recurrent += 1
            valid_iter += 1
        rows.append({
            "Triplet_Rule_ID": rule["Triplet_Rule_ID"],
            "Antecedent_Items": rule["Antecedent_Items"],
            "Bootstrap_Iterations": valid_iter,
            "Bootstrap_Recurrent_N": recurrent,
            "Bootstrap_Recurrence": recurrent / max(1, valid_iter),
        })
    return pd.DataFrame(rows)


def triplet_threshold_sensitivity(test: pd.DataFrame, triplets: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if triplets.empty:
        return pd.DataFrame(columns=[
            "Triplet_Rule_ID", "Antecedent_Items", "Has_Continuous_Threshold", "Threshold_Deltas",
            "Test_Hit_N_Min", "Test_Lift_Min", "Test_Lift_Max", "Test_Confidence_Min",
        ])
    y = pd.to_numeric(test[LABEL_COL], errors="coerce").fillna(0).astype(int)
    base = float(y.mean())
    manifest_map = manifest_lookup(manifest)
    test_flags = blind06.load_sentinel_flags(OUTPUTS["sentinel_flags_test"], test.index)
    rows = []

    def nan_reduce(values, reducer):
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return float(reducer(arr)) if arr.size else np.nan

    for _, rule in triplets.iterrows():
        items = tuple(x for x in str(rule["Antecedent_Items"]).split("||") if x)
        lifts, confs, hits = [], [], []
        has_any_cont = False
        for d in THRESHOLD_DELTAS:
            resolved, _, mask, has_cont = sentinel_aware_combined_mask(test, manifest_map, items, test_flags, float(d))
            has_any_cont = has_any_cont or has_cont
            hit = int(mask.sum()) if resolved else 0
            hits.append(hit)
            if hit > 0:
                conf = float(y[mask].mean())
                lift = conf / max(base, 1e-12)
            else:
                conf, lift = np.nan, np.nan
            confs.append(conf)
            lifts.append(lift)
        rows.append({
            "Triplet_Rule_ID": rule["Triplet_Rule_ID"],
            "Antecedent_Items": rule["Antecedent_Items"],
            "Has_Continuous_Threshold": int(has_any_cont),
            "Threshold_Deltas": "||".join(str(x) for x in THRESHOLD_DELTAS),
            "Test_Hit_N_Min": nan_reduce(hits, np.min) if hits else np.nan,
            "Test_Lift_Min": nan_reduce(lifts, np.min) if has_any_cont else np.nan,
            "Test_Lift_Max": nan_reduce(lifts, np.max) if has_any_cont else np.nan,
            "Test_Confidence_Min": nan_reduce(confs, np.min) if has_any_cont else np.nan,
        })
    return pd.DataFrame(rows)


def triplet_evidence_tiers(
    replay: pd.DataFrame,
    boot: pd.DataFrame,
    sens: pd.DataFrame,
    item_audit: pd.DataFrame,
) -> pd.DataFrame:
    if replay.empty:
        return pd.DataFrame()
    df = replay.merge(boot[["Triplet_Rule_ID", "Bootstrap_Recurrence"]], on="Triplet_Rule_ID", how="left")
    for col in ["Triplet_Rule_ID", "Has_Continuous_Threshold", "Test_Lift_Min", "Test_Hit_N_Min"]:
        if col not in sens.columns:
            sens[col] = np.nan
    df = df.merge(
        sens[["Triplet_Rule_ID", "Has_Continuous_Threshold", "Test_Lift_Min", "Test_Hit_N_Min"]],
        on="Triplet_Rule_ID",
        how="left",
        suffixes=("", "_Sensitivity"),
    )
    grades, uses = [], []
    fixed_hit_flags, threshold_hit_flags, threshold_lift_flags, threshold_stable_flags = [], [], [], []

    def numeric_scalar(value: object) -> float:
        return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])

    for _, r in df.iterrows():
        hit_n = int(r.get("Test_Hit_N", 0) or 0)
        replayable = bool(r.get("Resolved", False))
        stat = bool(
            replayable
            and hit_n >= REPLAY_TEST_HIT_MIN
            and r.get("q_value_BH", np.nan) <= CORE_Q_THRESHOLD
            and r.get("Risk_Ratio_vs_BaseRate", np.nan) >= CORE_RR_THRESHOLD
        )
        boot_ok = bool(r.get("Bootstrap_Recurrence", 0) >= CORE_BOOTSTRAP_THRESHOLD)
        has_cont = bool(int(r.get("Has_Continuous_Threshold", r.get("Has_Continuous_Threshold_Sensitivity", 0)) or 0))
        fixed_hit_ok = int(r.get("Test_Hit_N", 0) or 0) >= CORE_TEST_HIT_MIN
        if has_cont:
            threshold_hit_ok = numeric_scalar(r.get("Test_Hit_N_Min", np.nan)) >= CORE_TEST_HIT_MIN
            threshold_lift_ok = numeric_scalar(r.get("Test_Lift_Min", np.nan)) >= CORE_THRESHOLD_MIN_LIFT
            threshold_stable = bool(threshold_hit_ok and threshold_lift_ok)
        else:
            threshold_hit_ok = True
            threshold_lift_ok = True
            threshold_stable = True
        if stat and boot_ok and fixed_hit_ok and threshold_stable:
            if has_cont:
                grade = "core-confirmed"
                use = "priority_governance_target"
            else:
                grade = "binary-stable confirmed"
                use = "stable_categorical_governance_signal"
        elif stat:
            grade = "statistically confirmed"
            use = "confirmatory_monitoring_signal"
        elif replayable:
            grade = "replayable"
            use = "replayed_without_confirmatory_evidence"
        else:
            grade = "exploratory"
            use = "hypothesis_generation_only"
        grades.append(grade)
        uses.append(use)
        fixed_hit_flags.append(int(fixed_hit_ok))
        threshold_hit_flags.append(int(threshold_hit_ok))
        threshold_lift_flags.append(int(threshold_lift_ok))
        threshold_stable_flags.append(int(threshold_stable))
    df["Fixed_Test_Hit_OK"] = fixed_hit_flags
    df["Threshold_Hit_OK"] = threshold_hit_flags
    df["Threshold_Lift_OK"] = threshold_lift_flags
    df["Threshold_Stable_Flag"] = threshold_stable_flags
    df["Evidence_Tier"] = grades
    df["Governance_Use"] = uses
    df["Core_Confirmed_Flag"] = df["Evidence_Tier"].eq("core-confirmed").astype(int)
    df["Binary_Stable_Flag"] = df["Evidence_Tier"].eq("binary-stable confirmed").astype(int)
    return attach_triplet_physical_fields(df, item_audit)


def attach_triplet_physical_fields(tiers: pd.DataFrame, item_audit: pd.DataFrame) -> pd.DataFrame:
    out = tiers.copy()
    item_rate, item_missing, item_semantic, item_phys, item_unknown = {}, {}, {}, {}, {}
    if not item_audit.empty and "item" in item_audit.columns:
        for _, row in item_audit.iterrows():
            item = str(row.get("item", ""))
            item_rate[item] = float(pd.to_numeric(pd.Series([row.get("test_item_hit_sentinel_rate", 0)]), errors="coerce").fillna(0).iloc[0])
            item_missing[item] = str(row.get("Missing_Category_Type", "not_missing_category"))
            item_semantic[item] = str(row.get("Semantic_Group", ""))
            item_phys[item] = int(pd.to_numeric(pd.Series([row.get("Physical_Mechanism_Eligible", 1)]), errors="coerce").fillna(1).iloc[0])
            item_unknown[item] = int(pd.to_numeric(pd.Series([row.get("unknown_dominated_flag", 0)]), errors="coerce").fillna(0).iloc[0])
    dominated, max_rates, missing_flags, pure_flags, partial_flags, dup_flags, valid_flags = [], [], [], [], [], [], []
    unknown_item_flags = []
    for _, row in out.iterrows():
        items = [str(item) for item in str(row.get("Antecedent_Items", "")).split("||") if str(item)]
        rates = [item_rate.get(item, 0.0) for item in items]
        max_rate = max(rates) if rates else 0.0
        missing_types = [item_missing.get(item, "not_missing_category") for item in items]
        sems = [item_semantic.get(item, "") for item in items if item_semantic.get(item, "")]
        pure = int(any(t == "pure_missing_unknown" for t in missing_types))
        partial = int(any(t == "partial_unknown_detail" for t in missing_types))
        dup = int(len(sems) != len(set(sems)))
        unknown = int(any(item_unknown.get(item, 0) == 1 for item in items))
        dominated_flag = int(max_rate >= 0.50)
        valid = int(
            dominated_flag == 0
            and pure == 0
            and partial == 0
            and dup == 0
            and all(item_phys.get(item, 1) == 1 for item in items)
        )
        max_rates.append(max_rate)
        dominated.append(dominated_flag)
        missing_flags.append(int(pure or partial))
        pure_flags.append(pure)
        partial_flags.append(partial)
        dup_flags.append(dup)
        unknown_item_flags.append(unknown)
        valid_flags.append(valid)
    out["Rule_Sentinel_Dominated_Flag"] = dominated
    out["Max_Item_Test_Hit_Sentinel_Rate"] = max_rates
    out["Rule_Missing_Category_Flag"] = missing_flags
    out["Rule_Pure_Missing_Unknown_Flag"] = pure_flags
    out["Rule_Partial_Unknown_Detail_Flag"] = partial_flags
    out["Rule_Semantic_Duplicate_Flag"] = dup_flags
    out["Rule_Unknown_Dominated_Item_Flag"] = unknown_item_flags
    out["Physical_Mechanism_Valid_Flag"] = valid_flags
    out["Physical_Evidence_Tier"] = out["Evidence_Tier"]
    invalid = out["Physical_Mechanism_Valid_Flag"].astype(int).eq(0)
    out.loc[invalid, "Physical_Evidence_Tier"] = "data-availability-or-redundancy-dominated replay signal"
    threshold_stable = out.get("Threshold_Stable_Flag", pd.Series(1, index=out.index)).astype(int).eq(1)
    out["Manuscript_Eligible_Triplet_Flag"] = (
        out["Evidence_Tier"].isin(CONFIRMED_TIERS) & out["Physical_Mechanism_Valid_Flag"].astype(int).eq(1)
    ).astype(int)
    out["Main_Text_Eligible_Triplet_Flag"] = (
        out["Evidence_Tier"].isin(MAIN_TEXT_TIERS)
        & out["Physical_Mechanism_Valid_Flag"].astype(int).eq(1)
        & threshold_stable
    ).astype(int)
    out["Supplementary_Only_Triplet_Flag"] = (
        out["Main_Text_Eligible_Triplet_Flag"].astype(int).eq(0)
        | out["Evidence_Tier"].isin(["statistically confirmed", "replayable", "exploratory"])
    ).astype(int)
    return out


def build_main_text_triplet_rule_table(tiers: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Triplet_Rule_ID", "Antecedent_Items", "Antecedent_Display_CN", "Antecedent_Display_EN",
        "Mechanism_Axis", "Evidence_Tier", "Physical_Evidence_Tier",
        "Test_Hit_N", "Test_Confidence", "Risk_Ratio_vs_BaseRate", "q_value_BH",
        "Bootstrap_Recurrence", "Has_Continuous_Threshold", "Test_Lift_Min", "Test_Hit_N_Min",
        "Fixed_Test_Hit_OK", "Threshold_Hit_OK", "Threshold_Lift_OK", "Threshold_Stable_Flag",
        "Governance_Scene", "Governance_Use", "Governance_Interpretation_Template",
        "Interpretation_Boundary", "Physical_Mechanism_Valid_Flag",
        "Main_Text_Eligible_Triplet_Flag", "Manuscript_Eligible_Triplet_Flag",
    ]
    if tiers.empty:
        return pd.DataFrame(columns=keep)
    df = tiers[tiers["Main_Text_Eligible_Triplet_Flag"].astype(int).eq(1)].copy()
    if df.empty:
        return pd.DataFrame(columns=keep)
    priority = {"core-confirmed": 0, "binary-stable confirmed": 1}
    df["_Priority"] = df["Evidence_Tier"].map(priority).fillna(99)
    df["_RR"] = pd.to_numeric(df.get("Risk_Ratio_vs_BaseRate"), errors="coerce").fillna(-np.inf)
    df["_Hit"] = pd.to_numeric(df.get("Test_Hit_N"), errors="coerce").fillna(-1)
    df = df.sort_values(["_Priority", "_RR", "_Hit", "Triplet_Rule_ID"], ascending=[True, False, False, True])
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep]


def build_triplet_governance_scene_map(tiers: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Triplet_Rule_ID", "Antecedent_Items", "Antecedent_Display_CN", "Antecedent_Display_EN",
        "Mechanism_Axis", "Evidence_Tier", "Physical_Evidence_Tier", "Test_Hit_N",
        "Risk_Ratio_vs_BaseRate", "q_value_BH", "Bootstrap_Recurrence",
        "Has_Continuous_Threshold", "Test_Lift_Min", "Governance_Scene",
        "Threshold_Stable_Flag", "Threshold_Hit_OK", "Threshold_Lift_OK",
        "Governance_Use", "Governance_Interpretation_Template", "Interpretation_Boundary",
        "Physical_Mechanism_Valid_Flag", "Manuscript_Eligible_Triplet_Flag",
        "Main_Text_Eligible_Triplet_Flag", "Supplementary_Only_Triplet_Flag",
    ]
    out = tiers.copy()
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep] if not out.empty else pd.DataFrame(columns=keep)


def build_triplet_interpretation_check(tiers: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Triplet_Rule_ID", "Antecedent_Items", "Antecedent_Display_CN", "Antecedent_Display_EN",
        "Evidence_Tier", "Governance_Use", "Physical_Mechanism_Valid_Flag",
        "Threshold_Stable_Flag", "Threshold_Hit_OK", "Threshold_Lift_OK",
        "Manuscript_Eligible_Triplet_Flag", "Main_Text_Eligible_Triplet_Flag",
        "Interpretation_Check",
    ]
    if tiers.empty:
        return pd.DataFrame(columns=keep)
    out = tiers.copy()
    sensitive = (
        out["Evidence_Tier"].astype(str).eq("statistically confirmed")
        & pd.to_numeric(out.get("Threshold_Stable_Flag", pd.Series(1, index=out.index)), errors="coerce").fillna(0).astype(int).eq(0)
    )
    out["Interpretation_Check"] = out.get("Governance_Interpretation_Template", pd.Series("", index=out.index)).astype(str)
    out.loc[sensitive, "Interpretation_Check"] = (
        "Statistically confirmed but threshold-sensitive; use as supplementary monitoring evidence rather than a primary governance rule."
    )
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]


def validate_triplet_manuscript_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tiers: pd.DataFrame,
    main_text: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    rows = []

    def add(check: str, ok: bool, evidence: str, detail: object) -> None:
        rows.append({
            "Check_Item": check,
            "Status": "PASS" if ok else "FAIL",
            "Evidence_File": evidence,
            "Detail": detail,
        })

    failures: List[str] = []
    try:
        train_flags = blind06.load_sentinel_flags(OUTPUTS["sentinel_flags_train"], train.index)
        test_flags = blind06.load_sentinel_flags(OUTPUTS["sentinel_flags_test"], test.index)
        sentinel_ok = len(train_flags) == len(train) and len(test_flags) == len(test)
    except Exception as exc:
        sentinel_ok = False
        add("Sentinel flag row counts match train/test matrices.", False, f"{OUTPUTS['sentinel_flags_train']};{OUTPUTS['sentinel_flags_test']}", str(exc))
    else:
        add("Sentinel flag row counts match train/test matrices.", sentinel_ok, f"{OUTPUTS['sentinel_flags_train']};{OUTPUTS['sentinel_flags_test']}", f"train={len(train_flags)}/{len(train)}; test={len(test_flags)}/{len(test)}")
    manuscript = tiers[tiers.get("Manuscript_Eligible_Triplet_Flag", pd.Series(dtype=int)).astype(int).eq(1)] if not tiers.empty else pd.DataFrame()
    no_pure = manuscript.empty or not manuscript.get("Rule_Pure_Missing_Unknown_Flag", pd.Series(dtype=int)).astype(int).eq(1).any()
    no_dup = manuscript.empty or not manuscript.get("Rule_Semantic_Duplicate_Flag", pd.Series(dtype=int)).astype(int).eq(1).any()
    no_unknown = manuscript.empty or not manuscript.get("Rule_Unknown_Dominated_Item_Flag", pd.Series(dtype=int)).astype(int).eq(1).any()
    main_ok = main_text.empty or (
        main_text.get("Physical_Mechanism_Valid_Flag", pd.Series(dtype=int)).astype(int).eq(1).all()
        and main_text.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).isin(MAIN_TEXT_TIERS).all()
        and main_text.get("Main_Text_Eligible_Triplet_Flag", pd.Series(dtype=int)).astype(int).eq(1).all()
    )
    add("No pure_missing_unknown item appears in manuscript-eligible triplets.", no_pure, TRIPLET_EVIDENCE_TIERS, int((not manuscript.empty) and manuscript.get("Rule_Pure_Missing_Unknown_Flag", pd.Series(dtype=int)).astype(int).eq(1).sum()))
    add("No duplicated Semantic_Group appears in manuscript-eligible triplets.", no_dup, TRIPLET_EVIDENCE_TIERS, int((not manuscript.empty) and manuscript.get("Rule_Semantic_Duplicate_Flag", pd.Series(dtype=int)).astype(int).eq(1).sum()))
    add("No manuscript-eligible triplet has an unknown-dominated item.", no_unknown, TRIPLET_EVIDENCE_TIERS, int((not manuscript.empty) and manuscript.get("Rule_Unknown_Dominated_Item_Flag", pd.Series(dtype=int)).astype(int).eq(1).sum()))
    add("Main text triplet table only contains physical-valid core/binary-stable triplets.", main_ok, MAIN_TEXT_TRIPLET_RULE_TABLE, int(len(main_text)))
    for row in rows:
        if row["Status"] == "FAIL":
            failures.append(str(row["Check_Item"]))
    return pd.DataFrame(rows), failures


def main() -> None:
    required = [
        TRAIN_MATRIX, TEST_MATRIX, OUTPUTS["rule_manifest"], OUTPUTS["rules"], OUTPUTS["evidence_tiers"],
        OUTPUTS["sentinel_flags_train"], OUTPUTS["sentinel_flags_test"],
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required triplet extension inputs: " + "||".join(missing))
    assert_test_not_used_for_fit(
        "10_triplet_rule_extension",
        [TRAIN_MATRIX, OUTPUTS["rule_manifest"], OUTPUTS["rules"]],
        test_paths=[TEST_MATRIX, OUTPUTS["blind_replay"], OUTPUTS["threshold_sensitivity"]],
    )
    train = read_csv_smart(TRAIN_MATRIX)
    test = read_csv_smart(TEST_MATRIX)
    manifest = read_csv_smart(OUTPUTS["rule_manifest"])

    triplets, semantic_audit = build_train_candidates(train, manifest)
    replay = replay_triplets_sentinel_aware(test, manifest, triplets)
    item_audit = triplet_rule_item_replay_audit(train, test, replay, manifest)
    boot = triplet_bootstrap_stability(train, replay, manifest)
    sens = triplet_threshold_sensitivity(test, replay, manifest)
    tiers = triplet_evidence_tiers(replay, boot, sens, item_audit)
    main_text = build_main_text_triplet_rule_table(tiers)
    validation_check, validation_failures = validate_triplet_manuscript_outputs(train, test, tiers, main_text)
    if validation_failures and not tiers.empty:
        tiers["Manuscript_Eligible_Triplet_Flag"] = 0
        tiers["Main_Text_Eligible_Triplet_Flag"] = 0
        tiers["Supplementary_Only_Triplet_Flag"] = 1
        main_text = build_main_text_triplet_rule_table(tiers)
    scene_map = build_triplet_governance_scene_map(tiers)
    interpretation_check = build_triplet_interpretation_check(tiers)

    write_csv(replay, OUTPUTS["triplet_extension_report"])
    write_csv(item_audit, TRIPLET_ITEM_REPLAY_AUDIT)
    write_csv(boot, TRIPLET_BOOTSTRAP_STABILITY)
    write_csv(sens, TRIPLET_THRESHOLD_SENSITIVITY)
    write_csv(tiers, TRIPLET_EVIDENCE_TIERS)
    write_csv(main_text, MAIN_TEXT_TRIPLET_RULE_TABLE)
    write_csv(scene_map, TRIPLET_GOVERNANCE_SCENE_MAP)
    write_csv(interpretation_check, TRIPLET_INTERPRETATION_CHECK)
    write_csv(semantic_audit, TRIPLET_SEMANTIC_REDUNDANCY_AUDIT)

    main_text_n = int(len(main_text))
    manuscript_eligible_n = int(tiers.get("Manuscript_Eligible_Triplet_Flag", pd.Series(dtype=int)).astype(int).sum()) if not tiers.empty else 0
    core_confirmed_n = int(tiers.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).eq("core-confirmed").sum()) if not tiers.empty else 0
    binary_stable_confirmed_n = int(tiers.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).eq("binary-stable confirmed").sum()) if not tiers.empty else 0
    statistically_confirmed_n = int(tiers.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).eq("statistically confirmed").sum()) if not tiers.empty else 0
    ready = bool(main_text_n > 0 and not validation_failures)
    outputs = [
        OUTPUTS["triplet_extension_report"], TRIPLET_ITEM_REPLAY_AUDIT, TRIPLET_BOOTSTRAP_STABILITY,
        TRIPLET_THRESHOLD_SENSITIVITY, TRIPLET_EVIDENCE_TIERS, MAIN_TEXT_TRIPLET_RULE_TABLE,
        TRIPLET_GOVERNANCE_SCENE_MAP, TRIPLET_INTERPRETATION_CHECK, TRIPLET_SEMANTIC_REDUNDANCY_AUDIT,
    ]
    write_json({
        "stage": "10_triplet_rule_extension",
        "fit_source": "train",
        "selection_source": "train",
        "rule_generation_source": "train_only",
        "threshold_source": "frozen_train_derived_manifest",
        "evaluation_source": "test_blind_replay_only",
        "test_used_for_triplet_generation": False,
        "test_used_for_triplet_filtering": False,
        "test_used_for_threshold_derivation": False,
        "test_used_for_triplet_selection": False,
        "test_used_for_evaluation_only": True,
        "semantic_redundancy_filter": True,
        "semantic_group_source": "train_manifest_string_rules",
        "test_used_for_semantic_filtering": False,
        "semantic_duplicate_rules_excluded_from_primary": True,
        "semantic_duplicate_triplets_excluded_from_extension": True,
        "modifies_primary_rules": False,
        "modifies_manifest": False,
        "modifies_selected_features": False,
        "supplementary_only": not ready,
        "triplet_rules_are_supplementary_governance_scenario_extensions": True,
        "triplet_extension_not_used_for_manuscript": not ready,
        "triplet_extension_requires_sentinel_aware_update": False,
        "sentinel_aware_triplet_replay": True,
        "triplet_bootstrap_stability_done": True,
        "triplet_threshold_sensitivity_done": True,
        "triplet_threshold_hit_stability_enforced": True,
        "triplet_threshold_lift_stability_enforced": True,
        "triplet_core_confirmed_requires_threshold_stable": True,
        "triplet_main_text_requires_threshold_stable": True,
        "triplet_physical_validity_audit_done": True,
        "triplet_extension_excluded_from_final_manuscript_package": not ready,
        "triplet_manuscript_ready": ready,
        "triplet_rule_n": int(len(replay)),
        "core_confirmed_triplet_n": core_confirmed_n,
        "binary_stable_confirmed_triplet_n": binary_stable_confirmed_n,
        "statistically_confirmed_triplet_n": statistically_confirmed_n,
        "manuscript_eligible_triplet_n": manuscript_eligible_n,
        "main_text_eligible_triplet_n": main_text_n,
        "semantic_duplicate_triplet_candidates_excluded_n": int(len(semantic_audit)),
        "validation_failures": validation_failures,
        "validation_checks": validation_check.to_dict("records"),
        "outputs": outputs,
    }, TRIPLET_MANIFEST)
    print(f"Triplet extension finished. Triplets: {len(replay)}; main-text eligible: {main_text_n}")


if __name__ == "__main__":
    main()
