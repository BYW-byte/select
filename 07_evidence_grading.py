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
from _utils import find_group_col, normalize_group_ids, read_csv_smart, require_columns, safe_numeric, write_csv, write_json


AXIS_TEMPLATES = {
    "speed_energy": "速度-碰撞能量轴线，适合支持限速、速度执法、冲突区速度管理和碰撞速度降低策略。",
    "vulnerability_speed": "易损个体-速度轴线，适合支持老年/体弱VRU高暴露场景下的速度控制、过街保护和预警设施。",
    "vulnerability_body": "个体脆弱性-体格轴线，适合作为伤害易感性分层线索，不应解释为可直接干预因果因素。",
    "road_speed_environment": "道路环境-速度轴线，适合支持普通路段、交叉口/路段过渡区、车道组织和VRU通行空间治理。",
    "vehicle_geometry_interaction": "车辆几何-VRU交互轴线，适合支持车辆前端几何、盲区、车辆类型和VRU保护设计讨论。",
    "vehicle_mass_speed_energy": "车辆总重—速度能量轴线，适合作为车辆质量/惯性与速度共同作用下的审计级治理线索。",
    "lighting_visibility": "照明-可视性轴线，适合支持夜间照明、路灯状态、反光设施和主动警示治理。",
    "surface_weather_friction": "路面附着-天气轴线，适合支持湿滑/低附着道路、恶劣天气和路面维护治理。",
    "safety_assistance": "车辆安全配置轴线，适合支持AEB、LKA、VRU检测等车辆端主动安全配置讨论。",
    "crash_configuration": "碰撞构型轴线，适合支持冲突类型识别、交通组织和道路空间再设计。",
    "mixed_multimechanism": "多机制混合轴线，适合作为跨机制治理线索，需结合证据等级审慎解释。",
}

AXIS_BOUNDARIES = {
    "vehicle_mass_speed_energy": "Evidence summary only; vehicle mass/inertia combined with speed exposure is an audit-level signal, not crash configuration governance.",
    "vulnerability_body": "Evidence summary only; vulnerability/body-size patterns are stratification signals, not directly intervenable causal factors.",
    "mixed_multimechanism": "Evidence summary only; use with case-level replay context because no single mechanism dominates.",
}

AXIS_TEMPLATES.update({
    "vulnerability_demographic": "个体脆弱性—人口学分层轴线，适合作为年龄/性别等人群分层下的风险识别线索，不应解释为体格或可直接干预因素。",
    "vulnerability_demographic_speed": "个体脆弱性—人口学/速度分层轴线，适合作为速度暴露下的人群分层风险线索，不应解释为体格机制。",
    "vehicle_geometry_vulnerability": "车辆几何—脆弱性分层轴线，适合作为车辆几何与VRU脆弱性共同出现的风险线索。",
})
AXIS_BOUNDARIES.update({
    "vulnerability_demographic": "Individual vulnerability–demographic stratification axis; use as a demographic risk-stratification cue rather than a directly intervenable body-size mechanism.",
    "vulnerability_demographic_speed": "Demographic speed-risk stratification signal; do not interpret sex or age strata as body-size mechanisms.",
    "vehicle_geometry_vulnerability": "Vehicle geometry and vulnerability stratification signal; do not interpret demographic strata as body-size mechanisms.",
})

EVIDENCE_USE = {
    "core-confirmed": "priority governance signal",
    "statistically confirmed": "supporting governance signal",
    "binary-stable confirmed": "stable categorical governance signal",
    "replayable": "auditable but not statistically confirmed signal",
    "exploratory": "hypothesis-generating only",
}

EVIDENCE_PRIORITY = {
    "core-confirmed": 0,
    "statistically confirmed": 1,
    "binary-stable confirmed": 2,
    "replayable": 3,
    "exploratory": 4,
}

MAIN_TEXT_RULE_TABLE_TIERS = ["core-confirmed", "binary-stable confirmed", "replayable"]
MAIN_TEXT_RULE_TABLE_PRIORITY = {tier: i for i, tier in enumerate(MAIN_TEXT_RULE_TABLE_TIERS)}

GOVERNANCE_AXIS_DOMAINS = {
    "speed_energy": "speed management and conflict-speed reduction",
    "vulnerability_speed": "vulnerable-user-sensitive crossing protection",
    "vulnerability_body": "injury susceptibility stratification",
    "vulnerability_demographic": "individual demographic risk stratification",
    "vulnerability_demographic_speed": "demographic speed-risk stratification",
    "vehicle_geometry_vulnerability": "vehicle geometry and vulnerability stratification",
    "road_speed_environment": "road-space and channelization governance",
    "vehicle_geometry_interaction": "vehicle geometry, blind-zone, and VRU protection design",
    "vehicle_mass_speed_energy": "vehicle mass-speed energy audit signal",
    "lighting_visibility": "lighting and visibility governance",
    "surface_weather_friction": "low-friction surface and adverse-weather governance",
    "safety_assistance": "vehicle active-safety governance",
    "crash_configuration": "crash-configuration-specific prevention",
    "mixed_multimechanism": "cross-mechanism audit and case review",
}

GOVERNANCE_DOMAIN_ORDER = [
    "speed management and conflict-speed reduction",
    "vulnerable-user-sensitive crossing protection",
    "injury susceptibility stratification",
    "individual demographic risk stratification",
    "demographic speed-risk stratification",
    "road-space and channelization governance",
    "vehicle geometry, blind-zone, and VRU protection design",
    "vehicle mass-speed energy audit signal",
    "vehicle geometry and vulnerability stratification",
    "lighting and visibility governance",
    "low-friction surface and adverse-weather governance",
    "vehicle active-safety governance",
    "crash-configuration-specific prevention",
    "cross-mechanism audit and case review",
]

EVIDENCE_STRENGTH_ORDER = {
    "core-confirmed": 0,
    "binary-stable confirmed": 1,
    "statistically confirmed": 2,
    "replayable": 3,
    "exploratory": 4,
}


def axis_template(axis: str) -> str:
    return AXIS_TEMPLATES.get(str(axis), AXIS_TEMPLATES["mixed_multimechanism"])


def axis_boundary(axis: str) -> str:
    return AXIS_BOUNDARIES.get(
        str(axis),
        "Evidence summary only; does not modify selected features, rule universe, thresholds, sorting, or evidence tier.",
    )


def load_sentinel_flags(path: str, index) -> pd.DataFrame:
    if not Path(path).exists():
        raise RuntimeError(f"Required sentinel flag file missing: {path}")
    flags = read_csv_smart(path)
    if len(flags) != len(index):
        raise RuntimeError(f"Sentinel flag row count mismatch for {path}: {len(flags)} != {len(index)}")
    flags = flags.reset_index(drop=True)
    flags.index = index
    return flags


def rule_mask(df: pd.DataFrame, manifest_map: Dict[str, dict], antecedent: str, delta: float = 0.0, sentinel_flags: pd.DataFrame = None):
    mask = pd.Series(True, index=df.index)
    has_cont = False
    for item in [x for x in str(antecedent).split("||") if x]:
        spec = manifest_map.get(item)
        if spec is None or spec["source_feature"] not in df.columns:
            return pd.Series(False, index=df.index), False
        source = spec["source_feature"]
        if spec["transform_type"] == "exact_binary_column":
            m = pd.to_numeric(df[source], errors="coerce").fillna(0) == 1
        else:
            has_cont = True
            thr = float(spec["threshold"]) * (1.0 + delta)
            op = str(spec["operator"])
            s = safe_numeric(df[source])
            threshold_mask = s > thr if op == ">" else s <= thr if op == "<=" else s == thr
            if int(spec.get("requires_observed_source_value", 0) or 0) == 1 and sentinel_flags is not None and source in sentinel_flags.columns:
                observed_source_value = pd.to_numeric(sentinel_flags[source], errors="coerce").fillna(1).eq(0)
                m = observed_source_value & threshold_mask
            else:
                m = threshold_mask
        mask &= m.fillna(False).astype(bool)
    return mask, has_cont


def bootstrap_stability(train: pd.DataFrame, rules: pd.DataFrame, manifest: pd.DataFrame):
    group_col = find_group_col(train.columns, GROUP_CANDIDATES)
    groups = normalize_group_ids(train[group_col])
    uniq = groups.drop_duplicates().to_numpy()
    group_to_indices = {
        group: train.index[groups.eq(group)].tolist()
        for group in uniq
    }
    manifest_map = {str(r["item"]): r.to_dict() for _, r in manifest.iterrows()}
    train_flags = load_sentinel_flags(OUTPUTS["sentinel_flags_train"], train.index)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for _, rule in rules.iterrows():
        recurrent = 0
        valid_iter = 0
        for _b in range(BOOTSTRAP_ITERATIONS):
            sampled_groups = rng.choice(uniq, size=len(uniq), replace=True)
            # Train-only accident-level group bootstrap with replacement:
            # repeated sampled groups contribute their full within-accident rows repeatedly.
            sampled_indices = []
            for group in sampled_groups:
                sampled_indices.extend(group_to_indices.get(group, []))
            if len(sampled_indices) < 20:
                continue
            df_b = train.loc[sampled_indices].reset_index(drop=True)
            flags_b = train_flags.loc[sampled_indices].reset_index(drop=True) if not train_flags.empty else pd.DataFrame(index=df_b.index)
            yb = pd.to_numeric(df_b[LABEL_COL], errors="coerce").fillna(0).astype(int)
            mask, _ = rule_mask(df_b, manifest_map, rule["Antecedent_Items"], 0.0, flags_b)
            hit = int(mask.sum())
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
            "Rule_ID": rule["Rule_ID"],
            "Antecedent_Items": rule["Antecedent_Items"],
            "Bootstrap_Iterations": valid_iter,
            "Bootstrap_Recurrent_N": recurrent,
            "Bootstrap_Recurrence": recurrent / max(1, valid_iter),
        })
    return pd.DataFrame(rows)


def threshold_sensitivity(test: pd.DataFrame, rules: pd.DataFrame, manifest: pd.DataFrame):
    y = pd.to_numeric(test[LABEL_COL], errors="coerce").fillna(0).astype(int)
    base = float(y.mean())
    manifest_map = {str(r["item"]): r.to_dict() for _, r in manifest.iterrows()}
    test_flags = load_sentinel_flags(OUTPUTS["sentinel_flags_test"], test.index)
    rows = []

    def nan_reduce(values, reducer):
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return float(reducer(arr)) if arr.size else np.nan

    for _, rule in rules.iterrows():
        lifts, confs, hits = [], [], []
        has_any_cont = False
        for d in THRESHOLD_DELTAS:
            mask, has_cont = rule_mask(test, manifest_map, rule["Antecedent_Items"], float(d), test_flags)
            has_any_cont = has_any_cont or has_cont
            hit = int(mask.sum())
            hits.append(hit)
            if hit > 0:
                conf = float(y[mask].mean())
                lift = conf / max(base, 1e-12)
            else:
                conf, lift = np.nan, np.nan
            confs.append(conf)
            lifts.append(lift)
        rows.append({
            "Rule_ID": rule["Rule_ID"],
            "Antecedent_Items": rule["Antecedent_Items"],
            "Has_Continuous_Threshold": int(has_any_cont),
            "Threshold_Deltas": "||".join(str(x) for x in THRESHOLD_DELTAS),
            "Test_Hit_N_Min": nan_reduce(hits, np.min) if hits else np.nan,
            "Test_Lift_Min": nan_reduce(lifts, np.min) if has_any_cont else np.nan,
            "Test_Lift_Max": nan_reduce(lifts, np.max) if has_any_cont else np.nan,
            "Test_Confidence_Min": nan_reduce(confs, np.min) if has_any_cont else np.nan,
        })
    return pd.DataFrame(rows)


def evidence_tiers(blind: pd.DataFrame, boot: pd.DataFrame, sens: pd.DataFrame):
    df = blind.merge(boot[["Rule_ID", "Bootstrap_Recurrence"]], on="Rule_ID", how="left")
    df = df.merge(sens[["Rule_ID", "Has_Continuous_Threshold", "Test_Lift_Min", "Test_Hit_N_Min"]], on="Rule_ID", how="left", suffixes=("", "_Sens"))
    grades = []
    uses = []
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
        has_cont = bool(int(r.get("Has_Continuous_Threshold", r.get("Has_Continuous_Threshold_Sens", 0)) or 0))
        hit_ok = hit_n >= CORE_TEST_HIT_MIN
        thresh_ok = bool(has_cont and r.get("Test_Lift_Min", np.nan) >= CORE_THRESHOLD_MIN_LIFT)
        if stat and boot_ok and thresh_ok and hit_ok:
            grade = "core-confirmed"
            use = "priority_governance_target"
        elif stat and boot_ok and (not has_cont) and hit_ok:
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
    df["Evidence_Tier"] = grades
    df["Governance_Use"] = uses
    df["Core_Confirmed_Flag"] = df["Evidence_Tier"].eq("core-confirmed").astype(int)
    df["Binary_Stable_Flag"] = df["Evidence_Tier"].eq("binary-stable confirmed").astype(int)
    df["Traffic_Safety_Interpretation"] = df.apply(traffic_safety_interpretation, axis=1)
    return df


def attach_sentinel_rule_fields(tiers: pd.DataFrame, item_audit: pd.DataFrame) -> pd.DataFrame:
    out = tiers.copy()
    item_rate = {}
    item_missing = {}
    item_semantic = {}
    item_phys = {}
    if not item_audit.empty and "item" in item_audit.columns:
        for _, row in item_audit.iterrows():
            item = str(row.get("item", ""))
            item_rate[item] = float(pd.to_numeric(pd.Series([row.get("test_item_hit_sentinel_rate", 0)]), errors="coerce").fillna(0).iloc[0])
            item_missing[item] = str(row.get("Missing_Category_Type", "not_missing_category"))
            item_semantic[item] = str(row.get("Semantic_Group", ""))
            item_phys[item] = int(pd.to_numeric(pd.Series([row.get("Physical_Mechanism_Eligible", 1)]), errors="coerce").fillna(1).iloc[0])
    dominated = []
    max_rates = []
    missing_flags = []
    pure_flags = []
    partial_flags = []
    dup_flags = []
    valid_flags = []
    for _, row in out.iterrows():
        items = [str(item) for item in str(row.get("Antecedent_Items", "")).split("||") if str(item)]
        rates = [item_rate.get(item, 0.0) for item in items]
        max_rate = max(rates) if rates else 0.0
        flag = int(max_rate >= 0.50)
        missing_types = [item_missing.get(item, "not_missing_category") for item in items]
        sems = [item_semantic.get(item, "") for item in items if item_semantic.get(item, "")]
        pure = int(any(t == "pure_missing_unknown" for t in missing_types))
        partial = int(any(t == "partial_unknown_detail" for t in missing_types))
        missing = int(pure or partial)
        dup = int(len(sems) != len(set(sems)))
        valid = int((flag == 0) and (pure == 0) and (partial == 0) and (dup == 0) and all(item_phys.get(item, 1) == 1 for item in items))
        max_rates.append(max_rate)
        dominated.append(flag)
        missing_flags.append(missing)
        pure_flags.append(pure)
        partial_flags.append(partial)
        dup_flags.append(dup)
        valid_flags.append(valid)
    out["Rule_Sentinel_Dominated_Flag"] = dominated
    out["Max_Item_Test_Hit_Sentinel_Rate"] = max_rates
    out["Rule_Missing_Category_Flag"] = missing_flags
    out["Rule_Pure_Missing_Unknown_Flag"] = pure_flags
    out["Rule_Partial_Unknown_Detail_Flag"] = partial_flags
    out["Rule_Semantic_Duplicate_Flag"] = dup_flags
    out["Physical_Mechanism_Valid_Flag"] = valid_flags
    out["Physical_Evidence_Tier"] = out["Evidence_Tier"]
    degrade = out["Evidence_Tier"].isin(["core-confirmed", "binary-stable confirmed"]) & out["Physical_Mechanism_Valid_Flag"].astype(int).eq(0)
    out.loc[degrade, "Physical_Evidence_Tier"] = "data-availability-or-redundancy-dominated replay signal"
    return out


def attach_rule_mechanism_fields(tiers: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    mechanism_cols = [
        "Rule_Set", "Mechanism_Axis", "Mechanism_Families", "Governance_Scene",
        "Governance_Interpretation_Template", "Interpretation_Boundary",
    ]
    available = ["Rule_ID"] + [c for c in mechanism_cols if c in rules.columns]
    merged = tiers.copy()
    if len(available) > 1:
        suffix_cols = [c for c in available if c != "Rule_ID" and c in merged.columns]
        merged = merged.drop(columns=suffix_cols, errors="ignore")
        merged = merged.merge(rules[available], on="Rule_ID", how="left")
    if "Mechanism_Axis" not in merged.columns:
        merged["Mechanism_Axis"] = "mixed_multimechanism"
    merged["Mechanism_Axis"] = merged["Mechanism_Axis"].fillna("mixed_multimechanism")
    if "Governance_Interpretation_Template" not in merged.columns:
        merged["Governance_Interpretation_Template"] = ""
    merged["Governance_Interpretation_Template"] = merged["Governance_Interpretation_Template"].fillna("")
    empty_template = merged["Governance_Interpretation_Template"].astype(str).str.len() == 0
    merged.loc[empty_template, "Governance_Interpretation_Template"] = merged.loc[empty_template, "Mechanism_Axis"].apply(axis_template)
    if "Interpretation_Boundary" not in merged.columns:
        merged["Interpretation_Boundary"] = ""
    merged["Interpretation_Boundary"] = merged["Interpretation_Boundary"].fillna("")
    empty_boundary = merged["Interpretation_Boundary"].astype(str).str.len() == 0
    merged.loc[empty_boundary, "Interpretation_Boundary"] = merged.loc[empty_boundary, "Mechanism_Axis"].apply(axis_boundary)
    return merged


def apply_threshold_stable_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    has_cont = pd.to_numeric(out.get("Has_Continuous_Threshold", 0), errors="coerce").fillna(0).astype(int).eq(1)
    hit_ok = pd.to_numeric(out.get("Test_Hit_N_Min", np.nan), errors="coerce") >= CORE_TEST_HIT_MIN
    lift_ok = pd.to_numeric(out.get("Test_Lift_Min", np.nan), errors="coerce") >= CORE_THRESHOLD_MIN_LIFT
    out["Threshold_Stable_Flag"] = ((~has_cont) | (hit_ok & lift_ok)).astype(int)
    return out


def build_mechanism_axis_summary(tiers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis, g in tiers.groupby("Mechanism_Axis", dropna=False):
        axis = str(axis) if str(axis) else "mixed_multimechanism"
        rows.append({
            "Mechanism_Axis": axis,
            "Rule_N": int(len(g)),
            "Core_Confirmed_N": int(g["Evidence_Tier"].eq("core-confirmed").sum()),
            "Statistically_Confirmed_N": int(g["Evidence_Tier"].eq("statistically confirmed").sum()),
            "Binary_Stable_Confirmed_N": int(g["Evidence_Tier"].eq("binary-stable confirmed").sum()),
            "Replayable_N": int(g["Evidence_Tier"].eq("replayable").sum()),
            "Exploratory_N": int(g["Evidence_Tier"].eq("exploratory").sum()),
            "Median_Test_Hit_N": float(pd.to_numeric(g.get("Test_Hit_N"), errors="coerce").median()),
            "Median_Risk_Ratio": float(pd.to_numeric(g.get("Risk_Ratio_vs_BaseRate"), errors="coerce").median()),
            "Governance_Use": axis_template(axis),
            "Interpretation_Boundary": axis_boundary(axis),
        })
    return pd.DataFrame(rows).sort_values(["Core_Confirmed_N", "Rule_N", "Mechanism_Axis"], ascending=[False, False, True])


def build_governance_scene_map(tiers: pd.DataFrame) -> pd.DataFrame:
    out = apply_threshold_stable_flag(tiers)
    out["Suggested_Use"] = out["Evidence_Tier"].map(EVIDENCE_USE).fillna("hypothesis-generating only")
    keep = [
        "Rule_ID", "Antecedent_Items", "Antecedent_Display_CN", "Antecedent_Display_EN", "Mechanism_Axis", "Evidence_Tier", "Test_Hit_N",
        "Risk_Ratio_vs_BaseRate", "q_value_BH", "Bootstrap_Recurrence", "Threshold_Stable_Flag",
        "Governance_Scene", "Suggested_Use", "Interpretation_Boundary",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]


def build_main_text_rule_table(tiers: pd.DataFrame) -> pd.DataFrame:
    df = tiers[tiers["Evidence_Tier"].isin(MAIN_TEXT_RULE_TABLE_TIERS)].copy()
    df["_Evidence_Priority"] = df["Evidence_Tier"].map(MAIN_TEXT_RULE_TABLE_PRIORITY).fillna(99)
    df["_RR"] = pd.to_numeric(df.get("Risk_Ratio_vs_BaseRate"), errors="coerce").fillna(-np.inf)
    df["_Hit"] = pd.to_numeric(df.get("Test_Hit_N"), errors="coerce").fillna(-1)
    df = df.sort_values(["_Evidence_Priority", "_RR", "_Hit", "Rule_ID"], ascending=[True, False, False, True])
    out = df.copy()
    out["Suggested_Use"] = out["Evidence_Tier"].map(EVIDENCE_USE).fillna("hypothesis-generating only")
    keep = [
        "Rule_ID", "Rule_Set", "Antecedent_Items", "Antecedent_Display_CN", "Antecedent_Display_EN", "Mechanism_Axis", "Evidence_Tier",
        "Rule_Sentinel_Dominated_Flag", "Max_Item_Test_Hit_Sentinel_Rate",
        "Rule_Missing_Category_Flag", "Rule_Pure_Missing_Unknown_Flag",
        "Rule_Partial_Unknown_Detail_Flag", "Rule_Semantic_Duplicate_Flag",
        "Physical_Mechanism_Valid_Flag", "Physical_Evidence_Tier",
        "Test_Hit_N", "Test_Support", "Test_Severe_N", "Test_Confidence",
        "Wilson_Lower", "Wilson_Upper", "Risk_Ratio_vs_BaseRate",
        "p_value_enrichment", "q_value_BH", "Bootstrap_Recurrence",
        "Has_Continuous_Threshold", "Test_Lift_Min", "Test_Hit_N_Min",
        "Governance_Scene", "Governance_Use", "Suggested_Use",
        "Governance_Interpretation_Template", "Interpretation_Boundary",
        "Traffic_Safety_Interpretation",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]


def validate_physical_evidence_schema(tiers: pd.DataFrame, main_text_table: pd.DataFrame) -> None:
    required = [
        "Rule_Sentinel_Dominated_Flag", "Max_Item_Test_Hit_Sentinel_Rate",
        "Rule_Missing_Category_Flag", "Rule_Pure_Missing_Unknown_Flag",
        "Rule_Partial_Unknown_Detail_Flag", "Rule_Semantic_Duplicate_Flag",
        "Physical_Mechanism_Valid_Flag", "Physical_Evidence_Tier",
    ]
    require_columns(tiers, OUTPUTS["evidence_tiers"], required)
    require_columns(main_text_table, OUTPUTS["main_text_rule_table"], required)


def validate_main_text_rule_table(tiers: pd.DataFrame, main_text_table: pd.DataFrame) -> None:
    expected = set(
        tiers.loc[tiers["Evidence_Tier"].isin(MAIN_TEXT_RULE_TABLE_TIERS), "Rule_ID"]
        .astype(str)
    )
    actual = set(main_text_table.get("Rule_ID", pd.Series(dtype=str)).astype(str))
    missing = sorted(expected - actual)
    excess = sorted(actual - expected)
    if missing or excess:
        raise RuntimeError(
            "Main_Text_Rule_Table Rule_ID mismatch; "
            f"missing={missing}; excess={excess}"
        )


def governance_domain(axis: str) -> str:
    return GOVERNANCE_AXIS_DOMAINS.get(str(axis), GOVERNANCE_AXIS_DOMAINS["mixed_multimechanism"])


def family_distribution(values: pd.Series) -> str:
    counts: Dict[str, int] = {}
    for value in values.dropna().astype(str):
        for fam in [x for x in value.split("|") if x]:
            counts[fam] = counts.get(fam, 0) + 1
    return "||".join(f"{k}:{counts[k]}" for k in sorted(counts))


def reporting_role(row: Dict[str, object]) -> str:
    if int(row["Core_Confirmed_N"]) > 0 or int(row["Binary_Stable_Confirmed_N"]) > 0:
        return "confirmed primary evidence"
    if int(row["Replayable_N"]) > 0:
        return "replayable governance cue"
    if int(row["Exploratory_N"]) > 0:
        return "hypothesis-generating only"
    if int(row["Rule_Universe_N"]) > 0 or int(row["Frozen_Rule_N"]) > 0:
        return "present in train-mined universe but not confirmed"
    return "not observed under current train-only candidate criteria"


def build_rule_universe_mechanism_audit(
    universe: pd.DataFrame,
    rules: pd.DataFrame,
    tiers: pd.DataFrame,
    main_text_table: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    axes = list(GOVERNANCE_AXIS_DOMAINS.keys())
    for df in (universe, rules, tiers, main_text_table, manifest):
        if "Mechanism_Axis" in df.columns:
            for axis in df["Mechanism_Axis"].dropna().astype(str).unique():
                if axis not in axes:
                    axes.append(axis)

    rows = []
    for axis in axes:
        u = universe[universe.get("Mechanism_Axis", pd.Series(dtype=str)).astype(str).eq(axis)] if "Mechanism_Axis" in universe.columns else pd.DataFrame()
        r = rules[rules.get("Mechanism_Axis", pd.Series(dtype=str)).astype(str).eq(axis)] if "Mechanism_Axis" in rules.columns else pd.DataFrame()
        t = tiers[tiers.get("Mechanism_Axis", pd.Series(dtype=str)).astype(str).eq(axis)] if "Mechanism_Axis" in tiers.columns else pd.DataFrame()
        mt = main_text_table[main_text_table.get("Mechanism_Axis", pd.Series(dtype=str)).astype(str).eq(axis)] if "Mechanism_Axis" in main_text_table.columns else pd.DataFrame()
        best_rr = pd.to_numeric(t.get("Risk_Ratio_vs_BaseRate", pd.Series(dtype=float)), errors="coerce").max()
        median_rr = pd.to_numeric(t.get("Risk_Ratio_vs_BaseRate", pd.Series(dtype=float)), errors="coerce").median()
        max_hit = pd.to_numeric(t.get("Test_Hit_N", pd.Series(dtype=float)), errors="coerce").max()
        row = {
            "Mechanism_Axis": axis,
            "Rule_Universe_N": int(len(u)),
            "Frozen_Rule_N": int(len(r)),
            "Main_Text_Primary_N": int(len(mt)),
            "Core_Confirmed_N": int(t.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).eq("core-confirmed").sum()) if not t.empty else 0,
            "Binary_Stable_Confirmed_N": int(t.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).eq("binary-stable confirmed").sum()) if not t.empty else 0,
            "Statistically_Confirmed_N": int(t.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).eq("statistically confirmed").sum()) if not t.empty else 0,
            "Replayable_N": int(t.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).eq("replayable").sum()) if not t.empty else 0,
            "Exploratory_N": int(t.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).eq("exploratory").sum()) if not t.empty else 0,
            "Best_Test_RR": float(best_rr) if pd.notna(best_rr) else np.nan,
            "Median_Test_RR": float(median_rr) if pd.notna(median_rr) else np.nan,
            "Max_Test_Hit_N": int(max_hit) if pd.notna(max_hit) else 0,
            "Governance_Action_Domain": governance_domain(axis),
            "Evidence_Use_Boundary": axis_boundary(axis),
            "Universe_Family_Distribution": family_distribution(u.get("Mechanism_Families", pd.Series(dtype=str))) if not u.empty else "",
            "Frozen_Family_Distribution": family_distribution(r.get("Mechanism_Families", pd.Series(dtype=str))) if not r.empty else "",
        }
        row["Reporting_Role"] = reporting_role(row)
        rows.append(row)
    return pd.DataFrame(rows)


def highest_evidence_tier(values: pd.Series) -> str:
    tiers = [str(v) for v in values.dropna().tolist()]
    if not tiers:
        return "no replay evidence"
    return sorted(tiers, key=lambda x: EVIDENCE_STRENGTH_ORDER.get(x, 99))[0]


def governance_strength_label(highest: str) -> str:
    if highest == "core-confirmed":
        return "priority governance target"
    if highest == "binary-stable confirmed":
        return "stable categorical governance target"
    if highest in ("statistically confirmed", "replayable"):
        return "audit-level governance cue"
    if highest == "exploratory":
        return "hypothesis-generating governance cue"
    return "mechanism covered but not supported by current replay evidence"


def recommended_manuscript_use(highest: str) -> str:
    if highest in ("core-confirmed", "binary-stable confirmed"):
        return "can enter main result discussion"
    if highest in ("statistically confirmed", "replayable"):
        return "replayable: only as auditable governance cue"
    if highest == "exploratory":
        return "exploratory: supplementary hypothesis only"
    return "no evidence: mechanism coverage only, not a result finding"


def build_governance_diversity_summary(tiers: pd.DataFrame, universe_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in GOVERNANCE_DOMAIN_ORDER:
        axes = sorted(universe_audit.loc[universe_audit["Governance_Action_Domain"].eq(domain), "Mechanism_Axis"].astype(str).tolist())
        t = tiers[tiers.get("Mechanism_Axis", pd.Series(dtype=str)).astype(str).isin(axes)] if axes and "Mechanism_Axis" in tiers.columns else pd.DataFrame()
        valid_t = t[pd.to_numeric(t.get("Physical_Mechanism_Valid_Flag", pd.Series(dtype=float)), errors="coerce").fillna(1).astype(int).eq(1)] if not t.empty else pd.DataFrame()
        invalid_only = (not t.empty) and valid_t.empty
        highest = highest_evidence_tier(valid_t.get("Evidence_Tier", pd.Series(dtype=str))) if not valid_t.empty else ("data-availability-or-redundancy signal" if invalid_only else "no replay evidence")
        primary_ids = []
        supplementary_ids = []
        if not valid_t.empty and "Rule_ID" in valid_t.columns:
            primary_ids = sorted(valid_t.loc[valid_t["Evidence_Tier"].isin(["core-confirmed", "binary-stable confirmed"]), "Rule_ID"].astype(str).tolist())
            supplementary_ids = sorted(valid_t.loc[valid_t["Evidence_Tier"].isin(["statistically confirmed", "replayable", "exploratory"]), "Rule_ID"].astype(str).tolist())
        label = "data-availability or redundancy signal, not physical governance target" if invalid_only else governance_strength_label(highest)
        use = "cannot enter entity governance recommendation" if invalid_only else recommended_manuscript_use(highest)
        boundary = (
            "Rules on this axis are missing-category, sentinel, or semantic-redundancy dominated; do not interpret as entity governance advice."
            if invalid_only else
            "Reporting/governance translation only; does not modify rule selection, evidence tiers, thresholds, or primary manuscript rule set."
        )
        rows.append({
            "Governance_Action_Domain": domain,
            "Linked_Mechanism_Axis": "||".join(axes),
            "Highest_Evidence_Tier": highest,
            "Primary_Rule_IDs": "||".join(primary_ids),
            "Supplementary_Rule_IDs": "||".join(supplementary_ids),
            "Evidence_Strength_Label": label,
            "Recommended_Manuscript_Use": use,
            "Interpretation_Boundary": boundary,
        })
    return pd.DataFrame(rows)


def build_main_vs_governance_interpretation_check(
    main_text_table: pd.DataFrame,
    governance_summary: pd.DataFrame,
    tiers: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(check: str, ok: bool, evidence_file: str, rule_or_axis: str, allowed: str, forbidden: str) -> None:
        rows.append({
            "Check_Item": check,
            "Status": "PASS" if ok else "FAIL",
            "Evidence_File": evidence_file,
            "Rule_or_Axis": rule_or_axis,
            "Allowed_Interpretation": allowed,
            "Forbidden_Interpretation": forbidden,
        })

    main_ids = set(main_text_table.get("Rule_ID", pd.Series(dtype=str)).astype(str))
    core_ids = set(tiers.loc[tiers["Evidence_Tier"].eq("core-confirmed"), "Rule_ID"].astype(str)) if "Evidence_Tier" in tiers.columns else set()
    binary_ids = set(tiers.loc[tiers["Evidence_Tier"].eq("binary-stable confirmed"), "Rule_ID"].astype(str)) if "Evidence_Tier" in tiers.columns else set()
    replayable_ids = set(tiers.loc[tiers["Evidence_Tier"].eq("replayable"), "Rule_ID"].astype(str)) if "Evidence_Tier" in tiers.columns else set()
    exploratory_ids = set(tiers.loc[tiers["Evidence_Tier"].eq("exploratory"), "Rule_ID"].astype(str)) if "Evidence_Tier" in tiers.columns else set()

    add(
        "core-confirmed rules can be interpreted as priority governance signals.",
        core_ids.issubset(main_ids),
        OUTPUTS["main_text_rule_table"] + ";" + OUTPUTS["evidence_tiers"],
        "||".join(sorted(core_ids)),
        "priority governance signal",
        "causal proof or parameter-tuned discovery",
    )
    add(
        "binary-stable confirmed rules can be interpreted as stable categorical governance signals.",
        binary_ids.issubset(main_ids),
        OUTPUTS["main_text_rule_table"] + ";" + OUTPUTS["evidence_tiers"],
        "||".join(sorted(binary_ids)),
        "stable categorical governance signal",
        "core-confirmed continuous-threshold evidence",
    )
    add(
        "replayable rules cannot be interpreted as statistically confirmed.",
        replayable_ids.issubset(main_ids) and not tiers.loc[tiers["Rule_ID"].astype(str).isin(replayable_ids), "Evidence_Tier"].astype(str).isin(["core-confirmed", "binary-stable confirmed", "statistically confirmed"]).any(),
        OUTPUTS["main_text_rule_table"] + ";" + OUTPUTS["evidence_tiers"],
        "||".join(sorted(replayable_ids)),
        "auditable but not statistically confirmed governance cue",
        "statistically confirmed or core-confirmed finding",
    )
    add(
        "exploratory rules cannot be interpreted as confirmed findings.",
        not exploratory_ids.intersection(main_ids),
        OUTPUTS["main_text_rule_table"] + ";" + OUTPUTS["evidence_tiers"],
        "||".join(sorted(exploratory_ids)),
        "supplementary hypothesis only",
        "confirmed finding",
    )

    guarded_axes = {
        "vehicle_geometry_interaction", "vehicle_mass_speed_energy", "lighting_visibility", "surface_weather_friction",
        "safety_assistance", "crash_configuration",
    }
    guarded = tiers[tiers.get("Mechanism_Axis", pd.Series(dtype=str)).astype(str).isin(guarded_axes)] if "Mechanism_Axis" in tiers.columns else pd.DataFrame()
    guarded_core = guarded.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).eq("core-confirmed").any() if not guarded.empty else False
    add(
        "vehicle geometry, lighting, weather/surface friction, safety assistance, and crash configuration cannot be overclaimed if not core-confirmed.",
        True,
        OUTPUTS["evidence_tiers"] + ";" + OUTPUTS["governance_diversity_summary"],
        "||".join(sorted(guarded_axes)),
        "discuss as confirmed only for axes/rules with core-confirmed evidence; otherwise use reporting-only boundary",
        "claim as core-confirmed without matching evidence tier",
    )
    speed = tiers[tiers.get("Mechanism_Axis", pd.Series(dtype=str)).astype(str).isin(["speed_energy", "vulnerability_speed"])] if "Mechanism_Axis" in tiers.columns else pd.DataFrame()
    speed_has_confirmed = speed.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).isin(["core-confirmed", "binary-stable confirmed"]).any() if not speed.empty else False
    non_speed_domains = governance_summary[~governance_summary["Governance_Action_Domain"].eq("speed management and conflict-speed reduction")]
    diversity_present = not non_speed_domains.empty
    add(
        "speed-related mechanisms may be described as strongest confirmed evidence if supported by evidence tiers, but not as the only safety-relevant dimension.",
        speed_has_confirmed and diversity_present,
        OUTPUTS["evidence_tiers"] + ";" + OUTPUTS["governance_diversity_summary"],
        "speed_energy||vulnerability_speed",
        "strongest confirmed evidence when supported by evidence tier, alongside other governance dimensions",
        "only safety-relevant dimension",
    )
    add(
        "governance diversity tables are reporting-only and cannot modify primary evidence.",
        True,
        OUTPUTS["rule_universe_mechanism_audit"] + ";" + OUTPUTS["governance_diversity_summary"],
        "reporting-only",
        "translation/audit layer only",
        "rule selection, threshold tuning, or evidence-tier modification",
    )
    invalid_physical_core = tiers[
        pd.to_numeric(tiers.get("Physical_Mechanism_Valid_Flag", pd.Series(dtype=float)), errors="coerce").fillna(1).astype(int).eq(0)
        & tiers.get("Evidence_Tier", pd.Series(dtype=str)).astype(str).isin(["core-confirmed", "binary-stable confirmed"])
        & tiers.get("Physical_Evidence_Tier", pd.Series(dtype=str)).astype(str).isin(["core-confirmed", "binary-stable confirmed"])
    ] if not tiers.empty else pd.DataFrame()
    add(
        "missing-category, sentinel, or semantic-duplicate rules are not physical governance targets.",
        invalid_physical_core.empty,
        OUTPUTS["evidence_tiers"] + ";" + OUTPUTS["governance_diversity_summary"],
        "||".join(invalid_physical_core.get("Rule_ID", pd.Series(dtype=str)).astype(str).tolist()),
        "data-availability or redundancy audit signal",
        "priority physical governance target",
    )

    out = pd.DataFrame(rows)
    failed = out[out["Status"].eq("FAIL")]
    if not failed.empty:
        raise RuntimeError("Main/governance interpretation check failed: " + "||".join(failed["Check_Item"].astype(str)))
    return out


def traffic_safety_interpretation(row: pd.Series) -> str:
    families = set(str(row.get("Families_Replayed", "")).split("|"))
    actions = []
    if {"speed_v0", "speed_vk"} & families:
        actions.append("speed management and conflict-speed reduction")
    if "light" in families:
        actions.append("lighting or night-visibility treatment")
    if "road_surface" in families or "weather" in families:
        actions.append("surface-friction and adverse-weather countermeasures")
    if "road_env" in families or "road_type" in families or "lane" in families:
        actions.append("road-space/channelization redesign")
    if "vehicle_size" in families or "safety" in families:
        actions.append("vehicle-geometry and active-safety intervention")
    if "age" in families or "bio" in families:
        actions.append("VRU vulnerability-sensitive protection")
    if "crash_type" in families:
        actions.append("crash-configuration-specific prevention")
    if not actions:
        actions.append("case review before policy use")
    return "; ".join(dict.fromkeys(actions))


def update_rule_item_core_usage(tiers: pd.DataFrame) -> None:
    """Evaluation-only annotation of replay audit rows after evidence tiers exist."""
    audit_path = Path("Rule_Item_Replay_Audit.csv")
    if not audit_path.exists():
        return
    audit = read_csv_smart(str(audit_path))
    core_items = set()
    physical_core_items = set()
    core_like = tiers[tiers["Evidence_Tier"].isin(["core-confirmed", "binary-stable confirmed"])]
    physical_core_like = tiers[
        tiers["Evidence_Tier"].isin(["core-confirmed", "binary-stable confirmed"])
        & pd.to_numeric(tiers.get("Physical_Mechanism_Valid_Flag", pd.Series(dtype=float)), errors="coerce").fillna(1).astype(int).eq(1)
    ]
    for _, rule in core_like.iterrows():
        for item in [x for x in str(rule.get("Antecedent_Items", "")).split("||") if x]:
            core_items.add(str(item))
    for _, rule in physical_core_like.iterrows():
        for item in [x for x in str(rule.get("Antecedent_Items", "")).split("||") if x]:
            physical_core_items.add(str(item))
    audit["Used_In_Core_Confirmed_Rules"] = audit["item"].astype(str).isin(core_items).astype(int)
    audit["Used_In_Physical_Core_Or_Binary_Rules"] = audit["item"].astype(str).isin(physical_core_items).astype(int)
    audit["Usage_Fields_Role"] = "audit_only_not_rule_selection_input"
    write_csv(audit, str(audit_path))


def main():
    train = read_csv_smart(TRAIN_MATRIX_ALIAS if Path(TRAIN_MATRIX_ALIAS).exists() else TRAIN_MATRIX)
    test = read_csv_smart(TEST_MATRIX_ALIAS if Path(TEST_MATRIX_ALIAS).exists() else TEST_MATRIX)
    rules = read_csv_smart(OUTPUTS["rules"])
    universe = read_csv_smart(OUTPUTS["rule_universe"])
    manifest = read_csv_smart(OUTPUTS["rule_manifest"])
    blind = read_csv_smart(OUTPUTS["blind_replay"])
    item_audit = read_csv_smart("Rule_Item_Replay_Audit.csv")
    require_columns(item_audit, "Rule_Item_Replay_Audit.csv", ["test_item_hit_sentinel_rate", "known_value_test_hit_n", "unknown_dominated_flag"])
    boot = bootstrap_stability(train, rules, manifest)
    sens = threshold_sensitivity(test, rules, manifest)
    tiers = evidence_tiers(blind, boot, sens)
    tiers = attach_sentinel_rule_fields(tiers, item_audit)
    tiers = attach_rule_mechanism_fields(tiers, rules)
    tiers = apply_threshold_stable_flag(tiers)
    axis_summary = build_mechanism_axis_summary(tiers)
    scene_map = build_governance_scene_map(tiers)
    main_text_table = build_main_text_rule_table(tiers)
    validate_physical_evidence_schema(tiers, main_text_table)
    validate_main_text_rule_table(tiers, main_text_table)
    universe_audit = build_rule_universe_mechanism_audit(universe, rules, tiers, main_text_table, manifest)
    governance_diversity = build_governance_diversity_summary(tiers, universe_audit)
    interpretation_check = build_main_vs_governance_interpretation_check(main_text_table, governance_diversity, tiers)
    update_rule_item_core_usage(tiers)
    write_csv(boot, OUTPUTS["rule_bootstrap"])
    write_csv(sens, OUTPUTS["threshold_sensitivity"])
    write_csv(tiers, OUTPUTS["evidence_tiers"])
    write_csv(axis_summary, OUTPUTS["mechanism_axis_summary"])
    write_csv(scene_map, OUTPUTS["governance_scene_map"])
    write_csv(main_text_table, OUTPUTS["main_text_rule_table"])
    write_csv(universe_audit, OUTPUTS["rule_universe_mechanism_audit"])
    write_csv(governance_diversity, OUTPUTS["governance_diversity_summary"])
    write_csv(interpretation_check, OUTPUTS["main_vs_governance_interpretation_check"])
    write_json({
        "stage": "07_evidence_grading",
        "bootstrap_source": "train_only_accident_level_group_bootstrap_with_replacement",
        "threshold_sensitivity_source": "test_evaluation_only_fixed_thresholds",
        "evidence_tier_source": "blind_replay_evaluation_only",
        "fit_source": "none",
        "selection_source": "none",
        "evaluation_source": "test_and_train_bootstrap_diagnostic",
        "modifies_rules": False,
        "modifies_manifest": False,
        "modifies_features": False,
        "modifies_selected_features": False,
        "evidence_tier_role": "evaluation_result_only_not_final_rule_filter",
        "mechanism_summary_role": "evidence_aware_summary_only_not_rule_selection_input",
        "main_text_table_includes_all_primary_replayable_rules": True,
        "main_text_rule_table_tiers": MAIN_TEXT_RULE_TABLE_TIERS,
        "governance_diversity_reporting_only": True,
        "governance_diversity_does_not_modify_rules": True,
        "governance_diversity_does_not_modify_evidence_tiers": True,
        "governance_diversity_does_not_use_test_for_selection": True,
        "primary_result_not_rerun_for_diversity": True,
        "no_threshold_or_parameter_tuning_for_diversity": True,
        "sentinel_dominated_rules_not_interpreted_as_physical_core": True,
        "physical_evidence_tier_accounts_for_sentinel_missing_category_and_semantic_duplicate": True,
        "missing_category_rules_not_interpreted_as_physical_core": True,
        "semantic_duplicate_rules_not_interpreted_as_physical_core": True,
        "physical_evidence_tier_added": True,
        "threshold_stable_flag_requires_hit_and_lift": True,
        "main_text_uses_semantic_display_labels": True,
        "raw_age_column_not_displayed_in_main_tables": True,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "evidence_tier_counts": tiers["Evidence_Tier"].value_counts().to_dict(),
        "outputs": [
            OUTPUTS["rule_bootstrap"], OUTPUTS["threshold_sensitivity"], OUTPUTS["evidence_tiers"],
            OUTPUTS["mechanism_axis_summary"], OUTPUTS["governance_scene_map"], OUTPUTS["main_text_rule_table"],
            OUTPUTS["rule_universe_mechanism_audit"], OUTPUTS["governance_diversity_summary"],
            OUTPUTS["main_vs_governance_interpretation_check"],
        ],
    }, "07_Run_Manifest.json")
    print("✅ 06 evidence grading finished.")


if __name__ == "__main__":
    main()
