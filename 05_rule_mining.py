# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    HAS_MLXTEND = True
except Exception:
    HAS_MLXTEND = False

from _config import *
from _utils import (
    assert_test_not_used_for_fit,
    find_group_col,
    infer_family,
    infer_feature_semantic_group,
    infer_feature_source_group,
    is_binary_like,
    is_pure_missing_unknown_feature,
    is_raw_age_column,
    missing_category_type,
    read_csv_smart,
    require_columns,
    safe_numeric,
    write_csv,
    write_json,
)


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

AXIS_GOVERNANCE_SCENE = {
    "speed_energy": "speed enforcement and conflict-speed management",
    "vulnerability_speed": "VRU vulnerability-sensitive speed and crossing protection",
    "vulnerability_body": "injury susceptibility stratification",
    "road_speed_environment": "road-space, intersection, and lane organization governance",
    "vehicle_geometry_interaction": "vehicle geometry, blind-zone, and VRU protection design",
    "vehicle_mass_speed_energy": "vehicle mass-speed energy audit signal",
    "lighting_visibility": "night lighting and visibility governance",
    "surface_weather_friction": "low-friction road surface and weather governance",
    "safety_assistance": "vehicle active safety configuration governance",
    "crash_configuration": "conflict type and road-space redesign governance",
    "mixed_multimechanism": "cross-mechanism audit and hypothesis governance",
}

AXIS_BOUNDARIES = {
    "vehicle_mass_speed_energy": "Vehicle mass-speed energy axis; use as an audit-level signal for vehicle mass/inertia combined with speed exposure, not as crash configuration governance.",
    "vulnerability_body": "Train-only mechanism label; vulnerability/body-size patterns are stratification signals, not directly intervenable causal factors.",
    "mixed_multimechanism": "Train-only mechanism label; use only with evidence tier and case-level replay context because no single mechanism dominates.",
}

AXIS_TEMPLATES.update({
    "vulnerability_demographic": "个体脆弱性—人口学分层轴线，适合作为年龄/性别等人群分层下的风险识别线索，不应解释为体格或可直接干预因素。",
    "vulnerability_demographic_speed": "个体脆弱性—人口学/速度分层轴线，适合作为速度暴露下的人群分层风险线索，不应解释为体格机制。",
    "vehicle_geometry_vulnerability": "车辆几何—脆弱性分层轴线，适合作为车辆几何与VRU脆弱性共同出现的风险线索。",
})
AXIS_GOVERNANCE_SCENE.update({
    "vulnerability_demographic": "individual demographic risk stratification",
    "vulnerability_demographic_speed": "demographic speed-risk stratification",
    "vehicle_geometry_vulnerability": "vehicle geometry and vulnerability stratification",
})
AXIS_BOUNDARIES.update({
    "vulnerability_demographic": "Individual vulnerability–demographic stratification axis; use as a demographic risk-stratification cue rather than a directly intervenable body-size mechanism.",
    "vulnerability_demographic_speed": "Demographic speed-risk stratification signal; do not interpret sex or age strata as body-size mechanisms.",
    "vehicle_geometry_vulnerability": "Vehicle geometry and vulnerability stratification signal; do not interpret demographic strata as body-size mechanisms.",
})


def _text_blob(values) -> str:
    return " ".join(str(v) for v in values if pd.notna(v)).upper()


def _semantic_text(source_feature, item, family) -> str:
    raw = " ".join(str(v) for v in (source_feature, item, family) if pd.notna(v))
    return raw, raw.upper()


def sanitize_semantic_source(source_feature: str) -> str:
    text = str(source_feature).strip()
    safe = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return safe or "unknown"


def infer_semantic_group(source_feature, item, family) -> str:
    return infer_feature_semantic_group(str(source_feature or item or ""))


def semantic_duplicate_details(antecedent_items, manifest_map: Dict[str, dict]) -> List[Dict[str, str]]:
    groups: Dict[str, List[str]] = {}
    for item in [str(x) for x in antecedent_items if str(x)]:
        spec = manifest_map.get(item, {})
        group = str(spec.get("Semantic_Group") or infer_semantic_group(spec.get("source_feature", item), item, spec.get("family", "")))
        if group == "unknown_missing_category" and str(spec.get("Missing_Category_Type", "not_missing_category")) == "not_missing_category":
            group = infer_feature_semantic_group(spec.get("source_feature", item))
        groups.setdefault(group, []).append(item)
    return [
        {
            "Duplicate_Semantic_Group": group,
            "Duplicate_Items": "||".join(items),
        }
        for group, items in groups.items()
        if len(items) >= 2
    ]


def has_semantic_duplicate_rule(antecedent_items, manifest: pd.DataFrame) -> bool:
    manifest_map = {str(r["item"]): r.to_dict() for _, r in manifest.iterrows()}
    return bool(semantic_duplicate_details(antecedent_items, manifest_map))


def infer_mechanism_axis(families, source_features, items) -> str:
    fam = {str(x) for x in families if str(x) and str(x) != "nan"}
    text = _text_blob(list(source_features) + list(items))
    age_text = text
    if "VEHUSAGE" in age_text or re.search(r"(^|_)USAGE(_|$)", age_text) or "车辆用途" in age_text:
        age_text = re.sub(r"[A-Z0-9_]*VEHUSAGE[A-Z0-9_]*", " ", age_text)
        age_text = re.sub(r"(^|_)USAGE(_|$)", " ", age_text)
    has_speed_v0 = "speed_v0" in fam or "V0" in text or "INITIAL_SPEED" in text
    has_speed_vk = "speed_vk" in fam or "VK" in text or "IMPACT_SPEED" in text or "COLLISION_SPEED" in text
    has_speed = has_speed_v0 or has_speed_vk
    has_age = (
        "age" in fam
        or re.search(r"(^|_)AGE(_|$)", age_text) is not None
        or "FEATURE_AGE" in age_text
        or "AGE_YEARS" in age_text
        or "ALTER1" in age_text
        or "ALTERG" in age_text
        or "年龄" in age_text
        or "老年" in age_text
        or "青壮年" in age_text
        or "未成年" in age_text
    )
    has_vehicle_mass = "vehicle_mass" in fam or "GEWGES" in text or "碰撞时的总重" in text
    has_body = any(k in text for k in ("GEWP", "GROESP", "WEIGHT", "HEIGHT", "BMI", "体重", "身高"))
    has_sex = "GESCHL" in text or re.search(r"(^|_)SEX(_|$)", text) is not None or "GENDER" in text or "性别" in text
    has_bio = "bio" in fam or has_body
    has_road = bool({"road_env", "road_type", "lane"} & fam) or any(k in text for k in ("ROAD_ENV", "ROAD_TYPE", "LANE", "STRART", "SPUR", "交叉", "路段", "车道"))
    has_vehicle_geometry = bool({"vehicle_size", "vehicle_state"} & fam) or any(k in text for k in ("VEHICLE_WIDTH", "VEHICLE_HEIGHT", "VEHICLE_LENGTH", "BREITE", "HOEHE", "LAENGE", "车辆用途", "车高", "车宽", "车长"))
    has_light = "light" in fam or any(k in text for k in ("LIGHT", "NIGHT", "LICHT", "STRABEL", "照明", "路灯", "夜"))
    has_surface_weather = bool({"road_surface", "weather"} & fam) or any(k in text for k in ("MUE", "FRICTION", "SURFACE", "WEATHER", "WIND", "RAIN", "SNOW", "FOG", "雨", "雪", "雾", "路面"))
    has_safety = "safety" in fam or any(k in text for k in ("SAFETY", "AEB", "LKA", "SPURHAE", "ASSIST", "车道保持", "主动安全"))
    has_crash = "crash_type" in fam or any(k in text for k in ("CRASH_TYPE", "STOSS", "COLLISION_TYPE", "IMPACT_POINT", "碰撞角", "事故类型"))

    if has_speed_v0 and has_speed_vk:
        return "speed_energy"
    if has_speed and has_vehicle_mass:
        return "vehicle_mass_speed_energy"
    if has_vehicle_geometry and has_age:
        return "vehicle_geometry_vulnerability"
    if has_speed and has_age:
        return "vulnerability_speed"
    if has_speed and has_sex and not has_body:
        return "vulnerability_demographic_speed"
    if has_age and has_sex and not has_body:
        return "vulnerability_demographic"
    if (has_age and has_body) or has_body:
        return "vulnerability_body"
    if has_road and has_speed:
        return "road_speed_environment"
    if has_vehicle_geometry:
        return "vehicle_geometry_interaction"
    if has_vehicle_mass:
        return "vehicle_mass_speed_energy"
    if has_light:
        return "lighting_visibility"
    if has_surface_weather:
        return "surface_weather_friction"
    if has_safety:
        return "safety_assistance"
    if has_crash:
        return "crash_configuration"
    if has_speed:
        return "speed_energy"
    return "mixed_multimechanism"


def governance_scene(axis: str) -> str:
    return AXIS_GOVERNANCE_SCENE.get(str(axis), AXIS_GOVERNANCE_SCENE["mixed_multimechanism"])


def governance_template(axis: str) -> str:
    return AXIS_TEMPLATES.get(str(axis), AXIS_TEMPLATES["mixed_multimechanism"])


def interpretation_boundary(axis: str) -> str:
    return AXIS_BOUNDARIES.get(
        str(axis),
        "Train-only mechanism label; interpret only together with fixed rule thresholds, blind replay results, and evidence tier.",
    )


def load_selected_features() -> List[str]:
    df = read_csv_smart(OUTPUTS["selected_features"])
    for col in ["Selected_Features", "Feature", "feature"]:
        if col in df.columns:
            return [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
    raise KeyError("Selected feature file does not contain a feature column.")


def item_name(source: str, op: str = None, threshold=None) -> str:
    if op is None:
        return str(source)
    val = str(threshold).replace(".", "_").replace("-", "m")
    return f"{source}___{op.replace('<=','LE').replace('>','GT').replace('==','EQ')}_{val}"


def display_labels(source: str, item: str, op: str = None, threshold=None) -> Tuple[str, str]:
    src = str(source)
    sem = infer_feature_semantic_group(src)
    if src == "FEATURE_Age_Over60" or sem == "age_over60":
        return "年龄 > 60岁", "Age > 60 years"
    if "GESCHL" in src.upper() and ("1-" in src or src.endswith("_1") or "男性" in src):
        return "男性", "Male"
    if op is not None:
        try:
            thr = float(threshold)
            thr_text = f"{thr:g}"
        except Exception:
            thr_text = str(threshold)
        if sem == "precrash_speed":
            return f"初始速度 {op} {thr_text} km/h", f"Initial speed {op} {thr_text} km/h"
        if sem == "impact_speed":
            return f"碰撞速度 {op} {thr_text} km/h", f"Impact speed {op} {thr_text} km/h"
        if sem == "vehicle_geometry_length":
            return f"车辆长度 {op} {thr_text} mm", f"Vehicle length {op} {thr_text} mm"
        if sem == "vehicle_geometry_height":
            return f"车辆高度 {op} {thr_text} mm", f"Vehicle height {op} {thr_text} mm"
        if sem == "vehicle_geometry_width":
            return f"车辆宽度 {op} {thr_text} mm", f"Vehicle width {op} {thr_text} mm"
        if sem == "vehicle_mass":
            return f"车辆总重 {op} {thr_text}", f"Vehicle total weight {op} {thr_text}"
    return item, item


def derive_threshold(col: str, series: pd.Series):
    upper = str(col).upper()
    s = safe_numeric(series)
    if "FEATURE_AGE" in upper or "ALTER" in upper or "年龄" in col:
        return 60.0, ">", "domain_age_60"
    if "V0" in upper or "初始速度" in col:
        return 50.0, ">", "domain_speed_50"
    if ("VK" in upper and "VKREG" not in upper) or "碰撞速度" in col:
        return 50.0, ">", "domain_speed_50"
    if "MUE" in upper or "附着" in col:
        return 0.5, "<=", "domain_friction_0_5"
    if "GROESP" in upper or "身高" in col:
        return float(s.median()), ">", "train_median_height"
    if "GEWP" in upper or "体重" in col:
        return float(s.median()), ">", "train_median_weight"
    if "GEWGES" in upper or "碰撞时的总重" in col:
        return float(s.median()), ">", "train_median_vehicle_mass"
    if "车高" in col or "车宽" in col or "车长" in col or "HOEHE" in upper or "BREITE" in upper:
        return float(s.median()), ">", "train_median_vehicle_size"
    if "角" in col or "STOSS" in upper:
        return float(s.median()), ">", "train_median_collision_geometry"
    return float(s.median()), ">", "train_median_exploratory"


def load_sentinel_flags(path: str, index) -> Tuple[pd.DataFrame, List[str]]:
    warnings = []
    if not Path(path).exists():
        return pd.DataFrame(index=index), [f"{path}:missing_all_threshold_sources_default_observed"]
    flags = read_csv_smart(path)
    if len(flags) != len(index):
        warnings.append(f"{path}:row_count_mismatch_flags_ignored")
        return pd.DataFrame(index=index), warnings
    flags = flags.reset_index(drop=True)
    flags.index = index
    return flags, warnings


def build_items(df: pd.DataFrame, features: List[str], fit_manifest: bool = True, manifest: pd.DataFrame = None):
    item_df = pd.DataFrame(index=df.index)
    rows = []
    sentinel_flags, sentinel_warnings = load_sentinel_flags(OUTPUTS["sentinel_flags_train"], df.index)
    if fit_manifest:
        for f in features:
            if f not in df.columns:
                continue
            if is_raw_age_column(f) or infer_feature_semantic_group(f) == "raw_age_continuous":
                continue
            mtype = missing_category_type(f)
            if mtype == "pure_missing_unknown":
                continue
            s = df[f]
            sem_group = infer_feature_semantic_group(f)
            src_group = infer_feature_source_group(f)
            physical_eligible = int(mtype == "not_missing_category")
            if is_binary_like(s):
                name = str(f)
                label_cn, label_en = display_labels(f, name)
                item_df[name] = (pd.to_numeric(s, errors="coerce").fillna(0) == 1).astype(bool)
                axis = infer_mechanism_axis([infer_family(f)], [f], [name])
                rows.append({
                    "item": name, "source_feature": f, "family": infer_family(f),
                    "transform_type": "exact_binary_column", "operator": "==", "threshold": 1,
                    "Display_Item_Label_CN": label_cn, "Display_Item_Label_EN": label_en,
                    "threshold_source": "binary_column", "has_continuous_threshold": 0,
                    "requires_observed_source_value": 0,
                    "sentinel_flag_available": int(f in sentinel_flags.columns),
                    "missing_or_sentinel_values_can_trigger": 1,
                    "Missing_Category_Type": mtype,
                    "Semantic_Group": sem_group,
                    "Source_Group": src_group,
                    "Physical_Mechanism_Eligible": physical_eligible,
                    "Mechanism_Axis": axis,
                    "Mechanism_Families": infer_family(f),
                    "Governance_Scene": governance_scene(axis),
                    "Governance_Interpretation_Template": governance_template(axis),
                    "Interpretation_Boundary": interpretation_boundary(axis),
                })
            else:
                threshold, op, source = derive_threshold(f, s)
                name = item_name(f, op, threshold)
                label_cn, label_en = display_labels(f, name, op, threshold)
                sn = safe_numeric(s)
                sentinel_available = f in sentinel_flags.columns
                if sentinel_available:
                    observed_source_value = pd.to_numeric(sentinel_flags[f], errors="coerce").fillna(1).eq(0)
                else:
                    observed_source_value = pd.Series(True, index=df.index)
                    sentinel_warnings.append(f"{f}:sentinel_flag_missing_default_observed")
                threshold_mask = sn > threshold if op == ">" else sn <= threshold if op == "<=" else sn == threshold
                mask = observed_source_value & threshold_mask
                item_df[name] = mask.fillna(False).astype(bool)
                axis = infer_mechanism_axis([infer_family(f)], [f], [name])
                rows.append({
                    "item": name, "source_feature": f, "family": infer_family(f),
                    "transform_type": "threshold", "operator": op, "threshold": threshold,
                    "Display_Item_Label_CN": label_cn, "Display_Item_Label_EN": label_en,
                    "threshold_source": source, "has_continuous_threshold": 1,
                    "requires_observed_source_value": 1,
                    "sentinel_flag_available": int(sentinel_available),
                    "missing_or_sentinel_values_can_trigger": 0,
                    "Missing_Category_Type": mtype,
                    "Semantic_Group": sem_group,
                    "Source_Group": src_group,
                    "Physical_Mechanism_Eligible": physical_eligible,
                    "Mechanism_Axis": axis,
                    "Mechanism_Families": infer_family(f),
                    "Governance_Scene": governance_scene(axis),
                    "Governance_Interpretation_Template": governance_template(axis),
                    "Interpretation_Boundary": interpretation_boundary(axis),
                })
        manifest_df = pd.DataFrame(rows)
    else:
        manifest_df = manifest.copy()
        if "Semantic_Group" not in manifest_df.columns:
            manifest_df["Semantic_Group"] = manifest_df.apply(
                lambda r: infer_semantic_group(r.get("source_feature", ""), r.get("item", ""), r.get("family", "")),
                axis=1,
            )
        if "Missing_Category_Type" not in manifest_df.columns:
            manifest_df["Missing_Category_Type"] = manifest_df["source_feature"].map(missing_category_type)
        if "Source_Group" not in manifest_df.columns:
            manifest_df["Source_Group"] = manifest_df["source_feature"].map(infer_feature_source_group)
        if "Physical_Mechanism_Eligible" not in manifest_df.columns:
            manifest_df["Physical_Mechanism_Eligible"] = manifest_df["Missing_Category_Type"].astype(str).eq("not_missing_category").astype(int)
        for _, r in manifest_df.iterrows():
            item, f = str(r["item"]), str(r["source_feature"])
            if f not in df.columns:
                item_df[item] = False
                continue
            if r["transform_type"] == "exact_binary_column":
                item_df[item] = (pd.to_numeric(df[f], errors="coerce").fillna(0) == 1).astype(bool)
            else:
                sn = safe_numeric(df[f])
                thr = float(r["threshold"])
                op = str(r["operator"])
                threshold_mask = sn > thr if op == ">" else sn <= thr if op == "<=" else sn == thr
                if int(r.get("requires_observed_source_value", 0) or 0) == 1 and f in sentinel_flags.columns:
                    observed_source_value = pd.to_numeric(sentinel_flags[f], errors="coerce").fillna(1).eq(0)
                else:
                    observed_source_value = pd.Series(True, index=df.index)
                mask = observed_source_value & threshold_mask
                item_df[item] = mask.fillna(False).astype(bool)
    item_df[TARGET_ITEM] = (pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0) == 1).astype(bool)
    manifest_df.attrs["sentinel_warnings"] = sorted(set(sentinel_warnings))
    return item_df, manifest_df


def validate_rule_manifest_schema(manifest: pd.DataFrame) -> None:
    required = [
        "requires_observed_source_value",
        "sentinel_flag_available",
        "missing_or_sentinel_values_can_trigger",
        "Missing_Category_Type",
        "Semantic_Group",
        "Source_Group",
        "Physical_Mechanism_Eligible",
    ]
    require_columns(manifest, OUTPUTS["rule_manifest"], required)
    if manifest["Missing_Category_Type"].astype(str).eq("pure_missing_unknown").any():
        bad = manifest.loc[manifest["Missing_Category_Type"].astype(str).eq("pure_missing_unknown"), "item"].astype(str).head(20).tolist()
        raise RuntimeError("Pure missing/unknown items are forbidden in rule manifest: " + "||".join(bad))
    threshold = manifest[manifest["transform_type"].astype(str).eq("threshold")] if "transform_type" in manifest.columns else pd.DataFrame()
    if not threshold.empty:
        if not pd.to_numeric(threshold["requires_observed_source_value"], errors="coerce").fillna(0).eq(1).all():
            raise RuntimeError("All threshold items must require observed source value.")
        if not pd.to_numeric(threshold["missing_or_sentinel_values_can_trigger"], errors="coerce").fillna(1).eq(0).all():
            raise RuntimeError("Threshold items cannot be triggered by missing or sentinel values.")
    binary = manifest[manifest["transform_type"].astype(str).eq("exact_binary_column")] if "transform_type" in manifest.columns else pd.DataFrame()
    if not binary.empty:
        if not pd.to_numeric(binary["requires_observed_source_value"], errors="coerce").fillna(1).eq(0).all():
            raise RuntimeError("Exact binary items must not require observed source sentinel flags.")
        if not pd.to_numeric(binary["missing_or_sentinel_values_can_trigger"], errors="coerce").fillna(0).eq(1).all():
            raise RuntimeError("Exact binary items must record that encoded missing/sentinel categories can trigger.")


def write_semantic_redundancy_audit(rules: pd.DataFrame, manifest_map: Dict[str, dict], audit_path: str) -> pd.DataFrame:
    rows = []
    if rules is not None and not rules.empty:
        for _, r in rules.iterrows():
            details = semantic_duplicate_details(list(r["antecedents"]), manifest_map)
            for d in details:
                rows.append({
                    "Rule_ID": f"C{int(r.get('Candidate_Rank', 0) or 0):04d}" if "Candidate_Rank" in r else "",
                    "Antecedent_Items": r.get("Antecedent_Items", ""),
                    "Duplicate_Semantic_Group": d["Duplicate_Semantic_Group"],
                    "Duplicate_Items": d["Duplicate_Items"],
                    "Action": "excluded_from_primary_rules",
                    "Reason": "same semantic construct repeated in one rule",
                })
    audit = pd.DataFrame(rows, columns=[
        "Rule_ID", "Antecedent_Items", "Duplicate_Semantic_Group", "Duplicate_Items", "Action", "Reason",
    ])
    write_csv(audit, audit_path)
    return audit


def append_rule_audit_rows(audit_path: str, rows: List[Dict[str, str]]) -> None:
    old = read_csv_smart(audit_path) if Path(audit_path).exists() else pd.DataFrame()
    add = pd.DataFrame(rows)
    if old.empty:
        out = add
    elif add.empty:
        out = old
    else:
        out = pd.concat([old, add], ignore_index=True)
    write_csv(out, audit_path)


def mine_rules(item_df: pd.DataFrame, manifest: pd.DataFrame, audit_path: str = "Rule_Semantic_Redundancy_Audit.csv"):
    if not HAS_MLXTEND:
        raise ImportError("mlxtend is required for FPGrowth. Please install mlxtend.")
    frequent = fpgrowth(item_df, min_support=RULE_MIN_SUPPORT, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame(), pd.DataFrame()
    rules = association_rules(frequent, metric="lift", min_threshold=RULE_MIN_LIFT)
    rules = rules[rules["consequents"].apply(lambda x: set(x) == {TARGET_ITEM})].copy()
    rules["Antecedent_Items"] = rules["antecedents"].apply(lambda x: "||".join(sorted(list(x))))
    rules["Rule_Length"] = rules["antecedents"].apply(len)
    rules = rules[(rules["Rule_Length"] >= RULE_MIN_LEN) & (rules["Rule_Length"] <= RULE_MAX_LEN)].copy()
    if rules.empty:
        return rules, pd.DataFrame()
    family_map = dict(zip(manifest["item"], manifest["family"]))
    source_map = dict(zip(manifest["item"], manifest["source_feature"]))
    rules["Family_Signature"] = rules["antecedents"].apply(lambda x: "|".join(sorted(set(family_map.get(i, "other") for i in x))))
    rules["Family_Count"] = rules["Family_Signature"].apply(lambda x: len(set(x.split("|"))))
    missing_map = dict(zip(manifest["item"], manifest["Missing_Category_Type"]))
    pure_missing_rules = rules[rules["antecedents"].apply(lambda x: any(str(missing_map.get(i, "not_missing_category")) == "pure_missing_unknown" for i in x))].copy()
    rules = rules[~rules["antecedents"].apply(lambda x: any(str(missing_map.get(i, "not_missing_category")) == "pure_missing_unknown" for i in x))].copy()
    if rules.empty:
        return rules, pd.DataFrame()
    manifest_map = {str(r["item"]): r.to_dict() for _, r in manifest.iterrows()}
    rules["Semantic_Duplicate_Flag"] = rules["antecedents"].apply(
        lambda x: int(bool(semantic_duplicate_details(list(x), manifest_map)))
    )
    rules["Mechanism_Clarity"] = rules["antecedents"].apply(
        lambda x: float(np.mean([
            0.7 if "exploratory" in str(manifest_map.get(i, {}).get("threshold_source", "")) else 1.0
            for i in x
        ]))
    )
    rules["Score"] = (
        rules["lift"] * 0.42
        + rules["confidence"] * 0.32
        + rules["support"] * 3.2
        + rules["Family_Count"] * 0.07
        + rules["Mechanism_Clarity"] * 0.10
        - rules["Rule_Length"] * 0.035
    )
    rules["Mechanism_Families"] = rules["antecedents"].apply(
        lambda x: "|".join(sorted(set(str(family_map.get(i, "other")) for i in x)))
    )
    rules["Mechanism_Axis"] = rules["antecedents"].apply(
        lambda x: infer_mechanism_axis(
            [family_map.get(i, "other") for i in x],
            [source_map.get(i, "") for i in x],
            list(x),
        )
    )
    rules["Governance_Scene"] = rules["Mechanism_Axis"].apply(governance_scene)
    rules["Governance_Interpretation_Template"] = rules["Mechanism_Axis"].apply(governance_template)
    rules["Interpretation_Boundary"] = rules["Mechanism_Axis"].apply(interpretation_boundary)
    rules = rules.sort_values(["Score", "lift", "confidence", "support", "Rule_Length"], ascending=[False, False, False, False, True]).reset_index(drop=True)
    universe = rules.copy()
    universe.insert(0, "Candidate_Rank", np.arange(1, len(universe) + 1))
    duplicate_candidates = universe[universe["Semantic_Duplicate_Flag"].astype(int).eq(1)].copy()
    write_semantic_redundancy_audit(duplicate_candidates, manifest_map, audit_path)
    if not pure_missing_rules.empty:
        append_rule_audit_rows(audit_path, [{
            "Rule_ID": "",
            "Antecedent_Items": r.get("Antecedent_Items", ""),
            "Duplicate_Semantic_Group": "unknown_missing_category",
            "Duplicate_Items": "||".join([str(i) for i in r.get("antecedents", []) if str(missing_map.get(i, "")) == "pure_missing_unknown"]),
            "Action": "excluded_from_primary_rules",
            "Reason": "pure_missing_unknown_item_excluded",
        } for _, r in pure_missing_rules.iterrows()])
    rules = rules[rules["Semantic_Duplicate_Flag"].astype(int).eq(0)].copy().reset_index(drop=True)

    selected = []
    selected_sets = []
    sig_counts: Dict[str, int] = {}
    primary_counts: Dict[str, int] = {}
    axis_counts: Dict[str, int] = {}
    target_n = min(int(RULE_MAX_FINAL), int(RULE_FINAL_DISPLAY_MAX))

    def can_add(r, allow_signature_repeat: bool = False) -> bool:
        current_set = set(r["antecedents"])
        if any(prev.issubset(current_set) for prev in selected_sets):
            return False
        axis = str(r["Mechanism_Axis"])
        if axis_counts.get(axis, 0) >= RULE_AXIS_CAPS.get(axis, 2):
            return False
        sig = r["Family_Signature"]
        if (not allow_signature_repeat) and sig_counts.get(sig, 0) >= 1:
            return False
        primary = sig.split("|")[0]
        if primary_counts.get(primary, 0) >= 3:
            return False
        return True

    def add_rule(r, rule_set: str):
        selected.append((r, rule_set))
        current_set = set(r["antecedents"])
        selected_sets.append(current_set)
        sig = r["Family_Signature"]
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        primary = sig.split("|")[0]
        primary_counts[primary] = primary_counts.get(primary, 0) + 1
        axis = str(r["Mechanism_Axis"])
        axis_counts[axis] = axis_counts.get(axis, 0) + 1

    for _, r in rules.iterrows():
        if not can_add(r, allow_signature_repeat=False):
            continue
        add_rule(r, "Primary_Rule_Set")
        if len(selected) >= target_n:
            break

    for axis, min_n in RULE_AXIS_MIN_SOFT.items():
        if len(selected) >= target_n or axis_counts.get(axis, 0) >= int(min_n):
            continue
        candidates = rules[
            (rules["Mechanism_Axis"].astype(str) == str(axis))
            & (rules["Mechanism_Clarity"] >= 0.85)
        ]
        for _, r in candidates.iterrows():
            if not can_add(r, allow_signature_repeat=True):
                continue
            add_rule(r, "Governance_Coverage_Rule_Set")
            break

    out = pd.DataFrame([r for r, _rule_set in selected]).copy()
    rule_sets = [_rule_set for _r, _rule_set in selected]
    out.insert(0, "Rule_ID", [f"R{i+1:02d}" for i in range(len(out))])
    out["Rule_Set"] = rule_sets
    manifest_map = {str(r["item"]): r.to_dict() for _, r in manifest.iterrows()}
    out["Physical_Mechanism_Eligible"] = out["Antecedent_Items"].apply(
        lambda v: int(all(int(manifest_map.get(i, {}).get("Physical_Mechanism_Eligible", 0) or 0) == 1 for i in str(v).split("||") if i))
    )
    keep = [
        "Rule_ID", "Rule_Set", "Antecedent_Items", "support", "confidence", "lift", "leverage", "conviction",
        "Rule_Length", "Family_Signature", "Family_Count", "Score", "Mechanism_Axis", "Mechanism_Families",
        "Governance_Scene", "Governance_Interpretation_Template", "Interpretation_Boundary", "Semantic_Duplicate_Flag",
        "Physical_Mechanism_Eligible",
    ]
    universe_keep = [
        "Candidate_Rank", "Antecedent_Items", "support", "confidence", "lift", "leverage", "conviction",
        "Rule_Length", "Family_Signature", "Family_Count", "Score", "Mechanism_Axis", "Mechanism_Families",
        "Governance_Scene", "Governance_Interpretation_Template", "Interpretation_Boundary", "Semantic_Duplicate_Flag",
    ]
    return out[keep], universe[universe_keep]


def main():
    train_path = TRAIN_MATRIX_ALIAS if Path(TRAIN_MATRIX_ALIAS).exists() else TRAIN_MATRIX
    assert_test_not_used_for_fit(
        "05_rule_mining",
        [train_path, OUTPUTS["selected_features"]],
        test_paths=[TEST_MATRIX, OUTPUTS["blind_replay"], OUTPUTS["evidence_tiers"], OUTPUTS["threshold_sensitivity"], "Rule_Item_Replay_Audit.csv"],
    )
    train = read_csv_smart(train_path)
    features = load_selected_features()
    item_df, manifest = build_items(train, features, fit_manifest=True)
    sentinel_warnings = manifest.attrs.get("sentinel_warnings", [])
    validate_rule_manifest_schema(manifest)
    write_csv(manifest, OUTPUTS["rule_manifest"])
    rules, universe = mine_rules(item_df, manifest)
    write_csv(universe, OUTPUTS["rule_universe"])
    write_csv(rules, OUTPUTS["rules"])
    write_json({
        "stage": "05_rule_mining",
        "fit_source": "train",
        "selection_source": "train",
        "rule_mining_source": "train_only",
        "threshold_derivation_source": "train_or_domain_only",
        "transform_source": "train_item_matrix_from_train_fitted_rule_manifest",
        "evaluation_source": "none",
        "test_used_for_rule_mining": False,
        "test_used_for_rule_filtering": False,
        "test_used_for_threshold_derivation": False,
        "test_used_for_final_rule_sorting": False,
        "test_used_for_family_signature_limits": False,
        "selected_features_n": len(features),
        "item_n": int(item_df.shape[1] - 1),
        "mechanism_axis_source": "train_manifest_family_and_antecedent_source_features_only",
        "semantic_redundancy_filter": True,
        "semantic_group_source": "train_manifest_string_rules",
        "test_used_for_semantic_filtering": False,
        "rule_item_semantic_group_deduplication_enabled": True,
        "pure_missing_unknown_excluded_from_rule_item_universe": True,
        "raw_age_threshold_item_disabled_for_primary_rules": True,
        "age_over60_binary_item_used_for_rule_mining": True,
        "partial_unknown_detail_downgraded_for_physical_interpretation": True,
        "test_used_for_missing_category_or_semantic_rule_filtering": False,
        "semantic_duplicate_rules_excluded_from_primary": True,
        "continuous_threshold_items_require_observed_source_value": True,
        "sentinel_flags_used_for_threshold_item_masks": True,
        "missing_or_sentinel_values_can_trigger_threshold_items": False,
        "test_used_for_sentinel_threshold_masking": False,
        "sentinel_threshold_mask_warnings": sentinel_warnings,
        "sentinel_flag_warnings": sentinel_warnings,
        "axis_caps": RULE_AXIS_CAPS,
        "axis_min_soft": RULE_AXIS_MIN_SOFT,
        "final_rule_display_max": RULE_FINAL_DISPLAY_MAX,
        "final_rule_n": int(len(rules)),
        "outputs": [OUTPUTS["rule_manifest"], OUTPUTS["rule_universe"], OUTPUTS["rules"], "Rule_Semantic_Redundancy_Audit.csv"],
    }, "05_Run_Manifest.json")
    print(f"✅ 04 rule mining finished. Final rules: {len(rules)}")


if __name__ == "__main__":
    main()
