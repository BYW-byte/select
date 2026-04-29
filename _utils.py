# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest, chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def read_csv_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def write_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if "Manifest" in Path(path).name or Path(path).name.endswith("_Run_Manifest.json"):
        obj = dict(obj)
        obj.setdefault("run_id", os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S") + "_42")
        obj.setdefault("script_name", str(obj.get("stage") or Path(path).stem))
        obj.setdefault("script_version_note", "sentinel-aware audit hard-stop version")
        obj.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        obj.setdefault("semantic_deduplication_version", SEMANTIC_DEDUPLICATION_VERSION)
        obj.setdefault("missing_category_detector_version", MISSING_CATEGORY_DETECTOR_VERSION)
        obj.setdefault("age_semantic_feature_version", AGE_SEMANTIC_FEATURE_VERSION)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def require_columns(df: pd.DataFrame, path: str, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing required schema columns: " + "||".join(missing))


FORBIDDEN_TEST_DERIVED_INPUTS = {
    "ready_matrix_test.csv",
    "final_blind_test_validation_report.csv",
    "06f_rule_evidence_tiers.csv",
    "06d_thresholdsensitivity.csv",
    "rule_item_replay_audit.csv",
}


def assert_test_not_used_for_fit(stage_name: str, fit_inputs: Sequence[str], test_paths: Sequence[str] = None, evaluation_only: bool = False) -> None:
    """Hard-stop if a fit/selection/mining stage is wired to test-derived inputs.

    The independent test set is allowed only for final evaluation, blind replay,
    and fixed-threshold evaluation-only diagnostics. It must not feed feature
    selection, threshold derivation, rule mining/filtering, model selection, or
    hyperparameter selection.
    """
    if evaluation_only:
        return
    forbidden = set(FORBIDDEN_TEST_DERIVED_INPUTS)
    if test_paths:
        forbidden |= {Path(str(p)).name.lower() for p in test_paths}
    hits = []
    for p in fit_inputs or []:
        name = Path(str(p)).name.lower()
        if name in forbidden:
            hits.append(str(p))
    if hits:
        raise RuntimeError(
            f"{stage_name}: independent test or replay-derived file supplied to a fit/selection/mining stage: "
            + "||".join(hits)
        )


def normalize_group_ids(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out = out.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    if out.isna().any():
        filler = [f"MISSING_ACC_{i}" for i in range(int(out.isna().sum()))]
        out.loc[out.isna()] = filler
    return out.astype(str)


def find_group_col(columns: Sequence[str], candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in columns:
            return col
    for col in columns:
        text = str(col)
        if "FALL" in text.upper() or "事故编号" in text:
            return col
    raise KeyError("No accident/group identifier column found.")


def sanitize_column_name(name: str) -> str:
    out = re.sub(r"[\[\]<>,;:{}()'\"\s]", "_", str(name))
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "EMPTY_COL"


def make_unique(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__dup{seen[n]}")
    return out


def safe_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    extracted = series.astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


TEXT_SENTINEL_TOKENS = {
    "未知", "不详", "未测", "未填", "缺失", "不适用",
    "unknown", "missing", "n/a", "none", "nan",
}


def _feature_sentinel_values(feature_name: str) -> set:
    upper = str(feature_name).upper()
    values = set()
    if ("V0" in upper) or ("VK" in upper and "VKREG" not in upper):
        values.update({999, 9999, 99999, 999999})
    if any(k in upper for k in ("GEWP", "GEWPG", "GROESP", "GROESPG")):
        values.update({999, 9999, 99999, 999999})
    if "MUE" in upper:
        values.update({9, 99, 999, 9999, 99999})
    if any(k in upper for k in ("LAENGE", "LÄNGE", "L脛NGE", "BREITE", "HOEHE")):
        values.update({99999, 999999})
    if "TEMP" in upper:
        values.update({999, 9999, 99999})
    return values


def sentinel_mask(series: pd.Series, feature_name: str) -> pd.Series:
    """Return True for raw unknown/sentinel encodings in a feature column."""
    raw = pd.Series(series)
    text = raw.astype(str).str.strip()
    text_lower = text.str.lower()
    text_compact = text_lower.str.replace(r"\s+", "", regex=True)

    text_unknown = text_compact.isin(TEXT_SENTINEL_TOKENS)
    coded_unknown = text_compact.isin({"99-未知", "999-未知", "9999-未知", "99999-未知"})

    num = safe_numeric(raw)
    numeric_values = _feature_sentinel_values(feature_name)
    numeric_unknown = num.isin(numeric_values) if numeric_values else pd.Series(False, index=raw.index)
    return (text_unknown | coded_unknown | numeric_unknown).fillna(False).astype(bool)


def sentinel_aware_numeric(series: pd.Series, feature_name: str) -> pd.Series:
    out = safe_numeric(series).copy()
    out.loc[sentinel_mask(series, feature_name)] = np.nan
    return out


def is_binary_like(series: pd.Series) -> bool:
    s = pd.Series(series).dropna()
    if s.empty:
        return False
    vals = set(s.astype(str).str.lower().unique().tolist())
    return vals.issubset({"0", "1", "0.0", "1.0", "false", "true"})


def as_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    s = series.astype(str).str.lower().str.strip()
    return s.map({"1": 1, "1.0": 1, "true": 1, "0": 0, "0.0": 0, "false": 0}).astype("float")


def infer_family(feature: str) -> str:
    f = str(feature)
    fu = f.upper()
    compact = re.sub(r"[^A-Z0-9]+", "_", fu)
    tokens = set(x for x in compact.split("_") if x)
    if "GEWGES" in compact or "碰撞时的总重" in f:
        return "vehicle_mass"
    if "VEHUSAGE" in compact or "USAGE" in tokens or "车辆用途" in f:
        return "vehicle_state"
    if any(k in compact for k in ("INITIAL_SPEED", "PRECRASH_SPEED", "TRAVEL_SPEED", "SPEED_V0", "V_INIT")):
        return "speed_v0"
    if any(k in compact for k in ("COLLISION_SPEED", "IMPACT_SPEED", "CRASH_SPEED", "SPEED_VK", "DELTA_V")):
        return "speed_vk"
    if (
        "AGE" in tokens
        or "AGE_YEARS" in compact
        or "FEATURE_AGE" in compact
        or "YEARS_OLD" in compact
        or "ALTER1" in tokens
        or "ALTERG" in tokens
        or "年龄" in f
        or "老年" in f
        or "青壮年" in f
        or "未成年" in f
    ):
        return "age"
    if any(k in compact for k in ("GESCHL", "GENDER", "SEX")):
        return "bio"
    if any(k in compact for k in ("HEIGHT", "WEIGHT", "BMI", "GROESP", "GEWP")):
        return "bio"
    if any(k in compact for k in ("FRICTION", "SURFACE", "PAVEMENT", "MUE", "MU_")):
        return "road_surface"
    if any(k in compact for k in ("LIGHT", "LIGHTING", "DAYLIGHT", "DARK", "NIGHT", "DUSK", "DAWN", "LICHT")):
        return "light"
    if any(k in compact for k in ("ROAD_TYPE", "ROAD_CLASS", "STRART", "URBAN", "RURAL", "HIGHWAY")):
        return "road_type"
    if any(k in compact for k in ("INTERSECTION", "JUNCTION", "CROSSING", "ROAD_ENV", "ROAD_SECTION")):
        return "road_env"
    if any(k in compact for k in ("LANE", "SPUR", "SHOULDER", "MEDIAN")):
        return "lane"
    if any(k in compact for k in ("POWERTRAIN", "ENGINE_TYPE", "FUEL", "NEV", "ELECTRIC", "HYBRID")):
        return "vehicle_state"
    if any(k in compact for k in ("WEATHER", "WETTER", "RAIN", "SNOW", "FOG", "WIND", "VISIBILITY")):
        return "weather"
    if any(k in compact for k in ("VEHICLE_WIDTH", "VEHICLE_HEIGHT", "VEHICLE_LENGTH", "BREITE", "HOEHE", "LAENGE")):
        return "vehicle_size"
    safety_tokens = tokens
    if any(k in safety_tokens for k in ("AEB", "ABS", "ESC", "ASR", "ESP")) or any(k in compact for k in ("AIRBAG", "SEATBELT", "BELT", "SAFETY", "ASSIST")):
        return "safety"
    if any(k in compact for k in ("CRASH_TYPE", "COLLISION_TYPE", "IMPACT_POINT", "STOSS", "CONFLICT_TYPE")):
        return "crash_type"
    if "V0" in fu or "初始速度" in f:
        return "speed_v0"
    if ("VK" in fu and "VKREG" not in fu) or "碰撞速度" in f:
        return "speed_vk"
    if "GEWGES" in compact or "碰撞时的总重" in f:
        return "vehicle_mass"
    if (
        "AGE" in tokens
        or "AGE_YEARS" in compact
        or "FEATURE_AGE" in compact
        or "YEARS_OLD" in compact
        or "ALTER1" in tokens
        or "ALTERG" in tokens
        or "年龄" in f
        or "老年" in f
        or "青壮年" in f
        or "未成年" in f
    ):
        return "age"
    if "GEWP" in fu or "GROESP" in fu or "体重" in f or "身高" in f:
        return "bio"
    if "MUE" in fu or "附着" in f or "路面" in f:
        return "road_surface"
    if "LICHT" in fu or "路灯" in f or "LIGHT" in fu or "照明" in f or "夜" in f:
        return "light"
    if "STRART" in fu or "道路类型" in f or "公路" in f or "城市道路" in f:
        return "road_type"
    if "普通路段" in f or "交叉" in f or "路段" in f or "ROAD_ENV" in fu:
        return "road_env"
    if "SPUR" in fu or "车道" in f or "LANE" in fu:
        return "lane"
    if "WETTER" in fu or "天气" in f or "雨" in f or "雪" in f:
        return "weather"
    if "车宽" in f or "车高" in f or "车长" in f or "BREITE" in fu or "HOEHE" in fu or "LAENGE" in fu:
        return "vehicle_size"
    if "AEB" in fu or "SPURHAE" in fu or "车道保持" in f or "安全" in f:
        return "safety"
    if "碰撞" in f or "STOSS" in fu or "冲突" in f or "事故类型" in f:
        return "crash_type"
    if "ENGINE" in fu or "工程" in f:
        return "engineering"
    if "车辆状态" in f or "行驶状态" in f or "制动" in f:
        return "vehicle_state"
    return "other"


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) / total) + z ** 2 / (4 * total ** 2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def enrichment_pvalue(successes: int, total: int, base_rate: float) -> float:
    if total <= 0 or pd.isna(base_rate):
        return np.nan
    base_rate = float(min(max(base_rate, 1e-12), 1 - 1e-12))
    if HAS_SCIPY:
        try:
            return float(binomtest(successes, total, base_rate, alternative="greater").pvalue)
        except Exception:
            pass
    mean = total * base_rate
    var = total * base_rate * (1 - base_rate)
    if var <= 0:
        return np.nan
    z = (successes - mean) / math.sqrt(var)
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    vals = [(i, float(p)) for i, p in enumerate(pvalues) if not pd.isna(p)]
    out = [np.nan] * len(pvalues)
    if not vals:
        return out
    vals.sort(key=lambda x: x[1])
    m = len(vals)
    prev = 1.0
    for rank, (idx, p) in reversed(list(enumerate(vals, start=1))):
        q = min(prev, p * m / rank)
        out[idx] = q
        prev = q
    return out


def smd_numeric(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled <= 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def smd_binary(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    p1, p2 = float(a.mean()), float(b.mean())
    pooled = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if pooled <= 1e-12:
        return 0.0
    return (p1 - p2) / pooled


def bhattacharyya_safe_name(text: str, max_len: int = 140) -> str:
    out = re.sub(r"\s+", " ", str(text)).strip()
    return out[:max_len]


MISSING_CATEGORY_DETECTOR_VERSION = "missing-category-detector-v2"
SEMANTIC_DEDUPLICATION_VERSION = "semantic-deduplication-v2"
AGE_SEMANTIC_FEATURE_VERSION = "age-over60-semantic-v1"


def _norm_feature_text(name: str) -> str:
    return str(name or "").strip()


def _ascii_compact(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(name or "").upper()).strip("_")


def missing_category_type(name: str) -> str:
    text = _norm_feature_text(name)
    upper = text.upper()
    compact = _ascii_compact(text)
    lower = text.lower()
    if any(k in lower for k in ("yes_but_unknown_detail", "有_具体未知", "存在_具体未知")):
        return "partial_unknown_detail"
    if "具体未知" in text and any(k in text for k in ("有", "存在")):
        return "partial_unknown_detail"

    pure_tokens = {
        "UNKNOWN", "MISSING", "MISSING_OR_UNKNOWN", "N_A", "NA", "NONE", "NAN",
        "不详", "未知", "未记录", "未填", "缺失", "无资料", "不适用",
        "涓嶈", "鏈", "缂哄け", "涓嶉",
    }
    if "MISSING_OR_UNKNOWN" in compact:
        return "pure_missing_unknown"
    if any(tok in text for tok in ("不详", "未知", "未记录", "未填", "缺失", "无资料", "不适用")):
        return "pure_missing_unknown"
    if any(tok in upper for tok in ("UNKNOWN", "MISSING", "N/A")):
        return "pure_missing_unknown"
    if re.search(r"(^|[_\-])(?:9|99|999|9999)[_\-]?(?:UNKNOWN|MISSING|未知|不详)$", upper):
        return "pure_missing_unknown"
    tail = re.split(r"[_\-]", compact)[-1] if compact else ""
    if tail in pure_tokens:
        return "pure_missing_unknown"
    if compact in pure_tokens:
        return "pure_missing_unknown"
    return "not_missing_category"


def is_pure_missing_unknown_feature(name: str) -> bool:
    return missing_category_type(name) == "pure_missing_unknown"


def is_missing_category_feature(name: str) -> bool:
    return missing_category_type(name) != "not_missing_category"


def infer_feature_source_group(feature: str) -> str:
    f = _norm_feature_text(feature)
    upper = _ascii_compact(f)
    if re.search(r"BRP[XYZ]\b", upper) or any(k in upper for k in ("BRPX", "BRPY", "BRPZ")):
        return "BRP"
    if re.search(r"STOSSP[XYZ]\b", upper):
        return "STOSSP"
    known = [
        "STFUHO", "STRKL", "STRART", "STROB", "GESCHL", "PTYPE", "VEHUSAGE",
        "NEVTYPE", "SPURZ", "SPURHAE", "COLLWARN", "ACSCE", "BLICHT", "LICHT",
        "WINDV", "LAENGE", "BREITE", "HOEHE", "GEWP", "GEWPG", "GROESP",
        "GROESPG", "ALTER1", "V0", "VK",
    ]
    for key in known:
        if re.search(rf"(^|_){re.escape(key)}($|_)", upper):
            return key
    if upper.startswith("FEATURE_AGE_GROUP"):
        return "FEATURE_Age_Group"
    if upper.startswith("FEATURE_OPP_POWERTRAIN"):
        return "FEATURE_Opp_Powertrain"
    parts = [p for p in f.split("_") if p]
    if len(parts) >= 2 and parts[0] == "FEATURE":
        return "_".join(parts[:-1]) if len(parts) > 2 else f
    if len(parts) >= 3:
        return "_".join(parts[:-1])
    return sanitize_column_name(f) if f else "unknown_source"


def infer_feature_semantic_group(feature: str) -> str:
    if missing_category_type(feature) != "not_missing_category":
        return "unknown_missing_category"
    f = _norm_feature_text(feature)
    upper = _ascii_compact(f)
    raw = f.lower()

    def has(*keys: str) -> bool:
        return any(k.upper() in upper or k.lower() in raw for k in keys)

    if has("FEATURE_AGE_GROUP", "ALTER1", "ALTERG", "FEATURE_AGE_YEARS", "年龄年数记录", "年龄段") or re.search(r"(^|_)AGE(_|$)", upper):
        return "age_over60"
    if has("V0", "初始速度"):
        return "precrash_speed"
    if ("VK" in upper and "VKREG" not in upper) or "碰撞速度" in f:
        return "impact_speed"
    if has("GEWP", "GEWPG", "体重估计", "体重"):
        return "weight_body_size"
    if has("GROESP", "GROESPG", "身高估计", "身高"):
        return "height_body_size"
    if has("STFUHO", "事故现场道路环境"):
        return "road_environment_type"
    if has("STRKL", "STRART"):
        return "road_classification"
    if has("STROB", "路面情况"):
        return "road_surface"
    if has("LAENGE", "LÄNGE", "LANGE", "车长"):
        return "vehicle_geometry_length"
    if has("HOEHE", "车高"):
        return "vehicle_geometry_height"
    if has("BREITE", "车宽"):
        return "vehicle_geometry_width"
    if has("BRPX", "BRPY", "BRPZ", "STOSSPX", "STOSSPY", "STOSSPZ", "第一碰撞点", "碰撞点坐标"):
        return "crash_contact_geometry"
    if has("FEATURE_OPP_POWERTRAIN", "NEVTYPE", "EV", "HYBRID", "新能源", "动力"):
        return "vehicle_powertrain"
    if has("VEHUSAGE", "车辆用途"):
        return "vehicle_usage"
    if has("BLICHT", "LICHT", "路灯", "照明", "夜间"):
        return "lighting_visibility"
    if has("WINDV", "WEATHER", "天气", "风", "雨", "雪", "雾"):
        return "weather_wind"
    if has("ACSCE", "AEB", "LKA", "COLLWARN", "SPURHAE", "车道保持", "碰撞预警", "主动安全"):
        return "safety_assistance"
    if has("SPURZ", "车道"):
        return "lane_trace"
    if has("GESCHL", "性别"):
        return "sex"
    if has("PTYPE", "人员类型"):
        return "person_type"
    return "other_unique:" + sanitize_column_name(infer_feature_source_group(f) or f)


# Canonical semantic helpers for the manuscript-facing CIDAS VRU pipeline.
# Later definitions override the permissive compatibility helpers above.
_RAW_AGE_ALIASES = {"ALTER1", "FEATURE_AGE_YEARS", "AGE_YEARS"}


def is_raw_age_column(feature: str) -> bool:
    f = _norm_feature_text(feature)
    upper = _ascii_compact(f)
    if f in {"年龄年数记录_ALTER1", "年龄年数记录(ALTER1)", "FEATURE_Age_Years", "FEATURE_AGE_YEARS", "Age_Years", "ALTER1"}:
        return True
    return upper in _RAW_AGE_ALIASES or upper.endswith("_ALTER1") or upper.endswith("_AGE_YEARS")


def missing_category_type(name: str) -> str:
    text = _norm_feature_text(name)
    upper = text.upper()
    compact = _ascii_compact(text)
    lower = text.lower()
    if any(k in text for k in ("有_具体未知", "存在_具体未知")) or any(k in lower for k in ("yes_but_unknown_detail", "partial_unknown_detail")):
        return "partial_unknown_detail"
    if "具体未知" in text and any(k in text for k in ("有", "存在")):
        return "partial_unknown_detail"
    if "MISSING_OR_UNKNOWN" in compact:
        return "pure_missing_unknown"
    if any(tok in text for tok in ("未知", "不详", "缺失", "未记录", "未填", "无资料", "不适用")):
        return "pure_missing_unknown"
    if any(tok in upper for tok in ("UNKNOWN", "MISSING", "N/A")):
        return "pure_missing_unknown"
    if re.search(r"(^|[_\-])(?:9|99|999|9999)[_\-]?(?:UNKNOWN|MISSING|未知|不详)$", upper):
        return "pure_missing_unknown"
    if re.search(r"(?:^|_)(?:9|99|999|9999)(?:_|$)", compact) and any(tok in text for tok in ("未知", "不详")):
        return "pure_missing_unknown"
    return "not_missing_category"


def is_pure_missing_unknown_feature(name: str) -> bool:
    return missing_category_type(name) == "pure_missing_unknown"


def is_missing_category_feature(name: str) -> bool:
    return missing_category_type(name) != "not_missing_category"


def infer_feature_source_group(feature: str) -> str:
    f = _norm_feature_text(feature)
    upper = _ascii_compact(f)
    if f == "FEATURE_Age_Over60":
        return "FEATURE_Age_Over60"
    if re.search(r"(^|_)ALTERG(_|$)", upper) or f.startswith("年龄段_ALTERG_"):
        return "ALTERG"
    if re.search(r"(^|_)GEWGES(_|$)", upper) or "碰撞时的总重" in f:
        return "GEWGES"
    source_patterns = [
        ("事故类型_UTYP", r"(^|_)UTYP(_|$)|^事故类型_UTYP_"),
        ("STFUHO", r"(^|_)STFUHO(_|$)|事故现场道路环境_STFUHO_"),
        ("STRART", r"(^|_)STRART(_|$)"),
        ("STRKL", r"(^|_)STRKL(_|$)"),
        ("ACSCE", r"(^|_)ACSCE(_|$)"),
        ("URSWIS", r"(^|_)URSWIS[123]?(_|$)"),
        ("MARK", r"(^|_)MARK(_|$)"),
        ("BSPUR", r"(^|_)BSPUR(_|$)"),
        ("SPURZ", r"(^|_)SPURZ(_|$)"),
        ("GESCHL", r"(^|_)GESCHL(_|$)"),
        ("VEHUSAGE", r"(^|_)VEHUSAGE(_|$)"),
        ("FEATURE_Opp_Powertrain", r"^FEATURE_OPP_POWERTRAIN(_|$)"),
        ("FEATURE_Age_Group", r"^FEATURE_AGE_GROUP(_|$)"),
        ("BLICHT", r"(^|_)BLICHT(_|$)"),
        ("STROB", r"(^|_)STROB(_|$)"),
        ("WINDV", r"(^|_)WINDV(_|$)"),
    ]
    for group, pattern in source_patterns:
        if re.search(pattern, upper) or re.search(pattern, f):
            return group
    for key in ["V0", "VK", "LAENGE", "LANGE", "BREITE", "HOEHE", "GEWP", "GEWPG", "GROESP", "GROESPG", "ALTER1", "PTYPE", "MUE", "BRPX", "BRPY", "BRPZ", "STOSSPX", "STOSSPY", "STOSSPZ"]:
        if re.search(rf"(^|_){re.escape(key)}($|_)", upper):
            return "LAENGE" if key == "LANGE" else key
    parts = [p for p in f.split("_") if p]
    if len(parts) >= 3:
        return "_".join(parts[:-1])
    return sanitize_column_name(f) if f else "unknown_source"


def infer_feature_semantic_group(feature: str) -> str:
    if missing_category_type(feature) == "pure_missing_unknown":
        return "unknown_missing_category"
    f = _norm_feature_text(feature)
    upper = _ascii_compact(f)
    lower = f.lower()
    if f == "FEATURE_Age_Over60" or upper.startswith("FEATURE_AGE_GROUP_") or "老年_60" in f:
        return "age_over60"
    if re.search(r"(^|_)ALTERG(_|$)", upper) or f.startswith("年龄段_ALTERG_"):
        if any(x in f for x in ("老人", "老年", "71岁以上", "60岁以上")) or re.search(r"(^|_)ALTERG_?11($|_)", upper):
            return "age_elderly_group"
        return "age_group"
    if is_raw_age_column(f):
        return "raw_age_continuous"
    if re.search(r"(^|_)V0($|_)", upper):
        return "precrash_speed"
    if re.search(r"(^|_)VK($|_)", upper) and "VKREG" not in upper:
        return "impact_speed"
    if "GEWPG" in upper or re.search(r"(^|_)GEWP($|_)", upper) or "体重" in f:
        return "weight_body_size"
    if "GROESPG" in upper or "GROESP" in upper or "身高" in f:
        return "height_body_size"
    if "GEWGES" in upper or "碰撞时的总重" in f:
        return "vehicle_mass"
    if "LAENGE" in upper or "LANGE" in upper or "LÄNGE" in f or "车长" in f:
        return "vehicle_geometry_length"
    if "HOEHE" in upper or "车高" in f:
        return "vehicle_geometry_height"
    if "BREITE" in upper or "车宽" in f:
        return "vehicle_geometry_width"
    if any(k in upper for k in ("BRPX", "BRPY", "BRPZ", "STOSSPX", "STOSSPY", "STOSSPZ")) or any(k in f for k in ("第一碰撞点", "碰撞点坐标")):
        return "crash_contact_geometry"
    if "UTYP" in upper or "事故类型" in f:
        return "crash_configuration_type"
    if "STFUHO" in upper or "道路环境" in f:
        return "road_environment_type"
    if "STROB" in upper or "路面情况" in f:
        return "road_surface"
    if any(k in upper for k in ("BLICHT", "LICHT")) or any(k in f for k in ("路灯", "照明", "夜间")):
        return "lighting_visibility"
    if "WINDV" in upper or any(k in f for k in ("天气", "风", "雨", "雪", "雾")):
        return "weather_wind"
    if any(k in upper for k in ("ACSCE", "AEB", "LKA", "COLLWARN", "SPURHAE")) or any(k in f for k in ("车道保持", "碰撞预警", "主动安全")):
        return "safety_assistance"
    if "GESCHL" in upper or "性别" in f or lower in {"sex", "gender"}:
        return "sex"
    if "PTYPE" in upper or "人员类型" in f:
        return "person_type"
    return "other_unique:" + sanitize_column_name(infer_feature_source_group(f) or f)
