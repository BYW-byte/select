
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KEY_ACC = "事故编号(FALL)"
KEY_PART = "参与方编号(BETNR)"
KEY_PERSON = "人员序号(PNUMBER)"
KEYS = [KEY_ACC, KEY_PART, KEY_PERSON]

VRU_KEYWORDS = [
    "二/三轮摩托车", "二/三轮自行车", "二轮电动自行车", "二/三轮电动摩托车",
    "自行车", "摩托车", "电动自行车", "电动摩托车", "行人"
]


# ==========================================
# 通用工具函数
# ==========================================
def ensure_parent_dir(path: str) -> str:
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def safe_to_csv(df: pd.DataFrame, path: str, encoding: str = "utf-8-sig") -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding=encoding)


def normalize_keys(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    for k in KEYS:
        if k in df.columns:
            df[k] = (
                df[k]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
                .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
            )
    return df


def extract_first_number_as_str(series):
    out = series.astype(str).str.extract(r"(\d+)", expand=False)
    return out.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})


def extract_first_number(series):
    return pd.to_numeric(extract_first_number_as_str(series), errors="coerce")


def prefix_non_keys(df, prefix, keep_cols=None):
    keep_cols = set(keep_cols or [])
    rename_map = {}
    for c in df.columns:
        if c not in keep_cols:
            rename_map[c] = f"{prefix}{c}"
    return df.rename(columns=rename_map)


def merge_without_dup(base, other, on=None, left_on=None, right_on=None, how="left"):
    if on is not None:
        merge_keys_left = set(on)
        merge_keys_right = set(on)
    else:
        merge_keys_left = set(left_on or [])
        merge_keys_right = set(right_on or [])

    overlap_cols = [
        c for c in other.columns
        if c in base.columns and c not in merge_keys_right and c not in merge_keys_left
    ]
    if overlap_cols:
        other = other.drop(columns=overlap_cols, errors="ignore")

    return pd.merge(base, other, on=on, left_on=left_on, right_on=right_on, how=how)


def find_sheet_name(sheet_names, keywords):
    def sheet_rank(name):
        match = re.match(r"^(\d+)_", str(name))
        return int(match.group(1)) if match else 9999

    exact = [s for s in sheet_names if all(k in s for k in keywords)]
    if exact:
        return sorted(exact, key=sheet_rank)[0]

    fuzzy = [s for s in sheet_names if any(k in s for k in keywords)]
    if fuzzy:
        return sorted(fuzzy, key=sheet_rank)[0]
    return None


def is_vru_series(series):
    pattern = "|".join(map(re.escape, VRU_KEYWORDS))
    return series.astype(str).str.contains(pattern, na=False)


def append_stage_audit(rows: List[dict], stage: str, df: pd.DataFrame, sheet_name: str = "", note: str = "", **extra) -> None:
    row = {
        "Stage": stage,
        "Sheet": sheet_name,
        "Rows": int(len(df)),
        "Cols": int(df.shape[1]),
        "Unique_Accidents": int(df[KEY_ACC].nunique(dropna=True)) if KEY_ACC in df.columns else np.nan,
        "Unique_AccidentParty": int(df[[KEY_ACC, KEY_PART]].drop_duplicates().shape[0]) if (KEY_ACC in df.columns and KEY_PART in df.columns) else np.nan,
        "Unique_AccidentPartyPerson": int(df[KEYS].drop_duplicates().shape[0]) if set(KEYS).issubset(df.columns) else np.nan,
        "Note": note,
    }
    row.update(extra)
    rows.append(row)


def save_missingness_audit(df: pd.DataFrame, output_dir: str) -> None:
    rows = []
    n = len(df)
    for col in df.columns:
        nonnull = int(df[col].notna().sum())
        rows.append({
            "Feature": col,
            "NonNull_Count": nonnull,
            "Missing_Count": int(n - nonnull),
            "Missing_Ratio": round(float(1 - nonnull / max(1, n)), 6),
            "Dtype": str(df[col].dtype),
        })
    audit = pd.DataFrame(rows).sort_values(
        by=["Missing_Ratio", "NonNull_Count", "Feature"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    safe_to_csv(audit, os.path.join(output_dir, "01_Feature_Missingness_Audit.csv"))


def save_key_feature_audit(df: pd.DataFrame, output_dir: str) -> None:
    preferred = [
        KEY_ACC,
        "TARGET_MAIS_Merged",
        "FEATURE_Age_Years",
        "FEATURE_Age_Group",
        "VEH_意识到危险前的初始速度(V0)",
        "VEH_碰撞速度(VK)",
        "VEH_自动紧急制动系统(AEB1)",
        "VEH_新能源汽车类型(NEVTYPE)",
        "FEATURE_Opp_Is_PureEV",
        "FEATURE_Opp_Is_HybridEV",
        "FEATURE_Opp_Is_NEV",
        "FEATURE_Opp_Powertrain",
    ]
    rows = []
    for col in preferred:
        if col in df.columns:
            rows.append({
                "Feature": col,
                "NonNull_Count": int(df[col].notna().sum()),
                "Missing_Ratio": round(float(df[col].isna().mean()), 6),
                "Unique_Values": int(df[col].nunique(dropna=True)),
                "Dtype": str(df[col].dtype),
            })
    if rows:
        safe_to_csv(pd.DataFrame(rows), os.path.join(output_dir, "01_Key_Feature_NonNull_Audit.csv"))


def deduplicate_base_person_table(df_base, audit_dir="."):
    df_base = normalize_keys(df_base)
    if not set(KEYS).issubset(df_base.columns):
        print("  ⚠️ 人员底表缺少完整主键，跳过去重审计。")
        return df_base

    dup_mask = df_base.duplicated(subset=KEYS, keep=False)
    dup_rows = df_base[dup_mask].copy()
    if dup_rows.empty:
        print("  ✅ 人员底表主键唯一，无重复个体记录。")
        summary = pd.DataFrame([{
            "Duplicate_Row_Count": 0,
            "Duplicate_Group_Count": 0,
            "Deduplicated_Row_Count": int(len(df_base)),
        }])
        safe_to_csv(summary, os.path.join(audit_dir, "01_Base_Duplicate_Summary.csv"))
        return df_base

    dup_group_n = int(dup_rows[KEYS].drop_duplicates().shape[0])
    print(f"  ⚠️ 检测到 {len(dup_rows)} 条重复个体记录，涉及 {dup_group_n} 组主键重复。")

    # Suppress row-level duplicate dump; summary is retained for audit.
    # safe_to_csv(
    #     dup_rows.sort_values(KEYS).reset_index(drop=True),
    #     os.path.join(audit_dir, "01_Base_Duplicate_Rows.csv"),
    # )

    completeness = df_base.notna().sum(axis=1)
    df_base = (
        df_base.assign(__COMPLETE__=completeness)
        .sort_values(KEYS + ["__COMPLETE__"], ascending=[True, True, True, False])
        .drop_duplicates(subset=KEYS, keep="first")
        .drop(columns=["__COMPLETE__"])
        .copy()
    )

    summary = pd.DataFrame([{
        "Duplicate_Row_Count": int(len(dup_rows)),
        "Duplicate_Group_Count": dup_group_n,
        "Deduplicated_Row_Count": int(len(df_base)),
    }])
    safe_to_csv(summary, os.path.join(audit_dir, "01_Base_Duplicate_Summary.csv"))
    print(f"  ✅ 以信息完整度优先去重后，人员底表剩余 {len(df_base)} 行。")
    return df_base


def normalize_age_columns(df):
    df = df.copy()

    age_year_col = next((c for c in df.columns if "年龄年数记录" in c or c == "ALTER1"), None)
    if age_year_col:
        numeric_age = extract_first_number(df[age_year_col])
        numeric_age = numeric_age.where((numeric_age >= 0) & (numeric_age <= 120), pd.NA)
        df[age_year_col] = numeric_age
        df["FEATURE_Age_Years"] = numeric_age
        valid_n = int(numeric_age.notna().sum())
        print(f"  ✅ 年龄年数已规范化: [{age_year_col}] -> 数值年龄，有效 {valid_n}/{len(df)} 行。")

    age_group_col = next((c for c in df.columns if "年龄段" in c or c == "ALTERG"), None)
    if age_group_col:
        df[age_group_col] = (
            df[age_group_col]
            .astype(str)
            .str.strip()
            .replace({"nan": pd.NA, "None": pd.NA, "99-未知": pd.NA, "999-未知": pd.NA})
        )
        print(f"  ✅ 年龄段保留为分类变量: [{age_group_col}]")

    return df


# ==========================================
# 顶刊升级模块 1：时空风险特征工程
# ==========================================
def extract_temporal_features(df_merged):
    print("\n--> 🕒 启动时空风险特征提取...")
    preferred_cols = [
        c for c in df_merged.columns
        if "事故时间(UDAT)" in c or c.endswith("(UDAT)") or ("事故时间" in c and "救援" not in c)
    ]
    time_col = preferred_cols[0] if preferred_cols else next(
        (c for c in df_merged.columns if ("时间" in c or "TIME" in c.upper()) and "救援" not in c and "报警" not in c),
        None
    )

    if time_col:
        try:
            dt_series = pd.to_datetime(
                df_merged[time_col].astype(str).str.strip(),
                errors="coerce",
                format="mixed"
            )
            df_merged["FEATURE_Hour"] = dt_series.dt.hour
            df_merged["FEATURE_Is_RushHour"] = df_merged["FEATURE_Hour"].apply(
                lambda x: 1 if (pd.notna(x) and ((7 <= x <= 9) or (17 <= x <= 19))) else 0
            )
            df_merged["FEATURE_Is_Night"] = df_merged["FEATURE_Hour"].apply(
                lambda x: 1 if (pd.notna(x) and (x >= 19 or x <= 6)) else 0
            )
            df_merged["FEATURE_Is_Weekend"] = dt_series.dt.dayofweek.apply(
                lambda x: 1 if (pd.notna(x) and x >= 5) else 0
            )
            parsed_n = int(dt_series.notna().sum())
            print(f"  ✅ 成功提取早晚高峰/夜间/周末特征。时间成功解析 {parsed_n}/{len(dt_series)} 行。")
        except Exception as e:
            print(f"  ⚠️ 时间特征解析失败，保持原状: {e}")
    else:
        print("  ⚠️ 未检测到明确时间列，跳过时空特征提取。")

    return df_merged


# ==========================================
# 顶刊升级模块 1.2：对方机动车 / 新能源特征工程
# ==========================================
def engineer_counterparty_vehicle_features(df_merged):
    print("\n--> ⚡ 启动对方车辆动力系统特征工程...")

    nev_col = next((c for c in df_merged.columns if "VEH_新能源汽车类型" in c), None)
    motor_col = next((c for c in df_merged.columns if "VEH_发动机类型" in c), None)
    fuel_col = next((c for c in df_merged.columns if "VEH_燃料" in c), None)
    batpow_col = next((c for c in df_merged.columns if "VEH_动力电池容量" in c), None)
    battype_col = next((c for c in df_merged.columns if "VEH_动力电池种类" in c), None)

    def contains_keywords(col, keywords):
        if col is None or col not in df_merged.columns:
            return pd.Series(False, index=df_merged.index)
        pattern = "|".join(map(re.escape, keywords))
        return df_merged[col].astype(str).str.contains(pattern, na=False)

    pure_ev = (
        contains_keywords(nev_col, ["纯电"]) |
        contains_keywords(motor_col, ["电动机"]) |
        contains_keywords(fuel_col, ["电池"])
    )

    hybrid_ev = (
        contains_keywords(nev_col, ["插电式混合", "增程", "混合动力"]) |
        contains_keywords(motor_col, ["混合动力"]) |
        contains_keywords(fuel_col, ["混合动力"])
    )

    fuel_only = (
        contains_keywords(motor_col, ["汽油发动机", "柴油发动机"]) |
        contains_keywords(fuel_col, ["汽油", "柴油"])
    ) & (~pure_ev) & (~hybrid_ev)

    has_battery_struct = pd.Series(False, index=df_merged.index)
    if batpow_col and batpow_col in df_merged.columns:
        has_battery_struct = has_battery_struct | (pd.to_numeric(df_merged[batpow_col], errors="coerce").fillna(0) > 0)
    if battype_col and battype_col in df_merged.columns:
        has_battery_struct = has_battery_struct | (
            df_merged[battype_col].astype(str).str.contains("不适用|未知|nan", na=False) == False
        ) & df_merged[battype_col].notna()

    df_merged["FEATURE_Opp_Is_PureEV"] = pure_ev.astype(int)
    df_merged["FEATURE_Opp_Is_HybridEV"] = hybrid_ev.astype(int)
    df_merged["FEATURE_Opp_Is_NEV"] = (pure_ev | hybrid_ev | has_battery_struct).astype(int)
    df_merged["FEATURE_Opp_Powertrain"] = np.select(
        [pure_ev, hybrid_ev, fuel_only],
        ["纯电/电驱", "混动/增程", "燃油"],
        default="未知"
    )

    print(
        "  ✅ 对方车辆动力系统特征已生成："
        f"纯电={int(df_merged['FEATURE_Opp_Is_PureEV'].sum())}，"
        f"混动/增程={int(df_merged['FEATURE_Opp_Is_HybridEV'].sum())}，"
        f"新能源总计={int(df_merged['FEATURE_Opp_Is_NEV'].sum())}。"
    )
    return df_merged


# ==========================================
# 顶刊升级模块 1.5：MAIS 标签智能融合
# ==========================================
def merge_mais_labels(df_merged):
    print("\n--> 🎯 启动 MAIS 伤情标签智能融合 (最大化保留样本)...")

    col_15 = next((c for c in df_merged.columns if "MAIS15" in c.upper() or "2015" in c), None)
    col_05 = next((c for c in df_merged.columns if "MAIS05" in c.upper() or "2005" in c), None)

    if not col_15 or not col_05:
        print("  ⚠️ 未同时找到 MAIS15 和 MAIS05 列，跳过融合。")
        return df_merged

    def clean_mais(series):
        s = series.astype(str).str.extract(r"(\d)", expand=False)
        s = pd.to_numeric(s, errors="coerce")
        return s.where(s != 9, pd.NA)

    s15 = clean_mais(df_merged[col_15])
    s05 = clean_mais(df_merged[col_05])

    df_merged["TARGET_MAIS_Merged"] = s15.fillna(s05)

    valid_15 = int(s15.notna().sum())
    valid_05 = int(s05.notna().sum())
    valid_merged = int(df_merged["TARGET_MAIS_Merged"].notna().sum())

    print(f"  📊 融合前：MAIS15 有效={valid_15}, MAIS05 有效={valid_05}")
    print(f"  ✅ 融合后：TARGET 最终有效={valid_merged} (挽回了 {valid_merged - valid_15} 个标签！)")
    return df_merged


# ==========================================
# 顶刊升级模块 2：研究队列严格筛选与追踪
# ==========================================
def strict_cohort_selection(df_merged, return_audit: bool = False):
    print("\n--> 📊 启动研究队列纳排程序 (PRISMA Flowchart Data)...")

    flow_rows: List[dict] = []
    initial_n = int(len(df_merged))
    print(f"  [初始入组] 数据基座总样本数: N = {initial_n}")
    flow_rows.append({"Step": "Initial_Base", "Remaining_N": initial_n, "Dropped_N": 0, "Reason": "原始数据基座"})

    target_col = "TARGET_MAIS_Merged"
    if target_col in df_merged.columns:
        before = len(df_merged)
        df_merged = df_merged[df_merged[target_col].notna()].copy()
        after = len(df_merged)
        print(f"  [排除准则 1] 剔除伤情标签未知/缺失样本: 删除了 {before - after} 例, 剩余 N = {after}")
        flow_rows.append({"Step": "Exclude_Missing_Target", "Remaining_N": int(after), "Dropped_N": int(before - after), "Reason": "TARGET_MAIS_Merged 缺失"})
    else:
        flow_rows.append({"Step": "Exclude_Missing_Target", "Remaining_N": initial_n, "Dropped_N": 0, "Reason": "未找到 TARGET_MAIS_Merged"})

    role_col = next((c for c in df_merged.columns if "参与方类型" in c or "ARTTEIL" in c.upper()), None)
    if role_col:
        before = len(df_merged)
        mask_vru = is_vru_series(df_merged[role_col])
        df_merged = df_merged[mask_vru].copy()
        after = len(df_merged)
        print(f"  [排除准则 2] 锁定 VRU 队列: 删除了 {before - after} 例非目标角色, 剩余 N = {after}")
        flow_rows.append({"Step": "Restrict_VRU", "Remaining_N": int(after), "Dropped_N": int(before - after), "Reason": f"基于 {role_col} 锁定 VRU"})
    else:
        flow_rows.append({"Step": "Restrict_VRU", "Remaining_N": int(len(df_merged)), "Dropped_N": 0, "Reason": "未找到角色列"})

    print(f"--> 🏁 队列筛选完毕！最终有效研究队列: N = {len(df_merged)}")
    flow_rows.append({"Step": "Final_Cohort", "Remaining_N": int(len(df_merged)), "Dropped_N": 0, "Reason": "最终分析队列"})

    if return_audit:
        return df_merged, pd.DataFrame(flow_rows)
    return df_merged


# ==========================================
# 顶刊升级模块 3：Apriori 算法前置离散化
# ==========================================
def discretize_continuous_features(df_merged):
    print("\n--> 🧮 启动连续特征离散化 (关联规则挖掘必备)...")

    age_col = "FEATURE_Age_Years" if "FEATURE_Age_Years" in df_merged.columns else next(
        (c for c in df_merged.columns if "年龄年数记录" in c or c == "ALTER1"), None
    )
    if age_col:
        df_merged[age_col] = pd.to_numeric(df_merged[age_col], errors="coerce")
        df_merged["FEATURE_Age_Group"] = pd.cut(
            df_merged[age_col],
            bins=[-1, 18, 60, 150],
            labels=["未成年(<18)", "青壮年(18-60)", "老年(>60)"]
        )
        print(f"  ✅ 成功离散化: {age_col} -> FEATURE_Age_Group")

    speed_col = next((c for c in df_merged.columns if ("V0" in c or "初速" in c or "VELOCITY" in c.upper()) and "VK" not in c), None)
    if speed_col:
        df_merged[speed_col] = pd.to_numeric(df_merged[speed_col], errors="coerce")
        df_merged["VEH_FEATURE_Speed_Group"] = pd.cut(
            df_merged[speed_col],
            bins=[-1, 30, 50, 300],
            labels=["低速(≤30)", "中速(31-50)", "高速(>50)"]
        )
        print(f"  ✅ 成功离散化: {speed_col} -> VEH_FEATURE_Speed_Group")

    return df_merged


# ==========================================
# 参与方映射：优先使用 REKO 精确对方，再用事故内配对回退
# ==========================================
def build_counterparty_map_fallback(df_part):
    part_cols = [c for c in [KEY_ACC, KEY_PART, "参与方类型(ARTTEIL)"] if c in df_part.columns]
    df_part = df_part[part_cols].drop_duplicates().copy()
    df_part["SELF_IS_VRU"] = is_vru_series(df_part["参与方类型(ARTTEIL)"])

    self_df = df_part.rename(columns={
        KEY_PART: "SELF_BETNR",
        "参与方类型(ARTTEIL)": "SELF_ARTTEIL"
    })

    opp_df = df_part.rename(columns={
        KEY_PART: "OPP_BETNR",
        "参与方类型(ARTTEIL)": "OPP_ARTTEIL",
        "SELF_IS_VRU": "OPP_IS_VRU"
    })

    pair_df = self_df.merge(opp_df, on=KEY_ACC, how="left")
    pair_df = pair_df[pair_df["SELF_BETNR"] != pair_df["OPP_BETNR"]].copy()

    pair_df["priority"] = np.where(
        pair_df["SELF_IS_VRU"] & (~pair_df["OPP_IS_VRU"]), 0,
        np.where((~pair_df["SELF_IS_VRU"]) & pair_df["OPP_IS_VRU"], 1, 2)
    )

    pair_df = pair_df.sort_values([KEY_ACC, "SELF_BETNR", "priority", "OPP_BETNR"])
    pair_df = pair_df.drop_duplicates(subset=[KEY_ACC, "SELF_BETNR"], keep="first")

    out = pair_df[[KEY_ACC, "SELF_BETNR", "SELF_ARTTEIL", "OPP_BETNR", "OPP_ARTTEIL"]].copy()
    return out.rename(columns={"SELF_BETNR": KEY_PART})


def build_counterparty_map_from_reko(df_part, df_reko):
    role_col = "参与方类型(ARTTEIL)"
    if role_col not in df_part.columns:
        return pd.DataFrame(columns=[KEY_ACC, KEY_PART, "SELF_ARTTEIL", "OPP_BETNR", "OPP_ARTTEIL"])

    role_map = df_part[[KEY_ACC, KEY_PART, role_col]].drop_duplicates().copy()
    role_map = role_map.rename(columns={role_col: "ROLE"})

    df_reko = normalize_keys(df_reko)
    if KEY_ACC not in df_reko.columns or KEY_PART not in df_reko.columns or "碰撞对方参与方编号(KONBETEI)" not in df_reko.columns:
        return pd.DataFrame(columns=[KEY_ACC, KEY_PART, "SELF_ARTTEIL", "OPP_BETNR", "OPP_ARTTEIL"])

    df_reko["KONBETEI_CLEAN"] = extract_first_number_as_str(df_reko["碰撞对方参与方编号(KONBETEI)"])
    df_reko = df_reko[
        df_reko["KONBETEI_CLEAN"].notna() &
        (~df_reko["KONBETEI_CLEAN"].isin(["0"])) &
        (df_reko["KONBETEI_CLEAN"] != df_reko[KEY_PART])
    ].copy()

    for c in ["事故碰撞编号(ACOLLINO)", "参与方碰撞编号(PCOLLINO)"]:
        if c in df_reko.columns:
            df_reko[f"__SORT__{c}"] = pd.to_numeric(extract_first_number_as_str(df_reko[c]), errors="coerce").fillna(9999)
        else:
            df_reko[f"__SORT__{c}"] = 9999

    df_reko = df_reko.merge(
        role_map.rename(columns={KEY_PART: KEY_PART, "ROLE": "SELF_ARTTEIL"}),
        on=[KEY_ACC, KEY_PART],
        how="left"
    )
    df_reko = df_reko.merge(
        role_map.rename(columns={KEY_PART: "KONBETEI_CLEAN", "ROLE": "OPP_ARTTEIL"}),
        on=[KEY_ACC, "KONBETEI_CLEAN"],
        how="left"
    )

    df_reko["SELF_IS_VRU"] = is_vru_series(df_reko["SELF_ARTTEIL"])
    df_reko["OPP_IS_VRU"] = is_vru_series(df_reko["OPP_ARTTEIL"])

    df_reko["priority"] = np.select(
        [
            df_reko["SELF_IS_VRU"] & (~df_reko["OPP_IS_VRU"]),
            (~df_reko["SELF_IS_VRU"]) & df_reko["OPP_IS_VRU"]
        ],
        [0, 1],
        default=2
    )

    df_reko = df_reko.sort_values(
        by=[KEY_ACC, KEY_PART, "priority", "__SORT__事故碰撞编号(ACOLLINO)", "__SORT__参与方碰撞编号(PCOLLINO)", "KONBETEI_CLEAN"],
        ascending=[True, True, True, True, True, True]
    )

    out = df_reko.drop_duplicates(subset=[KEY_ACC, KEY_PART], keep="first")[
        [KEY_ACC, KEY_PART, "SELF_ARTTEIL", "KONBETEI_CLEAN", "OPP_ARTTEIL"]
    ].copy()
    out = out.rename(columns={"KONBETEI_CLEAN": "OPP_BETNR"})
    return out


def build_counterparty_map(df_part, df_reko=None, return_stats: bool = False):
    fallback_map = build_counterparty_map_fallback(df_part)
    stats: Dict[str, float] = {
        "Fallback_Pairs": int(len(fallback_map)),
        "REKO_Exact_Pairs": 0,
        "Final_Mapped_Pairs": int(fallback_map["OPP_BETNR"].notna().sum()) if "OPP_BETNR" in fallback_map.columns else 0,
        "Used_REKO_Pairs": 0,
        "Final_Mapping_Rate": round(float(fallback_map["OPP_BETNR"].notna().mean()), 6) if "OPP_BETNR" in fallback_map.columns else np.nan,
    }
    if df_reko is None:
        return (fallback_map, stats) if return_stats else fallback_map

    reko_map = build_counterparty_map_from_reko(df_part, df_reko)
    if reko_map.empty:
        return (fallback_map, stats) if return_stats else fallback_map

    merged = fallback_map.merge(
        reko_map.rename(columns={
            "SELF_ARTTEIL": "SELF_ARTTEIL_REKO",
            "OPP_BETNR": "OPP_BETNR_REKO",
            "OPP_ARTTEIL": "OPP_ARTTEIL_REKO"
        }),
        on=[KEY_ACC, KEY_PART],
        how="left"
    )

    used_reko_pairs = int(merged["OPP_BETNR_REKO"].notna().sum())
    merged["SELF_ARTTEIL"] = merged["SELF_ARTTEIL_REKO"].combine_first(merged["SELF_ARTTEIL"])
    merged["OPP_BETNR"] = merged["OPP_BETNR_REKO"].combine_first(merged["OPP_BETNR"])
    merged["OPP_ARTTEIL"] = merged["OPP_ARTTEIL_REKO"].combine_first(merged["OPP_ARTTEIL"])
    merged = merged.drop(columns=["SELF_ARTTEIL_REKO", "OPP_BETNR_REKO", "OPP_ARTTEIL_REKO"])

    stats = {
        "Fallback_Pairs": int(len(fallback_map)),
        "REKO_Exact_Pairs": int(reko_map["OPP_BETNR"].notna().sum()),
        "Used_REKO_Pairs": used_reko_pairs,
        "Final_Mapped_Pairs": int(merged["OPP_BETNR"].notna().sum()),
        "Final_Mapping_Rate": round(float(merged["OPP_BETNR"].notna().mean()), 6),
    }

    print(f"  ✅ 对方参与方映射已升级：REKO 精确命中 {stats['REKO_Exact_Pairs']} 个目标参与方，其余使用事故内回退映射。")
    return (merged, stats) if return_stats else merged


# ==========================================
# 通用合并模块
# ==========================================
def merge_accident_level(base_df, df_temp, sheet_name):
    df_temp = normalize_keys(df_temp)
    df_temp = df_temp.drop_duplicates(subset=[KEY_ACC], keep="first")
    print(f"  🌍 融合事故级宏观环境特征: [{sheet_name}]")
    return merge_without_dup(base_df, df_temp, on=[KEY_ACC], how="left")


def merge_target_level(base_df, df_temp, merge_keys, sheet_name):
    df_temp = normalize_keys(df_temp)
    df_temp = df_temp.drop_duplicates(subset=merge_keys, keep="first")
    print(f"  👤 融合目标个体/参与方特征: [{sheet_name}]")
    return merge_without_dup(base_df, df_temp, on=merge_keys, how="left")


def merge_road_level(base_df, df_temp, sheet_name):
    df_temp = normalize_keys(df_temp)
    keep_keys = [k for k in [KEY_ACC, KEY_PART] if k in df_temp.columns]
    if set(keep_keys) == {KEY_ACC, KEY_PART}:
        df_temp = df_temp.drop_duplicates(subset=[KEY_ACC, KEY_PART], keep="first")
        print(f"  🛣️ 融合目标参与方道路暴露特征(按事故+参与方): [{sheet_name}]")
        return merge_without_dup(base_df, df_temp, on=[KEY_ACC, KEY_PART], how="left")
    else:
        df_temp = df_temp.drop_duplicates(subset=[KEY_ACC], keep="first")
        print(f"  🛣️ 融合道路特征(退化为事故级): [{sheet_name}]")
        return merge_without_dup(base_df, df_temp, on=[KEY_ACC], how="left")


def merge_opponent_vehicle_sheet(base_df, df_temp, sheet_name, prefix="VEH_"):
    df_temp = normalize_keys(df_temp)
    if KEY_ACC not in df_temp.columns or KEY_PART not in df_temp.columns:
        print(f"  ⚠️ 跳过 [{sheet_name}]：缺少事故/参与方键。")
        return base_df

    df_temp = df_temp.drop_duplicates(subset=[KEY_ACC, KEY_PART], keep="first")
    df_temp = prefix_non_keys(df_temp, prefix=prefix, keep_cols=[KEY_ACC, KEY_PART])
    df_temp = df_temp.rename(columns={KEY_PART: "OPP_BETNR"})
    print(f"  🚘 融合对方机动车特征: [{sheet_name}]")
    return merge_without_dup(base_df, df_temp, on=[KEY_ACC, "OPP_BETNR"], how="left")


def prepare_reko_tables(df_reko):
    df_reko = normalize_keys(df_reko)
    if "碰撞对方参与方编号(KONBETEI)" in df_reko.columns:
        df_reko["KONBETEI_CLEAN"] = extract_first_number_as_str(df_reko["碰撞对方参与方编号(KONBETEI)"])
    else:
        df_reko["KONBETEI_CLEAN"] = pd.NA

    for c in ["事故碰撞编号(ACOLLINO)", "参与方碰撞编号(PCOLLINO)"]:
        if c in df_reko.columns:
            df_reko[f"__SORT__{c}"] = pd.to_numeric(extract_first_number_as_str(df_reko[c]), errors="coerce").fillna(9999)
        else:
            df_reko[f"__SORT__{c}"] = 9999

    df_reko["__HAS_TARGET_OPP"] = df_reko["KONBETEI_CLEAN"].notna() & (~df_reko["KONBETEI_CLEAN"].isin(["0"]))
    df_reko = df_reko.sort_values(
        by=[KEY_ACC, KEY_PART, "__HAS_TARGET_OPP", "__SORT__事故碰撞编号(ACOLLINO)", "__SORT__参与方碰撞编号(PCOLLINO)"],
        ascending=[True, True, False, True, True]
    )

    exact = df_reko[df_reko["__HAS_TARGET_OPP"]].drop_duplicates(subset=[KEY_ACC, KEY_PART, "KONBETEI_CLEAN"], keep="first")
    fallback = df_reko.drop_duplicates(subset=[KEY_ACC, KEY_PART], keep="first")
    return exact, fallback


def merge_reko_vehicle_sheet(base_df, df_reko, sheet_name, prefix="VEH_"):
    if KEY_ACC not in df_reko.columns or KEY_PART not in df_reko.columns:
        print(f"  ⚠️ 跳过 [{sheet_name}]：缺少事故/参与方键。")
        return base_df

    exact, fallback = prepare_reko_tables(df_reko)
    print(f"  🚑 融合对方车辆碰撞动力学特征(优先主碰撞): [{sheet_name}]")

    exact_merge = prefix_non_keys(
        exact.drop(columns=[c for c in exact.columns if c.startswith("__SORT__") or c == "__HAS_TARGET_OPP"], errors="ignore"),
        prefix=prefix,
        keep_cols=[KEY_ACC, KEY_PART, "KONBETEI_CLEAN"]
    ).rename(columns={KEY_PART: "OPP_BETNR", "KONBETEI_CLEAN": KEY_PART})

    out_df = merge_without_dup(
        base_df,
        exact_merge,
        on=[KEY_ACC, KEY_PART, "OPP_BETNR"],
        how="left"
    )

    fallback_merge = prefix_non_keys(
        fallback.drop(columns=[c for c in fallback.columns if c.startswith("__SORT__") or c in ["__HAS_TARGET_OPP", "KONBETEI_CLEAN"]], errors="ignore"),
        prefix=prefix,
        keep_cols=[KEY_ACC, KEY_PART]
    ).rename(columns={KEY_PART: "OPP_BETNR"})

    fallback_view = pd.merge(
        out_df[[KEY_ACC, KEY_PART, "OPP_BETNR"]],
        fallback_merge,
        on=[KEY_ACC, "OPP_BETNR"],
        how="left"
    )
    fill_cols = [c for c in fallback_merge.columns if c not in [KEY_ACC, "OPP_BETNR"]]
    for c in fill_cols:
        if c not in out_df.columns:
            out_df[c] = fallback_view[c]
        else:
            out_df[c] = out_df[c].where(out_df[c].notna(), fallback_view[c])

    return out_df


# ==========================================
# 主程序：数据基座构建
# ==========================================
def merge_cidas_from_excel(excel_path, output_file):
    print("=" * 72)
    print("🚀 启动 [01_数据基座构建]：对方车辆精确映射 + 年龄修正 + 审计导出")
    print("=" * 72)

    if not os.path.exists(excel_path):
        print(f"❌ 错误：找不到文件 '{excel_path}'")
        return

    output_dir = ensure_parent_dir(output_file)
    merge_audit_rows: List[dict] = []

    print("📂 正在加载 Excel 工作簿...")
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    base_sheet_name = find_sheet_name(sheet_names, ["3_人员数据"]) or find_sheet_name(sheet_names, ["PERSDAT"])
    part_sheet_name = find_sheet_name(sheet_names, ["2_事故参与方"]) or find_sheet_name(sheet_names, ["BETEIL"])
    acc_sheet_name = find_sheet_name(sheet_names, ["1_事故概览"]) or find_sheet_name(sheet_names, ["UMWELT"])
    road_sheet_name = find_sheet_name(sheet_names, ["10_道路数据"]) or find_sheet_name(sheet_names, ["STRASSE"])
    visit_sheet_name = find_sheet_name(sheet_names, ["11_回访数据"]) or find_sheet_name(sheet_names, ["BEFRAG"])
    fzg_sheet_name = find_sheet_name(sheet_names, ["12_车辆一般数据"]) or find_sheet_name(sheet_names, ["FZG"])
    pcsafe_sheet_name = find_sheet_name(sheet_names, ["18_乘用车安全配置"]) or find_sheet_name(sheet_names, ["PCSAFE"])
    tire_sheet_name = find_sheet_name(sheet_names, ["36_轮胎数据"]) or find_sheet_name(sheet_names, ["REIFEN"])
    reko_sheet_name = find_sheet_name(sheet_names, ["44_碰撞详细数据"]) or find_sheet_name(sheet_names, ["REKO"])

    sheet_audit = pd.DataFrame([
        {"Logical_Table": "Base_Person", "Detected_Sheet": base_sheet_name or "", "Found": int(bool(base_sheet_name))},
        {"Logical_Table": "Participant", "Detected_Sheet": part_sheet_name or "", "Found": int(bool(part_sheet_name))},
        {"Logical_Table": "Accident", "Detected_Sheet": acc_sheet_name or "", "Found": int(bool(acc_sheet_name))},
        {"Logical_Table": "Road", "Detected_Sheet": road_sheet_name or "", "Found": int(bool(road_sheet_name))},
        {"Logical_Table": "Visit", "Detected_Sheet": visit_sheet_name or "", "Found": int(bool(visit_sheet_name))},
        {"Logical_Table": "Vehicle", "Detected_Sheet": fzg_sheet_name or "", "Found": int(bool(fzg_sheet_name))},
        {"Logical_Table": "PCSAFE", "Detected_Sheet": pcsafe_sheet_name or "", "Found": int(bool(pcsafe_sheet_name))},
        {"Logical_Table": "Tire", "Detected_Sheet": tire_sheet_name or "", "Found": int(bool(tire_sheet_name))},
        {"Logical_Table": "REKO", "Detected_Sheet": reko_sheet_name or "", "Found": int(bool(reko_sheet_name))},
    ])
    # Nonessential intermediate audit suppressed for paper-run cleanliness.
    # safe_to_csv(sheet_audit, os.path.join(output_dir, "01_Sheet_Detection_Audit.csv"))

    if not base_sheet_name:
        raise RuntimeError("未找到人员底表，请确认 Excel 中是否存在 '3_人员数据' 或 'PERSDAT'。")

    df_base = normalize_keys(pd.read_excel(xls, sheet_name=base_sheet_name).copy())
    print(f"🟢 成功读取人员底表 [{base_sheet_name}]，初始总行数：{len(df_base)}。")
    append_stage_audit(merge_audit_rows, "Load_Base", df_base, sheet_name=base_sheet_name, note="读取人员底表")

    df_base = deduplicate_base_person_table(df_base, audit_dir=output_dir)
    append_stage_audit(merge_audit_rows, "Deduplicate_Base", df_base, sheet_name=base_sheet_name, note="按 KEY_ACC+KEY_PART+KEY_PERSON 去重")

    df_base = normalize_age_columns(df_base)
    append_stage_audit(merge_audit_rows, "Normalize_Age", df_base, note="规范年龄年数和年龄段")

    df_reko_raw = None
    if reko_sheet_name:
        df_reko_raw = normalize_keys(pd.read_excel(xls, sheet_name=reko_sheet_name).copy())

    mapping_audit_rows: List[dict] = []
    if part_sheet_name:
        df_part = normalize_keys(pd.read_excel(xls, sheet_name=part_sheet_name).copy())
        df_base = merge_target_level(df_base, df_part, [KEY_ACC, KEY_PART], part_sheet_name)
        append_stage_audit(merge_audit_rows, "Merge_Participant", df_base, sheet_name=part_sheet_name, note="融合目标参与方表")

        pair_map, mapping_stats = build_counterparty_map(df_part, df_reko=df_reko_raw, return_stats=True)
        df_base = merge_without_dup(df_base, pair_map, on=[KEY_ACC, KEY_PART], how="left")
        mapped_n = int(df_base["OPP_BETNR"].notna().sum()) if "OPP_BETNR" in df_base.columns else 0
        print(f"  ✅ 对方参与方映射完成：{mapped_n}/{len(df_base)} 行已识别到对方参与方编号。")
        append_stage_audit(
            merge_audit_rows,
            "Merge_Counterparty_Map",
            df_base,
            sheet_name=part_sheet_name,
            note="融合对方参与方映射",
            OPP_Mapped_Rows=mapped_n,
            OPP_Mapped_Rate=round(float(mapped_n / max(1, len(df_base))), 6),
        )
        mapping_stats["Rows_In_Base_After_Merge"] = int(len(df_base))
        mapping_stats["Mapped_Rows_In_Base"] = mapped_n
        mapping_stats["Mapped_Rate_In_Base"] = round(float(mapped_n / max(1, len(df_base))), 6)
        mapping_audit_rows.append(mapping_stats)
    else:
        print("  ⚠️ 未找到事故参与方表，无法构建对方机动车映射。")

    if acc_sheet_name:
        df_base = merge_accident_level(df_base, pd.read_excel(xls, sheet_name=acc_sheet_name).copy(), acc_sheet_name)
        append_stage_audit(merge_audit_rows, "Merge_Accident", df_base, sheet_name=acc_sheet_name, note="融合事故级宏观环境")

    if road_sheet_name:
        df_base = merge_road_level(df_base, pd.read_excel(xls, sheet_name=road_sheet_name).copy(), road_sheet_name)
        append_stage_audit(merge_audit_rows, "Merge_Road", df_base, sheet_name=road_sheet_name, note="融合道路暴露特征")

    if visit_sheet_name:
        visit_df = normalize_keys(pd.read_excel(xls, sheet_name=visit_sheet_name).copy())
        visit_keys = [KEY_ACC, KEY_PART]
        if KEY_PERSON in visit_df.columns:
            visit_keys.append(KEY_PERSON)
        df_base = merge_target_level(df_base, visit_df, visit_keys, visit_sheet_name)
        append_stage_audit(merge_audit_rows, "Merge_Visit", df_base, sheet_name=visit_sheet_name, note="融合回访表")

    if fzg_sheet_name and "OPP_BETNR" in df_base.columns:
        df_base = merge_opponent_vehicle_sheet(df_base, pd.read_excel(xls, sheet_name=fzg_sheet_name).copy(), fzg_sheet_name, prefix="VEH_")
        append_stage_audit(merge_audit_rows, "Merge_Opponent_Vehicle", df_base, sheet_name=fzg_sheet_name, note="融合对方车辆一般特征")

    if pcsafe_sheet_name and "OPP_BETNR" in df_base.columns:
        df_base = merge_opponent_vehicle_sheet(df_base, pd.read_excel(xls, sheet_name=pcsafe_sheet_name).copy(), pcsafe_sheet_name, prefix="VEH_")
        append_stage_audit(merge_audit_rows, "Merge_Opponent_PCSAFE", df_base, sheet_name=pcsafe_sheet_name, note="融合对方安全配置")

    if tire_sheet_name and "OPP_BETNR" in df_base.columns:
        df_base = merge_opponent_vehicle_sheet(df_base, pd.read_excel(xls, sheet_name=tire_sheet_name).copy(), tire_sheet_name, prefix="VEH_")
        append_stage_audit(merge_audit_rows, "Merge_Opponent_Tire", df_base, sheet_name=tire_sheet_name, note="融合对方轮胎特征")

    if reko_sheet_name and "OPP_BETNR" in df_base.columns and df_reko_raw is not None:
        df_base = merge_reko_vehicle_sheet(df_base, df_reko_raw, reko_sheet_name, prefix="VEH_")
        append_stage_audit(merge_audit_rows, "Merge_REKO_Vehicle", df_base, sheet_name=reko_sheet_name, note="融合 REKO 对方碰撞动力学")

    df_base = extract_temporal_features(df_base)
    append_stage_audit(merge_audit_rows, "Feature_Temporal", df_base, note="提取时间相关特征")

    df_base = engineer_counterparty_vehicle_features(df_base)
    append_stage_audit(merge_audit_rows, "Feature_OppPowertrain", df_base, note="提取对方车辆动力系统特征")

    df_base = merge_mais_labels(df_base)
    append_stage_audit(merge_audit_rows, "Merge_MAIS", df_base, note="MAIS05/15 融合")

    df_base, cohort_audit = strict_cohort_selection(df_base, return_audit=True)
    append_stage_audit(merge_audit_rows, "Cohort_Selection", df_base, note="限制到 VRU 且目标非缺失")

    df_base = discretize_continuous_features(df_base)
    append_stage_audit(merge_audit_rows, "Discretize_Continuous", df_base, note="为规则挖掘离散化年龄和速度")

    for tag in [
        "FEATURE_Age_Years",
        "VEH_意识到危险前的初始速度(V0)",
        "VEH_碰撞速度(VK)",
        "VEH_自动紧急制动系统(AEB1)",
        "VEH_新能源汽车类型(NEVTYPE)"
    ]:
        if tag in df_base.columns:
            print(f"  📌 质检 | {tag} 非空数: {int(df_base[tag].notna().sum())}")

    print("\n--> 🧹 启动顶刊级特征提纯：清除数据泄露、事后变量与纯标识符...")

    exact_black_list = [
        KEY_PART, KEY_PERSON, "人员索引编号(PSKZ)",
        "SELF_ARTTEIL", "OPP_BETNR", "OPP_ARTTEIL",
        "连续事故编号(UNR)", "交警队(REVIER)", "采集站点(ORT)", "事故采集进程状态(STATUS)",
        "权重因子(WFAKT)", "VEH_案件流程状态", "VEH_VIN码(FGSTNR)", "VEH_事故碰撞编号(ACOLLINO)",
        "VEH_参与方碰撞编号(PCOLLINO)", "VEH_碰撞对方参与方编号(KONBETEI)", "VEH_碰撞对方的碰撞编号(OPPOCOLLINO)",
        "VEH_挂车编号(TRAINO)", "人员伤口数(WOUNDNO)", "受伤人数(ANZVERL)", "死亡人数(ANZG)",
        "实际采集人数(NACP)", "治疗措施(BEHAND)", "技术救援措施1(TBERG1)", "技术救援措施2(TBERG2)",
        "技术救援措施3(TBERG3)", "是否有救护车到达事故现场(ANRTW)", "到达现场的救护车数量(NOANRTW)",
        "消防部门是否达到现场(ANRUEST)", "到达现场消防车数量(NOANRUEST)", "由已发生事故造成的事故(FOLGE1)",
        "导致其他事故发生的事故(FOLGE2)", "交警到达事故现场的形态(SITEPOLARR)", "事故调查小组(ARU)到达时的事故现场(EINTRUFO)",
        "事故对交通的影响(ABSOU)", "紧急救援人员的安全措施 1(ABSRET1)", "紧急救援人员的安全措施 2(ABSRET2)",
        "紧急救援人员的安全措施 3(ABSRET3)", "紧急救援人员采取的安全措施的评价(ABSRETQ)", "现场绘图(STEREO)",
        "经济损失( 车辆维修)万元(VEHLOSS)", "经济损失( 人员救治)万元(HUMLOSS)", "责任(ANTSCH)",
        "驾驶员肇事逃逸(VFLUCHT)", "事故描述(HERGANG)", "参与方所在道路名称(ROADNAME)", "VEH_车辆的特殊性(TEXTF)",
        "VEH_柱状物名称(COLUNAME)", "VEH_品牌型号(行驶证上的)(VML)", "VEH_车辆制造商(VMR)",
        "VEH_发动机型号(ENGMOD)", "VEH_车辆生产日期(PRODDATE)", "VEH_车辆注册日期(REGIDATE)",
        "VEH_救援的复杂程度(BERGKOMP)", "VEH_救援复杂程度的评价(BERGKW)", "VEH_燃油泄漏(BENZAUS)",
        "VEH_机油泄漏(OELAUS)", "VEH_其他液体泄漏1(FLAUS1)", "VEH_其他液体泄漏2(FLAUS2)", "VEH_其他液体泄漏3(FLAUS3)",
        "VEH_起火(BRANDURS)", "VEH_火势大小(BRAND)", "VEH_使用灭火器(FLB)", "VEH_灭火器使用者(FLBP)",
        "VEH_碰撞后的车况(FZGPOST)", "VEH_碰撞后的车辆姿态(FZGENDL)", "VEH_车辆最终位置(ENDLAGE)",
        "VEH_救援单(RESOR)", "VEH_车辆损坏程度(FZGB)", "时间", "日期", "姓名", "GPS", "视频", "录音", "勘查", "拼音", "备注", "检验编号", "车牌号",
        "事故分类(ACCDEG)", "案件流程状态", "事故时间(UDAT)", "出生日期(GEBDATUM)", "初次领证日期(FLICDATE)",
        "事故地点GPS纬度(ORTGPSNB)", "事故地点GPS经度(ORTGPSOL)", "从事故发生到得到救援的时间(TBERGM)",
        "报警时间(REPTIME)", "通知事故调查小组时间(ALAARUTIME)", "交警到达事故现场的时间(POLICETIME)",
        "事故调查小组达到现场时间(ARUARTIME)", "事故调查小组离开现场时间(ARUDEPTIME)", "被采集人员数(BNACP)",
        "现场勘查(SCENEIVG)", "停车场勘查(GARAMEA)", "现场录音(SCENEAUD)", "现场录像(SCENEVID)",
        "现场无人机勘查(SCENEDRONE)", "停车场无人机勘查(GARADRONE)", "监控视频(VIDEO)",
        "简明损伤定级标准身体各部位最大创伤值 (AIS2005 版)(MAIS05)",
        "简明损伤定级标准身体各部位最大创伤值 (AIS2015 版)(MAIS15)",
        "人员受伤情况(PVERL)",
        "VEH_Delta-V(DV)", "VEH_EES(EES)",
        "VEH_钻入后重叠部分的长度(UFAHRCM)", "VEH_钻入后重叠部分的高度(UFAHRH)", "VEH_护栏的接触长度(SLPANKEB)",
        "VEH_车上最大受力点X坐标(STOSSPX)", "VEH_车上最大受力点Y坐标(STOSSPY)", "VEH_车上最大受力点Z坐标(STOSSPZ)",
        "VEH_初速计算所用数据来源及方法(INIVELCAL)", "VEH_碰撞时的速度计算所用数据来源及方法(IMPVELCAL)",
        "回访地点(ORTBEF)", "调查方式(WIE)", "事故地点(ASITE)", "VEH_气囊特殊情况描述(VEHABDES)",
        "道路使用者的安全措施 1(ABSVT1)", "道路使用者的安全措施 2(ABSVT2)", "道路使用者的安全措施 3(ABSVT3)",
        "道路使用者采取的安全措施的好坏(ABSVTQ)", "事故参与者为何要采取安全措施(RUSECWHY)", "事故现场的辨识程度(ABSAUF)",
        "VEH_驾驶员正面安全气囊(DABST1)", "VEH_副驾驶员正面安全气囊(FABST1)", "VEH_驾驶员侧面安全气囊(DSABST1)",
        "VEH_副驾驶员侧面安全气囊(FSABST1)", "VEH_左侧侧面气帘(LCABST1)", "VEH_右侧侧面气帘(RCABST1)",
        "VEH_后排左侧侧面安全气囊(RLCABST1)", "VEH_后排右侧侧面安全气囊(RRCABST1)", "VEH_驾驶员膝部安全气囊(DKABST1)",
        "VEH_副驾驶员膝部安全气囊(FKABST1)", "VEH_其他气囊1(OTHABST1)", "VEH_其他气囊2(OTHABST2)"
    ]

    fuzzy_black_list = [
        "第一次碰撞", "第二次碰撞", "第三次碰撞", "第四次碰撞", "第五次碰撞",
        "第三轴", "第四轴", "第五轴", "第六轴",
        "轮胎品牌", "轮胎系列", "轮胎其他尺寸",
        "VDI", "乘员舱变形", "乘员舱损伤", "变形深度",
        "气囊", "气帘", "安全带", "方向盘", "仪表板", "座椅", "后视镜", "头枕", "内饰",
        "巡航", "空调", "导航", "多媒体", "车载电话", "天窗", "门锁", "玻璃升降", "防盗",
        "救护车", "交警", "案件", "笔录", "医院", "保险", "救援", "损失", "鉴定", "检验",
        "乘员", "核定载客"
    ]

    protected_cols = {
        KEY_ACC,
        "TARGET_MAIS_Merged",
        "FEATURE_Age_Years", "FEATURE_Age_Group",
        "FEATURE_Opp_Is_PureEV", "FEATURE_Opp_Is_HybridEV", "FEATURE_Opp_Is_NEV", "FEATURE_Opp_Powertrain"
    }

    cols_to_drop = []
    for c in df_base.columns:
        c_clean = c.strip()
        if c_clean.startswith("FEATURE_") or c_clean.startswith("VEH_FEATURE_") or c_clean in protected_cols:
            continue
        if c_clean in exact_black_list:
            cols_to_drop.append(c)
            continue
        if any(kw in c_clean for kw in fuzzy_black_list):
            cols_to_drop.append(c)

    df_base = df_base.drop(columns=cols_to_drop, errors="ignore").copy()
    df_base.columns = df_base.columns.str.strip()
    append_stage_audit(
        merge_audit_rows,
        "Feature_Prune_Final",
        df_base,
        note="去除泄露变量、事后变量和纯标识符；保留 KEY_ACC 供事故级分组切分",
        Dropped_Columns=int(len(cols_to_drop)),
    )

    print(f"  🔪 成功剔除 {len(cols_to_drop)} 个冗余特征，已保留事故编号、年龄修正变量与对方新能源汽车变量。")

    safe_to_csv(pd.DataFrame(merge_audit_rows), os.path.join(output_dir, "01_Merge_Audit.csv"))
    safe_to_csv(cohort_audit, os.path.join(output_dir, "01_Cohort_Flow_Audit.csv"))
    # Nonessential mapping/key-feature audits suppressed; keep cohort/merge/missingness audits only.
    # if mapping_audit_rows:
    #     safe_to_csv(pd.DataFrame(mapping_audit_rows), os.path.join(output_dir, "01_Counterparty_Mapping_Audit.csv"))

    # save_key_feature_audit(df_base, output_dir)
    save_missingness_audit(df_base, output_dir)

    print("\n" + "🔥" * 24)
    print("✅ 数据基座学术提纯完成！")
    print(f"📊 最终输出：{df_base.shape[0]} 样本 | {df_base.shape[1]} 特征。")
    df_base.to_csv(output_file, index=False, encoding="gb18030")
    print(f"💾 已保存至工作目录: {output_file}")
    print("🧾 审计文件已导出：01_Merge_Audit.csv / 01_Cohort_Flow_Audit.csv / 01_Feature_Missingness_Audit.csv")
    print("🔥" * 24)


if __name__ == "__main__":
    merge_cidas_from_excel("data.xlsx", "Cleaned_Data_Base.csv")
