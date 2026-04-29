# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _config import *  # noqa: F403,F401
from _utils import (
    as_binary,
    find_group_col,
    infer_family,
    is_binary_like,
    make_unique,
    normalize_group_ids,
    read_csv_smart,
    require_columns,
    safe_numeric,
    sanitize_column_name,
    sentinel_aware_numeric,
    sentinel_mask,
    smd_binary,
    smd_numeric,
    write_csv,
    write_json,
)


def infer_bound(col: str):
    upper = str(col).upper()
    if any(k in upper for k in ("LAENGE", "LÄNGE")) or "车长" in str(col):
        return PHYSICAL_BOUNDS["LAENGE"]
    if "BREITE" in upper or "车宽" in str(col):
        return PHYSICAL_BOUNDS["BREITE"]
    if "HOEHE" in upper or "车高" in str(col):
        return PHYSICAL_BOUNDS["HOEHE"]
    for key, bound in PHYSICAL_BOUNDS.items():
        if key.upper() in upper or key in str(col):
            if key == "VK" and "VKREG" in upper:
                continue
            return bound
    return None


def is_vehicle_dimension(col: str) -> bool:
    upper = str(col).upper()
    return any(k in upper for k in ("LAENGE", "LÄNGE", "BREITE", "HOEHE")) or any(k in str(col) for k in ("车长", "车宽", "车高"))


def is_engineering_continuous(col: str) -> bool:
    upper = str(col).upper()
    return any(p.upper() in upper or p in str(col) for p in CONTINUOUS_PATTERNS)


def choose_grouped_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    desired_test_n = int(round(len(X) * TEST_SIZE))
    candidates = []
    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for tr_idx, te_idx in sgkf.split(X, y, groups=groups):
            score = abs(len(te_idx) - desired_test_n) + 100.0 * abs(float(y.iloc[tr_idx].mean()) - float(y.iloc[te_idx].mean()))
            candidates.append((score, tr_idx, te_idx))
        if candidates:
            _, tr_idx, te_idx = min(candidates, key=lambda x: x[0])
            return tr_idx, te_idx, "StratifiedGroupKFold"
    except Exception:
        pass
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    return tr_idx, te_idx, "GroupShuffleSplit"


def split_column_types(df: pd.DataFrame, label_col: str, group_col: str):
    drop_cols = {label_col, group_col, TARGET_SOURCE_COL}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    engineering_cols, numeric_cols, categorical_cols = [], [], []
    for c in feature_cols:
        if is_engineering_continuous(c):
            engineering_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]) and not is_binary_like(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return engineering_cols, numeric_cols, categorical_cols


def preprocess_engineering(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]):
    tr_out, te_out, audit_rows = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index), []
    for c in cols:
        tr_raw = sentinel_aware_numeric(train[c], c)
        te_raw = sentinel_aware_numeric(test[c], c)
        bound = infer_bound(c)
        median = float(tr_raw.median()) if tr_raw.notna().any() else 0.0
        tr_imp = tr_raw.fillna(median)
        te_imp = te_raw.fillna(median)
        before_tr_min, before_tr_max = float(tr_imp.min()), float(tr_imp.max())
        before_te_min, before_te_max = float(te_imp.min()), float(te_imp.max())
        tr_clip, te_clip = tr_imp.copy(), te_imp.copy()
        lo, hi = (None, None) if bound is None else bound
        if bound is not None:
            tr_clip = tr_clip.clip(lo, hi)
            te_clip = te_clip.clip(lo, hi)
        tr_out[c] = tr_clip.astype(float)
        te_out[c] = te_clip.astype(float)
        for split_name, raw, imp, clip, mn, mx in [
            ("Train", tr_raw, tr_imp, tr_clip, before_tr_min, before_tr_max),
            ("Test", te_raw, te_imp, te_clip, before_te_min, before_te_max),
        ]:
            below_before = int((imp < lo).sum()) if lo is not None else 0
            above_before = int((imp > hi).sum()) if hi is not None else 0
            violation_after = int(((clip < lo) | (clip > hi)).sum()) if lo is not None else 0
            audit_rows.append({
                "Feature": c,
                "Family": infer_family(c),
                "Split": split_name,
                "Is_Vehicle_Dimension": int(is_vehicle_dimension(c)),
                "Vehicle_Dimension_Bound_Applied": int(is_vehicle_dimension(c) and bound is not None),
                "Imputation": "train_median_for_engineering_continuous",
                "Train_Median_Used": median,
                "Lower_Bound": lo,
                "Upper_Bound": hi,
                "Missing_Before": int(raw.isna().sum()),
                "Min_Before_Clip": mn,
                "Max_Before_Clip": mx,
                "Min_After_Clip": float(clip.min()) if len(clip) else np.nan,
                "Max_After_Clip": float(clip.max()) if len(clip) else np.nan,
                "Below_Bound_Before": below_before,
                "Above_Bound_Before": above_before,
                "Violation_After": violation_after,
                "Clip_Count": below_before + above_before,
                "Extreme_Value_Warning": int(is_vehicle_dimension(c) and (below_before + above_before) > 0),
            })
    return tr_out, te_out, pd.DataFrame(audit_rows)


def preprocess_numeric(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]):
    tr_out, te_out, audit_rows = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index), []
    for c in cols:
        tr_raw = sentinel_aware_numeric(train[c], c)
        te_raw = sentinel_aware_numeric(test[c], c)
        med = float(tr_raw.median()) if tr_raw.notna().any() else 0.0
        tr_out[c] = tr_raw.fillna(med).astype(float)
        te_out[c] = te_raw.fillna(med).astype(float)
        audit_rows.append({
            "Feature": c, "Family": infer_family(c), "Split": "Train/Test",
            "Imputation": "train_median_numeric", "Train_Median_Used": med,
            "Train_Missing_Before": int(tr_raw.isna().sum()), "Test_Missing_Before": int(te_raw.isna().sum()),
        })
    return tr_out, te_out, pd.DataFrame(audit_rows)


def preprocess_categorical(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]):
    missing_token = "MISSING_OR_UNKNOWN"
    tr = train[cols].copy() if cols else pd.DataFrame(index=train.index)
    te = test[cols].copy() if cols else pd.DataFrame(index=test.index)
    audit_rows = []
    for c in cols:
        if is_binary_like(tr[c]):
            tr[c] = as_binary(tr[c]).fillna(0).astype(int)
            te[c] = as_binary(te[c]).fillna(0).astype(int)
            audit_rows.append({"Feature": c, "Family": infer_family(c), "Transform": "binary_as_0_1"})
        else:
            tr[c] = tr[c].fillna(missing_token).astype(str)
            te[c] = te[c].fillna(missing_token).astype(str)
            train_levels = set(tr[c].dropna().astype(str).unique().tolist())
            test_levels = set(te[c].dropna().astype(str).unique().tolist())
            unseen_test_levels = sorted(test_levels - train_levels)
            audit_rows.append({
                "Feature": c,
                "Family": infer_family(c),
                "Transform": "missing_as_explicit_category_then_onehot",
                "Missing_Token": missing_token,
                "Train_Level_N": int(len(train_levels)),
                "Test_Unseen_Level_N": int(len(unseen_test_levels)),
                "Test_Unseen_Levels": "||".join(unseen_test_levels[:20]),
            })
    if not cols:
        return tr, te, pd.DataFrame(audit_rows)
    # one-hot only for non-binary object columns; binary columns remain as-is.
    non_binary = [c for c in cols if not is_binary_like(train[c])]
    binary = [c for c in cols if is_binary_like(train[c])]
    tr_bin = tr[binary].copy() if binary else pd.DataFrame(index=train.index)
    te_bin = te[binary].copy() if binary else pd.DataFrame(index=test.index)
    if non_binary:
        tr_oh = pd.get_dummies(tr[non_binary], prefix=non_binary, dummy_na=False)
        te_oh = pd.get_dummies(te[non_binary], prefix=non_binary, dummy_na=False)
        te_oh = te_oh.reindex(columns=tr_oh.columns, fill_value=0)
    else:
        tr_oh, te_oh = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    return pd.concat([tr_bin, tr_oh], axis=1), pd.concat([te_bin, te_oh], axis=1), pd.DataFrame(audit_rows)


def _sanitized_unique(cols: List[str]) -> List[str]:
    return make_unique([sanitize_column_name(c) for c in cols])


def build_sentinel_outputs(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_flags = pd.DataFrame(index=train.index)
    test_flags = pd.DataFrame(index=test.index)
    audit_rows = []
    sanitized_cols = _sanitized_unique([str(c) for c in cols])
    for raw_col, out_col in zip(cols, sanitized_cols):
        tr_sentinel = sentinel_mask(train[raw_col], raw_col)
        te_sentinel = sentinel_mask(test[raw_col], raw_col)
        tr_known = sentinel_aware_numeric(train[raw_col], raw_col)
        te_known = sentinel_aware_numeric(test[raw_col], raw_col)
        tr_unobserved = (tr_sentinel | tr_known.isna()).fillna(False).astype(int)
        te_unobserved = (te_sentinel | te_known.isna()).fillna(False).astype(int)
        train_flags[out_col] = tr_unobserved
        test_flags[out_col] = te_unobserved
        for split_name, raw_flag, unobserved, known in [
            ("Train", tr_sentinel, tr_unobserved, tr_known),
            ("Test", te_sentinel, te_unobserved, te_known),
        ]:
            audit_rows.append({
                "Feature": out_col,
                "Raw_Feature": raw_col,
                "Family": infer_family(raw_col),
                "Split": split_name,
                "Raw_Sentinel_N": int(raw_flag.sum()),
                "Raw_Sentinel_Rate": float(raw_flag.mean()) if len(raw_flag) else 0.0,
                "Converted_To_NaN_N": int(raw_flag.sum()),
                "Missing_After_Sentinel_N": int(known.isna().sum()),
            })
    return train_flags, test_flags, pd.DataFrame(audit_rows)


def add_age_over60_feature(train_raw: pd.DataFrame, test_raw: pd.DataFrame, train_m: pd.DataFrame, test_m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = None
    for candidate in ["年龄年数记录_ALTER1", "年龄年数记录(ALTER1)", "ALTER1", "FEATURE_Age_Years", "FEATURE_AGE_YEARS", "Age_Years"]:
        if candidate in train_raw.columns and candidate in test_raw.columns:
            source = candidate
            break
    if source is None:
        for c in train_raw.columns:
            text = str(c)
            if ("ALTER1" in text.upper() or "年龄年数记录" in text) and c in test_raw.columns:
                source = c
                break
    if source is None:
        raise RuntimeError("Cannot create FEATURE_Age_Over60: raw age source column not found.")

    tr_age = sentinel_aware_numeric(train_raw[source], source)
    te_age = sentinel_aware_numeric(test_raw[source], source)
    tr_missing = tr_age.isna() | sentinel_mask(train_raw[source], source)
    te_missing = te_age.isna() | sentinel_mask(test_raw[source], source)

    train_m = train_m.copy()
    test_m = test_m.copy()
    train_m["FEATURE_Age_Over60"] = (tr_age > 60).fillna(False).astype(int).values
    test_m["FEATURE_Age_Over60"] = (te_age > 60).fillna(False).astype(int).values

    audit = pd.DataFrame([{
        "source_feature": sanitize_column_name(source),
        "raw_source_feature": source,
        "derived_feature": "FEATURE_Age_Over60",
        "threshold": 60,
        "train_positive_n": int(train_m["FEATURE_Age_Over60"].sum()),
        "train_positive_rate": float(train_m["FEATURE_Age_Over60"].mean()) if len(train_m) else 0.0,
        "test_positive_n": int(test_m["FEATURE_Age_Over60"].sum()),
        "test_positive_rate": float(test_m["FEATURE_Age_Over60"].mean()) if len(test_m) else 0.0,
        "train_source_missing_sentinel_n": int(tr_missing.sum()),
        "train_source_missing_sentinel_rate": float(tr_missing.mean()) if len(tr_missing) else 0.0,
        "test_source_missing_sentinel_n": int(te_missing.sum()),
        "test_source_missing_sentinel_rate": float(te_missing.mean()) if len(te_missing) else 0.0,
        "unknown_age_treated_as_over60": 0,
    }])
    return train_m, test_m, audit


def validate_sentinel_outputs(train_m: pd.DataFrame, test_m: pd.DataFrame, train_flags: pd.DataFrame, test_flags: pd.DataFrame, audit: pd.DataFrame) -> None:
    required = [
        "Feature", "Family", "Split", "Raw_Sentinel_N", "Raw_Sentinel_Rate",
        "Converted_To_NaN_N", "Missing_After_Sentinel_N",
    ]
    require_columns(audit, OUTPUTS["sentinel_unknown_audit"], required)
    if len(train_flags) != len(train_m) or len(test_flags) != len(test_m):
        raise RuntimeError("Sentinel flag row counts do not match ready matrices.")
    if list(train_flags.index) != list(train_m.index) or list(test_flags.index) != list(test_m.index):
        raise RuntimeError("Sentinel flag index order does not match ready matrices.")
    missing_train = sorted(set(train_flags.columns) - set(train_m.columns))
    missing_test = sorted(set(test_flags.columns) - set(test_m.columns))
    if missing_train or missing_test:
        raise RuntimeError(
            "Sentinel flag columns must use Ready_Matrix feature names; missing="
            + "||".join((missing_train + missing_test)[:20])
        )


def align_train_test_matrices(
    train_m: pd.DataFrame,
    test_m: pd.DataFrame,
    engineering_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_train_cols = [str(c) for c in train_m.columns]
    raw_test_cols = [str(c) for c in test_m.columns]
    train_cols = _sanitized_unique(raw_train_cols)
    test_cols = _sanitized_unique(raw_test_cols)

    continuous_cols = set(_sanitized_unique([str(c) for c in engineering_cols + numeric_cols]))
    categorical_roots = [sanitize_column_name(c) for c in categorical_cols]

    audit_rows = []
    for i in range(max(len(raw_train_cols), len(raw_test_cols))):
        tr_raw = raw_train_cols[i] if i < len(raw_train_cols) else None
        te_raw = raw_test_cols[i] if i < len(raw_test_cols) else None
        tr_col = train_cols[i] if i < len(train_cols) else None
        te_col = test_cols[i] if i < len(test_cols) else None
        if tr_col in continuous_cols:
            kind = "continuous_or_numeric"
        elif any(str(tr_col).startswith(root + "_") or str(tr_col) == root for root in categorical_roots):
            kind = "categorical_dummy_or_binary"
        else:
            kind = "other"
        audit_rows.append({
            "Position": i,
            "Train_Raw_Column": tr_raw,
            "Test_Raw_Column": te_raw,
            "Train_Sanitized_Column": tr_col,
            "Test_Sanitized_Column": te_col,
            "Feature_Kind": kind,
            "Status": "aligned" if tr_col == te_col else "mismatch",
        })

    train_m = train_m.copy()
    test_m = test_m.copy()
    train_m.columns = train_cols
    test_m.columns = test_cols

    missing_in_test = [c for c in train_m.columns if c not in test_m.columns]
    extra_in_test = [c for c in test_m.columns if c not in train_m.columns]
    missing_continuous = [c for c in missing_in_test if c in continuous_cols]
    if missing_continuous:
        raise RuntimeError(f"Continuous/numeric columns missing from test matrix after sanitization: {missing_continuous[:20]}")

    allowed_dummy_missing = [c for c in missing_in_test if c not in continuous_cols]
    if allowed_dummy_missing:
        fill_block = pd.DataFrame(0, index=test_m.index, columns=allowed_dummy_missing)
        test_m = pd.concat([test_m, fill_block], axis=1)
        audit_rows.append({
            "Position": None,
            "Train_Raw_Column": None,
            "Test_Raw_Column": None,
            "Train_Sanitized_Column": "||".join(allowed_dummy_missing[:50]),
            "Test_Sanitized_Column": None,
            "Feature_Kind": "categorical_dummy_or_binary",
            "Status": f"test_missing_train_dummy_filled_zero_n={len(allowed_dummy_missing)}",
        })

    if extra_in_test:
        audit_rows.append({
            "Position": None,
            "Train_Raw_Column": None,
            "Test_Raw_Column": "||".join(extra_in_test[:50]),
            "Train_Sanitized_Column": None,
            "Test_Sanitized_Column": "||".join(extra_in_test[:50]),
            "Feature_Kind": "categorical_dummy_or_binary",
            "Status": f"test_only_columns_ignored_n={len(extra_in_test)}",
        })

    test_m = test_m.reindex(columns=train_m.columns)
    if list(test_m.columns) != list(train_m.columns):
        raise RuntimeError("Train/test matrix columns still differ after controlled alignment.")
    audit = pd.DataFrame(audit_rows)
    return train_m, test_m, audit


def audit_zero_collapse(
    train_m: pd.DataFrame,
    test_m: pd.DataFrame,
    engineering_cols: List[str],
    numeric_cols: List[str],
    group_col: str,
) -> pd.DataFrame:
    feature_cols = [c for c in train_m.columns if c not in {LABEL_COL, group_col}]
    continuous_cols = set(_sanitized_unique([str(c) for c in engineering_cols + numeric_cols]))
    rows = []
    for c in feature_cols:
        tr = pd.to_numeric(train_m[c], errors="coerce").fillna(0)
        te = pd.to_numeric(test_m[c], errors="coerce").fillna(0)
        train_nonzero = float((tr != 0).mean()) if len(tr) else 0.0
        test_nonzero = float((te != 0).mean()) if len(te) else 0.0
        is_cont = c in continuous_cols
        suspicious = train_nonzero >= 0.05 and test_nonzero == 0.0
        bound = infer_bound(c)
        lo, hi = (None, None) if bound is None else bound
        below_after = (int((tr < lo).sum()) + int((te < lo).sum())) if lo is not None else 0
        above_after = (int((tr > hi).sum()) + int((te > hi).sum())) if hi is not None else 0
        rows.append({
            "Feature": c,
            "Family": infer_family(c),
            "Feature_Kind": "continuous_or_numeric" if is_cont else "dummy_or_binary",
            "Is_Vehicle_Dimension": int(is_vehicle_dimension(c)),
            "Vehicle_Dimension_Bound_Applied": int(is_vehicle_dimension(c) and bound is not None),
            "Train_Nonzero_Rate": train_nonzero,
            "Test_Nonzero_Rate": test_nonzero,
            "Train_Unique_N": int(tr.nunique(dropna=True)),
            "Test_Unique_N": int(te.nunique(dropna=True)),
            "Train_Min": float(tr.min()) if len(tr) else np.nan,
            "Train_Max": float(tr.max()) if len(tr) else np.nan,
            "Test_Min": float(te.min()) if len(te) else np.nan,
            "Test_Max": float(te.max()) if len(te) else np.nan,
            "Violation_After": below_after + above_after,
            "Extreme_Value_Warning": int(is_vehicle_dimension(c) and (below_after + above_after) > 0),
            "Status": "suspicious_zero_collapse" if suspicious else "ok",
        })
    audit = pd.DataFrame(rows)
    return audit


def create_baseline(train_m: pd.DataFrame, test_m: pd.DataFrame, selected_cols: List[str] = None):
    rows = []
    cols = selected_cols or [c for c in train_m.columns if c not in {LABEL_COL} and c not in GROUP_CANDIDATES]
    for c in cols:
        if c not in train_m.columns or c == LABEL_COL:
            continue
        tr, te = train_m[c], test_m[c]
        if is_binary_like(tr):
            trb, teb = pd.to_numeric(tr, errors="coerce"), pd.to_numeric(te, errors="coerce")
            rows.append({
                "Characteristic": c, "Family": infer_family(c), "Type": "binary",
                "Train": f"{int((trb==1).sum())} ({trb.mean()*100:.1f}%)",
                "Test": f"{int((teb==1).sum())} ({teb.mean()*100:.1f}%)",
                "SMD_Train_minus_Test": smd_binary(trb, teb),
                "Train_Missing": int(trb.isna().sum()), "Test_Missing": int(teb.isna().sum()),
            })
        else:
            trn, ten = pd.to_numeric(tr, errors="coerce"), pd.to_numeric(te, errors="coerce")
            q1, q3 = trn.quantile(0.25), trn.quantile(0.75)
            tq1, tq3 = ten.quantile(0.25), ten.quantile(0.75)
            rows.append({
                "Characteristic": c, "Family": infer_family(c), "Type": "continuous",
                "Train": f"{trn.mean():.3f} ± {trn.std():.3f}; {trn.median():.3f} [{q1:.3f}, {q3:.3f}]",
                "Test": f"{ten.mean():.3f} ± {ten.std():.3f}; {ten.median():.3f} [{tq1:.3f}, {tq3:.3f}]",
                "SMD_Train_minus_Test": smd_numeric(trn, ten),
                "Train_Missing": int(trn.isna().sum()), "Test_Missing": int(ten.isna().sum()),
            })
    return pd.DataFrame(rows)


def main():
    df = read_csv_smart(CLEANED_DATA)
    df.columns = [str(c).strip() for c in df.columns]
    group_col = find_group_col(df.columns, GROUP_CANDIDATES)
    groups = normalize_group_ids(df[group_col])

    if LABEL_COL not in df.columns:
        if TARGET_SOURCE_COL not in df.columns:
            raise KeyError(f"Neither {LABEL_COL} nor {TARGET_SOURCE_COL} found.")
        target = pd.to_numeric(df[TARGET_SOURCE_COL], errors="coerce")
        df[LABEL_COL] = (target >= 3).astype(int)
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

    tr_idx, te_idx, split_method = choose_grouped_split(df, y, groups)
    train_raw, test_raw = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
    train_groups, test_groups = groups.iloc[tr_idx].reset_index(drop=True), groups.iloc[te_idx].reset_index(drop=True)
    train_raw = train_raw.reset_index(drop=True)
    test_raw = test_raw.reset_index(drop=True)

    eng_cols, num_cols, cat_cols = split_column_types(train_raw, LABEL_COL, group_col)
    sentinel_train_flags, sentinel_test_flags, sentinel_audit = build_sentinel_outputs(train_raw, test_raw, eng_cols + num_cols)
    tr_eng, te_eng, audit_eng = preprocess_engineering(train_raw, test_raw, eng_cols)
    tr_num, te_num, audit_num = preprocess_numeric(train_raw, test_raw, num_cols)
    tr_cat, te_cat, audit_cat = preprocess_categorical(train_raw, test_raw, cat_cols)

    train_m = pd.concat([tr_eng, tr_num, tr_cat], axis=1)
    test_m = pd.concat([te_eng, te_num, te_cat], axis=1)
    train_m, test_m, derived_audit = add_age_over60_feature(train_raw, test_raw, train_m, test_m)
    train_m, test_m, column_audit = align_train_test_matrices(train_m, test_m, eng_cols, num_cols, cat_cols)
    train_m[LABEL_COL] = y.iloc[tr_idx].reset_index(drop=True).astype(int)
    test_m[LABEL_COL] = y.iloc[te_idx].reset_index(drop=True).astype(int)
    train_m[group_col] = train_groups.values
    test_m[group_col] = test_groups.values
    zero_audit = audit_zero_collapse(train_m, test_m, eng_cols, num_cols, group_col)
    write_csv(column_audit, "Matrix_Column_Alignment_Audit.csv")
    write_csv(derived_audit, "Derived_Feature_Audit.csv")
    write_csv(zero_audit, "Matrix_Test_Zero_Collapse_Audit.csv")
    validate_sentinel_outputs(train_m, test_m, sentinel_train_flags, sentinel_test_flags, sentinel_audit)
    write_csv(sentinel_train_flags, OUTPUTS["sentinel_flags_train"])
    write_csv(sentinel_test_flags, OUTPUTS["sentinel_flags_test"])
    write_csv(sentinel_audit, OUTPUTS["sentinel_unknown_audit"])
    for p in [OUTPUTS["sentinel_flags_train"], OUTPUTS["sentinel_flags_test"], OUTPUTS["sentinel_unknown_audit"]]:
        if not Path(p).exists():
            raise RuntimeError(f"Required sentinel output was not generated: {p}")
    bad_cont = zero_audit[(zero_audit["Feature_Kind"] == "continuous_or_numeric") & (zero_audit["Status"] == "suspicious_zero_collapse")]
    if not bad_cont.empty:
        raise RuntimeError("Suspicious zero collapse in continuous/numeric test columns: " + "||".join(bad_cont["Feature"].head(20).astype(str)))
    bad_dummy_n = int(((zero_audit["Feature_Kind"] == "dummy_or_binary") & (zero_audit["Status"] == "suspicious_zero_collapse")).sum())
    dummy_total = max(1, int((zero_audit["Feature_Kind"] == "dummy_or_binary").sum()))
    if bad_dummy_n >= 25 and bad_dummy_n / dummy_total >= 0.10:
        raise RuntimeError(f"Suspicious zero collapse in many dummy columns: {bad_dummy_n}/{dummy_total}")

    # Output matrices once; aliases may point to the same canonical paths.
    write_csv(train_m, TRAIN_MATRIX)
    write_csv(test_m, TEST_MATRIX)
    if TRAIN_MATRIX_ALIAS != TRAIN_MATRIX:
        write_csv(train_m, TRAIN_MATRIX_ALIAS)
    if TEST_MATRIX_ALIAS != TEST_MATRIX:
        write_csv(test_m, TEST_MATRIX_ALIAS)

    overlap = len(set(train_groups) & set(test_groups))
    split_audit = pd.DataFrame([{
        "Split_Method": split_method,
        "Total_Rows": int(len(df)), "Train_Rows": int(len(train_m)), "Test_Rows": int(len(test_m)),
        "Total_Accidents": int(groups.nunique()), "Train_Accidents": int(train_groups.nunique()), "Test_Accidents": int(test_groups.nunique()),
        "Overlap_Accidents": int(overlap),
        "Train_Severe_Rate": float(train_m[LABEL_COL].mean()), "Test_Severe_Rate": float(test_m[LABEL_COL].mean()),
    }])
    write_csv(split_audit, OUTPUTS["split_audit"])

    audit = pd.concat([audit_eng, audit_num, audit_cat], ignore_index=True)
    write_csv(audit, OUTPUTS["preprocess_audit"])
    baseline_cols = [LABEL_COL] + [c for c in train_m.columns if c not in {LABEL_COL, group_col}][:80]
    write_csv(create_baseline(train_m, test_m, baseline_cols), OUTPUTS["baseline_table"])

    write_json({
        "stage": "02_preprocess_bounded",
        "input": CLEANED_DATA,
        "fit_source": "train",
        "transform_source": "train_fitted_transform_applied_to_train_and_test",
        "selection_source": "train",
        "evaluation_source": "none",
        "test_used_for_fit": False,
        "test_used_for_column_space_fit": False,
        "test_used_for_imputation_fit": False,
        "test_used_for_category_level_fit": False,
        "test_used_for_evaluation_only": False,
        "test_role": "held_out_transform_only_after_train_fitted_preprocessing",
        "outputs": sorted(set([TRAIN_MATRIX, TEST_MATRIX, TRAIN_MATRIX_ALIAS, TEST_MATRIX_ALIAS, OUTPUTS["split_audit"], OUTPUTS["preprocess_audit"], OUTPUTS["baseline_table"], "Matrix_Column_Alignment_Audit.csv", "Matrix_Test_Zero_Collapse_Audit.csv", "Derived_Feature_Audit.csv", OUTPUTS["sentinel_unknown_audit"], OUTPUTS["sentinel_flags_train"], OUTPUTS["sentinel_flags_test"]])),
        "group_col": group_col,
        "label_col": LABEL_COL,
        "engineering_imputation": "train_median_plus_fixed_physical_clipping",
        "vehicle_dimension_units": "millimetres",
        "vehicle_dimension_bounds": {
            "LAENGE": PHYSICAL_BOUNDS.get("LAENGE"),
            "BREITE": PHYSICAL_BOUNDS.get("BREITE"),
            "HOEHE": PHYSICAL_BOUNDS.get("HOEHE"),
        },
        "numeric_imputation": "train_median",
        "categorical_imputation": "missing_as_explicit_category_then_onehot",
        "sentinel_cleaning_before_numeric_conversion": True,
        "sentinel_cleaning_before_imputation": True,
        "sentinel_cleaning_before_physical_clipping": True,
        "sentinel_flags_audit_only_not_features": True,
        "age_over60_semantic_feature_created": True,
        "age_over60_source_feature": "年龄年数记录_ALTER1",
        "age_over60_threshold": 60,
        "age_raw_column_not_allowed_in_primary_selected_features": True,
        "sentinel_outputs": [
            OUTPUTS["sentinel_unknown_audit"],
            OUTPUTS["sentinel_flags_train"],
            OUTPUTS["sentinel_flags_test"],
        ],
    }, "02_Run_Manifest.json")
    print("✅ 02 preprocessing finished. Kept outputs: matrices, split audit, plausibility audit, baseline table.")


if __name__ == "__main__":
    main()
