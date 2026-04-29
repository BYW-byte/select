# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from _config import *
from _utils import find_group_col, infer_family, read_csv_smart, safe_numeric, write_csv, write_json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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


def main():
    train_path = TRAIN_MATRIX_ALIAS if Path(TRAIN_MATRIX_ALIAS).exists() else TRAIN_MATRIX
    test_path = TEST_MATRIX_ALIAS if Path(TEST_MATRIX_ALIAS).exists() else TEST_MATRIX
    if not Path(train_path).exists() or not Path(test_path).exists():
        raise FileNotFoundError("Ready train/test matrices are required for postprocess checks.")

    train = read_csv_smart(train_path)
    test = read_csv_smart(test_path)
    group_col = find_group_col(train.columns, GROUP_CANDIDATES)
    if list(train.columns) != list(test.columns):
        raise RuntimeError("Ready_Matrix_Train.csv and Ready_Matrix_Test.csv columns are not aligned.")

    rows = []
    feature_cols = [c for c in train.columns if c not in {LABEL_COL, group_col, TARGET_SOURCE_COL}]
    for c in feature_cols:
        bound = infer_bound(c)
        lo, hi = (None, None) if bound is None else bound
        tr = safe_numeric(train[c])
        te = safe_numeric(test[c])
        tr_nonzero = float((tr.fillna(0) != 0).mean()) if len(tr) else 0.0
        te_nonzero = float((te.fillna(0) != 0).mean()) if len(te) else 0.0
        suspicious = tr_nonzero >= 0.05 and te_nonzero == 0.0
        status = "suspicious_distribution_collapse" if suspicious else "ok"
        if bound is not None and lo is not None and lo > 0 and te_nonzero == 0.0:
            status = "physical_positive_lower_bound_all_zero"
        elif bound is not None and te_nonzero == 0.0 and any(k in str(c).upper() for k in ("V0", "VK", "MUE")) and "VKREG" not in str(c).upper():
            status = "suspicious_distribution_collapse"
        train_below = int((tr < lo).sum()) if lo is not None else 0
        train_above = int((tr > hi).sum()) if lo is not None else 0
        test_below = int((te < lo).sum()) if lo is not None else 0
        test_above = int((te > hi).sum()) if lo is not None else 0
        rows.append({
            "Split": "Train/Test",
            "Feature": c,
            "Family": infer_family(c),
            "Is_Vehicle_Dimension": int(is_vehicle_dimension(c)),
            "Vehicle_Dimension_Bound_Applied": int(is_vehicle_dimension(c) and bound is not None),
            "Lower_Bound": lo,
            "Upper_Bound": hi,
            "Train_N": int(tr.notna().sum()),
            "Test_N": int(te.notna().sum()),
            "Train_Nonzero_Rate": tr_nonzero,
            "Test_Nonzero_Rate": te_nonzero,
            "Train_Unique_N": int(tr.nunique(dropna=True)),
            "Test_Unique_N": int(te.nunique(dropna=True)),
            "Train_Min": float(tr.min()) if tr.notna().any() else None,
            "Train_Max": float(tr.max()) if tr.notna().any() else None,
            "Test_Min": float(te.min()) if te.notna().any() else None,
            "Test_Max": float(te.max()) if te.notna().any() else None,
            "Train_Below_Bound_Count": train_below,
            "Train_Above_Bound_Count": train_above,
            "Test_Below_Bound_Count": test_below,
            "Test_Above_Bound_Count": test_above,
            "Violation_Count": train_below + train_above + test_below + test_above,
            "Clip_Count": 0,
            "Extreme_Value_Warning": int(is_vehicle_dimension(c) and (train_below + train_above + test_below + test_above) > 0),
            "Status": status,
        })

    # Keep the original split-level physical-bound rows for manuscript continuity.
    for split, df in [("Train", train), ("Test", test)]:
        for c in feature_cols:
            b = infer_bound(c)
            if b is None:
                continue
            lo, hi = b
            s = safe_numeric(df[c])
            rows.append({
                "Split": split,
                "Feature": c,
                "Family": infer_family(c),
                "Is_Vehicle_Dimension": int(is_vehicle_dimension(c)),
                "Vehicle_Dimension_Bound_Applied": int(is_vehicle_dimension(c)),
                "Lower_Bound": lo,
                "Upper_Bound": hi,
                "N": int(s.notna().sum()),
                "Min": float(s.min()) if s.notna().any() else None,
                "Max": float(s.max()) if s.notna().any() else None,
                "Below_Bound_Count": int((s < lo).sum()),
                "Above_Bound_Count": int((s > hi).sum()),
                "Violation_Count": int(((s < lo) | (s > hi)).sum()),
                "Clip_Count": 0,
                "Extreme_Value_Warning": int(is_vehicle_dimension(c) and int(((s < lo) | (s > hi)).sum()) > 0),
                "Status": "physical_bound_check",
            })

    out = pd.DataFrame(rows)
    write_csv(out, "PostProcess_Matrix_Plausibility_Check.csv")
    write_json({
        "stage": "03_check_preprocess_outputs",
        "test_usage": "audit_only",
        "fit_source": "none",
        "transform_source": "none",
        "selection_source": "none",
        "evaluation_source": "none",
        "modifies_features": False,
        "modifies_thresholds": False,
        "modifies_rules": False,
        "postprocess_matrix_plausibility_check_role": "hard_stop_audit_only_not_selection_input",
        "outputs": ["PostProcess_Matrix_Plausibility_Check.csv"],
    }, "03_Run_Manifest.json")

    feature_rows = out[out["Status"].ne("physical_bound_check")].copy()
    positive_zero = feature_rows[feature_rows["Status"].eq("physical_positive_lower_bound_all_zero")]
    if not positive_zero.empty:
        raise RuntimeError("Positive-lower-bound physical variables are all zero in test: " + "||".join(positive_zero["Feature"].head(20).astype(str)))
    suspicious = feature_rows[feature_rows["Status"].eq("suspicious_distribution_collapse")]
    if len(suspicious) > 10:
        raise RuntimeError(f"Too many train-nonzero/test-zero distribution collapses: {len(suspicious)}")
    if int(out["Violation_Count"].fillna(0).sum()) > 0:
        raise RuntimeError("Physical bound violations detected in postprocess matrix check.")
    print("PostProcess_Matrix_Plausibility_Check.csv written with bound and distribution checks.")


if __name__ == "__main__":
    main()
