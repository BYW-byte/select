# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

from _config import *
from _utils import assert_test_not_used_for_fit, find_group_col, normalize_group_ids, read_csv_smart, write_csv, write_json, infer_family


def load_selected_features() -> List[str]:
    df = read_csv_smart(OUTPUTS["selected_features"])
    col = "Selected_Features" if "Selected_Features" in df.columns else df.columns[0]
    return [str(x) for x in df[col].dropna().tolist()]


def make_models(y_train: pd.Series):
    models = {
        "Logistic_Selected25": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]),
        "RandomForest_Selected25": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)),
        ]),
    }
    if HAS_XGB:
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        models["XGBoost_Selected25"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(eval_metric="logloss", n_estimators=350, learning_rate=0.05, max_depth=4,
                                      subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0, random_state=RANDOM_STATE,
                                      n_jobs=1, scale_pos_weight=neg / pos)),
        ])
    if HAS_CATBOOST:
        models["CatBoost_Selected25"] = CatBoostClassifier(iterations=350, depth=4, learning_rate=0.05, loss_function="Logloss", verbose=False, random_seed=RANDOM_STATE)
    return models


def predict_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def metric_row(name, y_true, score, dataset, feature_space="Selected25"):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= 0.5).astype(int)
    prev = float(y_true.mean())
    return {
        "Dataset": dataset,
        "Feature_Space": feature_space,
        "Model": name,
        "N": int(len(y_true)),
        "Prevalence": prev,
        "AUROC": roc_auc_score(y_true, score) if len(np.unique(y_true)) > 1 else np.nan,
        "AUPRC": average_precision_score(y_true, score),
        "AUPRC_Gain_Over_Prevalence": average_precision_score(y_true, score) - prev,
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "F1_5": fbeta_score(y_true, pred, beta=1.5, zero_division=0),
    }


def fixed_test(train, test, features):
    ytr = pd.to_numeric(train[LABEL_COL], errors="coerce").fillna(0).astype(int)
    yte = pd.to_numeric(test[LABEL_COL], errors="coerce").fillna(0).astype(int)
    Xtr = train[features].apply(pd.to_numeric, errors="coerce")
    Xte = test[features].apply(pd.to_numeric, errors="coerce")
    rows = []
    models = make_models(ytr)
    for name, model in models.items():
        model.fit(Xtr, ytr)
        score = predict_score(model, Xte)
        rows.append(metric_row(name, yte, score, "Fixed_Test"))
    return pd.DataFrame(rows), models.get("Logistic_Selected25")


def repeated_holdout(full, features, group_col, repeats=10):
    y = pd.to_numeric(full[LABEL_COL], errors="coerce").fillna(0).astype(int)
    X = full[features].apply(pd.to_numeric, errors="coerce")
    groups = normalize_group_ids(full[group_col])
    rows = []
    for seed in range(RANDOM_STATE, RANDOM_STATE + repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, te = next(gss.split(X, y, groups=groups))
        models = make_models(y.iloc[tr])
        for name, model in models.items():
            model.fit(X.iloc[tr], y.iloc[tr])
            score = predict_score(model, X.iloc[te])
            row = metric_row(name, y.iloc[te], score, f"Repeated_Group_Holdout_{seed}")
            row["Seed"] = seed
            rows.append(row)
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["Feature_Space", "Model"], as_index=False).agg(
        AUROC_mean=("AUROC", "mean"), AUROC_sd=("AUROC", "std"),
        AUPRC_mean=("AUPRC", "mean"), AUPRC_sd=("AUPRC", "std"),
        AUPRC_Gain_mean=("AUPRC_Gain_Over_Prevalence", "mean"),
        Recall_mean=("Recall", "mean"), Balanced_Accuracy_mean=("Balanced_Accuracy", "mean"), F1_5_mean=("F1_5", "mean"),
        Runs=("AUROC", "count"),
    )
    return detail, summary


def main():
    train_path = TRAIN_MATRIX_ALIAS if Path(TRAIN_MATRIX_ALIAS).exists() else TRAIN_MATRIX
    assert_test_not_used_for_fit(
        "08_model_baseline_repeated_holdout",
        [train_path, OUTPUTS["selected_features"]],
        test_paths=[TEST_MATRIX, OUTPUTS["blind_replay"], OUTPUTS["evidence_tiers"], OUTPUTS["threshold_sensitivity"], "Rule_Item_Replay_Audit.csv"],
    )
    train = read_csv_smart(train_path)
    test = read_csv_smart(TEST_MATRIX_ALIAS if Path(TEST_MATRIX_ALIAS).exists() else TEST_MATRIX)
    group_col = find_group_col(train.columns, GROUP_CANDIDATES)
    features = [f for f in load_selected_features() if f in train.columns]
    fixed, logit_model = fixed_test(train, test, features)
    fixed["Protocol"] = "train fit, fixed independent test evaluation"
    write_csv(fixed, OUTPUTS["model_fixed"])

    detail, summary = repeated_holdout(train, features, group_col, repeats=10)
    summary["Protocol"] = "train-only repeated grouped holdout / internal resampling"
    # Keep summary only by default; detail is not written to avoid output bloat.
    write_csv(summary, OUTPUTS["model_repeated"])

    # Logistic coefficient table for interpretability.
    coef_rows = []
    if logit_model is not None:
        clf = logit_model.named_steps["clf"]
        for feat, coef in zip(features, clf.coef_[0]):
            coef_rows.append({"Feature": feat, "Family": infer_family(feat), "Logistic_Coefficient": float(coef), "Abs_Coefficient": abs(float(coef))})
    coef_df = pd.DataFrame(coef_rows).sort_values("Abs_Coefficient", ascending=False) if coef_rows else pd.DataFrame()
    write_csv(coef_df, OUTPUTS["logistic_coef"])

    write_json({
        "stage": "09_model_baseline",
        "fixed_test_training_source": "train_only",
        "fixed_test_evaluation_source": "test_only",
        "repeated_holdout_source": "train_only",
        "fit_source": "train",
        "selection_source": "train",
        "evaluation_source": "test_only_for_fixed_evaluation_and_train_only_for_internal_holdout",
        "test_used_for_repeated_holdout": False,
        "test_used_for_model_selection": False,
        "test_used_for_hyperparameter_selection": False,
        "selected_features_n": len(features),
        "outputs": [OUTPUTS["model_fixed"], OUTPUTS["model_repeated"], OUTPUTS["logistic_coef"]],
    }, "09_Run_Manifest.json")
    print("✅ 09 model baseline finished.")


if __name__ == "__main__":
    main()
