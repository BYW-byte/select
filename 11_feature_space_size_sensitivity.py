# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    HAS_MLXTEND = True
except Exception:
    HAS_MLXTEND = False

from _config import *
from _utils import assert_test_not_used_for_fit, benjamini_hochberg, enrichment_pvalue, infer_family, read_csv_smart, wilson_ci, write_csv, write_json

fs04 = importlib.import_module("04_feature_selection_consensus")
rule05 = importlib.import_module("05_rule_mining")


def metric_pair(y_true: pd.Series, scores: np.ndarray) -> tuple:
    try:
        auroc = roc_auc_score(y_true, scores)
    except Exception:
        auroc = np.nan
    try:
        auprc = average_precision_score(y_true, scores)
    except Exception:
        auprc = np.nan
    return auroc, auprc


def model_evaluation(train: pd.DataFrame, test: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    y_train = pd.to_numeric(train[LABEL_COL], errors="coerce").fillna(0).astype(int)
    y_test = pd.to_numeric(test[LABEL_COL], errors="coerce").fillna(0).astype(int)
    X_train = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test = test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = {}

    models = {
        "Logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)),
        ]),
    }
    if HAS_XGB:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                scale_pos_weight=max(1.0, neg / max(1, pos)),
            )),
        ])

    for name in ("Logistic", "RandomForest", "XGBoost"):
        if name not in models:
            out[f"{name}_AUROC"] = np.nan
            out[f"{name}_AUPRC"] = np.nan
            continue
        try:
            model = models[name]
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            auroc, auprc = metric_pair(y_test, prob)
        except Exception:
            auroc, auprc = np.nan, np.nan
        out[f"{name}_AUROC"] = auroc
        out[f"{name}_AUPRC"] = auprc
    return out


def redundancy_index(train: pd.DataFrame, features: List[str]) -> float:
    if len(features) < 2:
        return 0.0
    X = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = X.corr().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vals = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    return float(vals.mean()) if len(vals) else 0.0


def item_complexity(item_df: pd.DataFrame) -> Dict[str, int]:
    item_cols = [c for c in item_df.columns if c != TARGET_ITEM and "MISSING_OR_UNKNOWN" not in str(c)]
    return {
        "Candidate_Item_N": len(item_cols),
        "Candidate_2Item_N": int(len(item_cols) * (len(item_cols) - 1) / 2),
        "Candidate_3Item_N": int(len(item_cols) * (len(item_cols) - 1) * (len(item_cols) - 2) / 6),
        "Frequent_Itemset_N": np.nan,
        "Train_Rule_Candidate_N": np.nan,
    }


def replay_summary(test: pd.DataFrame, rules: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, object]:
    if rules.empty:
        return {
            "Replayable_Rule_N": 0,
            "Test_Hit_N_Median": np.nan,
            "Test_Hit_N_Min": np.nan,
            "Test_RR_Median": np.nan,
            "q_lt_0_05_N": 0,
            "Confirmed_or_BinaryStable_Criterion_N_if_evaluated": 0,
            "Deprecated_CoreLike_N_if_evaluated": 0,
            "Criterion_Note": "screening-style count based on q/RR/hit only; not identical to strict core-confirmed evidence tier",
            "Rule_Axis_Coverage_N": 0,
            "Dominant_Axis": "",
            "Axis_Herfindahl_Index": np.nan,
        }
    y = pd.to_numeric(test[LABEL_COL], errors="coerce").fillna(0).astype(int)
    base_rate = float(y.mean())
    manifest_map = {str(r["item"]): r.to_dict() for _, r in manifest.iterrows()}
    rows = []
    for _, rule in rules.head(SIZE_SENS_RULE_TOPK).iterrows():
        mask = pd.Series(True, index=test.index)
        resolved = True
        for item in str(rule["Antecedent_Items"]).split("||"):
            spec = manifest_map.get(item)
            if spec is None or str(spec.get("source_feature")) not in test.columns:
                resolved = False
                mask &= False
                continue
            mask &= rule05.build_items(test, [], fit_manifest=False, manifest=pd.DataFrame([spec]))[0][item]
        hit_n = int(mask.sum())
        severe_n = int(y[mask].sum()) if hit_n else 0
        conf = severe_n / hit_n if hit_n else np.nan
        rr = conf / max(base_rate, 1e-12) if hit_n else np.nan
        p = enrichment_pvalue(severe_n, hit_n, base_rate)
        rows.append({
            "Rule_ID": rule.get("Rule_ID"),
            "Mechanism_Axis": rule.get("Mechanism_Axis", "mixed_multimechanism"),
            "Resolved": resolved,
            "Test_Hit_N": hit_n,
            "Risk_Ratio_vs_BaseRate": rr,
            "p_value_enrichment": p,
        })
    df = pd.DataFrame(rows)
    df["q_value_BH"] = benjamini_hochberg(df["p_value_enrichment"].tolist()) if len(df) else []
    axis_counts = df["Mechanism_Axis"].astype(str).value_counts()
    total_axis = max(1, int(axis_counts.sum()))
    herf = float(((axis_counts / total_axis) ** 2).sum()) if len(axis_counts) else np.nan
    q_lt = int((pd.to_numeric(df["q_value_BH"], errors="coerce") < CORE_Q_THRESHOLD).sum()) if len(df) else 0
    core_like = (
        (pd.to_numeric(df["q_value_BH"], errors="coerce") <= CORE_Q_THRESHOLD)
        & (pd.to_numeric(df["Risk_Ratio_vs_BaseRate"], errors="coerce") >= CORE_RR_THRESHOLD)
        & (pd.to_numeric(df["Test_Hit_N"], errors="coerce") >= CORE_TEST_HIT_MIN)
    )
    return {
        "Replayable_Rule_N": int((df["Test_Hit_N"] > 0).sum()) if len(df) else 0,
        "Test_Hit_N_Median": float(pd.to_numeric(df["Test_Hit_N"], errors="coerce").median()) if len(df) else np.nan,
        "Test_Hit_N_Min": float(pd.to_numeric(df["Test_Hit_N"], errors="coerce").min()) if len(df) else np.nan,
        "Test_RR_Median": float(pd.to_numeric(df["Risk_Ratio_vs_BaseRate"], errors="coerce").median()) if len(df) else np.nan,
        "q_lt_0_05_N": q_lt,
        "Confirmed_or_BinaryStable_Criterion_N_if_evaluated": int(core_like.sum()) if len(df) else 0,
        "Deprecated_CoreLike_N_if_evaluated": int(core_like.sum()) if len(df) else 0,
        "Criterion_Note": "screening-style count based on q/RR/hit only; not identical to strict core-confirmed evidence tier",
        "Rule_Axis_Coverage_N": int(axis_counts.shape[0]),
        "Dominant_Axis": str(axis_counts.index[0]) if len(axis_counts) else "",
        "Axis_Herfindahl_Index": herf,
    }


def family_summary(selected: pd.DataFrame, train: pd.DataFrame) -> Dict[str, object]:
    fam_counts = selected["Family"].astype(str).value_counts()
    covered = sorted(fam_counts.index.tolist())
    missing = sorted([fam for fam, min_n in MANDATORY_FAMILY_MIN.items() if fam_counts.get(fam, 0) < int(min_n)])
    features = selected["Feature"].astype(str).tolist()
    return {
        "Family_Coverage_N": len(covered),
        "Covered_Families": "|".join(covered),
        "Missing_Mandatory_Families": "|".join(missing),
        "Max_Family_Count": int(fam_counts.max()) if len(fam_counts) else 0,
        "Redundancy_Index": redundancy_index(train, features),
    }


def interpretation_complexity(row: Dict[str, object]) -> str:
    if row["Missing_Mandatory_Families"]:
        return "undercovered_mechanism_space"
    if row["Candidate_3Item_N"] > 5000 or row["Axis_Herfindahl_Index"] > 0.45:
        return "higher_rule_complexity_or_axis_concentration"
    if row["Selected_N"] == SELECTED_FEATURE_N:
        return "primary_balance_point"
    return "sensitivity_comparator_only"


def main() -> None:
    required = [TRAIN_MATRIX, TEST_MATRIX, OUTPUTS["consensus_scores"], OUTPUTS["screener_rankings"], "04_Run_Manifest.json"]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required feature-size sensitivity inputs: " + "||".join(missing))
    assert_test_not_used_for_fit(
        "11_feature_space_size_sensitivity",
        [TRAIN_MATRIX, OUTPUTS["consensus_scores"], OUTPUTS["screener_rankings"], "04_Run_Manifest.json"],
        test_paths=[TEST_MATRIX, OUTPUTS["blind_replay"], OUTPUTS["evidence_tiers"], OUTPUTS["threshold_sensitivity"]],
    )
    train = read_csv_smart(TRAIN_MATRIX)
    test = read_csv_smart(TEST_MATRIX)
    scores = read_csv_smart(OUTPUTS["consensus_scores"])

    rows = []
    long_rows = []
    complexity_rows = []
    for n in FEATURE_SIZE_SENSITIVITY_N:
        selected = fs04.compose_feature_set(scores, selected_n=int(n))
        features = selected["Feature"].astype(str).tolist()
        for _, r in selected.iterrows():
            long_rows.append({
                "Selected_N": int(n),
                "Is_Primary_Analysis": int(int(n) == SELECTED_FEATURE_N),
                "Selection_Order": int(r["Selection_Order"]),
                "Feature": r["Feature"],
                "Family": r["Family"],
                "ConsensusScore": r.get("ConsensusScore"),
                "SelectionReason": r.get("SelectionReason"),
            })
        item_df, manifest = rule05.build_items(train, features, fit_manifest=True)
        complexity = item_complexity(item_df)
        rule_mining_status = "ok"
        try:
            rules, _universe = rule05.mine_rules(item_df, manifest)
        except MemoryError:
            rules, _universe = pd.DataFrame(), pd.DataFrame()
            rule_mining_status = "memory_limited_supplementary_rule_enumeration_skipped"
        complexity["Train_Rule_Candidate_N"] = int(len(_universe)) if _universe is not None else 0
        complexity["Frequent_Itemset_N"] = np.nan
        replay = replay_summary(test, rules, manifest)
        fam = family_summary(selected, train)
        model = model_evaluation(train, test, features)
        row = {
            "Selected_N": int(n),
            "Is_Primary_Analysis": int(int(n) == SELECTED_FEATURE_N),
            **fam,
            **complexity,
            "Final_Train_Rule_N": int(len(rules)),
            **replay,
            **model,
            "Test_Used_For_Size_Selection": False,
            "Evaluation_Only": True,
            "Rule_Mining_Status": rule_mining_status,
        }
        row["Interpretation_Complexity"] = interpretation_complexity(row)
        rows.append(row)
        complexity_rows.append({
            "Selected_N": int(n),
            "Selected_Feature_List": "||".join(features),
            **complexity,
            "Final_Train_Rule_N": int(len(rules)),
            "Rule_Axis_Coverage_N": replay["Rule_Axis_Coverage_N"],
            "Dominant_Axis": replay["Dominant_Axis"],
            "Axis_Herfindahl_Index": replay["Axis_Herfindahl_Index"],
            "Rule_Mining_Status": rule_mining_status,
            "Sensitivity_Role": "does_not_modify_primary_rule_files",
        })

    summary = pd.DataFrame(rows)
    long = pd.DataFrame(long_rows)
    complexity_df = pd.DataFrame(complexity_rows)
    write_csv(summary, OUTPUTS["feature_size_sensitivity"])
    write_csv(long, OUTPUTS["feature_size_selected_features_long"])
    write_csv(complexity_df, OUTPUTS["feature_size_rule_complexity"])
    write_json({
        "stage": "11_feature_space_size_sensitivity",
        "sensitivity_type": "feature_space_size",
        "primary_selected_n": SELECTED_FEATURE_N,
        "candidate_selected_n": FEATURE_SIZE_SENSITIVITY_N,
        "feature_ranking_source": "train_only_consensus_scores",
        "fit_source": "train",
        "selection_source": "train_only",
        "test_usage": "evaluation_only",
        "sensitivity_role": "evaluation-only sensitivity; does not participate in selected features or primary rule set",
        "test_used_for_size_selection": False,
        "modifies_primary_selected_features": False,
        "modifies_primary_rules": False,
        "modifies_primary_rule_set": False,
        "memory_limited_supplementary_rule_enumeration_recorded": bool("Rule_Mining_Status" in summary.columns and summary["Rule_Mining_Status"].astype(str).ne("ok").any()),
        "modifies_evidence_tiers": False,
        "criterion_note": "Confirmed_or_BinaryStable_Criterion_N_if_evaluated is a screening-style count based on q/RR/hit only; not identical to strict core-confirmed evidence tier",
        "outputs": [
            OUTPUTS["feature_size_sensitivity"],
            OUTPUTS["feature_size_selected_features_long"],
            OUTPUTS["feature_size_rule_complexity"],
        ],
    }, "11_Feature_Size_Sensitivity_Manifest.json")
    print("Feature-space size sensitivity finished.")


if __name__ == "__main__":
    main()
