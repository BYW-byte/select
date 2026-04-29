# -*- coding: utf-8 -*-
"""Consensus-guided, mechanism-constrained feature composition.

This stage intentionally avoids making a single optimizer the methodological center.
It summarizes signals from standard feature-screening families, then composes a
compact, family-balanced rule-input space for downstream FPGrowth and blind replay.

Screeners:
- mutual information: marginal nonlinear dependence;
- L1 logistic: sparse linear signal;
- tree importance: nonlinear/interaction-prone signal from XGBoost or RF;
- random forest and extra trees: ensemble tree alternatives;
- univariate standardized signal: transparent marginal contrast;
- grouped-CV stability voting: repeated train-only ranking stability;
- light grouped permutation importance on a candidate pool: validation-oriented signal.
- GWO wrapper search: retained only as a candidate generator, not final authority.

Outputs are deliberately minimal but auditable:
- Feature_Screener_Rankings.csv
- Feature_Consensus_Scores.csv / Feature_Selection_Stability.csv
- Final_Consensus_Mechanism_Features.csv
- Feature_Selection_Baseline_Comparison.csv
- Feature_Family_Coverage_Audit.csv
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from _config import *
from _utils import (
    assert_test_not_used_for_fit,
    find_group_col,
    infer_family,
    infer_feature_semantic_group,
    infer_feature_source_group,
    is_missing_category_feature,
    is_pure_missing_unknown_feature,
    is_raw_age_column,
    missing_category_type,
    normalize_group_ids,
    read_csv_smart,
    write_csv,
    write_json,
)


def load_matrix() -> Tuple[pd.DataFrame, pd.Series, pd.Series, str]:
    train_path = TRAIN_MATRIX_ALIAS if Path(TRAIN_MATRIX_ALIAS).exists() else TRAIN_MATRIX
    assert_test_not_used_for_fit(
        "04_feature_selection_consensus",
        [train_path],
        test_paths=[TEST_MATRIX, OUTPUTS["blind_replay"], OUTPUTS["evidence_tiers"], OUTPUTS["threshold_sensitivity"], "Rule_Item_Replay_Audit.csv"],
    )
    train = read_csv_smart(train_path)
    group_col = find_group_col(train.columns, GROUP_CANDIDATES)
    y = pd.to_numeric(train[LABEL_COL], errors="coerce").fillna(0).astype(int)
    groups = normalize_group_ids(train[group_col])
    drop_cols = {LABEL_COL, group_col, TARGET_SOURCE_COL}
    feature_cols = [c for c in train.columns if c not in drop_cols]
    X = train[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y, groups, group_col


def minmax(values: Sequence[float]) -> pd.Series:
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def rank_to_score(rank: pd.Series, n_features: int) -> pd.Series:
    # Best rank gets close to 1; worst rank gets close to 0.
    r = pd.to_numeric(rank, errors="coerce").fillna(n_features)
    return (n_features - r + 1) / max(1, n_features)


def make_tree_model(kind: str, y: pd.Series):
    if kind == "xgb" and HAS_XGB:
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        scale_pos_weight = max(1.0, neg / max(1, pos))
        return xgb.XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=1,
            reg_lambda=3.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def safe_fit_importance(model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    try:
        model.fit(X, y)
        imp = getattr(model, "feature_importances_", np.zeros(X.shape[1]))
        return np.asarray(imp, dtype=float)
    except Exception:
        return np.zeros(X.shape[1], dtype=float)


def compute_base_screeners(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = pd.DataFrame({"Feature": X.columns})
    rows["Family"] = rows["Feature"].map(infer_family)

    # Mutual information.
    try:
        rows["MI"] = mutual_info_classif(X.values, y.values, random_state=RANDOM_STATE, discrete_features="auto")
    except Exception:
        rows["MI"] = 0.0

    # L1 logistic coefficient signal.
    try:
        logit = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=4000,
                random_state=RANDOM_STATE,
            )),
        ])
        logit.fit(X, y)
        rows["L1AbsCoef"] = np.abs(logit.named_steps["clf"].coef_[0])
    except Exception:
        rows["L1AbsCoef"] = 0.0

    # Tree screeners.
    rows["RFImportance"] = safe_fit_importance(make_tree_model("rf", y), X, y)
    rows["ExtraTreesImportance"] = safe_fit_importance(make_tree_model("extra_trees", y), X, y)
    if HAS_XGB:
        rows["XGBImportance"] = safe_fit_importance(make_tree_model("xgb", y), X, y)
    else:
        rows["XGBImportance"] = 0.0

    # Transparent univariate standardized mean-difference-like signal.
    univ = []
    for c in X.columns:
        a = pd.to_numeric(X.loc[y == 1, c], errors="coerce").dropna()
        b = pd.to_numeric(X.loc[y == 0, c], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            univ.append(0.0)
            continue
        pooled = np.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2.0)
        univ.append(0.0 if pooled <= 1e-12 else abs(float((a.mean() - b.mean()) / pooled)))
    rows["UnivariateSignal"] = univ

    screener_cols = ["MI", "L1AbsCoef", "RFImportance", "ExtraTreesImportance", "XGBImportance", "UnivariateSignal"]
    n = len(rows)
    for col in screener_cols:
        rows[f"{col}_N"] = minmax(rows[col])
        rows[f"{col}_Rank"] = rows[col].rank(ascending=False, method="min")
        rows[f"{col}_RankScore"] = rank_to_score(rows[f"{col}_Rank"], n)
        rows[f"{col}_TopK"] = (rows[f"{col}_Rank"] <= SCREENER_TOPK).astype(int)
    rows["MethodVotes_TopK"] = rows[[f"{c}_TopK" for c in screener_cols]].sum(axis=1)
    rows["MeanRank"] = rows[[f"{c}_Rank" for c in screener_cols]].mean(axis=1)
    rows["BaseConsensusScore"] = rows[[f"{c}_RankScore" for c in screener_cols]].mean(axis=1)
    return rows


def grouped_stability_votes(X: pd.DataFrame, y: pd.Series, groups: pd.Series, base_scores: pd.DataFrame) -> pd.Series:
    votes = pd.Series(0, index=X.columns, dtype=float)
    n_splits = min(STABILITY_SPLITS, max(2, int(pd.Series(groups).nunique())))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    split_count = 0
    for tr, _ in sgkf.split(X, y, groups=groups):
        split_count += 1
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        try:
            mi = mutual_info_classif(Xtr.values, ytr.values, random_state=RANDOM_STATE + split_count, discrete_features="auto")
            for f in pd.Series(mi, index=X.columns).sort_values(ascending=False).head(STABILITY_TOPK).index:
                votes[f] += 1
        except Exception:
            pass
        try:
            logit = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE + split_count)),
            ])
            logit.fit(Xtr, ytr)
            coefs = pd.Series(np.abs(logit.named_steps["clf"].coef_[0]), index=X.columns)
            for f in coefs.sort_values(ascending=False).head(STABILITY_TOPK).index:
                votes[f] += 1
        except Exception:
            pass
        try:
            rf = RandomForestClassifier(
                n_estimators=180,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE + split_count,
                n_jobs=-1,
            )
            rf.fit(Xtr, ytr)
            imp = pd.Series(rf.feature_importances_, index=X.columns)
            for f in imp.sort_values(ascending=False).head(STABILITY_TOPK).index:
                votes[f] += 1
        except Exception:
            pass
    denom = max(1, split_count * 3)  # MI, L1, RF per split
    return votes / denom


def grouped_permutation_signal(X: pd.DataFrame, y: pd.Series, groups: pd.Series, base_scores: pd.DataFrame) -> pd.Series:
    # Run a light grouped-CV permutation importance only on the consensus candidate pool.
    candidates = base_scores.sort_values("BaseConsensusScore", ascending=False)["Feature"].head(PERMUTATION_CANDIDATE_TOPK).tolist()
    out = pd.Series(0.0, index=X.columns)
    if not candidates:
        return out
    n_splits = min(3, max(2, int(pd.Series(groups).nunique())))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    imps = []
    for tr, va in sgkf.split(X[candidates], y, groups=groups):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ])
        try:
            model.fit(X.iloc[tr][candidates], y.iloc[tr])
            pi = permutation_importance(
                model,
                X.iloc[va][candidates],
                y.iloc[va],
                scoring="f1",
                n_repeats=PERMUTATION_REPEATS,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            imps.append(pd.Series(pi.importances_mean, index=candidates))
        except Exception:
            continue
    if imps:
        mean_imp = pd.concat(imps, axis=1).mean(axis=1).clip(lower=0)
        out.loc[mean_imp.index] = mean_imp
    return out


def _grouped_cv_subset_score(X: pd.DataFrame, y: pd.Series, groups: pd.Series, features: List[str]) -> float:
    if not features:
        return 0.0
    n_splits = min(3, max(2, int(pd.Series(groups).nunique())))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    vals = []
    for tr, va in sgkf.split(X[features], y, groups=groups):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", solver="liblinear")),
        ])
        try:
            model.fit(X.iloc[tr][features], y.iloc[tr])
            pred = model.predict(X.iloc[va][features])
            vals.append(fbeta_score(y.iloc[va], pred, beta=1.5, zero_division=0))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else 0.0


def gwo_candidate_signal(X: pd.DataFrame, y: pd.Series, groups: pd.Series, base_scores: pd.DataFrame) -> pd.Series:
    """Train-only binary GWO wrapper used as one candidate signal.

    The final feature set is still chosen by consensus, stability, and mechanism
    family constraints, so GWO can nominate features but cannot become the final
    authority.
    """
    pool = base_scores.sort_values("BaseConsensusScore", ascending=False)["Feature"].head(GWO_CANDIDATE_TOPK).tolist()
    out = pd.Series(0.0, index=X.columns)
    if len(pool) < 2:
        return out

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(pool)
    recurrence = pd.Series(0.0, index=pool)
    cache: Dict[Tuple[int, ...], float] = {}

    def fitness(position: np.ndarray) -> Tuple[float, np.ndarray]:
        mask = position > 0.5
        if not mask.any():
            mask[int(np.argmax(position))] = True
        key = tuple(np.flatnonzero(mask).tolist())
        if key in cache:
            return cache[key], mask
        feats = [pool[i] for i, keep in enumerate(mask) if keep]
        score = _grouped_cv_subset_score(X[pool], y, groups, feats)
        value = score - GWO_SIZE_PENALTY * (len(feats) / max(1, n))
        cache[key] = value
        return value, mask

    for _rep in range(GWO_REPEATS):
        wolves = rng.random((GWO_WOLVES, n))
        alpha = beta = delta = None
        alpha_score = beta_score = delta_score = -np.inf
        for t in range(GWO_ITERATIONS):
            for i in range(GWO_WOLVES):
                score, _ = fitness(wolves[i])
                if score > alpha_score:
                    delta, delta_score = beta, beta_score
                    beta, beta_score = alpha, alpha_score
                    alpha, alpha_score = wolves[i].copy(), score
                elif score > beta_score:
                    delta, delta_score = beta, beta_score
                    beta, beta_score = wolves[i].copy(), score
                elif score > delta_score:
                    delta, delta_score = wolves[i].copy(), score
            if alpha is None:
                continue
            if beta is None:
                beta = alpha.copy()
            if delta is None:
                delta = beta.copy()
            a = 2.0 - 2.0 * (t / max(1, GWO_ITERATIONS - 1))
            for i in range(GWO_WOLVES):
                updated = np.zeros(n)
                for leader in (alpha, beta, delta):
                    r1 = rng.random(n)
                    r2 = rng.random(n)
                    A = 2.0 * a * r1 - a
                    C = 2.0 * r2
                    D = np.abs(C * leader - wolves[i])
                    updated += leader - A * D
                wolves[i] = 1.0 / (1.0 + np.exp(-updated / 3.0))
        if alpha is not None:
            _, final_mask = fitness(alpha)
            recurrence.loc[[pool[i] for i, keep in enumerate(final_mask) if keep]] += 1.0

    out.loc[recurrence.index] = recurrence / max(1, GWO_REPEATS)
    return out


def build_consensus_scores(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    scores = compute_base_screeners(X, y)
    stab = grouped_stability_votes(X, y, groups, scores)
    perm = grouped_permutation_signal(X, y, groups, scores)
    gwo = gwo_candidate_signal(X, y, groups, scores)
    scores["GroupedStabilityFreq"] = scores["Feature"].map(stab).fillna(0.0)
    scores["GroupedPermutation"] = scores["Feature"].map(perm).fillna(0.0)
    scores["GroupedPermutation_N"] = minmax(scores["GroupedPermutation"])
    scores["GWO_CandidateFreq"] = scores["Feature"].map(gwo).fillna(0.0)
    scores["FamilyPriority"] = scores["Family"].map(FAMILY_PRIORITY).fillna(FAMILY_PRIORITY.get("other", 0.25))

    # Final consensus: intentionally not pure prediction. Stability and family priority make the
    # output aligned with a mechanism-informed rule-replay paper.
    scores["ConsensusScore"] = (
        0.40 * scores["BaseConsensusScore"]
        + 0.24 * scores["GroupedStabilityFreq"]
        + 0.14 * scores["GroupedPermutation_N"]
        + 0.07 * scores["GWO_CandidateFreq"]
        + 0.15 * scores["FamilyPriority"]
    )
    scores["Methods_Selected_TopK"] = scores["MethodVotes_TopK"]
    scores = scores.sort_values(["ConsensusScore", "Methods_Selected_TopK", "MeanRank"], ascending=[False, False, True]).reset_index(drop=True)
    scores["ConsensusRank"] = np.arange(1, len(scores) + 1)
    scores["Missing_Category_Type"] = scores["Feature"].map(missing_category_type)
    scores["Semantic_Group"] = scores["Feature"].map(infer_feature_semantic_group)
    scores["Source_Group"] = scores["Feature"].map(infer_feature_source_group)
    return scores


def compose_feature_set(scores: pd.DataFrame, selected_n: int = None) -> pd.DataFrame:
    selected_n = int(selected_n or SELECTED_FEATURE_N)
    selected: List[dict] = []
    selected_set = set()
    fam_counts: Dict[str, int] = defaultdict(int)
    semantic_counts: Dict[str, int] = defaultdict(int)
    source_counts: Dict[str, int] = defaultdict(int)
    audit: Dict[str, dict] = {}
    missing_family_due_to_no_valid_feature = []
    relaxed_source_group_cap_features = []
    relaxed_semantic_group_cap_features = []

    for _, row in scores.iterrows():
        audit[str(row["Feature"])] = {
            "Candidate_Feature": row["Feature"],
            "Family": row["Family"],
            "Missing_Category_Type": row["Missing_Category_Type"],
            "Semantic_Group": row["Semantic_Group"],
            "Source_Group": row["Source_Group"],
            "ConsensusRank": row["ConsensusRank"],
            "ConsensusScore": row["ConsensusScore"],
            "Selected": 0,
            "Excluded_As_Redundant": 0,
            "Excluded_As_Raw_Age_Column": int(is_raw_age_column(row["Feature"])),
            "Excluded_As_Pure_Missing_Unknown": int(row["Missing_Category_Type"] == "pure_missing_unknown"),
            "Replaced_By": "",
            "Reason": "raw_age_column_replaced_by_semantic_age_over60_feature" if is_raw_age_column(row["Feature"]) else ("pure_missing_unknown_excluded" if row["Missing_Category_Type"] == "pure_missing_unknown" else ""),
        }

    def cap_for(fam: str) -> int:
        return int(FAMILY_CAPS.get(fam, FAMILY_CAPS.get("other", 2)))

    def semantic_limit(group: str) -> int:
        return 2 if str(group) == "crash_contact_geometry" else 1

    def is_alterg_age_group_feature(feature: str, sem: str = None, src: str = None) -> bool:
        f = str(feature)
        upper = f.upper()
        sem = str(sem if sem is not None else infer_feature_semantic_group(f))
        src = str(src if src is not None else infer_feature_source_group(f))
        return (
            src == "ALTERG"
            or "ALTERG" in upper
            or f.startswith("年龄段_ALTERG_")
            or sem in {"age_group", "age_elderly_group"}
        )

    def add(row, reason: str, relax_cap: bool = False, relax_source: bool = False, relax_semantic: bool = False, allow_partial_unknown: bool = False) -> bool:
        f = str(row["Feature"])
        fam = str(row["Family"])
        mtype = str(row.get("Missing_Category_Type", missing_category_type(f)))
        sem = str(row.get("Semantic_Group", infer_feature_semantic_group(f)))
        src = str(row.get("Source_Group", infer_feature_source_group(f)))
        if f in selected_set:
            return False
        if is_raw_age_column(f) or sem == "raw_age_continuous":
            audit[f]["Excluded_As_Raw_Age_Column"] = 1
            audit[f]["Excluded_As_Redundant"] = 1
            audit[f]["Replaced_By"] = "FEATURE_Age_Over60"
            audit[f]["Reason"] = "raw_age_column_replaced_by_semantic_age_over60_feature"
            return False
        if f.startswith("FEATURE_Age_Group_") and any(x["Feature"] == "FEATURE_Age_Over60" for x in selected):
            audit[f]["Excluded_As_Redundant"] = 1
            audit[f]["Replaced_By"] = "FEATURE_Age_Over60"
            audit[f]["Reason"] = "age_group_replaced_by_semantic_age_over60_feature"
            return False
        if any(x["Feature"] == "FEATURE_Age_Over60" for x in selected) and is_alterg_age_group_feature(f, sem, src):
            audit[f]["Excluded_As_Redundant"] = 1
            audit[f]["Replaced_By"] = "FEATURE_Age_Over60"
            audit[f]["Reason"] = "age_group_alias_replaced_by_semantic_age_over60_feature"
            return False
        if mtype == "pure_missing_unknown":
            audit[f]["Excluded_As_Pure_Missing_Unknown"] = 1
            audit[f]["Reason"] = "pure_missing_unknown_excluded_from_primary"
            return False
        if mtype == "partial_unknown_detail" and not allow_partial_unknown:
            audit[f]["Reason"] = "partial_unknown_detail_deferred_audit_only"
            return False
        if not relax_cap and fam_counts[fam] >= cap_for(fam):
            return False
        if (not relax_semantic) and semantic_counts[sem] >= semantic_limit(sem):
            audit[f]["Excluded_As_Redundant"] = 1
            audit[f]["Replaced_By"] = "||".join([x["Feature"] for x in selected if str(x.get("Semantic_Group")) == sem])
            audit[f]["Reason"] = "semantic_group_cap"
            return False
        if sem == "age_over60" and semantic_counts[sem] >= 1:
            audit[f]["Excluded_As_Redundant"] = 1
            audit[f]["Reason"] = "age_over60_strict_cap"
            return False
        if (not relax_source) and source_counts[src] >= 1:
            audit[f]["Excluded_As_Redundant"] = 1
            audit[f]["Replaced_By"] = "||".join([x["Feature"] for x in selected if str(x.get("Source_Group")) == src])
            audit[f]["Reason"] = "source_group_cap"
            return False
        rec = row.to_dict()
        if mtype == "partial_unknown_detail":
            reason = reason + "|partial_unknown_detail_audit_only"
        rec["SelectionReason"] = reason
        rec["Semantic_Dedup_Status"] = "selected_relaxed_semantic_cap" if relax_semantic else "selected_primary_semantic_unique"
        rec["Source_Dedup_Status"] = "selected_relaxed_source_cap" if relax_source else "selected_primary_source_unique"
        rec["Physical_Mechanism_Eligible"] = int(mtype == "not_missing_category")
        rec["Redundancy_Reason"] = ""
        rec["Replaced_By"] = ""
        selected.append(rec)
        selected_set.add(f)
        fam_counts[fam] += 1
        semantic_counts[sem] += 1
        source_counts[src] += 1
        audit[f]["Selected"] = 1
        audit[f]["Reason"] = reason
        if relax_source:
            relaxed_source_group_cap_features.append(f)
        if relax_semantic:
            relaxed_semantic_group_cap_features.append(f)
        return True

    # 1) Family minimum coverage from consensus-ranked variables.
    age_rows = scores[scores["Feature"].astype(str).eq("FEATURE_Age_Over60")]
    if age_rows.empty:
        raise RuntimeError("FEATURE_Age_Over60 is required in candidate matrix but was not found.")
    add(age_rows.iloc[0], reason="mandatory_semantic_age_over60_feature")

    # 2) Family minimum coverage from consensus-ranked variables.
    for fam, min_n in MANDATORY_FAMILY_MIN.items():
        fam_rows = scores[scores["Family"] == fam].sort_values("ConsensusScore", ascending=False)
        before = fam_counts[fam]
        for _, row in fam_rows.iterrows():
            if fam_counts[fam] >= min_n:
                break
            add(row, reason=f"mandatory_family_min:{fam}")
        if fam_counts[fam] < min_n and before == fam_counts[fam]:
            missing_family_due_to_no_valid_feature.append(fam)

    # 2) Fill by consensus score under family caps.
    for _, row in scores[scores["Missing_Category_Type"].eq("not_missing_category")].iterrows():
        if len(selected) >= selected_n:
            break
        add(row, reason="consensus_fill_under_family_cap")

    # 3) If still short, relax caps but keep consensus order.
    for _, row in scores[scores["Missing_Category_Type"].eq("not_missing_category")].iterrows():
        if len(selected) >= selected_n:
            break
        add(row, reason="consensus_fill_relaxed_cap", relax_cap=True)
    # 4) Controlled semantic/source relaxation, still no pure missing and no age duplicates.
    for _, row in scores[scores["Missing_Category_Type"].eq("not_missing_category")].iterrows():
        if len(selected) >= selected_n:
            break
        add(row, reason="consensus_fill_relaxed_source_cap", relax_cap=True, relax_source=True)
    for _, row in scores[scores["Missing_Category_Type"].eq("not_missing_category")].iterrows():
        if len(selected) >= selected_n:
            break
        if str(row["Semantic_Group"]) == "age_over60":
            continue
        add(row, reason="consensus_fill_relaxed_semantic_cap", relax_cap=True, relax_source=True, relax_semantic=True)
    # 5) Partial unknown detail is last-resort audit-only signal.
    for _, row in scores[scores["Missing_Category_Type"].eq("partial_unknown_detail")].iterrows():
        if len(selected) >= selected_n:
            break
        add(row, reason="consensus_fill_partial_unknown_last_resort", relax_cap=True, relax_source=True, relax_semantic=True, allow_partial_unknown=True)

    out = pd.DataFrame(selected).head(selected_n).copy()
    if len(out) != selected_n:
        write_csv(pd.DataFrame(audit.values()), "Feature_Semantic_Redundancy_Audit.csv")
        raise RuntimeError(f"Unable to select {selected_n} non-pure-missing features after semantic/source deduplication; selected={len(out)}")
    if "FEATURE_Age_Over60" in set(out["Feature"].astype(str)):
        alterg_selected = [
            str(r["Feature"])
            for _, r in out.iterrows()
            if is_alterg_age_group_feature(r["Feature"], r.get("Semantic_Group"), r.get("Source_Group"))
        ]
        if alterg_selected:
            raise RuntimeError("ALTERG age-group aliases are forbidden when FEATURE_Age_Over60 is selected: " + "||".join(alterg_selected))
        for feature, rec in audit.items():
            if is_alterg_age_group_feature(feature, rec.get("Semantic_Group"), rec.get("Source_Group")) and int(rec.get("Selected", 0) or 0) == 0:
                rec["Excluded_As_Redundant"] = 1
                rec["Replaced_By"] = "FEATURE_Age_Over60"
                rec["Reason"] = "age_group_alias_replaced_by_semantic_age_over60_feature"
    out["Selection_Order"] = np.arange(1, len(out) + 1)
    out.attrs["semantic_audit"] = pd.DataFrame(audit.values())
    out.attrs["missing_family_due_to_no_valid_feature"] = missing_family_due_to_no_valid_feature
    out.attrs["relaxed_source_group_cap_features"] = relaxed_source_group_cap_features
    out.attrs["relaxed_semantic_group_cap_features"] = relaxed_semantic_group_cap_features
    return out


def cv_diagnostic(X: pd.DataFrame, y: pd.Series, groups: pd.Series, features: List[str]) -> float:
    if not features:
        return np.nan
    n_splits = min(3, max(2, int(pd.Series(groups).nunique())))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, va in sgkf.split(X[features], y, groups=groups):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ])
        try:
            model.fit(X.iloc[tr][features], y.iloc[tr])
            pred = model.predict(X.iloc[va][features])
            scores.append(fbeta_score(y.iloc[va], pred, beta=1.5, zero_division=0))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else np.nan


def method_top_features(scores: pd.DataFrame, method: str) -> List[str]:
    col_map = {
        "MutualInfo_TopK": "MI",
        "L1Logistic_TopK": "L1AbsCoef",
        "RFImportance_TopK": "RFImportance",
        "ExtraTrees_TopK": "ExtraTreesImportance",
        "XGBImportance_TopK": "XGBImportance",
        "Univariate_TopK": "UnivariateSignal",
        "Consensus_TopK": "ConsensusScore",
    }
    col = col_map[method]
    return scores.sort_values(col, ascending=False)["Feature"].head(SELECTED_FEATURE_N).tolist()


def main():
    assert_test_not_used_for_fit(
        "04_feature_selection_consensus",
        [TRAIN_MATRIX_ALIAS if Path(TRAIN_MATRIX_ALIAS).exists() else TRAIN_MATRIX],
        test_paths=[TEST_MATRIX, OUTPUTS["blind_replay"], OUTPUTS["evidence_tiers"], OUTPUTS["threshold_sensitivity"], "Rule_Item_Replay_Audit.csv"],
    )
    X, y, groups, group_col = load_matrix()
    scores = build_consensus_scores(X, y, groups)
    selected = compose_feature_set(scores)
    semantic_audit = selected.attrs.get("semantic_audit", pd.DataFrame())
    missing_family_due_to_no_valid_feature = selected.attrs.get("missing_family_due_to_no_valid_feature", [])
    relaxed_source_group_cap_features = selected.attrs.get("relaxed_source_group_cap_features", [])
    relaxed_semantic_group_cap_features = selected.attrs.get("relaxed_semantic_group_cap_features", [])
    selected_features = selected["Feature"].tolist()
    if "FEATURE_Age_Over60" not in selected_features:
        raise RuntimeError("FEATURE_Age_Over60 must be included in final selected 25.")
    raw_age_selected = [f for f in selected_features if is_raw_age_column(f) or infer_feature_semantic_group(f) == "raw_age_continuous"]
    if raw_age_selected:
        raise RuntimeError("Raw age columns are forbidden in final selected 25: " + "||".join(raw_age_selected))
    alterg_age_selected = [
        f for f in selected_features
        if infer_feature_source_group(f) == "ALTERG" or infer_feature_semantic_group(f) in {"age_group", "age_elderly_group"} or "ALTERG" in str(f).upper()
    ]
    if alterg_age_selected:
        raise RuntimeError("ALTERG age-group aliases are forbidden in final selected 25 when FEATURE_Age_Over60 exists: " + "||".join(alterg_age_selected))

    # Full screener ranking table.
    ranking_cols = [
        "Feature", "Family", "ConsensusRank", "ConsensusScore", "BaseConsensusScore", "GroupedStabilityFreq", "GroupedPermutation", "GWO_CandidateFreq", "FamilyPriority",
        "Methods_Selected_TopK", "MeanRank",
        "MI", "MI_Rank", "L1AbsCoef", "L1AbsCoef_Rank", "RFImportance", "RFImportance_Rank",
        "ExtraTreesImportance", "ExtraTreesImportance_Rank", "XGBImportance", "XGBImportance_Rank",
        "UnivariateSignal", "UnivariateSignal_Rank",
    ]
    ranking_cols = [c for c in ranking_cols if c in scores.columns]
    write_csv(scores[ranking_cols], OUTPUTS["screener_rankings"])
    write_csv(scores, OUTPUTS["consensus_scores"])
    write_csv(scores, OUTPUTS["feature_stability"])
    write_csv(
        scores.loc[scores["Missing_Category_Type"].ne("not_missing_category"), [
            "Feature", "Family", "Missing_Category_Type", "Semantic_Group", "Source_Group",
            "ConsensusRank", "ConsensusScore",
        ]].rename(columns={"Feature": "Candidate_Feature"}),
        "Categorical_Missing_Level_Audit.csv",
    )

    # Compatibility output expected by downstream scripts.
    selected_out = selected[[
        "Feature", "Family", "ConsensusScore", "BaseConsensusScore", "GroupedStabilityFreq", "GroupedPermutation",
        "GWO_CandidateFreq", "FamilyPriority", "Methods_Selected_TopK", "MeanRank", "SelectionReason", "Selection_Order",
        "Missing_Category_Type", "Semantic_Group", "Source_Group", "Semantic_Dedup_Status", "Source_Dedup_Status",
        "Physical_Mechanism_Eligible", "Redundancy_Reason", "Replaced_By",
    ]].copy()
    selected_out = selected_out.rename(columns={"Feature": "Selected_Features"})
    write_csv(selected_out, OUTPUTS["selected_features"])
    if not semantic_audit.empty:
        write_csv(semantic_audit, "Feature_Semantic_Redundancy_Audit.csv")
    else:
        write_csv(pd.DataFrame(columns=[
            "Candidate_Feature", "Family", "Missing_Category_Type", "Semantic_Group", "Source_Group",
            "ConsensusRank", "ConsensusScore", "Selected", "Excluded_As_Redundant",
            "Excluded_As_Pure_Missing_Unknown", "Replaced_By", "Reason",
        ]), "Feature_Semantic_Redundancy_Audit.csv")

    # Method comparison: existing screeners are run as comparators, not as the paper's innovation.
    rows = []
    selected_set = set(selected_features)
    methods = [
        "StableConsensusFamily_Final", "Consensus_TopK", "MutualInfo_TopK", "L1Logistic_TopK",
        "RFImportance_TopK", "ExtraTrees_TopK", "XGBImportance_TopK", "Univariate_TopK",
    ]
    for method in methods:
        if method == "StableConsensusFamily_Final":
            feats = selected_features
        else:
            feats = method_top_features(scores, method)
        rows.append({
            "Method": method,
            "Selected_N": len(feats),
            "CV_F1_5_GroupAware": cv_diagnostic(X, y, groups, feats),
            "Overlap_With_Final": len(set(feats) & selected_set) / max(1, len(feats)),
            "Selected_Features": "||".join(feats),
        })
    write_csv(pd.DataFrame(rows), OUTPUTS["feature_method_comparison"])

    fam_rows = []
    for fam in sorted(set(scores["Family"].tolist()) | set(MANDATORY_FAMILY_MIN.keys())):
        sel_fam = selected_out[selected_out["Family"] == fam]
        score_fam = scores[scores["Family"] == fam]
        fam_rows.append({
            "Family": fam,
            "Selected_Count": int(len(sel_fam)),
            "Candidate_Count": int(len(score_fam)),
            "Mandatory_Min": int(MANDATORY_FAMILY_MIN.get(fam, 0)),
            "Family_Cap": int(FAMILY_CAPS.get(fam, FAMILY_CAPS.get("other", 2))),
            "Min_Satisfied": int(len(sel_fam) >= int(MANDATORY_FAMILY_MIN.get(fam, 0))),
            "Mean_Selected_ConsensusScore": float(sel_fam["ConsensusScore"].mean()) if len(sel_fam) else np.nan,
            "Selected_Features": "||".join(sel_fam["Selected_Features"].astype(str).tolist()),
        })
    write_csv(pd.DataFrame(fam_rows), OUTPUTS["family_audit"])

    write_json({
        "stage": "04_feature_selection_consensus",
        "method": "consensus_guided_mechanism_constrained_feature_composition",
        "fit_source": "train",
        "selection_source": "train",
        "feature_selection_source": "train_only",
        "transform_source": "precomputed_train_matrix_only",
        "evaluation_source": "internal_grouped_cv_on_train_only",
        "test_used_for_feature_selection": False,
        "test_used_for_family_constraint": False,
        "test_used_for_gwo_candidate_generation": False,
        "test_used_for_permutation_importance": False,
        "test_used_for_model_selection": False,
        "test_used_for_hyperparameter_selection": False,
        "feature_semantic_deduplication_enabled": True,
        "pure_missing_unknown_excluded_from_primary_feature_selection": True,
        "age_alias_duplicate_excluded": True,
        "age_over60_semantic_feature_used": True,
        "raw_age_column_excluded_from_primary_selected_features": True,
        "one_hot_source_group_cap_enabled": True,
        "feature_semantic_redundancy_audit_generated": True,
        "test_used_for_semantic_deduplication": False,
        "missing_family_due_to_no_valid_feature": missing_family_due_to_no_valid_feature,
        "relaxed_source_group_cap_used": bool(relaxed_source_group_cap_features),
        "relaxed_source_group_cap_features": relaxed_source_group_cap_features,
        "relaxed_semantic_group_cap_used": bool(relaxed_semantic_group_cap_features),
        "relaxed_semantic_group_cap_features": relaxed_semantic_group_cap_features,
        "screeners": ["mutual_information", "l1_logistic", "random_forest", "extra_trees", "xgboost_optional", "univariate_signal", "grouped_stability", "grouped_permutation", "gwo_candidate_generator"],
        "selected_n": len(selected_features),
        "group_col": group_col,
        "mandatory_family_min": MANDATORY_FAMILY_MIN,
        "family_caps": FAMILY_CAPS,
        "outputs": [
            OUTPUTS["screener_rankings"], OUTPUTS["consensus_scores"], OUTPUTS["selected_features"],
            OUTPUTS["feature_method_comparison"], OUTPUTS["family_audit"], "Feature_Semantic_Redundancy_Audit.csv",
            "Categorical_Missing_Level_Audit.csv",
        ],
    }, "04_Run_Manifest.json")
    print(f"✅ 03 consensus feature composition finished. Selected {len(selected_features)} features.")


if __name__ == "__main__":
    main()
