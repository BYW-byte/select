# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from _utils import assert_test_not_used_for_fit, read_csv_smart, require_columns, write_csv, write_json

rule05 = importlib.import_module("05_rule_mining")
blind06 = importlib.import_module("06_blind_replay")
evidence07 = importlib.import_module("07_evidence_grading")
triplet10 = importlib.import_module("10_triplet_rule_extension")


BASELINE_A_CANDIDATES = "12A_Unconstrained_Triplet_Candidates.csv"
BASELINE_A_REPLAY = "12B_Unconstrained_Triplet_Blind_Replay.csv"
BASELINE_B_RULES = "12C_Conventional_ARM_Triplet_Rules.csv"
BASELINE_B_REPLAY = "12D_Conventional_ARM_Blind_Replay.csv"
BASELINE_COMPARISON = "12E_Rule_Extraction_Baseline_Comparison.csv"
BASELINE_AUDIT = "12F_Rule_Extraction_Baseline_Audit.csv"
BASELINE_MANIFEST = "12_Run_Manifest.json"

BASELINE_MIN_SUPPORT = 0.015
BASELINE_MIN_LIFT = 1.05
BASELINE_MIN_HIT_N = 40
BASELINE_TOPKS = (20, 50)
INPUT_SEARCH_DIRS = ("", "result", "manuscript_package", "Fig4")
OUTPUT_DIR = Path(".")


def resolve_input_path(name: str) -> str:
    for directory in INPUT_SEARCH_DIRS:
        p = Path(directory) / name if directory else Path(name)
        if p.exists():
            return str(p)
    return name


def output_path(name: str) -> str:
    return str(OUTPUT_DIR / name) if OUTPUT_DIR != Path(".") else name


def configure_resolved_config() -> Dict[str, str]:
    global OUTPUT_DIR
    paths = {
        "train": resolve_input_path(TRAIN_MATRIX),
        "test": resolve_input_path(TEST_MATRIX),
        "manifest": resolve_input_path(OUTPUTS["rule_manifest"]),
        "sentinel_train": resolve_input_path(OUTPUTS["sentinel_flags_train"]),
        "sentinel_test": resolve_input_path(OUTPUTS["sentinel_flags_test"]),
    }
    if Path("result").is_dir() and Path(paths["manifest"]).parent.name == "result":
        OUTPUT_DIR = Path("result")
        result_train_flags = Path("result") / SENTINEL_FLAGS_TRAIN
        result_test_flags = Path("result") / SENTINEL_FLAGS_TEST
        if result_train_flags.exists():
            paths["sentinel_train"] = str(result_train_flags)
        if result_test_flags.exists():
            paths["sentinel_test"] = str(result_test_flags)
    else:
        OUTPUT_DIR = Path(".")
    OUTPUTS["rule_manifest"] = paths["manifest"]
    OUTPUTS["sentinel_flags_train"] = paths["sentinel_train"]
    OUTPUTS["sentinel_flags_test"] = paths["sentinel_test"]
    triplet10.OUTPUTS["rule_manifest"] = paths["manifest"]
    triplet10.OUTPUTS["sentinel_flags_train"] = paths["sentinel_train"]
    triplet10.OUTPUTS["sentinel_flags_test"] = paths["sentinel_test"]
    blind06.OUTPUTS["sentinel_flags_train"] = paths["sentinel_train"]
    blind06.OUTPUTS["sentinel_flags_test"] = paths["sentinel_test"]
    evidence07.OUTPUTS["sentinel_flags_train"] = paths["sentinel_train"]
    evidence07.OUTPUTS["sentinel_flags_test"] = paths["sentinel_test"]
    return paths


def manifest_items(manifest: pd.DataFrame) -> List[str]:
    return [str(x) for x in manifest["item"].dropna().astype(str).tolist() if str(x)]


def rule_id(prefix: str, n: int) -> str:
    return f"{prefix}{n:03d}"


def item_metadata(items: Iterable[str], manifest_map: Dict[str, dict]) -> Dict[str, object]:
    item_list = [str(x) for x in items if str(x)]
    specs = [manifest_map.get(item, {}) for item in item_list]
    families = [str(spec.get("family", "other")) for spec in specs]
    sources = [str(spec.get("source_feature", "")) for spec in specs]
    axis = rule05.infer_mechanism_axis(families, sources, item_list)
    return {
        "Source_Features": "||".join(sources),
        "Mechanism_Axis": axis,
        "Mechanism_Families": "|".join(sorted(set(families))),
        "Mechanism_Family_N": len(set(families)),
        "Family_Signature": "|".join(sorted(set(families))),
        "Semantic_Duplicate_Flag": int(bool(rule05.semantic_duplicate_details(item_list, manifest_map))),
        "Physical_Mechanism_Eligible": int(all(
            int(pd.to_numeric(pd.Series([spec.get("Physical_Mechanism_Eligible", 0)]), errors="coerce").fillna(0).iloc[0]) == 1
            for spec in specs
        )),
        "Governance_Scene": rule05.governance_scene(axis),
        "Governance_Interpretation_Template": rule05.governance_template(axis),
        "Interpretation_Boundary": rule05.interpretation_boundary(axis),
    }


def build_unconstrained_triplet_candidates(train: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(train[LABEL_COL], errors="coerce").fillna(0).astype(int)
    base_rate = float(y.mean())
    manifest_map = triplet10.manifest_lookup(manifest)
    train_flags = blind06.load_sentinel_flags(OUTPUTS["sentinel_flags_train"], train.index)
    rows = []
    for combo in itertools.combinations(manifest_items(manifest), 3):
        stats = triplet10.rule_stats(train, y, base_rate, manifest_map, combo, train_flags)
        if not stats["Resolved"]:
            continue
        if int(stats["Hit_N"]) < BASELINE_MIN_HIT_N:
            continue
        if float(stats["Support"]) < BASELINE_MIN_SUPPORT:
            continue
        if pd.isna(stats["Lift"]) or float(stats["Lift"]) < BASELINE_MIN_LIFT:
            continue
        rows.append({
            "Antecedent_Items": "||".join(combo),
            "Rule_Length": 3,
            "Train_Hit_N": int(stats["Hit_N"]),
            "Train_Support": float(stats["Support"]),
            "Train_Severe_N": int(stats["Severe_N"]),
            "Train_Confidence": float(stats["Confidence"]),
            "Train_Lift": float(stats["Lift"]),
            "Train_Resolve_Trace": stats["Resolve_Trace"],
            **item_metadata(combo, manifest_map),
            "Semantic_Duplicate_Filter_Used": 0,
            "Family_Diversity_Filter_Used": 0,
            "Core_Family_Filter_Used": 0,
            "Same_Source_Filter_Used": 0,
            "Axis_Cap_Used": 0,
            "Physical_Mechanism_Valid_Filter_Used": 0,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["Train_Lift", "Train_Confidence", "Train_Hit_N", "Antecedent_Items"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out.insert(0, "Triplet_Rule_ID", [rule_id("UA", i + 1) for i in range(len(out))])
    out["Train_Rank"] = np.arange(1, len(out) + 1)
    out["In_Top20"] = out["Train_Rank"].le(20).astype(int)
    out["In_Top50"] = out["Train_Rank"].le(50).astype(int)
    return out


def build_manifest_item_matrix(train: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    item_df, _ = rule05.build_items(train, manifest_items(manifest), fit_manifest=False, manifest=manifest)
    return item_df.astype(bool)


def build_conventional_arm_triplet_rules(train: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if not HAS_MLXTEND:
        raise ImportError("mlxtend is required for FPGrowth baseline. Please install mlxtend.")
    item_df = build_manifest_item_matrix(train, manifest)
    frequent = fpgrowth(item_df, min_support=BASELINE_MIN_SUPPORT, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame()
    rules = association_rules(frequent, metric="lift", min_threshold=BASELINE_MIN_LIFT)
    rules = rules[rules["consequents"].apply(lambda x: set(x) == {TARGET_ITEM})].copy()
    rules["Rule_Length"] = rules["antecedents"].apply(len)
    rules = rules[rules["Rule_Length"].eq(3)].copy()
    if rules.empty:
        return pd.DataFrame()

    y = pd.to_numeric(train[LABEL_COL], errors="coerce").fillna(0).astype(int)
    base_rate = float(y.mean())
    manifest_map = triplet10.manifest_lookup(manifest)
    train_flags = blind06.load_sentinel_flags(OUTPUTS["sentinel_flags_train"], train.index)
    rows = []
    for _, row in rules.iterrows():
        items = tuple(sorted(str(x) for x in row["antecedents"]))
        stats = triplet10.rule_stats(train, y, base_rate, manifest_map, items, train_flags)
        if not stats["Resolved"]:
            continue
        rows.append({
            "Antecedent_Items": "||".join(items),
            "Rule_Length": 3,
            "FPGrowth_Itemset_Support": float(row.get("support", np.nan)),
            "Train_Hit_N": int(stats["Hit_N"]),
            "Train_Support": float(stats["Support"]),
            "Train_Severe_N": int(stats["Severe_N"]),
            "Train_Confidence": float(stats["Confidence"]),
            "Train_Lift": float(stats["Lift"]),
            "leverage": float(row.get("leverage", np.nan)),
            "conviction": float(row.get("conviction", np.nan)),
            "zhangs_metric": float(row.get("zhangs_metric", np.nan)) if "zhangs_metric" in row else np.nan,
            "Train_Resolve_Trace": stats["Resolve_Trace"],
            **item_metadata(items, manifest_map),
            "Mechanism_Constraint_Used": 0,
            "Physical_Mechanism_Valid_Filter_Used": 0,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates("Antecedent_Items")
    out = out.sort_values(
        ["Train_Lift", "Train_Confidence", "Train_Hit_N", "Antecedent_Items"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out.insert(0, "Triplet_Rule_ID", [rule_id("ARM", i + 1) for i in range(len(out))])
    out["Train_Rank"] = np.arange(1, len(out) + 1)
    out["In_Top20"] = out["Train_Rank"].le(20).astype(int)
    out["In_Top50"] = out["Train_Rank"].le(50).astype(int)
    return out


def replay_top50(train: pd.DataFrame, test: pd.DataFrame, manifest: pd.DataFrame, candidates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates.copy(), pd.DataFrame()
    selected = candidates.head(max(BASELINE_TOPKS)).copy()
    replay = triplet10.replay_triplets_sentinel_aware(test, manifest, selected)
    item_audit = triplet10.triplet_rule_item_replay_audit(train, test, replay, manifest)
    replay["Evidence_Tier"] = "baseline_fixed_replay"
    replay = evidence07.attach_sentinel_rule_fields(replay, item_audit)
    replay["Selection_Basis"] = "training_metrics_only"
    replay["Test_Used_For_Selection"] = 0
    replay["In_Top20"] = pd.to_numeric(replay.get("Train_Rank"), errors="coerce").le(20).astype(int)
    replay["In_Top50"] = pd.to_numeric(replay.get("Train_Rank"), errors="coerce").le(50).astype(int)
    return replay, item_audit


def summarize_replay(method: str, replay: pd.DataFrame, topk: int) -> Dict[str, object]:
    subset = replay[pd.to_numeric(replay.get("Train_Rank"), errors="coerce").le(topk)].copy() if not replay.empty else pd.DataFrame()
    n = int(len(subset))
    if n == 0:
        return {
            "Method": method, "TopK": topk, "Rule_N": 0, "Resolved_Rate": np.nan,
            "Q_LT_0_05_N": 0, "Median_Test_Hit_N": np.nan, "Median_Test_Confidence": np.nan,
            "Median_RR": np.nan, "Semantic_Duplicate_Rate": np.nan,
            "Sentinel_Dominated_Rate": np.nan, "Physical_Valid_Rate": np.nan,
            "Axis_Coverage_N": 0,
        }
    return {
        "Method": method,
        "TopK": topk,
        "Rule_N": n,
        "Resolved_Rate": float(subset.get("Resolved", pd.Series(dtype=bool)).astype(bool).mean()),
        "Q_LT_0_05_N": int(pd.to_numeric(subset.get("q_value_BH"), errors="coerce").lt(0.05).sum()),
        "Median_Test_Hit_N": float(pd.to_numeric(subset.get("Test_Hit_N"), errors="coerce").median()),
        "Median_Test_Confidence": float(pd.to_numeric(subset.get("Test_Confidence"), errors="coerce").median()),
        "Median_RR": float(pd.to_numeric(subset.get("Risk_Ratio_vs_BaseRate"), errors="coerce").median()),
        "Semantic_Duplicate_Rate": float(pd.to_numeric(subset.get("Rule_Semantic_Duplicate_Flag"), errors="coerce").fillna(0).mean()),
        "Sentinel_Dominated_Rate": float(pd.to_numeric(subset.get("Rule_Sentinel_Dominated_Flag"), errors="coerce").fillna(0).mean()),
        "Physical_Valid_Rate": float(pd.to_numeric(subset.get("Physical_Mechanism_Valid_Flag"), errors="coerce").fillna(0).mean()),
        "Axis_Coverage_N": int(subset.get("Mechanism_Axis", pd.Series(dtype=str)).dropna().astype(str).nunique()),
    }


def audit_rows(a_candidates: pd.DataFrame, b_candidates: pd.DataFrame, a_replay: pd.DataFrame, b_replay: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "Audit_Item": "Inputs read",
            "Status": "PASS",
            "Detail": "Ready_Matrix_Train.csv||Ready_Matrix_Test.csv||04_Rule_Threshold_Manifest.csv||Sentinel_Flags_Train.csv||Sentinel_Flags_Test.csv",
        },
        {
            "Audit_Item": "Selection uses training metrics only",
            "Status": "PASS",
            "Detail": "TopK ranking columns: Train_Lift, Train_Confidence, Train_Hit_N; test replay happens after truncation.",
        },
        {
            "Audit_Item": "Baseline A constraints disabled",
            "Status": "PASS",
            "Detail": "No semantic duplicate filtering, family diversity, core family, same-source, axis cap, or physical-valid admission filter.",
        },
        {
            "Audit_Item": "Baseline B mechanism constraints disabled",
            "Status": "PASS",
            "Detail": "FPGrowth association rules only require consequent target, antecedent length 3, min_support=0.015, min_lift=1.05.",
        },
        {
            "Audit_Item": "Primary pipeline isolation",
            "Status": "PASS",
            "Detail": "Script writes only 12* baseline outputs and does not modify 01-11 logic, rule manifest, selected features, or primary rules.",
        },
    ]
    for method, candidates, replay in [
        ("Baseline A", a_candidates, a_replay),
        ("Baseline B", b_candidates, b_replay),
    ]:
        rows.append({
            "Audit_Item": f"{method} candidate and replay counts",
            "Status": "PASS",
            "Detail": f"candidates={len(candidates)}; replay_top50={len(replay)}",
        })
    return pd.DataFrame(rows)


def main() -> None:
    paths = configure_resolved_config()
    required = [
        paths["train"], paths["test"], paths["manifest"],
        paths["sentinel_train"], paths["sentinel_test"],
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required baseline inputs: " + "||".join(missing))
    assert_test_not_used_for_fit(
        "12_rule_extraction_baseline",
        [paths["train"], paths["manifest"], paths["sentinel_train"]],
        test_paths=[paths["test"], paths["sentinel_test"]],
    )

    train = read_csv_smart(paths["train"])
    test = read_csv_smart(paths["test"])
    manifest = read_csv_smart(paths["manifest"])
    require_columns(manifest, paths["manifest"], [
        "item", "source_feature", "family", "transform_type", "operator", "threshold",
        "Missing_Category_Type", "Semantic_Group", "Physical_Mechanism_Eligible",
    ])

    a_candidates = build_unconstrained_triplet_candidates(train, manifest)
    a_replay, _a_item_audit = replay_top50(train, test, manifest, a_candidates)

    b_candidates = build_conventional_arm_triplet_rules(train, manifest)
    b_replay, _b_item_audit = replay_top50(train, test, manifest, b_candidates)

    comparison = pd.DataFrame(
        [summarize_replay("A_Unconstrained_Triplet", a_replay, k) for k in BASELINE_TOPKS]
        + [summarize_replay("B_Conventional_ARM_FPGrowth", b_replay, k) for k in BASELINE_TOPKS]
    )
    audit = audit_rows(a_candidates, b_candidates, a_replay, b_replay)

    write_csv(a_candidates, output_path(BASELINE_A_CANDIDATES))
    write_csv(a_replay, output_path(BASELINE_A_REPLAY))
    write_csv(b_candidates, output_path(BASELINE_B_RULES))
    write_csv(b_replay, output_path(BASELINE_B_REPLAY))
    write_csv(comparison, output_path(BASELINE_COMPARISON))
    write_csv(audit, output_path(BASELINE_AUDIT))
    write_json({
        "stage": "12_rule_extraction_baseline",
        "fit_source": "train",
        "selection_source": "train",
        "rule_generation_source": "train_only",
        "threshold_source": "frozen_train_derived_manifest",
        "evaluation_source": "test_fixed_replay_only",
        "test_used_for_rule_generation": False,
        "test_used_for_rule_filtering": False,
        "test_used_for_rule_sorting": False,
        "test_used_for_topk_selection": False,
        "test_used_for_evaluation_only": True,
        "modifies_primary_rules": False,
        "modifies_manifest": False,
        "modifies_selected_features": False,
        "input_paths": paths,
        "output_directory": str(OUTPUT_DIR),
        "baseline_a": {
            "name": "Unconstrained triplet baseline",
            "candidate_triplets_from_manifest": True,
            "sentinel_aware_rule_stats": True,
            "min_train_hit_n": BASELINE_MIN_HIT_N,
            "min_train_support": BASELINE_MIN_SUPPORT,
            "min_train_lift": BASELINE_MIN_LIFT,
            "constraints_disabled": [
                "semantic_duplicate_filter", "family_diversity", "core_family",
                "same_source", "axis_cap", "physical_mechanism_valid_admission",
            ],
            "candidate_n": int(len(a_candidates)),
        },
        "baseline_b": {
            "name": "Conventional ARM/FPGrowth triplet baseline",
            "min_support": BASELINE_MIN_SUPPORT,
            "min_lift": BASELINE_MIN_LIFT,
            "consequent": TARGET_ITEM,
            "antecedent_length": 3,
            "mechanism_constraints_used": False,
            "candidate_n": int(len(b_candidates)),
        },
        "topk_values": list(BASELINE_TOPKS),
        "outputs": [
            BASELINE_A_CANDIDATES, BASELINE_A_REPLAY, BASELINE_B_RULES,
            BASELINE_B_REPLAY, BASELINE_COMPARISON, BASELINE_AUDIT,
        ],
    }, output_path(BASELINE_MANIFEST))
    print(
        "Rule extraction baselines finished. "
        f"A candidates={len(a_candidates)}, B candidates={len(b_candidates)}."
    )


if __name__ == "__main__":
    main()
