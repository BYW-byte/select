# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from _config import RANDOM_STATE
from file_output_registry import (
    CLEAN_OUTPUT_PATTERNS,
    FINAL_PACKAGE_FILES,
    FINAL_ZIP_NAME,
    MANUSCRIPT_PACKAGE_DIR,
    OBSOLETE_OUTPUT_FILES,
    cleanup_reason,
    is_generated_output_dir,
    is_generated_output_file,
    is_source_or_input_file,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRIMARY_SCRIPTS = [
    "01_build_cohort.py",
    "02_preprocess_bounded.py",
    "03_check_preprocess_outputs.py",
    "04_feature_selection_consensus.py",
    "05_rule_mining.py",
    "06_blind_replay.py",
    "07_evidence_grading.py",
    "08_model_baseline.py",
]

SUPPLEMENTARY_SCRIPTS = [
    "10_triplet_rule_extension.py",
    "11_feature_space_size_sensitivity.py",
]


def clean_outputs() -> int:
    explicit = set(FINAL_PACKAGE_FILES) | set(OBSOLETE_OUTPUT_FILES) | {FINAL_ZIP_NAME}
    patterns = list(CLEAN_OUTPUT_PATTERNS)
    removed = []

    def remove_file(p: Path) -> None:
        try:
            p.chmod(0o666)
        except Exception:
            pass
        p.unlink()

    def remove_dir_if_possible(p: Path) -> bool:
        try:
            shutil.rmtree(p)
            return True
        except PermissionError:
            return False
        except OSError:
            return False

    for name in sorted(explicit):
        p = Path(name)
        if p.exists() and p.is_file() and is_generated_output_file(p) and not is_source_or_input_file(p):
            remove_file(p)
            removed.append(p.name)
    for pattern in patterns:
        for p in Path(".").glob(pattern):
            if p.is_file() and is_generated_output_file(p) and not is_source_or_input_file(p):
                remove_file(p)
                removed.append(p.name)
    package_dir = Path(MANUSCRIPT_PACKAGE_DIR)
    if package_dir.exists() and package_dir.is_dir():
        resolved = package_dir.resolve()
        if resolved.parent == Path(".").resolve():
            file_count = len([p for p in package_dir.rglob("*") if p.is_file()])
            if remove_dir_if_possible(package_dir):
                removed.extend([f"{MANUSCRIPT_PACKAGE_DIR}/"] * max(file_count, 1))
    for p in Path(".").iterdir():
        if p.is_dir() and is_generated_output_dir(p):
            resolved = p.resolve()
            if resolved.parent == Path(".").resolve():
                file_count = len([x for x in p.rglob("*") if x.is_file()])
                if remove_dir_if_possible(p):
                    removed.extend([f"{p.name}/"] * max(file_count, 1))
    removed_n = len(sorted(set(removed)))
    if removed:
        reasons = {}
        for item in sorted(set(removed)):
            reasons[cleanup_reason(item)] = reasons.get(cleanup_reason(item), 0) + 1
        print(f"Old files removed before run: {removed_n}; reasons={reasons}")
    else:
        print("Old files removed before run: 0")
    return removed_n


def run_script(script: str) -> int:
    print("\n" + "=" * 80)
    print(f"Running {script}")
    print("=" * 80)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RUN_ID"] = RUN_ID
    return subprocess.run([sys.executable, script], env=env).returncode


RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{RANDOM_STATE}"
os.environ["RUN_ID"] = RUN_ID
print(f"RUN_ID={RUN_ID}")
OLD_FILES_REMOVED_BEFORE_RUN = clean_outputs()

for script in PRIMARY_SCRIPTS:
    if run_script(script) != 0:
        raise SystemExit(f"Stage failed: {script}")

supplementary_failures = []
for script in SUPPLEMENTARY_SCRIPTS:
    code = run_script(script)
    if code != 0:
        supplementary_failures.append(script)
        print(f"Supplementary stage failed without modifying primary results: {script}")

if run_script("09_leakage_control_audit.py") != 0:
    raise SystemExit("Stage failed: 09_leakage_control_audit.py")

if supplementary_failures:
    print("Supplementary stages needing review: " + "||".join(supplementary_failures))
print("\nFull consensus pipeline finished.")
