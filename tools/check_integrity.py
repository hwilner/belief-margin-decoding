#!/usr/bin/env python3
"""Integrity checks for belief-margin-decoding.

Every check here is designed to FAIL. A checker that cannot fail is a comment.

Run it directly::

    python tools/check_integrity.py

or from CI. Exit code is 1 if any check fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Paths that must exist for the repository to be coherent.
REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "docs/INTRODUCTION.md",
    "docs/EXTENDED_INTRODUCTION.md",
    "docs/METHODS.md",
    "docs/PROJECT.md",
    "docs/ROADMAP.md",
    "docs/API_REFERENCE.md",
    "results/RESULTS.md",
    "src/bmd/__init__.py",
    "src/bmd/data.py",
    "src/bmd/model.py",
    "src/bmd/subspace.py",
    "src/bmd/intervene.py",
    "src/bmd/evaluate.py",
    "src/bmd/matrix.py",
    "src/run_phase0.py",
    "tests/test_data.py",
    "tests/test_subspace.py",
    "tests/test_intervene.py",
    "tests/test_evaluate.py",
    "tests/test_matrix.py",
    "tools/check_integrity.py",
    ".github/workflows/ci.yml",
    ".github/ISSUE_TEMPLATE/task.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
]

#: Patterns that look like a *value* rather than a mention. "no NaN values
#: were produced" is prose; "| NaN |" in a results table is a defect. Only the
#: latter is a failure.
FORBIDDEN_IN_RESULTS = (
    re.compile(r"\|\s*(nan|-?inf(inity)?)\s*\|", re.IGNORECASE),
    re.compile(r"=\s*(nan|-?inf(inity)?)\b", re.IGNORECASE),
    re.compile(r":\s*(nan|-?inf(inity)?)\s*$", re.IGNORECASE | re.MULTILINE),
)

_BACKTICK_PATH = re.compile(r"`([A-Za-z0-9_./-]+\.(?:md|py|toml|txt|json|svg|yml|yaml|csv))`")
_TABLE_ROW = re.compile(r"^\|")


def _results_allowed_missing() -> bool:
    """True while the results document is legitimately not written yet.

    ``docs/PROJECT.md`` marks task G7 (write RESULTS.md) as TODO until the
    Phase-0 run exists, so the document's absence is a pre-registered state
    rather than an integrity failure.
    """
    project = ROOT / "docs" / "PROJECT.md"
    if not project.exists():
        return True
    for line in project.read_text().splitlines():
        if not _TABLE_ROW.match(line):
            continue
        if "| G7 " in line and "TODO" in line:
            return True
    return False


def check_required_paths() -> tuple[bool, str]:
    missing = []
    for rel in REQUIRED_PATHS:
        if rel == "results/RESULTS.md" and _results_allowed_missing():
            continue
        if not (ROOT / rel).exists():
            missing.append(rel)
    if missing:
        return False, "missing required paths: " + ", ".join(missing)
    return True, f"{len(REQUIRED_PATHS)} required paths present"


def check_project_done_paths() -> tuple[bool, str]:
    """Every path mentioned in a DONE row of docs/PROJECT.md must exist."""
    project = ROOT / "docs" / "PROJECT.md"
    if not project.exists():
        return False, "docs/PROJECT.md is missing"
    referenced = []
    for line in project.read_text().splitlines():
        if not _TABLE_ROW.match(line):
            continue
        if "DONE" not in line:
            continue
        referenced.extend(_BACKTICK_PATH.findall(line))
    missing = sorted({p for p in referenced if not (ROOT / p).exists()})
    if missing:
        return False, "DONE rows reference missing paths: " + ", ".join(missing)
    return True, f"{len(referenced)} path references in DONE rows all exist"


def check_results_are_valid() -> tuple[bool, str]:
    """Committed results must not contain invalid JSON numerics."""
    results_dir = ROOT / "results"
    if not results_dir.exists():
        return True, "no results directory yet"
    offenders = []
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".json"}:
            continue
        text = path.read_text(errors="ignore")
        for pattern in FORBIDDEN_IN_RESULTS:
            for match in pattern.finditer(text):
                offenders.append(
                    f"{path.relative_to(ROOT)} contains {match.group(0)!r}"
                )
                break
    if offenders:
        return False, "; ".join(offenders)
    return True, "no invalid numerics in results/"


def check_readme_states_a_measurement() -> tuple[bool, str]:
    readme = ROOT / "README.md"
    if not readme.exists():
        return False, "README.md is missing"
    text = readme.read_text()
    if "measured" not in text.lower():
        return False, "README.md does not mention a measurement"
    return True, "README.md names a measurement"


def check_result_files_parse() -> tuple[bool, str]:
    """Every committed JSON result must parse and contain no NaN."""
    results_dir = ROOT / "results"
    if not results_dir.exists():
        return True, "no results directory yet"
    offenders = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            text = path.read_text()
        except OSError as exc:
            offenders.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        try:
            json.loads(text, parse_constant=_reject_constant)
        except ValueError as exc:
            offenders.append(f"{path.relative_to(ROOT)}: {exc}")
    if offenders:
        return False, "; ".join(offenders)
    return True, "all committed result JSON parses strictly"


def _reject_constant(name: str):
    raise ValueError(f"invalid JSON constant {name!r}")


def check_matrix_arms_are_unique() -> tuple[bool, str]:
    """The matrix configuration must not produce duplicate arm labels."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from bmd.matrix import MatrixConfig, build_arms, expected_arm_count
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"could not import bmd.matrix: {exc}"
    config = MatrixConfig()
    arms = build_arms(config)
    labels = [a.label for a in arms]
    if len(set(labels)) != len(labels):
        dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
        return False, "duplicate arm labels: " + ", ".join(dupes)
    if len(arms) != expected_arm_count(config):
        return False, (
            f"arm count mismatch: build_arms produced {len(arms)}, "
            f"expected_arm_count says {expected_arm_count(config)}"
        )
    return True, f"{len(arms)} arms, all labels unique"


CHECKS = [
    ("required paths", check_required_paths),
    ("PROJECT.md DONE paths", check_project_done_paths),
    ("results contain no invalid numerics", check_results_are_valid),
    ("result JSON parses strictly", check_result_files_parse),
    ("README names a measurement", check_readme_states_a_measurement),
    ("matrix arms are unique", check_matrix_arms_are_unique),
]


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:  # pragma: no cover - defensive
            ok, detail = False, f"checker raised {type(exc).__name__}: {exc}"
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failures += 1
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
