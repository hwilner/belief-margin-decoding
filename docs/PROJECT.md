# Project — atomic task breakdown

This file is the **pre-registered work plan** for the repository. It was written *before* the
implementation, and it is the definition of "done" used by the publication gate in §9.

Every task is atomic (one deliverable, one verifiable artefact), carries a **status**, and is
**mapped to the document** it feeds. Tasks that were already completed during the project's
origination (the feasibility pilot, the literature audit) are listed here too — they were real
work and they are part of the project record.

Status vocabulary: `DONE` · `IN PROGRESS` · `TODO` · `N/A` (dropped, with reason).

---

## 0. How tasks map to documents

| Tag | Document | Meaning |
|---|---|---|
| `EXT` | `docs/EXTENDED_INTRODUCTION.md` | zero-background explainer |
| `MET` | `docs/METHODS.md` | Done / Intended / Undecided, with selection rules |
| `RES` | `results/RESULTS.md` | what was measured |
| `CODE` | `src/bmd/**` | implementation |
| `TEST` | `tests/**` | test suite |
| `INFRA` | packaging, CI, templates | infrastructure |
| `DOC` | `README.md`, `INTRODUCTION.md`, `ROADMAP.md`, `API_REFERENCE.md` | documentation |

---

## A. Origination work — already completed before this repository existed

| ID | Task | Tags | Status |
|---|---|---|---|
| A1 | Feasibility pilot: CPU-only load of `Qwen2.5-0.5B-Instruct`; measure load wall-clock | CODE, RES | DONE |
| A2 | Feasibility pilot: last-token hidden-state extraction; measure ms/text | CODE, RES | DONE |
| A3 | Feasibility pilot: closed-form Fisher-LDA fit; measure fit wall-clock and ‖w‖ | CODE, RES | DONE |
| A4 | Feasibility pilot: per-token margin cost, full vocabulary vs. top-k restriction | CODE, RES | DONE |
| A5 | Feasibility pilot: greedy generation throughput on CPU | CODE, RES | DONE |
| A6 | Literature novelty audit (arXiv occupancy map for belief-in-prompt / belief-as-reward / logit steering / ToM benchmarks / privacy leakage) | EXT, MET | DONE |
| A7 | Verification of HuggingFace model and dataset IDs via the HF API | CODE | DONE |
| A8 | Feasibility pilot: implementation traps (fancy-index copy of the embedding matrix; margin scale blow-up) | CODE, MET | DONE |

---

## B. `docs/EXTENDED_INTRODUCTION.md`

| ID | Task | Tags | Status |
|---|---|---|---|
| B1 | §1 — the problem: what "misreading the partner" means, with a concrete wrong answer | EXT | TODO |
| B2 | §2 — the idea in one picture (mermaid flow) + a hand-checkable numeric toy example | EXT | TODO |
| B3 | §3 — the four ideas recapped (dot product, Gaussian, hyperplane, softmax) | EXT | TODO |
| B4 | §4 — what "belief-consistent" and "belief-violating" continuations are, side by side | EXT | TODO |
| B5 | §5 — what this repository does **not** do (no training, no partner model at deployment, no RL) | EXT | TODO |
| B6 | §6 — how to read the results (including how to read a null result) | EXT | TODO |

---

## C. `docs/METHODS.md`

| ID | Task | Tags | Status |
|---|---|---|---|
| C1 | Done — synthetic BigToM-style item generator with ground truth (protagonist belief vs. world state) | MET, CODE | TODO |
| C2 | Done — base-model answer collection and automatic belief-consistency labelling | MET, CODE | TODO |
| C3 | Done — per-layer hidden-state extraction | MET, CODE | TODO |
| C4 | Done — closed-form Fisher-LDA belief subspace with regularisation and **margin normalisation** | MET, CODE | TODO |
| C5 | Done — the four intervention mechanisms (`none`, `prompt`, `activation`, `logit`) with a layer sweep | MET, CODE | TODO |
| C6 | Done — the two curves: probe accuracy per layer ("read") and intervention effect per layer ("bite") | MET, CODE | TODO |
| C7 | Done — fluency guard (ΔPPL and MAUVE) | MET, CODE | TODO |
| C8 | Intended — loader for the real `Hi-ToM/Hi-ToM_Dataset` | MET, CODE | TODO |
| C9 | Intended — second-order belief subspace (what the partner believes about *me*) | MET, CODE | TODO |
| C10 | Intended — confidence-gated α | MET, CODE | TODO |
| C11 | Undecided **U1** — subspace fitting target: self-generated answers vs. annotated answers | MET | TODO |
| C12 | Undecided **U2** — intervention mechanism: activation-add vs. logit-projection vs. SASA-style per-token margin | MET | TODO |
| C13 | Undecided **U3** — α selection: fixed grid vs. auto-calibrated | MET | TODO |
| C14 | Undecided **U4** — significance test: Wilcoxon signed-rank + bootstrap vs. paired t-test | MET | TODO |
| C15 | Undecided **U5** — utility floor: MAUVE vs. pairwise win-rate vs. ΔPPL | MET | TODO |
| C16 | Undecided **U6** — layer-selection policy: a priori fixed layer vs. post-hoc best layer (leakage risk) | MET | TODO |

Every undecided choice must record **both options** and a **selection rule fixed in advance**.

---

## D. Implementation — `src/bmd/**`

| ID | Task | Tags | Status |
|---|---|---|---|
| D1 | Package skeleton `src/bmd/__init__.py` with public exports | CODE | TODO |
| D2 | `src/bmd/data.py` — synthetic item generator (deterministic, seeded) | CODE | TODO |
| D3 | `src/bmd/model.py` — model loading, per-layer hidden-state extraction, answer scoring | CODE | TODO |
| D4 | `src/bmd/subspace.py` — Fisher-LDA learner, margin computation, margin normalisation | CODE | TODO |
| D5 | `src/bmd/intervene.py` — activation steering and logit steering | CODE | TODO |
| D6 | `src/bmd/evaluate.py` — metrics (accuracy, ΔPPL, MAUVE) and statistics (Wilcoxon, bootstrap CI) | CODE | TODO |
| D7 | `src/bmd/matrix.py` — Phase-0 falsification matrix runner, JSON output | CODE | TODO |
| D8 | `src/run_phase0.py` — CLI entry point | CODE | TODO |
| D9 | `src/bmd/real_data.py` — `Hi-ToM/Hi-ToM_Dataset` loader | CODE | TODO |

---

## E. Tests — `tests/**`

| ID | Task | Tags | Status |
|---|---|---|---|
| E1 | Generator determinism and label correctness | TEST | TODO |
| E2 | Subspace separates trivially separable data; save/load round-trip | TEST | TODO |
| E3 | Margin normalisation is scale-invariant | TEST | TODO |
| E4 | Activation steering moves the intended logit direction | TEST | TODO |
| E5 | Logit steering sign and magnitude are correct | TEST | TODO |
| E6 | Metrics against hand-computed values; bootstrap CI on known data | TEST | TODO |
| E7 | Matrix runner emits the expected arms and never writes NaN | TEST | TODO |
| E8 | The whole suite runs on CPU with no CUDA available | TEST | TODO |

---

## F. Infrastructure

| ID | Task | Tags | Status |
|---|---|---|---|
| F1 | `pyproject.toml` — packaging, pytest configuration | INFRA | TODO |
| F2 | `requirements.txt` / `requirements-dev.txt` (CPU-only torch) | INFRA | TODO |
| F3 | `.github/workflows/ci.yml` — Python 3.10 / 3.11 / 3.12, CPU-only torch, no model download in CI | INFRA | TODO |
| F4 | `tools/check_integrity.py` — checks that can fail | INFRA | TODO |
| F5 | `.github/ISSUE_TEMPLATE/task.md` | INFRA | TODO |
| F6 | `.github/PULL_REQUEST_TEMPLATE.md` | INFRA | TODO |
| F7 | `CONTRIBUTING.md`, `LICENSE`, `.gitignore` | INFRA | TODO |
| F8 | `README.md` — headline is the **measured** result | DOC | TODO |

---

## G. Results — `results/RESULTS.md`

| ID | Task | Tags | Status |
|---|---|---|---|
| G1 | Run 1 — probe accuracy per layer (the **read** curve) | RES | TODO |
| G2 | Run 2 — intervention effect per layer × mechanism (the **bite** curve) | RES | TODO |
| G3 | Run 3 — base vs. prompt-injection vs. intervention ToM accuracy | RES | TODO |
| G4 | Run 4 — fluency guard: ΔPPL and MAUVE per arm | RES | TODO |
| G5 | Run 5 — α sweep | RES | TODO |
| G6 | The dissociation analysis: does the best-reading layer also bite hardest? | RES | TODO |
| G7 | `results/RESULTS.md` written from measured output only, including an explicit **"what was not run"** section | RES | TODO |

---

## H. Remaining documentation

| ID | Task | Tags | Status |
|---|---|---|---|
| H1 | `docs/INTRODUCTION.md` — claim and caveats up front | DOC | TODO |
| H2 | `docs/ROADMAP.md` — phases, each with its principal failure mode | DOC | TODO |
| H3 | `docs/API_REFERENCE.md` | DOC | TODO |
| H4 | `docs/PROJECT.md` — this file | DOC | DONE |
| H5 | `docs/figures/` — concept figure (SVG) and the Phase-0 plot | DOC | TODO |

---

## 9. Publication gate — **pre-registered**

The repository starts **private**. It is flipped to **public** when **all four** conditions hold:

- **(a)** at least **50 %** of the `CODE` tasks (D1–D9) are `DONE`;
- **(b)** at least **50 %** of the `RESULTS` tasks (G1–G7) are `DONE`;
- **(c)** `results/RESULTS.md` exists and reports **at least one fully measured arm** with a
  pre-registered metric, **including the arms that failed**;
- **(d)** CI is green on `main`.

Rationale for the shape of the gate: publishing a code shell with no measurements, or measurements
with no code, are both worse than publishing neither. The 50 % thresholds are set so that the
repository becomes public once it is *submittable*, not once it is *finished*.

**This gate is fixed before the first result was measured and is not adjusted afterwards.**
