# Contributing

## Workflow

1. Pick a task ID from `docs/PROJECT.md` and open (or reference) an issue for it.
2. Branch: `feat/<task-id>-<short-slug>`.
3. One task per pull request.
4. Open a PR using the template. The template asks for the measurement, the utility
   floor check, and the negative results — fill in all three.
5. Merge with `--no-ff` once CI is green.

## Rules

- **Every behaviour change needs a test** in `tests/`. No exceptions for "it's just a
  refactor" — refactors change behaviour by accident more often than not.
- **Measurements must state** the dataset, the model, the seed, the hardware and the
  wall-clock. A number without those is not a result.
- **Negative results are reported.** If an arm does not work, it goes in
  `results/RESULTS.md` with the same prominence as the arms that do. A null result that
  is not reported is a hidden result, and it is the worst kind.
- **Pre-registered thresholds are not adjusted after the fact.** The thresholds live in
  `docs/METHODS.md` and `docs/PROJECT.md` §9. If they turn out to be wrong, that is a
  finding to publish, not a number to quietly move.
- **The publication gate** lives in `docs/PROJECT.md` §9. It is fixed in advance and is
  not renegotiated to make the timeline easier.

## Running things locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # Linux CI path
pip install -r requirements-dev.txt && pip install -e .
pytest -q
python tools/check_integrity.py
```

On macOS install the default PyPI torch wheel instead (it is MPS-capable); a CUDA
wheel is several gigabytes and cannot be used there at all.

## Adding an arm to the matrix

Arms are declared in `src/bmd/matrix.py`. Two rules:

1. A mechanism that changes the forward pass must ship a **control** — a matched
   intervention whose direction is random or whose labels are shuffled. Without it, a
   positive result is indistinguishable from generic confidence steering.
2. A new arm must state its **principal failure mode** in `docs/METHODS.md`.
