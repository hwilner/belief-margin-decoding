#!/usr/bin/env bash
# Run the Phase-0 matrix on a SECOND MODEL, on Apple Silicon.
#
# WHY
# ---
# The recorded Phase-0 run used Qwen2.5-0.5B-Instruct, and the base model sits
# at **chance** on the benchmark (0.370 against a 0.333 chance level, binomial
# p = 0.294). That is the single largest threat to validity in the whole
# result: a model that cannot do the task may not encode a partner belief
# that is worth steering, and "we could not steer the belief" would then be a
# statement about the model rather than about the mechanism.
#
# Qwen2.5-1.5B-Instruct is the natural second point. It fits on an 8 GB Mac in
# bfloat16 (3.1 GB) and it is the same family, so the comparison is about
# scale rather than about architecture.
#
# WHAT THIS RUNS
# --------------
# 1. The full Phase-0 matrix (read curve + bite curve) on the 1.5B model.
# 2. The held-out, item-level cross-validated analysis (the leakage-controlled
#    read curve).
# 3. The matched activation controls (real / random / sign-flipped).
# 4. The random-direction sweep, so the second model gets the same treatment
#    as the first: a distribution of matched random directions, not one draw.
# 5. The read-vs-bite dissociation analysis.
#
# EVERYTHING GOES IN ITS OWN DIRECTORY. The 0.5B results are not touched.
#
# PREREQUISITES (macOS, Apple Silicon)
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install --upgrade pip
#   pip install torch transformers numpy scipy pytest ruff
#   pip install -e .
#
#   Note: do NOT pass --index-url .../whl/cpu. On macOS the default PyPI torch
#   wheel is the one with Metal (MPS) support. The CPU-only index is for Linux.
#
# USAGE
#   bash tools/run_second_model_macos.sh
#
# Environment overrides:
#   PYTHON      interpreter to use (default: .venv/bin/python, else python3)
#   MODEL_ID    model to run (default: Qwen/Qwen2.5-1.5B-Instruct)
#   DTYPE       torch dtype (default: bfloat16; use float32 on an M1/M2 with
#               little memory, or float16 if bfloat16 is unsupported)
#   N_ITEMS     items (default: 200)

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x ".venv/bin/python" ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
N_ITEMS="${N_ITEMS:-200}"
LAYERS="4,8,12,16,20,24"
OUT="results/qwen1_5b"

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

echo "model:  $MODEL_ID"
echo "dtype:  $DTYPE"
echo "items:  $N_ITEMS"
echo "output: $OUT"
"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__, "| mps available:", torch.backends.mps.is_available())
PY

mkdir -p "$OUT"

# 1. The matrix. The base pass is the expensive part (~1.2 h for 0.5B on a
#    2-vCPU container; materially faster on MPS). It is cached and resumable.
"$PYTHON" src/run_phase0.py \
  --model-id "$MODEL_ID" \
  --dtype "$DTYPE" \
  --n-items "$N_ITEMS" \
  --layers "$LAYERS" \
  --act-alphas 1.0 \
  --score-alphas 0.25,0.5,1,2,4 \
  --out-dir "$OUT"

# 2. The leakage-controlled read curve: item-level 5-fold cross-validation.
"$PYTHON" tools/analyze_heldout.py \
  --cache "$OUT/base_cache.npz" \
  --out-dir "$OUT/heldout" \
  --n-items "$N_ITEMS" \
  --seed 0 \
  --layers "$LAYERS"

# 3. The matched activation controls at the layer that mattered most (L4).
"$PYTHON" src/run_controls.py \
  --model-id "$MODEL_ID" \
  --dtype "$DTYPE" \
  --layer 4 \
  --alpha 1.0 \
  --n-items "$N_ITEMS" \
  --seed 0 \
  --layers "$LAYERS" \
  --cache "$OUT/base_cache.npz" \
  --out-dir "$OUT/controls"

# 4. The random-direction sweep: a distribution, not a single draw.
"$PYTHON" src/run_controls.py \
  --model-id "$MODEL_ID" \
  --dtype "$DTYPE" \
  --layer 4 \
  --random-draws 4 \
  --skip-real \
  --skip-reversed \
  --n-items "$N_ITEMS" \
  --seed 0 \
  --layers "$LAYERS" \
  --cache "$OUT/base_cache.npz" \
  --out-dir "$OUT/controls"

# 5. Read-vs-bite dissociation for this model.
"$PYTHON" tools/read_bite_dissociation.py \
  --heldout "$OUT/heldout/heldout_analysis.json" \
  --ablation-dir "$OUT/ablation" \
  --controls-dir "$OUT/controls" \
  --out-dir "$OUT/dissociation"

echo
echo "=== done: $OUT ==="
echo "Compare against the 0.5B run with:"
echo "  python tools/read_bite_dissociation.py   # the 0.5B numbers"
