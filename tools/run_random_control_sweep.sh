#!/usr/bin/env bash
# Repeated random-direction controls for the belief-margin-decoding Phase-0 matrix.
#
# WHY THIS EXISTS
# ---------------
# The Phase-0 matrix contains exactly one random-direction control at L4
# (accuracy 0.430 against the real direction's 0.480). The paired test came
# out at p = 0.189 -- not significant. But that is an underpowered comparison,
# not evidence of absence: one random draw answers "is the real direction
# better than *this* direction?", which is a much weaker question than "is the
# real direction better than a random direction?". The second question needs a
# distribution, and a distribution needs more than one draw.
#
# This script runs several independent random directions per layer, each with
# its own recorded seed, so that the real arm can be placed inside the
# distribution of matched random directions:
#
#   * how many random draws beat the real arm?
#   * where does the real arm sit in percentile?
#   * what is the paired (by item) p-value against each draw?
#
# Draw 0 at L4 uses seed 977, which is the seed of the original single-draw
# control, so that arm is re-run and serves as a determinism check: the
# predictions must match the stored ``act_random.json`` exactly.
#
# The real arm is *not* re-run here (--skip-real): the control code path was
# verified to reproduce the matrix code path byte-identically (the L4
# ``act_real.json`` predictions equal the ``act-L4-a1.json`` predictions
# exactly), so the stored matrix arms are valid real arms at every layer.
#
# USAGE
#   bash tools/run_random_control_sweep.sh
#
# Environment: run from the repository root with the project virtualenv on
# PATH. Override PYTHON to use a different interpreter.

set -euo pipefail

PYTHON="${PYTHON:-/workspace/venv/bin/python}"
cd "$(dirname "$0")/.."

export HF_HUB_DISABLE_SSL_VERIFICATION=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# layer:number_of_random_draws
# L4 gets the most draws: it is the only layer where the belief direction had
# any effect at all, so it is the one where "is this specific to the belief?"
# is the live question. L16 gets the second-most: it is where the belief reads
# best and steering does nothing, so it separates "this layer is not
# steerable" from "this direction does not steer".
SWEEP=(
  "4:6"
  "8:2"
  "12:2"
  "16:4"
  "20:2"
)

for entry in "${SWEEP[@]}"; do
  layer="${entry%%:*}"
  draws="${entry##*:}"
  echo "=== random control sweep: layer ${layer}, ${draws} draws ==="
  "$PYTHON" src/run_controls.py \
    --layer "$layer" \
    --random-draws "$draws" \
    --skip-real \
    --skip-reversed \
    --n-items 200 \
    --seed 0 \
    --layers 4,8,12,16,20,24 \
    --out-dir results/controls
done

echo "=== sweep complete ==="
"$PYTHON" tools/read_bite_dissociation.py
