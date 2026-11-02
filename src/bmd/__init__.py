"""Belief-Margin Decoding (BMD).

Training-free, logit-level steering of a frozen language model toward a
continuation that is consistent with an explicitly modelled partner belief.

Public API
----------
data        -- synthetic ToM item generator with ground truth
subspace    -- closed-form belief subspace (Fisher LDA) and probe variants
intervene   -- activation steering, logit bias, margin scoring
evaluate    -- metrics and statistics
model       -- model loading and per-layer state extraction
matrix      -- the Phase-0 falsification matrix runner
"""

from .data import (
    OPTION_LETTERS,
    ToMItem,
    generate_items,
    render_candidate,
    render_prompt,
)
from .evaluate import (
    bootstrap_ci,
    error_breakdown,
    paired_test,
    tom_accuracy,
)
from .intervene import ActivationSteerer, LogitBias, margin_scores
from .model import extract_states, load_model
from .subspace import BeliefSubspace, fit_subspace, normalize_margins

__version__ = "0.1.0"

__all__ = [
    "OPTION_LETTERS",
    "ToMItem",
    "generate_items",
    "render_candidate",
    "render_prompt",
    "BeliefSubspace",
    "fit_subspace",
    "normalize_margins",
    "ActivationSteerer",
    "LogitBias",
    "margin_scores",
    "bootstrap_ci",
    "error_breakdown",
    "paired_test",
    "tom_accuracy",
    "extract_states",
    "load_model",
    "__version__",
]
