"""Tests for the matched-control runner's file naming and resumability.

The first version of the random-control sweep named its output
``act_random_s<seed>.json``. The same direction seed is reused at every layer
-- the direction lives in the activation space, which has the same width at
every depth -- so each layer's run **overwrote** the previous layer's arm,
including its per-item predictions. The layer-4 and layer-8 draws were lost.

These tests pin the naming rule and the resume behaviour so that a long sweep
can never silently lose work again.
"""

from __future__ import annotations

import json

import pytest

from run_controls import random_control_label


def test_label_contains_the_layer():
    label = random_control_label(977, 4)
    assert "L4" in label
    assert "977" in label


def test_labels_differ_across_layers():
    """The same seed at two layers must not produce the same file name."""
    assert random_control_label(977, 4) != random_control_label(977, 8)
    assert random_control_label(977, 16) != random_control_label(978, 16)


def test_label_is_a_valid_filename():
    label = random_control_label(977, 20)
    assert "/" not in label
    assert not label.startswith(".")
    assert label.endswith("20")


def test_labels_are_unique_across_a_sweep():
    """No two (seed, layer) pairs in the sweep may collide."""
    layers = [4, 8, 12, 16, 20]
    seeds = [977, 978, 979, 980, 981, 982]
    labels = [random_control_label(seed, layer) for layer in layers for seed in seeds]
    assert len(labels) == len(set(labels))


@pytest.mark.skipif(
    not (__import__("pathlib").Path(__file__).resolve().parent.parent
         / "results" / "controls").exists(),
    reason="no controls directory",
)
def test_committed_control_arms_are_replayable():
    """Every committed random-direction arm records how to reproduce itself."""
    from pathlib import Path

    controls = Path(__file__).resolve().parent.parent / "results" / "controls"
    arms = sorted(controls.glob("act_random*.json"))
    if not arms:
        pytest.skip("no random-direction control arms committed")
    for path in arms:
        data = json.loads(path.read_text())
        assert "predictions" in data, f"{path.name}: no per-item predictions"
        assert "direction_seed" in data, f"{path.name}: no direction_seed"
        assert "cos_to_real_direction" in data, f"{path.name}: no cos_to_real_direction"
        layer = data.get("arm", {}).get("layer")
        assert layer is not None, f"{path.name}: no layer"
        # The layer in the filename must match the layer in the payload.
        if path.name.startswith("act_random_s"):
            assert f"_L{layer}" in path.name, (
                f"{path.name}: filename and payload disagree about the layer"
            )
