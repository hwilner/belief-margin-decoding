"""Tests for the falsification matrix configuration. Satisfies docs/PROJECT.md task E7."""

from __future__ import annotations

from bmd.data import generate_items
from bmd.matrix import (
    MECHANISMS,
    MatrixConfig,
    build_arms,
    expected_arm_count,
    oracle_prompt,
)


def test_arm_count_matches_expected():
    config = MatrixConfig()
    arms = build_arms(config)
    assert len(arms) == expected_arm_count(config)


def test_arm_labels_are_unique():
    config = MatrixConfig()
    labels = [a.label for a in build_arms(config)]
    assert len(set(labels)) == len(labels)


def test_first_two_arms_are_the_baselines():
    arms = build_arms(MatrixConfig())
    assert arms[0].mechanism == "none"
    assert arms[1].mechanism == "prompt_oracle"


def test_every_mechanism_is_covered():
    arms = build_arms(MatrixConfig())
    covered = {a.mechanism for a in arms}
    assert covered == set(MECHANISMS)


def test_oracle_prompt_states_the_believed_location():
    item = generate_items(1, seed=0)[0]
    prompt = oracle_prompt(item)
    assert item.protagonist in prompt
    assert item.options[item.correct_index] in prompt


def test_control_arm_is_present_for_every_scored_layer():
    config = MatrixConfig(layers=(4, 8), score_alphas=(1.0,))
    labels = {a.label for a in build_arms(config)}
    for layer in (4, 8):
        assert f"margin_shuffled-L{layer}-a1" in labels
        assert f"margin-L{layer}-a1" in labels
        assert f"logit-L{layer}-a1" in labels
