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


def test_act_arms_stop_at_the_last_hookable_layer():
    """A 24-layer model cannot be activation-steered at layer 24.

    The hook adds to the *input* of a decoder layer, so the last decoder
    layer's output (hidden_states[-1]) can be probed and can carry a
    score-level intervention, but there is no decoder layer after it whose
    input could be perturbed. The first version of the runner built an `act`
    arm for every requested layer and crashed on the last one, after the
    whole run had already been paid for.
    """
    config = MatrixConfig(layers=(4, 12, 24), act_alphas=(1.0,), score_alphas=(1.0,))
    arms = build_arms(config, n_model_layers=24)
    labels = {a.label for a in arms}
    assert "act-L12-a1" in labels
    assert "act-L24-a1" not in labels
    # The score-level interventions are still available at the last layer.
    assert "margin-L24-a1" in labels
    assert "logit-L24-a1" in labels
    assert "margin_shuffled-L24-a1" in labels


def test_expected_arm_count_tracks_the_model_depth():
    config = MatrixConfig(layers=(4, 12, 24), act_alphas=(1.0,), score_alphas=(1.0,))
    assert expected_arm_count(config, n_model_layers=24) == len(build_arms(config, n_model_layers=24))
    assert expected_arm_count(config, n_model_layers=25) == len(build_arms(config, n_model_layers=25))
