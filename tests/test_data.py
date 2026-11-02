"""Tests for the synthetic ToM item generator. Satisfies docs/PROJECT.md task E1."""

from __future__ import annotations

import pytest

from bmd.data import (
    generate_items,
    object_of,
    render_candidate,
    render_prompt,
    render_statement,
)


def test_generation_is_deterministic():
    a = generate_items(20, seed=0)
    b = generate_items(20, seed=0)
    assert a == b


def test_different_seeds_give_different_items():
    a = generate_items(20, seed=0)
    b = generate_items(20, seed=1)
    assert a != b


def test_indices_are_pairwise_distinct():
    for item in generate_items(50, seed=3):
        indices = {item.correct_index, item.world_index, item.other_index}
        assert len(indices) == 3
        assert indices <= {0, 1, 2}


def test_options_have_no_duplicates():
    for item in generate_items(50, seed=3):
        assert len(set(item.options)) == 3


def test_prompt_contains_protagonist_and_question():
    for item in generate_items(20, seed=1):
        prompt = render_prompt(item)
        assert item.protagonist in prompt
        assert item.question in prompt


def test_candidate_mentions_its_location():
    item = generate_items(1, seed=0)[0]
    candidate = render_candidate(item, item.correct_index)
    assert item.options[item.correct_index] in candidate


def test_statement_mentions_protagonist_and_location():
    item = generate_items(1, seed=0)[0]
    statement = render_statement(item, item.correct_index)
    assert item.protagonist in statement
    assert item.options[item.correct_index] in statement


def test_object_of_matches_field():
    for item in generate_items(10, seed=0):
        assert object_of(item) == item.obj


def test_out_of_range_index_raises():
    item = generate_items(1, seed=0)[0]
    with pytest.raises(IndexError):
        render_candidate(item, 3)
    with pytest.raises(IndexError):
        render_statement(item, -1)


def test_non_positive_n_raises():
    with pytest.raises(ValueError):
        generate_items(0, seed=0)
    with pytest.raises(ValueError):
        generate_items(-5, seed=0)
