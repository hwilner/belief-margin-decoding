"""Synthetic Theory-of-Mind items with exact ground truth.

The generator builds a three-agent scene in which every agent's belief about
one object's location is known by construction:

    P  ("away")   did not witness the move -> believes the ORIGINAL location
    C  ("told")   was told a location         -> believes the location it was told
    O  ("mover")  performed the move          -> believes the TRUE location

Exactly one of the three beliefs coincides with the world state, so the
protagonist can never be the mover: a protagonist always has a belief that
differs from reality. That gives a three-way question with two *distinguishable*
error modes:

    correct    the protagonist's believed location
    world      the true location (a perspective-blind answer)
    other      the third agent's believed location (a wrong-attribution answer)

Clause order is randomised so that neither "first location mentioned" nor
"last location mentioned" is a winning strategy, and the two non-protagonist
clauses are swapped with the protagonist clause independently.

Everything is generated from a seeded ``random.Random`` so results are exactly
reproducible across machines and Python versions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

OPTION_LETTERS: tuple[str, str, str] = ("A", "B", "C")

#: Single-noun locations. Every candidate therefore has the same token shape
#: (" <noun>."), which removes the length bias that would otherwise confound a
#: likelihood-scored multiple choice.
LOCATIONS: tuple[str, ...] = (
    "drawer",
    "bed",
    "shelf",
    "toolbox",
    "curtain",
    "table",
    "cupboard",
    "staircase",
    "attic",
    "lamp",
)

NAMES: tuple[str, ...] = (
    "Alice",
    "Bruno",
    "Carmen",
    "Dmitri",
    "Elena",
    "Farid",
    "Greta",
    "Hassan",
)

OBJECTS: tuple[str, ...] = (
    "key",
    "letter",
    "envelope",
    "wallet",
    "notebook",
    "spanner",
)


@dataclass(frozen=True)
class ToMItem:
    """One Theory-of-Mind item.

    Attributes
    ----------
    item_id:
        Stable identifier, unique within a generated set.
    story:
        The narrative the model reads.
    question:
        The question asked about the protagonist's belief.
    protagonist:
        Name of the agent the question is about.
    options:
        The three candidate location phrases, in the order shown to the model.
    correct_index:
        Index into ``options`` holding the protagonist's believed location.
    world_index:
        Index into ``options`` holding the true location.
    other_index:
        Index into ``options`` holding the third agent's believed location.
    """

    item_id: str
    story: str
    question: str
    protagonist: str
    obj: str
    options: tuple[str, str, str]
    correct_index: int
    world_index: int
    other_index: int


def _clause_away(rng: random.Random, name: str, obj: str, mover: str, loc: str) -> str:
    """The world event. Names the protagonist so that "the location in the
    clause that mentions the protagonist" is the WRONG answer, which breaks
    the cheapest surface heuristic."""
    return f"While {name} was away, {mover} moved the {obj} to the {loc}."


def _clause_before(rng: random.Random, obj: str, loc: str) -> str:
    """The original location -- what the protagonist still believes."""
    return f"Before that, the {obj} had been in the {loc}."


def _clause_told(rng: random.Random, name: str, obj: str, loc: str) -> str:
    """A second agent's belief, which serves as the wrong-attribution distractor."""
    return f"{name} was told the {obj} was in the {loc}."


def generate_items(n: int, seed: int = 0) -> list[ToMItem]:
    """Generate ``n`` deterministic Theory-of-Mind items.

    Parameters
    ----------
    n:
        Number of items to generate.
    seed:
        Seed for the internal ``random.Random``. Two calls with the same
        ``(n, seed)`` produce byte-identical items.

    Returns
    -------
    list[ToMItem]
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rng = random.Random(seed)
    items: list[ToMItem] = []

    for i in range(n):
        obj = rng.choice(OBJECTS)
        away_name, told_name, mover_name = rng.sample(NAMES, 3)
        loc_original, loc_true, loc_told = rng.sample(LOCATIONS, 3)

        clauses = [
            _clause_away(rng, away_name, obj, mover_name, loc_true),
            _clause_before(rng, obj, loc_original),
            _clause_told(rng, told_name, obj, loc_told),
        ]
        rng.shuffle(clauses)
        story = " ".join(clauses)

        # The protagonist is either the agent who missed the move or the agent
        # who was told. Both hold a belief that differs from the world state.
        if rng.random() < 0.5:
            protagonist = away_name
            believed, other_loc = loc_original, loc_told
        else:
            protagonist = told_name
            believed, other_loc = loc_told, loc_original

        locations = [believed, loc_true, other_loc]
        order = list(range(3))
        rng.shuffle(order)
        options = tuple(locations[j] for j in order)
        correct_index = order.index(0)
        world_index = order.index(1)
        other_index = order.index(2)

        question = f"Where does {protagonist} think the {obj} is?"

        items.append(
            ToMItem(
                item_id=f"tom-{seed}-{i:05d}",
                story=story,
                question=question,
                protagonist=protagonist,
                obj=obj,
                options=options,  # type: ignore[arg-type]
                correct_index=correct_index,
                world_index=world_index,
                other_index=other_index,
            )
        )
    return items


def render_prompt(item: ToMItem) -> str:
    """Render the prompt the model sees."""
    return (
        f"{item.story}\n\n"
        f"Question: {item.question}\n"
        f"Answer: {item.protagonist} thinks the {item.obj} is"
    )


def render_candidate(item: ToMItem, index: int) -> str:
    """Render the candidate continuation for one option.

    The continuation is ``" <location>."`` so that every candidate has the same
    token shape and likelihood comparisons are not confounded by length.
    """
    if not 0 <= index < 3:
        raise IndexError(f"option index out of range: {index}")
    return f" {item.options[index]}."


def render_statement(item: ToMItem, index: int) -> str:
    """Render a full belief statement for one option.

    Used to build the labelled (prompt, response) pairs that the belief
    subspace is fitted on.
    """
    if not 0 <= index < 3:
        raise IndexError(f"option index out of range: {index}")
    return f"{item.protagonist} thinks the {item.obj} is in the {item.options[index]}."


def object_of(item: ToMItem) -> str:
    """Return the object name of an item."""
    return item.obj
