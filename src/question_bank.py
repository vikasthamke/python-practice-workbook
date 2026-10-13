"""Canonical question bank: loads, indexes and looks up the 100 questions.

Question content lives in ``src/questions/`` (one module per difficulty band).
Everything else – the notebook, the catalog, the practice functions and the test
suite – is generated from or validated against this single source of truth.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .schema import CATEGORIES, CATEGORY_ALIASES, DIFFICULTIES, Question


class QuestionNotFoundError(LookupError, ValueError):
    """Raised for question numbers outside Q001–Q100 or malformed identifiers."""


@lru_cache(maxsize=1)
def all_questions() -> tuple[Question, ...]:
    from .questions import (advanced_a, advanced_b, beginner, expert, intermediate_a,
                            intermediate_b)

    questions = [q for module in (beginner, intermediate_a, intermediate_b, advanced_a,
                                  advanced_b, expert) for q in module.QUESTIONS]
    return tuple(sorted(questions, key=lambda q: q.id))


@lru_cache(maxsize=1)
def _index() -> dict[int, Question]:
    return {q.id: q for q in all_questions()}


def parse_question_number(value: int | str) -> int:
    """Accept 7, "7", "Q7", "q007" or "Q007" and return the integer 7."""
    if isinstance(value, bool):
        raise QuestionNotFoundError("Question number must be an int like 7 or a string like 'Q007'.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and re.fullmatch(r"\s*[Qq]?0*(\d{1,3})\s*", value):
        number = int(re.fullmatch(r"\s*[Qq]?0*(\d{1,3})\s*", value).group(1))
    else:
        raise QuestionNotFoundError(
            f"Could not understand question number {value!r}. Use an int like 7 or a "
            "string like 'Q007'.")
    if not 1 <= number <= 100:
        raise QuestionNotFoundError(
            f"Question {value!r} does not exist. Valid questions are Q001–Q100 "
            "(pass 1–100). Use list_questions() to browse.")
    return number


def get_question(value: int | str) -> Question:
    return _index()[parse_question_number(value)]


def resolve_difficulty(value: str) -> str:
    for d in DIFFICULTIES:
        if value.strip().lower() == d.lower():
            return d
    raise ValueError(f"Unknown difficulty {value!r}. Choose from: {', '.join(DIFFICULTIES)}.")


def resolve_category(value: str) -> str:
    text = value.strip().lower()
    if text in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[text]
    for c in CATEGORIES:
        if text == c.lower():
            return c
    matches = [c for c in CATEGORIES if text and text in c.lower()]
    if len(matches) == 1:
        return matches[0]
    options = ", ".join(sorted(CATEGORY_ALIASES))
    raise ValueError(f"Unknown or ambiguous category {value!r}. Use a full category name "
                     f"or one of these aliases: {options}.")


def filter_questions(difficulty: str | None = None, category: str | None = None) -> list[Question]:
    questions = list(all_questions())
    if difficulty:
        d = resolve_difficulty(difficulty)
        questions = [q for q in questions if q.difficulty == d]
    if category:
        c = resolve_category(category)
        questions = [q for q in questions if q.category == c]
    return questions
