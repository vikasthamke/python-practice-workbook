"""Data model for workbook questions.

Every question in the bank is a :class:`Question` instance. Test cases are either
:class:`T` (call the learner's function with arguments and compare the result) or
:class:`CT` (a custom test function that receives the learner's implementation and
makes assertions; used for classes, decorators, generators and file workflows).

Fixture markers (:class:`File`, :class:`DataFile`, :class:`TmpPath`) can appear
anywhere inside a test's arguments. The runner materialises them in a fresh
temporary directory and substitutes the resulting path string before calling the
learner's function, so every test is deterministic and isolated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample_data"

DIFFICULTIES = ("Beginner", "Intermediate", "Advanced", "Expert")

DIFFICULTY_TARGETS = {"Beginner": 20, "Intermediate": 35, "Advanced": 30, "Expert": 15}

CATEGORIES = (
    "Python fundamentals and control flow",
    "Strings and text processing",
    "Lists, tuples, sets, and dictionaries",
    "Functions, comprehensions, and iterators",
    "Sorting, searching, and algorithmic problem-solving",
    "Files, exceptions, and data serialization",
    "Object-oriented programming and reusable components",
    "Debugging, testing, and code quality",
    "Data processing and analytical transformations",
    "Performance, memory, and scalability",
    "Integrated data and AI engineering scenarios",
)

CATEGORY_TARGETS = dict(zip(CATEGORIES, (10, 8, 12, 10, 10, 8, 7, 8, 10, 7, 10)))

# Short aliases accepted by list_questions(category=...) and start_test(category=...).
CATEGORY_ALIASES = {
    "fundamentals": CATEGORIES[0],
    "strings": CATEGORIES[1],
    "data-structures": CATEGORIES[2],
    "functions": CATEGORIES[3],
    "algorithms": CATEGORIES[4],
    "files": CATEGORIES[5],
    "oop": CATEGORIES[6],
    "debugging": CATEGORIES[7],
    "data-processing": CATEGORIES[8],
    "performance": CATEGORIES[9],
    "ai-engineering": CATEGORIES[10],
}

FUND, STR, DS, FUNC, ALG, FILE, OOP, DBG, DATA, PERF, INT = CATEGORIES

# Target roles (Section 1 of the brief).
DS_ROLE = "Data Scientist"
MLE = "Machine Learning Engineer"
AIE = "AI Engineer"
DE = "Data Engineer"
AAI = "Applied AI Engineer"
PYDEV = "Python Developer (data/AI systems)"
ALL_ROLES = [DS_ROLE, MLE, AIE, DE, AAI, PYDEV]

_MISSING = object()


# --------------------------------------------------------------------------- fixtures
@dataclass(frozen=True)
class File:
    """A file created inside the test's temporary directory; replaced by its path."""

    name: str
    content: str | bytes = ""
    gzip: bool = False

    def __repr__(self) -> str:  # shown in failure messages
        return f"<path to temp file {self.name!r}>"


@dataclass(frozen=True)
class DataFile:
    """A file from ``data/sample_data``; replaced by its absolute path."""

    name: str

    def __repr__(self) -> str:
        return f"<path to data/sample_data/{self.name}>"


@dataclass(frozen=True)
class TmpPath:
    """A path inside the test's temporary directory that does not exist yet."""

    name: str

    def __repr__(self) -> str:
        return f"<path to new temp file {self.name!r}>"


# ------------------------------------------------------------------------ test cases
@dataclass
class T:
    """Call-and-compare test: ``entry(*args, **kwargs)`` must equal ``expected``.

    Options:
      raises       name of the exception class the call must raise (subclasses accepted)
      approx       absolute tolerance for float comparison (default 1e-9)
      unordered    compare a returned list as a multiset
      materialize  call ``list()`` on the result before comparing (generator questions)
      time_limit   seconds; the call fails if it takes longer (performance questions)
      example      render this case as a worked example in the question statement
    """

    args: tuple = ()
    expected: Any = _MISSING
    kwargs: dict = field(default_factory=dict)
    note: str = ""
    raises: str | None = None
    approx: float | None = None
    unordered: bool = False
    materialize: bool = False
    time_limit: float | None = None
    example: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            raise TypeError("T.args must be a tuple, e.g. T((5,), 25)")
        if self.raises is None and self.expected is _MISSING:
            raise ValueError("T needs either expected=... or raises=...")

    @property
    def has_expected(self) -> bool:
        return self.expected is not _MISSING


@dataclass
class CT:
    """Custom test: ``func(impl, tmp_dir)`` raises AssertionError on failure."""

    func: Callable[[Any, Any], None]
    note: str = ""


# -------------------------------------------------------------------------- question
@dataclass
class Question:
    id: int
    title: str
    difficulty: str
    category: str
    roles: list[str]
    skills: list[str]
    minutes: int
    scenario: str
    problem: str
    signature: str
    entry: str
    inputs: list[str]
    output: str
    constraints: list[str]
    criteria: list[str]
    hints: list[str]
    solution: str
    explanation: str
    edge_cases: list[str]
    mistakes: list[str]
    tests: list[T | CT]
    complexity: str
    extension: str
    examples: list[str] = field(default_factory=list)  # doctest-style snippets
    starter: str | None = None
    mutates_input: bool = False

    # ---- derived fields -------------------------------------------------------------
    @property
    def qid(self) -> str:
        return f"Q{self.id:03d}"

    @property
    def is_class(self) -> bool:
        return self.signature.lstrip().startswith("class ")

    @property
    def starter_code(self) -> str:
        """Learner workspace: the signature followed by a stub body."""
        if self.starter is not None:
            return self.starter.strip("\n") + "\n"
        sig = self.signature.strip("\n")
        indent = re.match(r"\s*", sig.splitlines()[-1]).group(0) + "    "
        return f"{sig}\n{indent}# TODO: write your solution here\n{indent}pass\n"

    @property
    def example_tests(self) -> list[T]:
        return [t for t in self.tests if isinstance(t, T) and t.example]

    # Field names required by Section 10 of the brief, mapped onto this dataclass.
    def as_record(self) -> dict[str, Any]:
        return {
            "id": self.qid,
            "title": self.title,
            "difficulty": self.difficulty,
            "category": self.category,
            "secondary_skills": self.skills,
            "relevant_roles": self.roles,
            "estimated_minutes": self.minutes,
            "scenario": self.scenario,
            "problem_statement": self.problem,
            "function_signature": self.signature,
            "input_specification": self.inputs,
            "output_specification": self.output,
            "constraints": self.constraints,
            "examples": [t for t in self.example_tests] + list(self.examples),
            "acceptance_criteria": self.criteria,
            "starter_code": self.starter_code,
            "hints": self.hints,
            "reference_solution": self.solution,
            "solution_explanation": self.explanation,
            "edge_cases": self.edge_cases,
            "common_mistakes": self.mistakes,
            "test_cases": self.tests,
            "complexity_analysis": self.complexity,
            "extension_challenge": self.extension,
        }


REQUIRED_TEXT_FIELDS = (
    "title", "scenario", "problem", "signature", "entry", "output", "solution",
    "explanation", "complexity", "extension",
)
REQUIRED_LIST_FIELDS = (
    "roles", "skills", "inputs", "constraints", "criteria", "hints", "edge_cases",
    "mistakes", "tests",
)
