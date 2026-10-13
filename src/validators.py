"""Test execution engine and question-bank consistency checks.

Two kinds of validation live here:

1. **Solution validation** – :func:`run_tests` executes a question's test cases
   against an implementation (a learner's function/class or the reference
   solution) and returns structured :class:`CaseResult` objects.
2. **Question-bank validation** – :func:`audit_question_bank` checks counts,
   distributions and required fields; :func:`run_doctest_examples` verifies the
   worked examples shown in question statements.

Security note
-------------
Learner functions passed to ``check()`` from a notebook are *already defined in the
learner's own kernel*; this module only calls them, it never ``eval``s or
``exec``s learner text. Reference solutions are trusted workbook content and are
compiled with :func:`load_reference`. Learner *source strings* are executed only in
a separate, time-limited subprocess (see ``src/sandbox_runner.py``), which is
process isolation, not a security sandbox.
"""

from __future__ import annotations

import copy
import doctest
import math
import reprlib
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from .schema import (
    CATEGORY_TARGETS,
    CT,
    DIFFICULTIES,
    DIFFICULTY_TARGETS,
    SAMPLE_DATA_DIR,
    REQUIRED_LIST_FIELDS,
    REQUIRED_TEXT_FIELDS,
    DataFile,
    File,
    Question,
    T,
    TmpPath,
)

DATA_DIR = SAMPLE_DATA_DIR
DEFAULT_TIMEOUT = 10.0  # seconds per test case (enforced where SIGALRM is available)
FLOAT_TOLERANCE = 1e-9

_repr = reprlib.Repr()
_repr.maxstring = 120
_repr.maxother = 120
_repr.maxlist = 12
_repr.maxdict = 10
_repr.maxlevel = 4


def short_repr(value: Any, limit: int = 400) -> str:
    text = _repr.repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ------------------------------------------------------------------------- timeouts
class CaseTimeout(Exception):
    """Raised inside a test case that exceeded its time budget."""


class _time_limit:
    """Context manager that interrupts the main thread after ``seconds``.

    Uses ``signal.setitimer`` when available (macOS/Linux main thread, which is
    where Jupyter executes cells). Elsewhere it is a no-op and only the measured
    duration is reported.
    """

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.active = (
            bool(seconds)
            and hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )

    def _handler(self, signum, frame):  # pragma: no cover - timing dependent
        raise CaseTimeout(f"timed out after {self.seconds:g}s (possible infinite loop?)")

    def __enter__(self):
        if self.active:
            self._old = signal.signal(signal.SIGALRM, self._handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old)
        return False


# ------------------------------------------------------------------------ comparison
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(expected: Any, got: Any, approx: float | None = None, path: str = "result") -> str | None:
    """Return ``None`` when ``got`` matches ``expected``; otherwise a description."""
    tol = FLOAT_TOLERANCE if approx is None else approx
    if isinstance(expected, bool) or isinstance(got, bool):
        if type(expected) is bool and type(got) is bool and expected == got:
            return None
        return f"{path}: expected {expected!r}, got {short_repr(got)}"
    if _is_number(expected) and _is_number(got):
        if isinstance(expected, float) or isinstance(got, float):
            if math.isnan(expected) and math.isnan(got):
                return None
            if math.isclose(expected, got, rel_tol=FLOAT_TOLERANCE, abs_tol=tol):
                return None
        elif expected == got:
            return None
        return f"{path}: expected {expected!r}, got {got!r}"
    if expected is None:
        return None if got is None else f"{path}: expected None, got {short_repr(got)}"
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return f"{path}: expected a dict, got {type(got).__name__} {short_repr(got)}"
        missing = [k for k in expected if k not in got]
        extra = [k for k in got if k not in expected]
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing keys {short_repr(missing)}")
            if extra:
                parts.append(f"unexpected keys {short_repr(extra)}")
            return f"{path}: " + "; ".join(parts)
        for key in expected:
            problem = compare(expected[key], got[key], approx, f"{path}[{key!r}]")
            if problem:
                return problem
        return None
    if isinstance(expected, (list, tuple)):
        want = type(expected).__name__
        if not isinstance(got, (list, tuple)) or isinstance(got, tuple) != isinstance(expected, tuple):
            return f"{path}: expected a {want}, got {type(got).__name__} {short_repr(got)}"
        if len(expected) != len(got):
            return (f"{path}: expected {want} of length {len(expected)}, got length "
                    f"{len(got)}: {short_repr(got)}")
        for i, (e, g) in enumerate(zip(expected, got)):
            problem = compare(e, g, approx, f"{path}[{i}]")
            if problem:
                return problem
        return None
    if expected == got and type(expected) is type(got):
        return None
    if expected == got:
        return None
    return f"{path}: expected {short_repr(expected)}, got {short_repr(got)}"


def compare_unordered(expected: list, got: Any, approx: float | None = None) -> str | None:
    if not isinstance(got, list):
        return f"result: expected a list, got {type(got).__name__} {short_repr(got)}"
    if len(expected) != len(got):
        return f"result: expected {len(expected)} items, got {len(got)}: {short_repr(got)}"
    remaining = list(got)
    for item in expected:
        for i, candidate in enumerate(remaining):
            if compare(item, candidate, approx) is None:
                del remaining[i]
                break
        else:
            return f"result: missing expected item {short_repr(item)} (order does not matter)"
    return None


# ------------------------------------------------------------------- fixture helpers
def _resolve(value: Any, tmp: Path) -> Any:
    if isinstance(value, File):
        target = tmp / value.name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = value.content.encode() if isinstance(value.content, str) else value.content
        if value.gzip:
            import gzip

            data = gzip.compress(data, mtime=0)
        target.write_bytes(data)
        return str(target)
    if isinstance(value, DataFile):
        return str(DATA_DIR / value.name)
    if isinstance(value, TmpPath):
        return str(tmp / value.name)
    if isinstance(value, tuple):
        return tuple(_resolve(v, tmp) for v in value)
    if isinstance(value, list):
        return [_resolve(v, tmp) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, tmp) for k, v in value.items()}
    return value


def _fresh(value: Any) -> Any:
    """Deep-copy test inputs so one test can never leak state into another."""
    try:
        return copy.deepcopy(value)
    except Exception:  # generators, locks, ... are passed through unchanged
        return value


def format_call(entry: str, args: tuple, kwargs: dict) -> str:
    parts = [short_repr(a, 200) for a in args]
    parts += [f"{k}={short_repr(v, 200)}" for k, v in kwargs.items()]
    return f"{entry}({', '.join(parts)})"


def _describe_exception(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    where = ""
    for frame in reversed(frames):
        if frame.filename != __file__:
            where = f" (line {frame.lineno}: `{(frame.line or '').strip()}`)"
            break
    return f"{type(exc).__name__}: {exc}{where}"


def _exception_matches(exc: BaseException, name: str) -> bool:
    return any(cls.__name__ == name for cls in type(exc).__mro__)


# ---------------------------------------------------------------------- run results
@dataclass
class CaseResult:
    index: int
    passed: bool
    note: str
    call: str = ""
    expected: str = ""
    got: str = ""
    message: str = ""
    duration: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CheckResult:
    """Outcome of ``check()``. Renders as Markdown in Jupyter, text elsewhere."""

    qid: str
    title: str
    cases: list[CaseResult] = field(default_factory=list)
    mode: str = "learning"
    source: str = "function"
    message: str = ""

    @property
    def n_passed(self) -> int:
        return sum(c.passed for c in self.cases)

    @property
    def n_total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> bool:
        return bool(self.cases) and self.n_passed == self.n_total

    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    def _lines(self, markdown: bool) -> list[str]:
        icon = "✅" if self.passed else "❌"
        head = f"{icon} {self.qid} – {self.title}: {self.n_passed}/{self.n_total} tests passed"
        lines = [f"**{head}**" if markdown else head]
        if self.message:
            lines.append(self.message)
        failures = self.failures()
        if self.passed:
            if self.mode == "learning":
                lines.append(f"Great work. Compare with the reference via answer({int(self.qid[1:])}) and think "
                             "about readability, edge cases and complexity.")
            return lines
        shown = failures[:3]
        for c in shown:
            label = f"Test {c.index + 1}" + (f" – {c.note}" if c.note else "")
            if self.mode == "testing":
                kind = c.message.split(":")[0] if c.message else "failed"
                lines.append(f"- {label}: failed ({kind})")
                continue
            lines.append(f"- {label}")
            if c.call:
                lines.append(f"    call:     {c.call}")
            if c.expected:
                lines.append(f"    expected: {c.expected}")
            if c.got:
                lines.append(f"    got:      {c.got}")
            if c.message:
                lines.append(f"    problem:  {c.message}")
        if len(failures) > len(shown):
            lines.append(f"- ... and {len(failures) - len(shown)} more failing test(s)")
        if self.mode == "learning":
            lines.append(f"Stuck? Try hint({int(self.qid[1:])}, 1). Hints get more specific at levels 2 and 3.")
        return lines

    def __str__(self) -> str:
        return "\n".join(self._lines(markdown=False))

    __repr__ = __str__

    def _repr_markdown_(self) -> str:
        lines = self._lines(markdown=True)
        body = "\n".join(lines[1:])
        return lines[0] + ("\n\n```text\n" + body + "\n```" if body else "")

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "passed": self.passed, "n_passed": self.n_passed,
            "n_total": self.n_total, "message": self.message,
            "cases": [c.to_dict() for c in self.cases],
        }


# ------------------------------------------------------------------- reference code
def load_reference(question: Question) -> Any:
    """Compile the (trusted) reference solution and return its entry point."""
    namespace = reference_namespace(question)
    if question.entry not in namespace:
        raise LookupError(f"{question.qid}: reference solution does not define {question.entry!r}")
    return namespace[question.entry]


def reference_namespace(question: Question) -> dict:
    name = f"ppw_reference_{question.qid}"
    module = types.ModuleType(name)
    sys.modules[name] = module  # lets dataclasses/typing resolve the module
    exec(compile(question.solution, f"<reference {question.qid}>", "exec"), module.__dict__)
    return module.__dict__


# ------------------------------------------------------------------------ test runner
def run_case(question: Question, impl: Any, case: T | CT, index: int,
             timeout: float | None = DEFAULT_TIMEOUT) -> CaseResult:
    tmp = Path(tempfile.mkdtemp(prefix=f"ppw_{question.qid}_"))
    try:
        if isinstance(case, CT):
            return _run_custom(case, impl, tmp, index, timeout)
        return _run_call(question, case, impl, tmp, index, timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_custom(case: CT, impl: Any, tmp: Path, index: int, timeout: float | None) -> CaseResult:
    result = CaseResult(index=index, passed=False, note=case.note)
    start = time.perf_counter()
    try:
        with _time_limit(timeout):
            case.func(impl, tmp)
        result.passed = True
    except AssertionError as exc:
        result.message = str(exc) or "an assertion in this test failed"
    except CaseTimeout as exc:
        result.message = f"Timeout: {exc}"
    except Exception as exc:
        result.message = _describe_exception(exc)
    result.duration = time.perf_counter() - start
    return result


def _run_call(question: Question, case: T, impl: Any, tmp: Path, index: int,
              timeout: float | None) -> CaseResult:
    args = _resolve(_fresh(case.args), tmp)
    kwargs = _resolve(_fresh(case.kwargs), tmp)
    result = CaseResult(index=index, passed=False, note=case.note,
                        call=format_call(question.entry, case.args, case.kwargs))
    if case.raises:
        result.expected = f"raises {case.raises}"
    else:
        result.expected = short_repr(case.expected)

    snapshot = None
    if not question.mutates_input:
        try:
            snapshot = copy.deepcopy((args, kwargs))
        except Exception:
            snapshot = None

    start = time.perf_counter()
    try:
        with _time_limit(max(timeout or 0, (case.time_limit or 0) * 5) or None):
            got = impl(*args, **kwargs)
            if case.materialize and got is not None:
                got = list(got)
    except CaseTimeout as exc:
        result.message = f"Timeout: {exc}"
        return result
    except Exception as exc:
        result.duration = time.perf_counter() - start
        if case.raises and _exception_matches(exc, case.raises):
            result.passed = True
            result.got = f"raised {type(exc).__name__}"
        else:
            result.got = f"raised {type(exc).__name__}"
            result.message = _describe_exception(exc)
        return result
    result.duration = time.perf_counter() - start
    result.got = short_repr(got)

    if case.raises:
        result.message = f"Expected {case.raises} to be raised, but the call returned normally"
        return result
    if case.unordered:
        problem = compare_unordered(case.expected, got, case.approx)
    else:
        problem = compare(case.expected, got, case.approx)
    if problem:
        result.message = problem
        return result
    if snapshot is not None and snapshot != (args, kwargs):
        result.message = ("Mutation: your function modified its input arguments; this "
                          "question requires inputs to be left unchanged")
        return result
    if case.time_limit is not None and result.duration > case.time_limit:
        result.message = (f"Too slow: took {result.duration:.2f}s, limit is "
                          f"{case.time_limit:.2f}s – look for a more efficient approach")
        return result
    result.passed = True
    return result


def run_tests(question: Question, impl: Any, *, reverse: bool = False,
              timeout: float | None = DEFAULT_TIMEOUT) -> list[CaseResult]:
    indexed = list(enumerate(question.tests))
    if reverse:
        indexed.reverse()
    results = [run_case(question, impl, case, i, timeout) for i, case in indexed]
    return sorted(results, key=lambda r: r.index)


# ------------------------------------------------------------- doctest-style examples
def run_doctest_examples(question: Question, namespace: dict) -> tuple[int, str]:
    """Run the question's doctest snippets. Returns (failures, report)."""
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    out: list[str] = []
    failures = 0
    for i, snippet in enumerate(question.examples):
        test = parser.get_doctest(snippet, dict(namespace), f"{question.qid}-example-{i + 1}",
                                  None, 0)
        runner.run(test, out=out.append)
    failures = runner.summarize(verbose=False).failed
    return failures, "".join(out)


# -------------------------------------------------------------- question-bank audit
def audit_question_bank(questions: Iterable[Question]) -> list[str]:
    """Return a list of human-readable problems (empty when the bank is consistent)."""
    questions = list(questions)
    problems: list[str] = []
    if len(questions) != 100:
        problems.append(f"expected 100 questions, found {len(questions)}")
    ids = [q.id for q in questions]
    if sorted(ids) != list(range(1, 101)):
        problems.append("question ids must be exactly 1..100 with no gaps or duplicates")
    titles = [q.title.lower() for q in questions]
    if len(set(titles)) != len(titles):
        problems.append("duplicate question titles found")
    entries = [q.entry for q in questions]
    if len(set(entries)) != len(entries):
        dupes = sorted({e for e in entries if entries.count(e) > 1})
        problems.append(f"duplicate entry-point names: {dupes}")

    for difficulty, target in DIFFICULTY_TARGETS.items():
        n = sum(q.difficulty == difficulty for q in questions)
        if n != target:
            problems.append(f"{difficulty}: expected {target} questions, found {n}")
    for category, target in CATEGORY_TARGETS.items():
        n = sum(q.category == category for q in questions)
        if n != target:
            problems.append(f"{category}: expected {target} questions, found {n}")

    order = {d: i for i, d in enumerate(DIFFICULTIES)}
    for q in questions:
        tag = q.qid
        if q.difficulty not in DIFFICULTY_TARGETS:
            problems.append(f"{tag}: unknown difficulty {q.difficulty!r}")
        if q.category not in CATEGORY_TARGETS:
            problems.append(f"{tag}: unknown category {q.category!r}")
        for name in REQUIRED_TEXT_FIELDS:
            if not str(getattr(q, name)).strip():
                problems.append(f"{tag}: field {name!r} is empty")
        for name in REQUIRED_LIST_FIELDS:
            values = getattr(q, name)
            if not values or any(not str(v).strip() for v in values if not isinstance(v, (T, CT))):
                problems.append(f"{tag}: list field {name!r} is empty or has blank items")
        if len(q.hints) != 3:
            problems.append(f"{tag}: expected 3 progressive hints, found {len(q.hints)}")
        if len(q.tests) < 4:
            problems.append(f"{tag}: expected at least 4 test cases, found {len(q.tests)}")
        if not q.example_tests and not q.examples:
            problems.append(f"{tag}: no worked example (mark a T(example=True) or add a doctest)")
        if q.entry not in q.signature:
            problems.append(f"{tag}: entry {q.entry!r} not found in signature")
        if q.signature.strip() not in q.starter_code:
            problems.append(f"{tag}: starter code does not contain the signature")
        if q.solution.strip() == q.starter_code.strip():
            problems.append(f"{tag}: starter code equals the reference solution")
        if q.minutes <= 0:
            problems.append(f"{tag}: estimated minutes must be positive")
    for prev, cur in zip(questions, questions[1:]):
        if order.get(cur.difficulty, 0) < order.get(prev.difficulty, 0):
            problems.append(f"{cur.qid}: difficulty decreases relative to {prev.qid}")
    return problems
