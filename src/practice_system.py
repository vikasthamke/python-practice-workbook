"""Interactive practice functions for the Python Practice Workbook.

Learning Mode (default)
    show_question(n)        full problem statement (no solution)
    hint(n, level=1)        progressive hints: 1 = nudge, 2 = approach, 3 = pseudocode
    check(n, fn)            run the automated tests against your function or class
    answer(n)               reference solution, explanation, edge cases, complexity
    list_questions(...)     browse/filter the 100 questions
    get_progress()          attempted / passed / remaining, by difficulty and category

Testing Mode
    exam = start_test(difficulty="Intermediate", count=5)   # hints and answers are locked
    exam.show(n); exam.submit(n, fn)   (or simply check(n, fn))
    report = end_test()                                       # score + summary
    start_test(questions=report.failed)                       # retake what you missed

Everything is local; progress is saved as JSON (see ``progress_tracker``).
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from collections.abc import Iterable

from . import sandbox_runner
from .progress_tracker import ProgressTracker
from .question_bank import (
    all_questions,
    filter_questions,
    get_question,
    parse_question_number,
    resolve_category,
    resolve_difficulty,
)
from .rendering import Rendered, render_answer, render_hint, render_question
from .schema import CATEGORIES, DIFFICULTIES, SAMPLE_DATA_DIR, T, Question
from .validators import CaseResult, CheckResult, compare, run_tests, short_repr

__all__ = [
    "DATA_DIR", "answer", "check", "current_test", "end_test", "get_progress", "hint",
    "list_questions", "reset_progress", "show_question", "start_test", "workbook_help",
]

DATA_DIR = SAMPLE_DATA_DIR
_active_test: TestSession | None = None


def _tracker() -> ProgressTracker:
    # Re-read every time so several notebooks/terminals never overwrite each other's progress.
    return ProgressTracker()


def _locked_message(action: str) -> Rendered:
    return Rendered(f"🔒 **{action} are disabled while a test is running.** Finish it with `end_test()` "
                    "to return to Learning Mode.")


# ===================================================================== learning mode
def show_question(question_number: int | str) -> Rendered:
    """Display the complete problem statement, examples and constraints (never the solution)."""
    question = get_question(question_number)
    testing = _active_test is not None
    return Rendered(render_question(question, testing=testing))


def hint(question_number: int | str, level: int = 1) -> Rendered:
    """Return hint ``level`` (1 = conceptual nudge, 2 = specific approach, 3 = pseudocode)."""
    question = get_question(question_number)
    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= len(question.hints):
        raise ValueError(f"Hint level must be an int from 1 to {len(question.hints)} "
                         f"(1 = nudge, 2 = approach, 3 = pseudocode); got {level!r}.")
    if _active_test is not None:
        return _locked_message("Hints")
    _tracker().record_hint(question.qid, level)
    return Rendered(render_hint(question, level))


def answer(question_number: int | str) -> Rendered:
    """Reveal the reference solution with explanation, edge cases, mistakes and complexity."""
    question = get_question(question_number)
    if _active_test is not None:
        return _locked_message("Solutions")
    tracker = _tracker()
    attempted = tracker.has_attempted(question.qid)
    tracker.record_solution_view(question.qid)
    return Rendered(render_answer(question, attempted))


def _looks_like_source(text: str) -> bool:
    return "\n" in text or bool(re.match(r"\s*(def|class|import|from|async)\b", text))


def check(question_number: int | str, user_code_or_result: Any, *, record: bool = True) -> CheckResult:
    """Validate a solution.

    ``user_code_or_result`` may be:
      * your function or class (recommended) – all automated tests are run;
      * a string of Python source code – executed in a separate, time-limited process;
      * a plain value – compared with the expected output of the first worked example.
    """
    question = get_question(question_number)
    mode = "testing" if _active_test is not None else "learning"
    if _active_test is not None and question.qid not in _active_test.questions:
        raise ValueError(f"{question.qid} is not part of the running test "
                         f"({', '.join(_active_test.questions)}). Use end_test() first.")
    value = user_code_or_result
    if callable(value):
        result = CheckResult(question.qid, question.title, run_tests(question, value), mode=mode)
    elif isinstance(value, str) and _looks_like_source(value):
        payload = sandbox_runner.run_source(question.qid, value)
        if "error" in payload:
            result = CheckResult(question.qid, question.title, [], mode=mode, source="subprocess",
                                 message=payload["error"])
        else:
            cases = [CaseResult(**c) for c in payload["cases"]]
            result = CheckResult(question.qid, question.title, cases, mode=mode, source="subprocess")
    else:
        result = _check_value(question, value, mode)
    if record and result.source != "value":
        _tracker().record_attempt(question.qid, result.n_passed, result.n_total)
        if _active_test is not None:
            _active_test._record(question.qid, result)
    return result


def _check_value(question: Question, value: Any, mode: str) -> CheckResult:
    examples = [t for t in question.example_tests if isinstance(t, T) and t.has_expected]
    if not examples:
        return CheckResult(question.qid, question.title, [], mode=mode, source="value",
                           message=f"This question is checked by running your code. Define "
                                   f"`{question.entry}` and call check({question.id}, {question.entry}).")
    example = examples[0]
    problem = compare(example.expected, value, example.approx)
    case = CaseResult(index=0, passed=problem is None, note="value compared with worked example 1",
                      expected=short_repr(example.expected), got=short_repr(value), message=problem or "")
    return CheckResult(question.qid, question.title, [case], mode=mode, source="value",
                       message=f"Result-only check (not recorded). For full validation run "
                               f"check({question.id}, {question.entry}).")


class QuestionList(list):
    """List of question summaries that renders as a table."""

    def _rows(self) -> list[str]:
        return [f"| {r['id']} | {r['title']} | {r['difficulty']} | {r['category']} | {r['minutes']} | {r['status']} |"
                for r in self]

    def _repr_markdown_(self) -> str:
        head = "| ID | Title | Difficulty | Category | Min | Status |\n|---|---|---|---|---|---|"
        return head + "\n" + "\n".join(self._rows()) + f"\n\n{len(self)} question(s)."

    def __str__(self) -> str:
        lines = [f"{r['id']}  {r['difficulty']:<12}  {r['status']:<11}  {r['title']}  [{r['category']}]" for r in self]
        return "\n".join(lines + [f"{len(self)} question(s)."])

    __repr__ = __str__


def list_questions(difficulty: str | None = None, category: str | None = None) -> QuestionList:
    """List questions (id, title, difficulty, category, skills, status), optionally filtered."""
    tracker = _tracker()
    return QuestionList({
        "id": q.qid, "number": q.id, "title": q.title, "difficulty": q.difficulty, "category": q.category,
        "skills": list(q.skills), "roles": list(q.roles), "minutes": q.minutes, "status": tracker.status(q.qid),
    } for q in filter_questions(difficulty, category))


# ========================================================================= progress
class ProgressReport(dict):
    """Progress summary that renders as Markdown tables."""

    def _repr_markdown_(self) -> str:
        lines = [f"**Progress:** {self['passed']} passed · {self['attempted_not_passed']} attempted · "
                 f"{self['remaining']} not started (of {self['total']}) — "
                 f"{self['passed_independently']} passed without hints or solutions."]
        for title, key in (("Difficulty", "by_difficulty"), ("Category", "by_category")):
            lines.append(f"\n| {title} | Passed | Attempted | Not started | Total |\n|---|---|---|---|---|")
            for name, row in self[key].items():
                lines.append(f"| {name} | {row['passed']} | {row['attempted']} | {row['not started']} | {row['total']} |")
        if self["next_suggestion"]:
            lines.append(f"\n**Suggested next question:** {self['next_suggestion']}")
        if self.get("warning"):
            lines.append(f"\n⚠️ {self['warning']}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self._repr_markdown_()

    __repr__ = __str__


def get_progress() -> ProgressReport:
    """Summarize attempted, passed and remaining questions by difficulty and category."""
    tracker = _tracker()
    questions = all_questions()
    statuses = {q.qid: tracker.status(q.qid) for q in questions}

    def table(groups: Iterable[str], attribute: str) -> dict:
        result = {}
        for group in groups:
            members = [q for q in questions if getattr(q, attribute) == group]
            counts = {"passed": 0, "attempted": 0, "not started": 0}
            for q in members:
                counts[statuses[q.qid]] += 1
            result[group] = {**counts, "total": len(members)}
        return result

    entries = tracker.data["questions"]
    next_q = next((q for q in questions if statuses[q.qid] != "passed"), None)
    return ProgressReport(
        total=len(questions),
        passed=sum(s == "passed" for s in statuses.values()),
        attempted_not_passed=sum(s == "attempted" for s in statuses.values()),
        remaining=sum(s == "not started" for s in statuses.values()),
        passed_independently=sum(1 for e in entries.values() if e.get("passed_independently")),
        hints_requested=sum(len(e.get("hints_used", [])) for e in entries.values()),
        solutions_viewed=sum(1 for e in entries.values() if e.get("solution_viewed")),
        by_difficulty=table(DIFFICULTIES, "difficulty"),
        by_category=table(CATEGORIES, "category"),
        questions={qid: dict(entries[qid]) for qid in sorted(entries)},
        tests_taken=len(tracker.data["tests"]),
        next_suggestion=f"{next_q.qid} – {next_q.title}" if next_q else None,
        progress_file=str(tracker.path),
        warning=tracker.warning,
    )


def reset_progress(question_numbers: Iterable[int | str] | None = None) -> Rendered:
    """Delete saved progress for all questions, or only for the given question numbers."""
    tracker = _tracker()
    if question_numbers is None:
        tracker.reset()
        return Rendered(f"All progress cleared ({tracker.path}).")
    qids = [get_question(n).qid for n in question_numbers]
    tracker.reset(qids)
    return Rendered(f"Progress cleared for {', '.join(qids)}.")


# ===================================================================== testing mode
@dataclass
class TestReport:
    """Score and per-question summary of a finished test."""

    name: str
    results: dict[str, dict]
    started_at: str
    duration_seconds: float

    @property
    def score(self) -> int:
        return sum(r["passed"] for r in self.results.values())

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def percent(self) -> float:
        return round(100 * self.score / self.total, 1) if self.total else 0.0

    @property
    def failed(self) -> list[str]:
        return [qid for qid, r in self.results.items() if not r["passed"]]

    def retake(self, only_failed: bool = True) -> TestSession:
        """Start a new test with the failed questions (or all questions)."""
        return start_test(questions=self.failed if only_failed else list(self.results), name=f"{self.name} (retake)")

    def _repr_markdown_(self) -> str:
        lines = [f"### Test finished: {self.name}",
                 f"**Score: {self.score}/{self.total} questions fully passed ({self.percent}%)** · "
                 f"time {self.duration_seconds / 60:.1f} min",
                 "\n| Question | Result | Tests passed | Attempts |\n|---|---|---|---|"]
        for qid, r in self.results.items():
            icon = "✅" if r["passed"] else ("❌" if r["attempts"] else "⏭️ not attempted")
            lines.append(f"| {qid} {r['title']} | {icon} | {r['tests_passed']}/{r['tests_total']} | {r['attempts']} |")
        if self.failed:
            lines.append(f"\nRetake the ones you missed with `report.retake()` or "
                         f"`start_test(questions={self.failed})`. Solutions are available again via `answer(n)`.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self._repr_markdown_()

    __repr__ = __str__


@dataclass
class TestSession:
    """A timed set of questions answered without hints or solutions."""

    questions: list[str]
    name: str = "Practice test"
    started: float = field(default_factory=time.time)
    results: dict[str, dict] = field(default_factory=dict)
    active: bool = True

    def __post_init__(self) -> None:
        for qid in self.questions:
            q = get_question(qid)
            self.results[qid] = {"title": q.title, "passed": False, "tests_passed": 0,
                                 "tests_total": len(q.tests), "attempts": 0}

    def _record(self, qid: str, result: CheckResult) -> None:
        entry = self.results[qid]
        entry["attempts"] += 1
        entry["tests_passed"] = max(entry["tests_passed"], result.n_passed)
        entry["passed"] = entry["passed"] or result.passed

    def show(self, question_number: int | str) -> Rendered:
        qid = get_question(question_number).qid
        if qid not in self.questions:
            raise ValueError(f"{qid} is not part of this test: {', '.join(self.questions)}")
        return Rendered(render_question(get_question(qid), testing=True))

    def submit(self, question_number: int | str, user_code: Any) -> CheckResult:
        if not self.active:
            raise RuntimeError("This test has already finished. Start a new one with start_test().")
        return check(question_number, user_code)

    def status(self) -> Rendered:
        rows = [f"| {qid} | {r['title']} | {'✅' if r['passed'] else '—'} | "
                f"{r['tests_passed']}/{r['tests_total']} | {r['attempts']} |"
                for qid, r in self.results.items()]
        return Rendered(f"**{self.name}** – {len(self.questions)} questions\n\n| Question | Title | Passed | Tests | Attempts |\n"
                        "|---|---|---|---|---|\n" + "\n".join(rows))

    def finish(self) -> TestReport:
        global _active_test
        if not self.active:
            raise RuntimeError("This test has already finished.")
        self.active = False
        if _active_test is self:
            _active_test = None
        started_at = datetime.fromtimestamp(self.started, timezone.utc).replace(microsecond=0).isoformat()
        report = TestReport(self.name, {k: dict(v) for k, v in self.results.items()}, started_at,
                            time.time() - self.started)
        _tracker().record_test({"name": self.name, "started_at": started_at, "questions": self.questions,
                                "score": report.score, "total": report.total,
                                "duration_seconds": round(report.duration_seconds, 1)})
        return report

    def _repr_markdown_(self) -> str:
        listing = "\n".join(f"- `{qid}` {r['title']}" for qid, r in self.results.items())
        return (f"### 📝 {self.name} started — Testing Mode\n\n{listing}\n\n"
                "Hints and solutions are locked. Use `exam.show(n)` (or `show_question(n)`) to read a question, "
                "`check(n, your_function)` to submit, and `end_test()` for your score.")

    def __str__(self) -> str:
        return self._repr_markdown_()

    __repr__ = __str__


def _parse_selection(questions) -> list[str]:
    if isinstance(questions, str):
        match = re.fullmatch(r"\s*[Qq]?(\d{1,3})\s*-\s*[Qq]?(\d{1,3})\s*", questions)
        if match:
            lo, hi = (parse_question_number(g) for g in match.groups())
            if lo > hi:
                raise ValueError(f"Invalid range {questions!r}: start is after end.")
            return [get_question(n).qid for n in range(lo, hi + 1)]
        questions = [part for part in re.split(r"[,\s]+", questions) if part]
    qids = [get_question(n).qid for n in questions]
    return list(dict.fromkeys(qids))


def start_test(questions=None, *, difficulty: str | None = None, category: str | None = None,
               count: int | None = None, seed: int | None = None, name: str | None = None) -> TestSession:
    """Enter Testing Mode with a selected set of questions.

    Select by explicit ``questions`` (``[3, 'Q010']``, ``range(21, 26)`` or ``'Q021-Q030'``) or by
    ``difficulty`` / ``category`` filters; ``count`` picks a random subset (``seed`` makes it repeatable).
    """
    global _active_test
    if _active_test is not None:
        raise RuntimeError("A test is already running. Call end_test() before starting another.")
    if questions is not None:
        qids = _parse_selection(questions)
    else:
        qids = [q.qid for q in filter_questions(difficulty, category)]
    if count is not None:
        if count < 1:
            raise ValueError("count must be >= 1")
        rng = random.Random(seed)
        qids = sorted(rng.sample(qids, min(count, len(qids))))
    if not qids:
        raise ValueError("No questions match this selection.")
    label = name or _default_name(difficulty, category, len(qids))
    _active_test = TestSession(qids, label)
    return _active_test


def _default_name(difficulty, category, n) -> str:
    parts = [resolve_difficulty(difficulty) if difficulty else None, resolve_category(category) if category else None]
    detail = " · ".join(p for p in parts if p)
    return f"Practice test ({n} questions{', ' + detail if detail else ''})"


def end_test() -> TestReport:
    """Finish the running test, return the score report and go back to Learning Mode."""
    if _active_test is None:
        raise RuntimeError("No test is running. Start one with start_test(...).")
    return _active_test.finish()


def current_test() -> TestSession | None:
    """The running TestSession, or None in Learning Mode."""
    return _active_test


def workbook_help() -> Rendered:
    """Quick reference of all commands."""
    return Rendered(
        "| Command | What it does |\n|---|---|\n"
        "| `list_questions(difficulty=None, category=None)` | Browse and filter questions |\n"
        "| `show_question(n)` | Full problem statement (no solution) |\n"
        "| `hint(n, level=1)` | Progressive hints: 1 nudge · 2 approach · 3 pseudocode |\n"
        "| `check(n, my_function)` | Run the automated tests (also accepts source code as a string) |\n"
        "| `answer(n)` | Reference solution, explanation, edge cases, complexity |\n"
        "| `get_progress()` / `reset_progress()` | View or clear your saved progress |\n"
        "| `start_test(...)` / `end_test()` | Enter / leave Testing Mode (hints and answers locked) |\n"
        f"| `DATA_DIR` | Path to the sample datasets: `{DATA_DIR}` |"
    )
