"""Markdown rendering of questions, hints and solutions.

Used by the practice functions (``show_question``, ``hint``, ``answer``), the
notebook generator and the catalog generator, so every view shows identical text.
"""

from __future__ import annotations

import pprint

from .schema import CT, DataFile, File, Question, T, TmpPath

SECTION = "####"


class Rendered(str):
    """A string that Jupyter renders as Markdown and a REPL prints readably."""

    def _repr_markdown_(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return str(self)


class _SortedSet:
    """Displays a set with sorted elements so generated text is deterministic."""

    def __init__(self, items):
        self.items = items

    def __repr__(self) -> str:
        return "{" + ", ".join(sorted(repr(i) for i in self.items)) + "}" if self.items else "set()"


def _stable(value):
    if isinstance(value, (set, frozenset)):
        return _SortedSet(value)
    if isinstance(value, list):
        return [_stable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_stable(v) for v in value)
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items()}
    return value


def _fmt(value, indent: int = 0) -> str:
    text = pprint.pformat(_stable(value), width=88 - indent, sort_dicts=False)
    pad = " " * indent
    return text.replace("\n", "\n" + pad)


def _arg_text(value) -> str:
    if isinstance(value, (File, TmpPath)):
        return repr(value.name)
    if isinstance(value, DataFile):
        return f"DATA_DIR / {value.name!r}"
    if isinstance(value, list) and any(isinstance(v, (File, TmpPath, DataFile)) for v in value):
        return "[" + ", ".join(_arg_text(v) for v in value) + "]"
    return _fmt(value, 4)


def format_call(question: Question, case: T) -> str:
    parts = [_arg_text(a) for a in case.args]
    parts += [f"{k}={_arg_text(v)}" for k, v in case.kwargs.items()]
    one_line = f"{question.entry}({', '.join(parts)})"
    if len(one_line) <= 88 and "\n" not in one_line:
        return one_line
    inner = ",\n".join("    " + p for p in parts)
    return f"{question.entry}(\n{inner},\n)"


def _file_notes(case: T) -> list[str]:
    notes = []
    values = list(case.args) + list(case.kwargs.values())
    flat = []
    for v in values:
        flat.extend(v if isinstance(v, (list, tuple)) else [v])
    for v in flat:
        if isinstance(v, File):
            content = v.content if isinstance(v.content, str) else v.content.decode("utf-8", "replace")
            notes.append(f"where `{v.name}` is a file containing:\n\n```text\n{content}\n```" if content
                         else f"where `{v.name}` is an **empty** file")
        elif isinstance(v, DataFile):
            notes.append("`DATA_DIR` is the `data/sample_data/` folder (exported by the practice system)")
        elif isinstance(v, TmpPath):
            notes.append(f"`{v.name}` does not exist before the call")
    return notes


def format_example(question: Question, case: T) -> str:
    call = format_call(question, case)
    lines = [f"```python\n{call}\n```"]
    lines += _file_notes(case)
    if case.raises:
        lines.append(f"Expected: raises `{case.raises}`")
    else:
        result = _fmt(case.expected)
        lines.append(f"Expected output:\n\n```python\n{result}\n```")
    return "\n\n".join(lines)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_question(question: Question, *, notebook: bool = False, testing: bool = False) -> str:
    """Full problem statement (sections 1–12, 16 and 18). Never includes the solution."""
    q = question
    n = q.id
    out = [
        f"### Question {q.qid} — {q.title}",
        f"**Difficulty:** {q.difficulty} &nbsp;|&nbsp; **Primary category:** {q.category}  \n"
        f"**Relevant roles:** {', '.join(q.roles)}  \n"
        f"**Skills tested:** {', '.join(q.skills)}  \n"
        f"**Estimated time:** {q.minutes} minutes",
        f"{SECTION} 1. Scenario\n\n{q.scenario}",
        f"{SECTION} 2. Problem statement\n\n{q.problem}",
        f"{SECTION} 3. Function signature\n\n```python\n{q.signature}\n```",
        f"{SECTION} 4. Input specification\n\n{_bullets(q.inputs)}",
        f"{SECTION} 5. Expected output\n\n{q.output}",
        f"{SECTION} 6. Rules and constraints\n\n{_bullets(q.constraints)}",
    ]
    examples = q.example_tests
    if examples:
        first, rest = examples[0], examples[1:]
        note = f" ({first.note})" if first.note else ""
        out.append(f"{SECTION} 7. Example input{note}\n\n```python\n{format_call(q, first)}\n```"
                   + ("\n\n" + "\n\n".join(_file_notes(first)) if _file_notes(first) else ""))
        expected = f"raises `{first.raises}`" if first.raises else f"```python\n{_fmt(first.expected)}\n```"
        out.append(f"{SECTION} 8. Expected output\n\n{expected}")
        extra = [f"**{case.note or 'Example'}**\n\n{format_example(q, case)}" for case in rest]
        extra += [f"**Interactive example**\n\n```python\n{snippet.strip()}\n```" for snippet in q.examples]
        out.append(f"{SECTION} 9. Additional examples\n\n" + ("\n\n".join(extra) if extra else "_None – see the tests below._"))
    else:
        snippet, *others = q.examples
        out.append(f"{SECTION} 7. Example input\n\nAn interactive session (lines starting with `>>>` are input):\n\n"
                   f"```python\n{snippet.strip()}\n```")
        out.append(f"{SECTION} 8. Expected output\n\nThe lines without `>>>` in the session above are the expected "
                   "results (a `Traceback` line means that exception is expected).")
        extra = [f"```python\n{s.strip()}\n```" for s in others]
        out.append(f"{SECTION} 9. Additional examples\n\n" + ("\n\n".join(extra) if extra else
                                                             "The automated tests in section 16 cover further boundary cases."))
    out.append(f"{SECTION} 10. Acceptance criteria\n\n" + "\n".join(f"- [ ] {c}" for c in q.criteria))
    if notebook:
        out.append(f"{SECTION} 11. Learner workspace\n\nWrite your solution in the code cell below, then run the "
                   f"check cell after it.")
    else:
        out.append(f"{SECTION} 11. Learner workspace\n\nStart from this code:\n\n```python\n{q.starter_code.rstrip()}\n```")
    if not testing:
        out.append(f"{SECTION} 12. Hints\n\nStuck? Run `hint({n}, level=1)`. Levels 2 and 3 are more specific "
                   "(level 3 is pseudocode).")
        out.append(f"{SECTION} 13–15, 17. Reference solution, edge cases, common mistakes, complexity\n\n"
                   f"Hidden until you ask for them with `answer({n})` – make a genuine attempt first.")
    test_notes = "\n".join(f"{i}. {t.note or ('custom check' if isinstance(t, CT) else 'test')}"
                           for i, t in enumerate(q.tests, start=1))
    out.append(f"{SECTION} 16. Testing\n\nValidate your work with `check({n}, {q.entry})`. It runs "
               f"{len(q.tests)} automated tests:\n\n{test_notes}")
    out.append(f"{SECTION} 18. Extension challenge (optional)\n\n{q.extension}")
    return "\n\n".join(out)


def render_hint(question: Question, level: int) -> str:
    text = question.hints[level - 1]
    body = f"```text\n{text}\n```" if level == 3 else text
    more = f"\n\nNeed more? Try `hint({question.id}, level={level + 1})`." if level < len(question.hints) else ""
    return f"**💡 Hint {level} of {len(question.hints)} – {question.qid} {question.title}**\n\n{body}{more}"


def render_answer(question: Question, attempted: bool) -> str:
    q = question
    parts = [f"### Reference solution — {q.qid} {q.title}"]
    if not attempted:
        parts.append("> ⚠️ You have not submitted an attempt for this question yet. You learn far more by "
                     "trying first – consider closing this and running `hint(...)` instead.")
    parts += [
        f"```python\n{q.solution.strip()}\n```",
        f"{SECTION} Why this works\n\n{q.explanation}",
        f"{SECTION} Edge cases\n\n{_bullets(q.edge_cases)}",
        f"{SECTION} Common mistakes\n\n{_bullets(q.mistakes)}",
        f"{SECTION} Complexity\n\n{q.complexity}",
        f"{SECTION} Tests your solution must pass\n\n" + "\n".join(
            f"{i}. {t.note or 'test'}" for i, t in enumerate(q.tests, start=1)),
        f"{SECTION} Extension challenge\n\n{q.extension}",
        "_Compare this with your own solution: correctness first, then readability, edge cases and complexity. "
        "Passing tests does not automatically mean a solution is maintainable or optimal._",
    ]
    return "\n\n".join(parts)
