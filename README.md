# Python Practice Workbook — 100 Questions for Data Science, ML & AI Engineering

A complete, local, test-driven practice system: **100 realistic Python coding problems**, each with a precise
specification, worked examples, acceptance criteria, three progressive hints, automated tests and a fully
explained reference solution — plus progress tracking and a timed **Testing Mode**.

The workbook is about *Python programming*. Scenarios come from data and AI engineering (logs, API payloads,
model predictions, token usage, evaluation records, RAG pipelines…), but no ML theory or third-party data
library is needed: every question and solution uses the Python standard library.

---

## 1. Overview

- **Notebook:** `notebooks/Python_Practice_Workbook_100_Questions.ipynb` — 100 numbered sections, each with the
  full statement, an editable workspace, a check cell and a (commented) hint/solution cell.
- **Practice system:** `show_question`, `hint`, `check`, `answer`, `list_questions`, `get_progress`,
  `start_test` / `end_test` — usable from the notebook or any Python REPL.
- **Single source of truth:** the question bank in `src/questions/`. The notebook and `docs/question_catalog.md`
  are generated from it, and every reference solution, example and notebook cell was executed and verified.

## 2. Who it is for

Aspiring and early-career **Data Scientists, ML Engineers, AI Engineers, Data Engineers, Applied AI Engineers
and Python developers** who can read basic Python and want to become confident solving problems independently —
for interviews, take-home assignments and day-to-day engineering work. It does not promise a job or an
interview; it gives you structured, honest practice and feedback.

## 3. Skills covered

Core Python (control flow, truthiness, identity, unpacking, scope, mutability) · data structures (lists, tuples,
sets, dicts, nested data, grouping, de-duplication) · strings and regular expressions · functions, closures,
decorators, generators, type hints · algorithms (binary search, two pointers, sliding windows, heaps, interval
merging, topological sort, dynamic programming, union-find) · files, CSV, JSON, JSON Lines, gzip, atomic writes,
exceptions · object-oriented design (properties, special methods, composition, descriptors) · debugging and
testing (mutable defaults, copies, late binding, mutation during iteration, timezone bugs, writing tests) ·
data processing (pivots, joins, sessionization, percentiles, as-of joins, SCD2) · performance and memory
(set lookups, streaming, bounded heaps, k-way merge, reservoir sampling, profiling with `tracemalloc`) ·
integrated AI-engineering scenarios (LLM output validation, token costs, retries, batch inference, streaming
tool calls, context packing, RAG chunking).

## 4. Difficulty distribution

| Difficulty | Questions | IDs | Purpose |
|---|---:|---|---|
| Beginner | 20 | Q001–Q020 | Core Python problem-solving |
| Intermediate | 35 | Q021–Q055 | Practical programming and data handling |
| Advanced | 30 | Q056–Q085 | Complex transformations, algorithms and debugging |
| Expert | 15 | Q086–Q100 | Engineering-quality solutions, optimization, realistic challenges |
| **Total** | **100** | | |

## 5. Category distribution

| Category | Questions | Alias for filters |
|---|---:|---|
| Python fundamentals and control flow | 10 | `fundamentals` |
| Strings and text processing | 8 | `strings` |
| Lists, tuples, sets, and dictionaries | 12 | `data-structures` |
| Functions, comprehensions, and iterators | 10 | `functions` |
| Sorting, searching, and algorithmic problem-solving | 10 | `algorithms` |
| Files, exceptions, and data serialization | 8 | `files` |
| Object-oriented programming and reusable components | 7 | `oop` |
| Debugging, testing, and code quality | 8 | `debugging` |
| Data processing and analytical transformations | 10 | `data-processing` |
| Performance, memory, and scalability | 7 | `performance` |
| Integrated data and AI engineering scenarios | 10 | `ai-engineering` |

The full list with skills, roles and time estimates is in [`docs/question_catalog.md`](docs/question_catalog.md).

## 6. Learning Mode (default)

- Every question follows the same 18-part format: scenario, problem statement, signature, input specification,
  expected output, rules, example input/output, additional (boundary) examples, acceptance criteria, workspace,
  hints, reference solution, edge cases, common mistakes, tests, complexity and an optional extension.
- `hint(n, level)` gives three progressive hints: **1** a conceptual nudge, **2** a specific approach or Python
  feature, **3** pseudocode.
- `check(n, your_function)` runs the question's automated tests (615 in total) and shows, for each failing test,
  the call, the expected value, what you returned and *why* it differs (wrong value, wrong type, exception,
  input mutated, too slow…). It never shows the solution.
- `answer(n)` reveals the reference solution with explanation, edge cases, common mistakes and complexity — only
  when you explicitly ask, and with a reminder if you have not attempted the question yet.

## 7. Testing Mode

- `start_test(...)` selects questions by explicit list/range, difficulty, category, or a random `count` (with an
  optional `seed`).
- While a test runs, **hints and solutions are locked**, and failure messages only say *which* tests failed.
- `end_test()` returns a score report (fully passed questions, tests passed per question, attempts, time).
- `report.retake()` starts a new test with the questions you missed. Retakes are unlimited.

```python
exam = start_test(questions="Q021-Q030")      # or difficulty="Advanced", category="strings", count=5, seed=1
exam.show(23)                                 # statement without hint references
check(23, my_solution)                        # submit as often as you like
report = end_test()                           # score + summary, back to Learning Mode
report.retake()                               # practise the misses again
```

## 8. Project structure

```
python-practice-workbook/
├── README.md, requirements.txt, .gitignore
├── notebooks/Python_Practice_Workbook_100_Questions.ipynb   # your main entry point
├── data/
│   ├── README.md                 # dataset descriptions
│   └── sample_data/              # 13 small synthetic datasets
├── src/
│   ├── practice_system.py        # show_question, hint, check, answer, list_questions, get_progress, tests
│   ├── question_bank.py          # loading and lookup of the 100 questions
│   ├── validators.py             # test runner, comparisons, question-bank audit
│   ├── progress_tracker.py       # local JSON progress file
│   ├── sandbox_runner.py         # runs source-code submissions in a separate, time-limited process
│   ├── rendering.py              # Markdown for questions, hints and answers
│   ├── schema.py                 # Question / test-case data model, targets
│   └── questions/                # the question content (beginner … expert)
└── docs/
    ├── question_catalog.md       # all 100 questions at a glance
    └── learning_guide.md         # how to practise effectively + weekly plan
```

## 9. Installation

Requires **Python 3.10 or newer** (tested on 3.10.21 and 3.14.6).

```bash
cd python-practice-workbook
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell): `py -3 -m venv .venv` then `.venv\Scripts\Activate.ps1` (Command Prompt:
`.venv\Scripts\activate.bat`), then `python -m pip install -r requirements.txt`.

## 10. Launch the notebook

```bash
jupyter notebook notebooks/Python_Practice_Workbook_100_Questions.ipynb
```

JupyterLab works too (`pip install jupyterlab`, then `jupyter lab`). Run the **setup cell** at the top first;
it locates the project folder and imports the practice functions. You can also practise from a plain REPL
started in the project root: `from src.practice_system import *`.

## 11. Requesting hints

```python
hint(12)            # level 1: conceptual nudge
hint(12, level=2)   # level 2: specific approach / Python feature
hint(12, level=3)   # level 3: pseudocode
```

Every notebook question has a cell with these calls commented out. Invalid question numbers or levels raise a
`ValueError` that explains the valid range.

## 12. Viewing solutions

```python
answer(12)   # reference solution + why it works + edge cases + common mistakes + complexity
```

Solutions live only in the question bank – they are **not** in the notebook – so you cannot see them by accident.

## 13. Validating answers

```python
check(12, merge_configs)                 # recommended: pass your function or class
check(7, "total_revenue_usd")            # quick result-only check against worked example 1 (not recorded)
check(19, """def last_n(items, n): ...""")  # source code string: runs in a separate, time-limited process
```

How validation works:

- Your function (already defined in your own kernel) is called in-process with deep copies of each test's
  inputs, so tests cannot leak state into each other. The checker never `eval`s/`exec`s text you type.
- Results are compared structurally with a float tolerance of 1e-9; list vs tuple and bool vs int are
  distinguished; some tests check exceptions, input mutation, time limits, laziness or peak memory.
- Each test case has a 10-second timeout (on macOS/Linux, via `SIGALRM`) so an infinite loop does not hang the notebook.
- **Source strings** are executed in a fresh `python -I` subprocess with a 60-second timeout. This is *process
  isolation, not a security sandbox*: the code runs with your user permissions. Only check code you trust.
- Several valid implementations pass: tests specify behaviour, not a particular algorithm (except that
  performance tests require a suitable complexity).

## 14. Progress tracking

- Saved automatically to `.progress/progress.json` inside the project (git-ignored). Set the environment variable
  `PPW_PROGRESS_FILE` to use a different file.
- Per question it records: attempts, whether it passed, best number of tests passed, hint levels used, whether
  the solution was viewed, whether it was **passed independently** (before any hint or solution), and timestamps.
  Test sessions record their score and duration. No personal information and no account are involved.
- `get_progress()` shows passed / attempted / not-started counts, broken down by difficulty and category, and
  suggests the next question.
- Reset with `reset_progress()` (everything) or `reset_progress([3, 7])` (selected questions), or delete the file.
  A corrupted progress file is moved aside to `progress.json.corrupt` and a fresh one is started.

## 15. Suggested practice strategy

1. Work through the levels in order, but feel free to filter by category to target weak areas.
2. Read the *entire* statement; write down inputs, outputs and constraints before coding.
3. Solve, then test the normal case and the edge cases yourself before running `check`.
4. Use `hint(n, 1)` only after a genuine attempt; escalate one level at a time.
5. After passing, read `answer(n)` anyway – compare readability, edge cases and complexity.
6. Re-solve questions you needed help with a few days later, without hints.
7. Every week, run a Testing Mode session on the material you covered.

See [`docs/learning_guide.md`](docs/learning_guide.md) for a detailed workflow and an 8-week plan.

## 16. Limitations and assumptions

- **Not a security sandbox.** Function checks run in your own kernel; source-string checks run in a separate
  process with your user's permissions. Do not use this checker for untrusted code without a container/VM.
- **Timeouts** use `SIGALRM`, which exists only on macOS/Linux and only in the main thread; on Windows, an
  infinite loop in an in-process check must be interrupted manually (source-string checks still time out).
- **Performance and memory tests** use generous thresholds measured on a typical laptop; an extremely slow or
  heavily loaded machine could exceed them. They check *growth* problems (e.g. O(n²) vs O(n log n)), not
  micro-optimizations.
- The statistical uniformity test for reservoir sampling (Q094) uses wide (≈6σ) bounds; a correct implementation
  failing it is astronomically unlikely but not impossible.
- Tests verify behaviour, not style: passing does not guarantee readable or idiomatic code – compare with the
  reference and the learning guide's review checklist.
- All datasets are small and synthetic; they contain no personal data or real credentials.
- Questions reflect broadly applicable engineering expectations, not any specific company's interview process.
