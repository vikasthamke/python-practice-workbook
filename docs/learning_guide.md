# Learning Guide

This workbook is built for **deliberate practice**: short, focused problems with immediate, specific feedback.
Watching solutions is not practice — writing code that fails, understanding why, and fixing it is. This guide
explains how to get the most out of each question and proposes a sustainable weekly plan.

## The recommended workflow for every question

1. **Read the entire problem statement** – scenario, rules, examples and acceptance criteria. Many failed checks
   come from a rule that was skimmed (an inclusive boundary, "do not modify the input", tie-breaking order).
2. **Identify inputs, outputs and constraints.** Write them down in a comment: types, empty inputs, invalid
   values, required exceptions, ordering of the result.
3. **Work out a solution independently.** Sketch the approach in plain words or pseudocode before typing code.
   Ask: which data structure fits? What is the complexity?
4. **Test the normal case yourself.** Call your function on the worked example in a scratch cell.
5. **Test edge cases yourself.** Empty input, a single element, duplicates, boundary values, `None`, wrong types.
   Only then run `check(n, your_function)`.
6. **Request Hint 1 only when you are stuck** – genuinely stuck for 10–15 minutes, not after the first error.
7. **Move to Hints 2 and 3 only when needed.** Each level narrows the search; try again after each one.
8. **Review the reference solution after a genuine attempt** with `answer(n)` – even when you passed.
9. **Compare your solution with the reference.** Look at correctness, naming, structure, edge-case handling,
   complexity and use of the standard library. Neither version is automatically "the right one"; understand the
   trade-offs.
10. **Record recurring mistakes** in a personal log (see the template below). Patterns appear quickly:
    off-by-one boundaries, forgetting `None`, mutating inputs, truthiness checks that drop `0`…
11. **Retry difficult questions later without looking at the solution** – a few days later is ideal. Use
    `reset_progress([n])` if you want the progress view to reflect the retry.
12. **Practise explaining your approach** as if in an interview: the idea, why it is correct, its complexity, and
    which edge cases you handled. Say it out loud or write three sentences.

## Passing tests is necessary, not sufficient

A green check means your code satisfies the specified behaviour on the test cases. It does **not** mean the code
is maintainable or optimal. Before moving on, review your solution with this checklist:

- **Correctness:** Could any input allowed by the specification break it? Are exceptions the documented ones?
- **Readability:** Would a teammate understand it in one pass? Are names meaningful? Is the function small?
- **Edge cases:** Empty input, duplicates, boundaries, `None`, very large inputs.
- **Side effects:** Does it modify arguments, global state or files it should not touch?
- **Complexity:** What is the time and memory cost? Is there an unnecessary nested loop or repeated work?
- **Testing:** Which test would you add if you were the reviewer?

## Using hints well

| Level | What it gives you | When to use it |
|---|---|---|
| 1 | A conceptual nudge – the *idea* | You do not know where to start |
| 2 | A concrete approach or Python feature | You have an idea but cannot turn it into code |
| 3 | Pseudocode / algorithm outline | You understand the approach but keep getting details wrong |

`get_progress()` counts questions **passed independently** (before any hint or solution). Watching that number
grow is a better motivator than raw completion.

## Testing Mode as a weekly checkpoint

Learning Mode builds skill; Testing Mode measures it. Once a week, run a short timed session on the material
you covered:

```python
exam = start_test(category="algorithms", count=4, seed=12)
# ... solve without hints ...
report = end_test()
report.retake()   # immediately or a few days later
```

Treat a failed question as a signal, not a verdict: review it in Learning Mode, then retake it.

## A sustainable 8-week plan (≈ 5–7 hours per week)

The plan is a suggestion – adapt the pace to your schedule. No fixed number of days or questions guarantees job
readiness; consistent practice with honest review is what builds skill.

| Week | Questions | Focus | Weekly checkpoint |
|---|---|---|---|
| 1 | Q001–Q020 | Fundamentals, strings, core data structures; build the habit of reading specs carefully | `start_test(difficulty="Beginner", count=5)` |
| 2 | Q021–Q031 | Control flow with state, parsing text, nested data | Retake anything missed in week 1 |
| 3 | Q032–Q046 | Generators, decorators, closures, sorting, files, first classes | `start_test(questions="Q021-Q039", count=5)` |
| 4 | Q047–Q055 | Debugging classics, data transformations, first performance fix | `start_test(category="debugging", count=3)` |
| 5 | Q056–Q070 | Advanced parsing, recursion, algorithms (heaps, DP, graphs), streaming files | `start_test(questions="Q056-Q070", count=4)` |
| 6 | Q071–Q085 | Component design, test writing, joins/sessionization, AI-engineering pipelines | `start_test(difficulty="Advanced", count=5)` |
| 7 | Q086–Q092 | Concurrency, near-duplicate clustering, checkpoints, descriptors, point-in-time data | Retake week 5–6 misses |
| 8 | Q093–Q100 | Memory-bounded processing, batch inference, streaming LLM responses, RAG building blocks | `start_test(difficulty="Expert", count=4)` |

Suggested rhythm for a practice session (60–90 minutes):

1. **10 min** – retry one previously difficult question from memory.
2. **40–60 min** – two or three new questions using the workflow above.
3. **10–15 min** – review reference solutions and update your mistake log.

## Mistake log template

| Date | Question | What went wrong | Root cause | What I'll do next time |
|---|---|---|---|---|
| 2024-05-02 | Q012 | `dropout=0` override ignored | Used `if value:` (truthiness) | Use `is not None` when only `None` means "missing" |

## Preparing for interviews with this workbook

- Practise **thinking aloud**: restate the problem, ask about edge cases, propose an approach, state complexity,
  then code.
- After passing a question, try the **extension challenge** – interviewers often ask a follow-up of that kind.
- For performance questions, be ready to explain *why* the faster version is faster (e.g. set lookups are O(1) on
  average; sorting enables two pointers) and what assumptions that relies on.
- For debugging questions, practise describing the bug's root cause in one sentence before fixing it.
