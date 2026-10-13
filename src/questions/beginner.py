"""Beginner questions Q001–Q020: core Python problem-solving with good habits."""

from ..schema import (
    AAI, AIE, ALG, DATA, DBG, DE, DS, DS_ROLE, FILE, FUNC, FUND, MLE, PYDEV, STR,
    DataFile, File, Question, T, TmpPath,
)

QUESTIONS = [
    # ------------------------------------------------------------------------ Q001
    Question(
        id=1,
        title="Classify Model Latency into SLA Tiers",
        difficulty="Beginner",
        category=FUND,
        roles=[MLE, AIE, PYDEV],
        skills=["if/elif/else chains", "boundary conditions", "input validation", "isinstance"],
        minutes=10,
        scenario=(
            "Your team's model-serving dashboard colours every request by how quickly the model "
            "answered. The service-level agreement (SLA) defines three tiers, and the dashboard "
            "needs a small, well-tested helper that turns a latency in milliseconds into a tier name."
        ),
        problem=(
            "Write `latency_tier(latency_ms)` that returns `\"fast\"` when the latency is below "
            "100 ms, `\"acceptable\"` when it is at least 100 ms but below 500 ms, and `\"slow\"` "
            "when it is 500 ms or more. Reject impossible or wrongly typed inputs."
        ),
        signature="def latency_tier(latency_ms: float) -> str:",
        entry="latency_tier",
        inputs=[
            "`latency_ms` is an `int` or `float` measured in milliseconds.",
            "`bool` values are **not** valid numbers here (even though `True` is an `int` in Python).",
            "Negative values are impossible and must be rejected.",
        ],
        output="A `str`: exactly one of `\"fast\"`, `\"acceptable\"`, `\"slow\"`.",
        constraints=[
            "`0 <= latency_ms < 100` → `\"fast\"`.",
            "`100 <= latency_ms < 500` → `\"acceptable\"` (100 itself is acceptable).",
            "`latency_ms >= 500` → `\"slow\"` (500 itself is slow).",
            "Raise `ValueError` for negative latencies.",
            "Raise `TypeError` when the value is not an `int`/`float`, or is a `bool`.",
        ],
        criteria=[
            "Returns the correct tier for values strictly inside each range.",
            "Boundary values 100 and 500 land in the higher tier.",
            "Negative input raises `ValueError`; strings and booleans raise `TypeError`.",
        ],
        hints=[
            "Check the invalid cases first, then test the ranges from smallest to largest.",
            "`isinstance(x, bool)` must be checked before `isinstance(x, (int, float))`, because "
            "`bool` is a subclass of `int`. Once the low ranges are handled, each `elif` only needs "
            "one comparison.",
            "if latency is bool or not int/float: raise TypeError\n"
            "if latency < 0: raise ValueError\n"
            "if latency < 100: return 'fast'\n"
            "if latency < 500: return 'acceptable'\n"
            "return 'slow'",
        ],
        solution='''
def latency_tier(latency_ms: float) -> str:
    """Map a latency in milliseconds to an SLA tier name."""
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)):
        raise TypeError(f"latency_ms must be a number, got {type(latency_ms).__name__}")
    if latency_ms < 0:
        raise ValueError(f"latency_ms cannot be negative: {latency_ms}")
    if latency_ms < 100:
        return "fast"
    if latency_ms < 500:
        return "acceptable"
    return "slow"
''',
        explanation=(
            "Validation happens first so the tier logic can assume a clean, non-negative number. "
            "Because the ranges are checked in ascending order, each branch only needs an upper "
            "bound: reaching `latency_ms < 500` already implies `latency_ms >= 100`."
        ),
        edge_cases=[
            "`100` and `500` are boundaries: the strict `<` comparisons push them to the higher tier.",
            "`0` is valid and fast.",
            "`True` would otherwise be treated as `1`; the explicit `bool` check rejects it.",
        ],
        mistakes=[
            "Using `<=` for the boundaries, which classifies exactly 100 ms as fast.",
            "Writing `if 100 < latency_ms < 500`, which leaves exactly 100 unclassified.",
            "Checking `isinstance(x, int)` before excluding `bool`.",
        ],
        tests=[
            T((42,), "fast", example=True, note="typical fast request"),
            T((250.0,), "acceptable", example=True, note="inside the middle tier"),
            T((100,), "acceptable", example=True, note="boundary: 100 belongs to acceptable"),
            T((99.9,), "fast", note="just below the first boundary"),
            T((499.99,), "acceptable", note="just below the second boundary"),
            T((500,), "slow", note="boundary: 500 belongs to slow"),
            T((0,), "fast", note="zero latency"),
            T((-1,), raises="ValueError", note="negative latency"),
            T(("100",), raises="TypeError", note="string input"),
            T((True,), raises="TypeError", note="bool input"),
        ],
        complexity="O(1) time and O(1) space: a fixed number of comparisons.",
        extension=(
            "Accept the tier boundaries as a parameter, e.g. `[(100, 'fast'), (500, 'acceptable')]`, "
            "so the SLA can change without editing the function."
        ),
    ),
    # ------------------------------------------------------------------------ Q002
    Question(
        id=2,
        title="Split a Dataset into Batch Index Ranges",
        difficulty="Beginner",
        category=FUND,
        roles=[MLE, DE, DS_ROLE],
        skills=["range with step", "loops", "min()", "tuples", "argument validation"],
        minutes=10,
        scenario=(
            "A training job processes records in mini-batches. Before loading data, the job "
            "computes the half-open index ranges `[start, end)` that each batch will cover so "
            "batches can be scheduled and logged."
        ),
        problem=(
            "Write `batch_ranges(total, batch_size)` that returns the list of `(start, end)` index "
            "pairs that cover `range(total)` in consecutive batches of `batch_size`. The last batch "
            "may be smaller."
        ),
        signature="def batch_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:",
        entry="batch_ranges",
        inputs=[
            "`total`: number of records, an `int` ≥ 0.",
            "`batch_size`: records per batch, an `int` ≥ 1.",
        ],
        output=(
            "A `list` of `(start, end)` tuples, in increasing order, where `start` is inclusive and "
            "`end` is exclusive. Every index from 0 to `total - 1` appears in exactly one batch."
        ),
        constraints=[
            "Return `[]` when `total` is 0.",
            "The final batch ends exactly at `total` (never beyond it).",
            "Raise `ValueError` if `total < 0` or `batch_size < 1`.",
        ],
        criteria=[
            "Batches are contiguous, non-overlapping and cover all records.",
            "The final partial batch is included and correctly truncated.",
            "Invalid arguments raise `ValueError`.",
        ],
        hints=[
            "The start of every batch is a multiple of `batch_size`.",
            "`range(0, total, batch_size)` yields every batch start; the end is the smaller of "
            "`start + batch_size` and `total`.",
            "validate inputs\nresult = []\nfor start in range(0, total, batch_size):\n"
            "    end = min(start + batch_size, total)\n    result.append((start, end))\nreturn result",
        ],
        solution='''
def batch_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    """Return half-open (start, end) ranges covering range(total)."""
    if total < 0:
        raise ValueError("total must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]
''',
        explanation=(
            "`range(0, total, batch_size)` produces each batch start exactly once. Clamping the "
            "end with `min()` handles the final partial batch without any special-case branch."
        ),
        edge_cases=[
            "`total == 0` → the range is empty, so the result is `[]`.",
            "`total` is an exact multiple of `batch_size` → no partial batch is produced.",
            "`batch_size > total` → a single batch `(0, total)`.",
        ],
        mistakes=[
            "Computing the number of batches as `total // batch_size`, which drops the last partial batch.",
            "Using inclusive end indices, which makes adjacent batches overlap when sliced.",
            "Forgetting to reject `batch_size = 0`, which raises a confusing error from `range()`.",
        ],
        tests=[
            T((10, 4), [(0, 4), (4, 8), (8, 10)], example=True, note="last batch is partial"),
            T((8, 4), [(0, 4), (4, 8)], example=True, note="exact multiple"),
            T((0, 5), [], example=True, note="boundary: no records"),
            T((3, 10), [(0, 3)], note="batch larger than total"),
            T((1, 1), [(0, 1)], note="single record"),
            T((5, 1), [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], note="batch size one"),
            T((10, 0), raises="ValueError", note="zero batch size"),
            T((-1, 3), raises="ValueError", note="negative total"),
        ],
        complexity="O(total / batch_size) time and space – one tuple per batch.",
        extension=(
            "Write a generator version `iter_batches(records, batch_size)` that yields the actual "
            "slices of a list instead of index pairs."
        ),
    ),
    # ------------------------------------------------------------------------ Q003
    Question(
        id=3,
        title="Coalesce Configuration Values",
        difficulty="Beginner",
        category=FUND,
        roles=[PYDEV, DE, MLE],
        skills=["truthiness", "`is None` identity checks", "loops with early return", "default arguments"],
        minutes=10,
        scenario=(
            "A pipeline reads a setting such as `retry_count` from several places in priority "
            "order: command-line flag, environment, config file. A source that does not define the "
            "setting reports `None`. Legitimate values such as `0`, `False` or `\"\"` must still "
            "win over lower-priority sources."
        ),
        problem=(
            "Write `coalesce(values, default=None)` that returns the first element of `values` "
            "that is not `None`. If every element is `None` (or the list is empty), return `default`."
        ),
        signature="def coalesce(values: list, default=None):",
        entry="coalesce",
        inputs=[
            "`values`: a list of arbitrary Python objects, ordered from highest to lowest priority. May be empty.",
            "`default`: the value to return when no element qualifies (defaults to `None`).",
        ],
        output="The first element that is not `None`, returned unchanged (same object), or `default`.",
        constraints=[
            "Falsy values (`0`, `False`, `\"\"`, `[]`) are real values and must be returned.",
            "Only `None` means \"not set\".",
            "Do not modify `values`.",
        ],
        criteria=[
            "Returns `0` from `[None, 0, 5]`.",
            "Returns `default` for an empty list or a list of only `None`.",
            "Uses an identity check (`is not None`), not truthiness.",
        ],
        hints=[
            "What is the difference between `if value:` and `if value is not None:`?",
            "Loop over the values and return as soon as you see one that is not `None`.",
            "for value in values:\n    if value is not None:\n        return value\nreturn default",
        ],
        solution='''
def coalesce(values: list, default=None):
    """Return the first value that is not None, otherwise ``default``."""
    for value in values:
        if value is not None:
            return value
    return default
''',
        explanation=(
            "`is not None` is an identity check that only excludes the `None` singleton. A "
            "truthiness check (`if value`) would wrongly skip `0`, `False` and empty strings, "
            "which are deliberate configuration values."
        ),
        edge_cases=[
            "Empty list → `default`.",
            "`[None, False]` → `False` (falsy but set).",
            "`default` itself may be any object, including `None`.",
        ],
        mistakes=[
            "Using `next((v for v in values if v), default)`, which skips falsy values.",
            "Writing `value != None`; it usually works, but `is not None` is the idiomatic, "
            "unambiguous identity check.",
            "Using `values[0] or values[1] or ...`, which has the same truthiness bug.",
        ],
        tests=[
            T(([None, 0, 5],), 0, example=True, note="zero is a real value"),
            T(([None, None], "n/a"), "n/a", example=True, note="boundary: nothing set → default"),
            T(([],), None, note="empty list with implicit default"),
            T(([None, "", "x"],), "", note="empty string is a real value"),
            T(([False, True],), False, note="False is a real value"),
            T(([None, [], [1]],), [], note="empty list element is a real value"),
            T((["cli", "env"], "default"), "cli", note="first value wins"),
        ],
        complexity="O(n) time in the worst case, O(1) extra space; it stops at the first match.",
        extension="Accept any iterable (including a generator) and make sure you stop consuming it after the first match.",
    ),
    # ------------------------------------------------------------------------ Q004
    Question(
        id=4,
        title="Exponential Backoff Retry Schedule",
        difficulty="Beginner",
        category=FUND,
        roles=[AIE, DE, PYDEV],
        skills=["for loops", "exponentiation", "min()", "round()", "input validation"],
        minutes=15,
        scenario=(
            "Calls to an external LLM API occasionally fail with rate-limit errors. The client "
            "retries using exponential backoff: each wait is longer than the last, but never "
            "longer than a cap. You need to compute the planned wait times up front so they can "
            "be logged and unit-tested."
        ),
        problem=(
            "Write `backoff_schedule(base, factor, max_retries, cap)` that returns the list of "
            "delays (in seconds) before each retry. Retry *i* (starting at 0) waits "
            "`base * factor ** i` seconds, limited to at most `cap`. Round every delay to 3 decimals."
        ),
        signature="def backoff_schedule(base: float, factor: float, max_retries: int, cap: float) -> list[float]:",
        entry="backoff_schedule",
        inputs=[
            "`base`: first delay in seconds, > 0.",
            "`factor`: growth multiplier, ≥ 1.",
            "`max_retries`: number of retries, an `int` ≥ 0.",
            "`cap`: maximum delay, ≥ `base`.",
        ],
        output="A `list[float]` with exactly `max_retries` delays, each rounded with `round(x, 3)`.",
        constraints=[
            "Delay *i* is `min(base * factor ** i, cap)`, rounded to 3 decimal places.",
            "Return `[]` when `max_retries` is 0.",
            "Raise `ValueError` if `base <= 0`, `factor < 1`, `max_retries < 0` or `cap < base`.",
        ],
        criteria=[
            "Produces the correct number of delays.",
            "Delays grow geometrically and are capped.",
            "Floating-point noise such as `0.30000000000000004` is removed by rounding.",
            "Invalid parameters raise `ValueError`.",
        ],
        hints=[
            "Each delay depends only on its retry index *i*.",
            "Loop `for i in range(max_retries)` and apply `min()` and `round()` to each value.",
            "validate the four parameters\nreturn [round(min(base * factor ** i, cap), 3) for i in range(max_retries)]",
        ],
        solution='''
def backoff_schedule(base: float, factor: float, max_retries: int, cap: float) -> list[float]:
    """Return capped exponential backoff delays, rounded to milliseconds."""
    if base <= 0:
        raise ValueError("base must be > 0")
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if cap < base:
        raise ValueError("cap must be >= base")
    return [round(min(base * factor ** i, cap), 3) for i in range(max_retries)]
''',
        explanation=(
            "The delay for retry *i* is a closed-form expression, so a comprehension over "
            "`range(max_retries)` is enough. `min()` applies the cap and `round(..., 3)` removes "
            "binary floating-point artefacts so the schedule is easy to log and compare."
        ),
        edge_cases=[
            "`max_retries = 0` → `[]`.",
            "`factor = 1` → constant delays equal to `base`.",
            "Large `i` → every delay equals `cap`.",
        ],
        mistakes=[
            "Starting the exponent at 1, which skips the base delay.",
            "Applying the cap before multiplying, or forgetting it entirely.",
            "Comparing unrounded floats such as `0.30000000000000004` with `0.3` in tests.",
        ],
        tests=[
            T((0.5, 2, 5, 4), [0.5, 1.0, 2.0, 4.0, 4.0], example=True, note="capped growth"),
            T((0.1, 3, 4, 10), [0.1, 0.3, 0.9, 2.7], example=True, note="rounding removes float noise"),
            T((1, 2, 0, 30), [], example=True, note="boundary: no retries"),
            T((2, 1, 3, 2), [2, 2, 2], note="factor 1 keeps delays constant"),
            T((0, 2, 3, 10), raises="ValueError", note="base must be positive"),
            T((1, 0.5, 3, 10), raises="ValueError", note="factor below 1"),
            T((1, 2, -1, 10), raises="ValueError", note="negative retries"),
            T((5, 2, 3, 1), raises="ValueError", note="cap smaller than base"),
        ],
        complexity="O(max_retries) time and space.",
        extension="Add optional 'full jitter': each delay becomes `random.uniform(0, delay)` using a seeded `random.Random` passed as a parameter.",
    ),
    # ------------------------------------------------------------------------ Q005
    Question(
        id=5,
        title="Normalize Bounding-Box Coordinates",
        difficulty="Beginner",
        category=FUND,
        roles=[MLE, AIE, DS_ROLE],
        skills=["tuple unpacking", "multiple assignment", "min/max", "immutable return values"],
        minutes=10,
        scenario=(
            "An object-detection labelling tool exports boxes as two opposite corners "
            "`(x1, y1, x2, y2)`, but annotators may drag from any corner, so `x1` can be larger "
            "than `x2`. The training pipeline expects `(x, y, width, height)` with `(x, y)` as the "
            "top-left corner."
        ),
        problem=(
            "Write `normalize_bbox(box)` that converts two opposite corners in any order into "
            "`(x, y, width, height)`, where `x`/`y` are the smallest coordinates and width/height "
            "are non-negative."
        ),
        signature="def normalize_bbox(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:",
        entry="normalize_bbox",
        inputs=[
            "`box`: a tuple or list of exactly four numbers `(x1, y1, x2, y2)`.",
            "Coordinates may be ints or floats; corners may be given in any order.",
        ],
        output="A 4-tuple `(x, y, width, height)`.",
        constraints=[
            "`x = min(x1, x2)`, `y = min(y1, y2)`, `width = abs(x2 - x1)`, `height = abs(y2 - y1)`.",
            "Zero-width or zero-height boxes are allowed.",
            "Raise `ValueError` if `box` does not contain exactly four values.",
            "Always return a `tuple`, even if the input is a list.",
        ],
        criteria=[
            "Correct output for corners given in any order.",
            "Returns a tuple.",
            "Wrong-length input raises `ValueError`.",
        ],
        hints=[
            "Unpack the four values into named variables first – it makes the rest readable.",
            "`x1, y1, x2, y2 = box` already raises `ValueError` when the length is wrong. Then use "
            "`min()` and `abs()`.",
            "x1, y1, x2, y2 = box\nreturn (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))",
        ],
        solution='''
def normalize_bbox(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert two opposite corners into (x, y, width, height)."""
    if len(box) != 4:
        raise ValueError(f"box must have 4 values, got {len(box)}")
    x1, y1, x2, y2 = box
    return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
''',
        explanation=(
            "Tuple unpacking names the four coordinates, so the formula reads like the "
            "specification. `min` picks the top-left corner and `abs` guarantees non-negative "
            "sizes regardless of drag direction."
        ),
        edge_cases=[
            "Corners already ordered → output equals `(x1, y1, x2 - x1, y2 - y1)`.",
            "Degenerate boxes (zero width or height) are returned with a 0 dimension.",
            "A list input still yields a tuple.",
        ],
        mistakes=[
            "Assuming `x1 <= x2`, which produces negative widths for reversed boxes.",
            "Returning a list when the specification says tuple.",
            "Indexing `box[0]`… repeatedly instead of unpacking, which hides mistakes.",
        ],
        tests=[
            T(((10, 20, 50, 80),), (10, 20, 40, 60), example=True, note="ordered corners"),
            T(((50, 80, 10, 20),), (10, 20, 40, 60), example=True, note="reversed corners"),
            T(((5, 5, 5, 9),), (5, 5, 0, 4), example=True, note="boundary: zero width"),
            T(([30, 10, 10, 40],), (10, 10, 20, 30), note="list input, mixed order"),
            T(((0.5, 0.25, 0.1, 0.75),), (0.1, 0.25, 0.4, 0.5), note="float coordinates"),
            T(((1, 2, 3),), raises="ValueError", note="too few values"),
        ],
        complexity="O(1) time and space.",
        extension="Add `iou(box_a, box_b)` that computes intersection-over-union of two normalized boxes.",
    ),
    # ------------------------------------------------------------------------ Q006
    Question(
        id=6,
        title="Longest Streak of Successful Health Checks",
        difficulty="Beginner",
        category=FUND,
        roles=[PYDEV, DE, MLE],
        skills=["loops with running state", "string normalization", "max tracking"],
        minutes=10,
        scenario=(
            "A monitoring job pings a model endpoint every minute and stores the status text it "
            "received. Reliability reports include the longest uninterrupted run of successful checks."
        ),
        problem=(
            "Write `longest_success_streak(statuses)` that returns the length of the longest run "
            "of consecutive successful checks. A check is successful when its status, ignoring "
            "surrounding whitespace and letter case, equals `\"ok\"`."
        ),
        signature="def longest_success_streak(statuses: list[str]) -> int:",
        entry="longest_success_streak",
        inputs=["`statuses`: a list of strings such as `\"ok\"`, `\"OK \"`, `\"error\"`, `\"timeout\"`. May be empty."],
        output="An `int` ≥ 0.",
        constraints=[
            "Any status other than `ok` (after `strip().lower()`) breaks the streak.",
            "Return 0 for an empty list or when no check succeeded.",
            "Solve it in a single pass.",
        ],
        criteria=[
            "Finds the longest run anywhere in the list, including at the end.",
            "Normalizes whitespace and case.",
            "Returns 0 when there are no successes.",
        ],
        hints=[
            "Keep two numbers while you loop: the current streak and the best streak so far.",
            "On success, increase the current streak and update the best. On failure, reset current to 0.",
            "best = current = 0\nfor s in statuses:\n    if s.strip().lower() == 'ok':\n        current += 1\n"
            "        best = max(best, current)\n    else:\n        current = 0\nreturn best",
        ],
        solution='''
def longest_success_streak(statuses: list[str]) -> int:
    """Length of the longest run of consecutive 'ok' statuses."""
    best = current = 0
    for status in statuses:
        if status.strip().lower() == "ok":
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
''',
        explanation=(
            "Updating `best` every time the streak grows (rather than only when it breaks) means "
            "a streak that runs to the very end of the list is still counted."
        ),
        edge_cases=[
            "Streak at the end of the list is counted because `best` is updated on every success.",
            "Empty list → 0.",
            "`' OK '` counts as success after normalization.",
        ],
        mistakes=[
            "Updating `best` only when a failure occurs, which misses a trailing streak.",
            "Comparing with `== 'ok'` without normalizing.",
        ],
        tests=[
            T((["ok", "ok", "error", "ok", "ok", "ok", "timeout"],), 3, example=True, note="middle streak"),
            T(([],), 0, example=True, note="boundary: empty list"),
            T((["error", "ok", "ok"],), 2, note="streak at the end"),
            T(([" OK ", "ok", "Ok"],), 3, note="case and whitespace"),
            T((["error", "timeout"],), 0, note="no successes"),
            T((["ok"],), 1, note="single success"),
        ],
        complexity="O(n) time, O(1) extra space.",
        extension="Return the start and end index of the longest streak as well as its length.",
    ),
    # ------------------------------------------------------------------------ Q007
    Question(
        id=7,
        title="Normalize Dataset Column Names",
        difficulty="Beginner",
        category=STR,
        roles=[DS_ROLE, DE, MLE],
        skills=["str.lower/strip", "character classification", "join", "regular expressions (optional)"],
        minutes=15,
        scenario=(
            "Spreadsheets exported by business teams have column headers such as "
            "`\"  Total Revenue (USD) \"` or `\"user-ID\"`. Before loading them into a feature "
            "table you convert every header into a predictable `snake_case` identifier."
        ),
        problem=(
            "Write `normalize_column(name)` that lower-cases the name, turns every run of characters "
            "that are not ASCII letters or digits into a single underscore, removes leading and "
            "trailing underscores, and prefixes `col_` when the result starts with a digit."
        ),
        signature="def normalize_column(name: str) -> str:",
        entry="normalize_column",
        inputs=["`name`: any string, possibly with spaces, punctuation, mixed case or surrounding whitespace."],
        output="A non-empty `str` containing only `a–z`, `0–9` and single underscores.",
        constraints=[
            "Only ASCII letters `a–z` and digits `0–9` are kept; every other run of characters becomes one `_`.",
            "No leading, trailing or repeated underscores in the result.",
            "If the cleaned name starts with a digit, prefix it with `col_`.",
            "Raise `ValueError` if nothing remains after cleaning (e.g. `\"%%%\"`).",
        ],
        criteria=[
            "`\"  Total Revenue (USD) \"` → `\"total_revenue_usd\"`.",
            "Repeated separators collapse to one underscore.",
            "Names starting with digits receive the `col_` prefix.",
            "Empty results raise `ValueError`.",
        ],
        hints=[
            "Think of the name as tokens of letters/digits separated by \"junk\".",
            "`re.findall(r\"[a-z0-9]+\", name.lower())` returns the tokens; joining them with `_` "
            "automatically avoids repeated or edge underscores.",
            "tokens = re.findall('[a-z0-9]+', name.lower())\nif not tokens: raise ValueError\n"
            "result = '_'.join(tokens)\nif result[0].isdigit(): result = 'col_' + result\nreturn result",
        ],
        solution='''
import re


def normalize_column(name: str) -> str:
    """Convert a raw column header into a snake_case identifier."""
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    if not tokens:
        raise ValueError(f"column name {name!r} has no letters or digits")
    result = "_".join(tokens)
    if result[0].isdigit():
        result = "col_" + result
    return result
''',
        explanation=(
            "Extracting the alphanumeric tokens and joining them with `_` handles collapsing, "
            "trimming and replacement in one step, because separators only ever appear *between* "
            "tokens."
        ),
        edge_cases=[
            "Already-clean names are unchanged.",
            "`\"2023 Sales\"` → `\"col_2023_sales\"`.",
            "Only punctuation → `ValueError`.",
        ],
        mistakes=[
            "Replacing characters one at a time, which leaves `__` sequences.",
            "Calling `str.isalnum()`, which also accepts non-ASCII letters such as `é`.",
            "Forgetting to strip trailing underscores produced by `)` at the end.",
        ],
        tests=[
            T(("  Total Revenue (USD) ",), "total_revenue_usd", example=True, note="spaces and punctuation"),
            T(("2023 Sales",), "col_2023_sales", example=True, note="leading digit"),
            T(("%%%",), raises="ValueError", example=True, note="boundary: nothing left"),
            T(("user-ID",), "user_id", note="hyphen and case"),
            T(("Click__Through--Rate",), "click_through_rate", note="repeated separators"),
            T(("already_clean",), "already_clean", note="unchanged"),
            T(("p95 latency (ms)",), "p95_latency_ms", note="digits inside"),
        ],
        complexity="O(n) time and space in the length of the name.",
        extension="Write `normalize_columns(names)` that also de-duplicates collisions by appending `_2`, `_3`, …",
    ),
    # ------------------------------------------------------------------------ Q008
    Question(
        id=8,
        title="Mask Secrets Before Logging",
        difficulty="Beginner",
        category=STR,
        roles=[AIE, PYDEV, DE],
        skills=["string slicing", "string multiplication", "default arguments", "validation"],
        minutes=10,
        scenario=(
            "Debug logs sometimes include API tokens. Security policy says only the last few "
            "characters of a secret may appear in logs, so engineers can tell keys apart without "
            "exposing them."
        ),
        problem=(
            "Write `mask_secret(secret, visible=4)` that replaces every character except the last "
            "`visible` characters with `*`. If the secret has `visible` characters or fewer, mask "
            "all of it."
        ),
        signature="def mask_secret(secret: str, visible: int = 4) -> str:",
        entry="mask_secret",
        inputs=[
            "`secret`: any string, possibly empty.",
            "`visible`: number of trailing characters to keep, an `int` ≥ 0 (default 4).",
        ],
        output="A `str` with the same length as `secret`.",
        constraints=[
            "Output length always equals input length.",
            "If `len(secret) <= visible`, return only asterisks.",
            "`visible = 0` masks everything.",
            "Raise `ValueError` if `visible` is negative.",
        ],
        criteria=[
            "Keeps exactly the last `visible` characters when the secret is longer.",
            "Never reveals a short secret in full.",
            "Rejects negative `visible`.",
        ],
        hints=[
            "Split the string into a part to hide and a part to show.",
            "`'*' * n` builds n asterisks. Be careful: `secret[-0:]` is the *whole* string.",
            "if visible < 0: raise\nif len(secret) <= visible: return '*' * len(secret)\n"
            "hidden = len(secret) - visible\nreturn '*' * hidden + secret[hidden:]",
        ],
        solution='''
def mask_secret(secret: str, visible: int = 4) -> str:
    """Mask all but the last ``visible`` characters of a secret."""
    if visible < 0:
        raise ValueError("visible must be >= 0")
    if len(secret) <= visible:
        return "*" * len(secret)
    hidden = len(secret) - visible
    return "*" * hidden + secret[hidden:]
''',
        explanation=(
            "Slicing from a positive index (`secret[hidden:]`) avoids the `secret[-0:]` trap, "
            "where a negative-zero slice returns the entire string and would leak the secret."
        ),
        edge_cases=[
            "Empty secret → `\"\"`.",
            "`visible = 0` → all asterisks (would leak with `secret[-visible:]`).",
            "Secret shorter than `visible` → fully masked.",
        ],
        mistakes=[
            "`'*' * (len(s) - visible) + s[-visible:]` – leaks the whole secret when `visible` is 0.",
            "Returning the short secret unmasked.",
        ],
        tests=[
            T(("sk-test-abcd1234",), "************1234", example=True, note="default visible=4"),
            T(("abc",), "***", example=True, note="boundary: shorter than visible"),
            T(("token-xyz", 0), "*********", example=True, note="boundary: visible=0"),
            T(("abcdef", 2), "****ef", note="custom visible"),
            T(("",), "", note="empty secret"),
            T(("abcd",), "****", note="length equal to visible"),
            T(("abc", -1), raises="ValueError", note="negative visible"),
        ],
        complexity="O(n) time and space.",
        extension="Write `redact_tokens(text)` that finds every `sk-...` token in a log line and masks it with this function.",
    ),
    # ------------------------------------------------------------------------ Q009
    Question(
        id=9,
        title="Parse Key=Value Settings Strings",
        difficulty="Beginner",
        category=STR,
        roles=[PYDEV, DE, AIE],
        skills=["str.split with maxsplit", "strip", "dict building", "error messages"],
        minutes=15,
        scenario=(
            "Experiment runs are launched with a compact settings string such as "
            "`\"model=classifier-v2; temperature=0.2;max_tokens=256\"`. The launcher must parse it "
            "into a dictionary and give clear errors for malformed input."
        ),
        problem=(
            "Write `parse_kv(text)` that splits `text` on `;`, parses each non-empty segment as "
            "`key=value`, strips whitespace around keys and values, and returns a dictionary of "
            "strings."
        ),
        signature="def parse_kv(text: str) -> dict[str, str]:",
        entry="parse_kv",
        inputs=["`text`: a string of `;`-separated segments. May be empty or contain empty segments."],
        output="A `dict[str, str]`. Values stay as strings (no type conversion).",
        constraints=[
            "Ignore segments that are empty or whitespace only.",
            "Split each segment on the **first** `=` only, so values may contain `=`.",
            "When a key repeats, the later value wins.",
            "Raise `ValueError` for a segment without `=` or with an empty key.",
        ],
        criteria=[
            "Parses well-formed strings with irregular spacing.",
            "Keeps `=` characters inside values.",
            "Raises `ValueError` on malformed segments.",
        ],
        hints=[
            "Handle one segment at a time and decide: skip it, parse it, or reject it.",
            "`segment.split('=', 1)` splits only once; `partition('=')` is another option that "
            "tells you whether a separator was found.",
            "result = {}\nfor seg in text.split(';'):\n    if not seg.strip(): continue\n"
            "    key, sep, value = seg.partition('=')\n    if not sep or not key.strip(): raise ValueError\n"
            "    result[key.strip()] = value.strip()\nreturn result",
        ],
        solution='''
def parse_kv(text: str) -> dict[str, str]:
    """Parse 'k1=v1; k2=v2' into a dict of strings."""
    result: dict[str, str] = {}
    for segment in text.split(";"):
        if not segment.strip():
            continue
        key, sep, value = segment.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"segment {segment.strip()!r} is missing '='")
        if not key:
            raise ValueError(f"segment {segment.strip()!r} has an empty key")
        result[key] = value.strip()
    return result
''',
        explanation=(
            "`str.partition` splits on the first separator and reports whether it was found, "
            "which covers both the 'missing =' error and values that contain `=`."
        ),
        edge_cases=[
            "`\"\"` → `{}`.",
            "Trailing `;` produces an empty segment that is skipped.",
            "`\"filter=a=b\"` → `{'filter': 'a=b'}`.",
        ],
        mistakes=[
            "`key, value = segment.split('=')` crashes on values containing `=`.",
            "Not stripping whitespace, producing keys like `' temperature'`.",
            "Silently ignoring malformed segments.",
        ],
        tests=[
            T(("model=classifier-v2; temperature=0.2;max_tokens=256",),
              {"model": "classifier-v2", "temperature": "0.2", "max_tokens": "256"}, example=True,
              note="irregular spacing"),
            T(("",), {}, example=True, note="boundary: empty string"),
            T(("a=1;;b=2;",), {"a": "1", "b": "2"}, note="empty segments"),
            T(("filter=a=b",), {"filter": "a=b"}, note="value contains '='"),
            T(("a=1;a=2",), {"a": "2"}, note="later key wins"),
            T(("note=",), {"note": ""}, note="empty value is allowed"),
            T(("debug",), raises="ValueError", note="missing '='"),
            T(("=5",), raises="ValueError", note="empty key"),
        ],
        complexity="O(n) time and space in the length of the text.",
        extension="Add optional type conversion: values that look like ints, floats or `true`/`false` are converted.",
    ),
    # ------------------------------------------------------------------------ Q010
    Question(
        id=10,
        title="De-duplicate IDs While Preserving Order",
        difficulty="Beginner",
        category=DS,
        roles=[DE, DS_ROLE, PYDEV],
        skills=["sets for membership", "list building", "order preservation", "time complexity"],
        minutes=10,
        scenario=(
            "An event collector sometimes delivers the same user ID several times. Downstream "
            "jobs need each ID once, in the order it was first seen, and the collector can "
            "produce tens of thousands of IDs per batch."
        ),
        problem=(
            "Write `dedupe_preserve_order(items)` that returns a new list containing each item "
            "once, in order of first appearance."
        ),
        signature="def dedupe_preserve_order(items: list) -> list:",
        entry="dedupe_preserve_order",
        inputs=["`items`: a list of hashable values (strings, ints, tuples). May be empty."],
        output="A new `list` with duplicates removed, first occurrences kept in their original order.",
        constraints=[
            "Do not modify `items`.",
            "Values are compared with `==` (so `'a'` and `'A'` are different).",
            "Must handle 20,000 unique items in well under a second – avoid `x in some_list` inside the loop.",
        ],
        criteria=[
            "Order of first occurrence is preserved.",
            "Input list is unchanged.",
            "Runs in linear time.",
        ],
        hints=[
            "You need two things: an ordered result and a fast way to ask \"have I seen this?\".",
            "Membership checks on a `set` are O(1) on average; on a list they are O(n).",
            "seen = set(); result = []\nfor item in items:\n    if item not in seen:\n"
            "        seen.add(item); result.append(item)\nreturn result",
        ],
        solution='''
def dedupe_preserve_order(items: list) -> list:
    """Remove duplicates while keeping first-seen order."""
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
''',
        explanation=(
            "The set answers membership questions in O(1) on average while the list keeps the "
            "order. (`list(dict.fromkeys(items))` is an equally valid one-liner because dicts "
            "preserve insertion order.)"
        ),
        edge_cases=[
            "Empty list → `[]`.",
            "All duplicates → a single element.",
            "Case-sensitive comparison: `'a'` and `'A'` both remain.",
        ],
        mistakes=[
            "`list(set(items))` loses the original order.",
            "Checking `if item not in result` makes the function O(n²).",
        ],
        tests=[
            T(([3, 1, 3, 2, 1],), [3, 1, 2], example=True, note="ints"),
            T(([],), [], example=True, note="boundary: empty"),
            T((["a", "A", "a"],), ["a", "A"], note="case sensitive"),
            T((["u1", "u1", "u1"],), ["u1"], note="all duplicates"),
            T(([("a", 1), ("a", 1), ("b", 2)],), [("a", 1), ("b", 2)], note="tuples"),
            T((list(range(20000)) * 2,), list(range(20000)), time_limit=1.0, note="20k ids must be fast"),
        ],
        complexity="O(n) average time and O(n) space.",
        extension="Add a `key` parameter so records (dicts) can be de-duplicated by one field, keeping the first record.",
    ),
    # ------------------------------------------------------------------------ Q011
    Question(
        id=11,
        title="Count Normalized Label Frequencies",
        difficulty="Beginner",
        category=DS,
        roles=[DS_ROLE, MLE, AIE],
        skills=["frequency counting", "dict.get / collections.Counter", "data cleaning"],
        minutes=10,
        scenario=(
            "Before training a spam classifier, you check class balance. The labels came from "
            "several annotators, so the same label appears as `\"Spam\"`, `\" spam \"` or "
            "`\"SPAM\"`, and some rows have an empty or missing label."
        ),
        problem=(
            "Write `label_counts(labels)` that returns how many times each label occurs after "
            "stripping whitespace and lower-casing, skipping `None` and empty labels."
        ),
        signature="def label_counts(labels: list) -> dict[str, int]:",
        entry="label_counts",
        inputs=["`labels`: a list whose items are strings or `None`. May be empty."],
        output="A `dict` mapping each normalized label to its count. Keys with a count of zero never appear.",
        constraints=[
            "Normalize with `label.strip().lower()`.",
            "Skip `None` and labels that are empty after stripping.",
            "Return a plain `dict` (a `Counter` is a `dict` subclass and is also accepted).",
        ],
        criteria=[
            "Variants of the same label are counted together.",
            "Missing and blank labels are skipped.",
            "Empty input returns `{}`.",
        ],
        hints=[
            "Clean each label first, then count it.",
            "`counts[key] = counts.get(key, 0) + 1`, or use `collections.Counter`.",
            "counts = {}\nfor label in labels:\n    if label is None: continue\n    key = label.strip().lower()\n"
            "    if not key: continue\n    counts[key] = counts.get(key, 0) + 1\nreturn counts",
        ],
        solution='''
def label_counts(labels: list) -> dict[str, int]:
    """Count labels after normalizing case and whitespace."""
    counts: dict[str, int] = {}
    for label in labels:
        if label is None:
            continue
        key = label.strip().lower()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts
''',
        explanation="Normalizing before counting merges variants; `dict.get` with a default avoids a separate 'first time' branch.",
        edge_cases=["`[]` → `{}`.", "`[None, '  ']` → `{}`.", "Labels differing only in case are merged."],
        mistakes=[
            "Calling `.strip()` on `None` (AttributeError).",
            "Counting before normalizing, which splits one class into several keys.",
        ],
        tests=[
            T((["Spam", "ham", " spam ", "HAM", "spam", "", " ", None],), {"spam": 3, "ham": 2},
              example=True, note="mixed formatting"),
            T(([],), {}, example=True, note="boundary: empty"),
            T(([None, "  "],), {}, note="only missing values"),
            T((["Positive"],), {"positive": 1}, note="single label"),
            T((["a", "b", "A", "c", "B", "a"],), {"a": 3, "b": 2, "c": 1}, note="several labels"),
        ],
        complexity="O(n) time; O(k) space for k distinct labels.",
        extension="Return the counts sorted from most to least common, with alphabetical tie-breaking.",
    ),
    # ------------------------------------------------------------------------ Q012
    Question(
        id=12,
        title="Merge Defaults with Overrides",
        difficulty="Beginner",
        category=DS,
        roles=[MLE, PYDEV, DE],
        skills=["dict copying", "dict iteration", "None vs falsy", "non-mutation"],
        minutes=10,
        scenario=(
            "A training script has default hyper-parameters. A user supplies overrides from a "
            "form where untouched fields arrive as `None`. Untouched fields must keep their "
            "defaults, but an explicit `0` (for example `dropout = 0`) is a real override."
        ),
        problem=(
            "Write `merge_configs(defaults, overrides)` that returns a new dict containing all "
            "defaults, updated with every override whose value is not `None`."
        ),
        signature="def merge_configs(defaults: dict, overrides: dict) -> dict:",
        entry="merge_configs",
        inputs=["`defaults`: dict of setting → value.", "`overrides`: dict of setting → value or `None`. Both may be empty."],
        output="A new `dict`.",
        constraints=[
            "Override values of `None` are ignored.",
            "Other overrides (including `0`, `False`, `\"\"`) replace defaults.",
            "Keys that appear only in `overrides` (with non-`None` values) are added.",
            "Neither input dictionary may be modified.",
        ],
        criteria=[
            "Correct merge with None-skipping.",
            "Falsy overrides are applied.",
            "Inputs are left unchanged.",
        ],
        hints=[
            "Start from a *copy* of the defaults.",
            "Loop through `overrides.items()` and only assign when `value is not None`.",
            "merged = dict(defaults)\nfor k, v in overrides.items():\n    if v is not None: merged[k] = v\nreturn merged",
        ],
        solution='''
def merge_configs(defaults: dict, overrides: dict) -> dict:
    """Return defaults updated with all non-None overrides."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
''',
        explanation="`dict(defaults)` makes a shallow copy so the caller's defaults are untouched; `is not None` keeps falsy overrides.",
        edge_cases=["Both empty → `{}`.", "Override `0` replaces a default.", "Override-only key with `None` is not added."],
        mistakes=[
            "`defaults.update(overrides)` mutates the caller's dict and copies `None` values.",
            "`if value:` drops legitimate `0`/`False` overrides.",
        ],
        tests=[
            T(({"lr": 0.01, "epochs": 10, "device": "cpu"}, {"epochs": 20, "device": None, "seed": 42}),
              {"lr": 0.01, "epochs": 20, "device": "cpu", "seed": 42}, example=True, note="typical merge"),
            T(({"dropout": 0.5}, {"dropout": 0}), {"dropout": 0}, example=True, note="boundary: falsy override"),
            T(({}, {}), {}, note="both empty"),
            T(({"a": 1}, {}), {"a": 1}, note="no overrides"),
            T(({}, {"x": None}), {}, note="None-only override is not added"),
            T(({"debug": True}, {"debug": False}), {"debug": False}, note="False override"),
        ],
        complexity="O(len(defaults) + len(overrides)) time and space.",
        extension="Make the merge recursive so nested dicts are merged key by key instead of replaced.",
    ),
    # ------------------------------------------------------------------------ Q013
    Question(
        id=13,
        title="Detect Training/Serving Feature Skew",
        difficulty="Beginner",
        category=DS,
        roles=[MLE, DS_ROLE, AAI],
        skills=["set difference", "set intersection", "sorted()", "dict of lists"],
        minutes=10,
        scenario=(
            "A model was trained on one list of feature columns, but the online service sends a "
            "slightly different list. Before deploying, you want a report of which features are "
            "missing at serving time, which are unexpected, and which match."
        ),
        problem=(
            "Write `compare_feature_sets(training_features, serving_features)` that returns a dict "
            "with three sorted lists: `missing_in_serving`, `unexpected_in_serving` and `shared`."
        ),
        signature="def compare_feature_sets(training_features: list[str], serving_features: list[str]) -> dict[str, list[str]]:",
        entry="compare_feature_sets",
        inputs=["Two lists of feature names (strings). Either may be empty or contain duplicates."],
        output=(
            "`{'missing_in_serving': [...], 'unexpected_in_serving': [...], 'shared': [...]}`, every "
            "list sorted ascending with no duplicates. All three keys are always present."
        ),
        constraints=["Names are compared exactly (case-sensitive).", "Do not modify the inputs."],
        criteria=["Correct set differences and intersection.", "Lists are sorted and duplicate-free.", "All keys always present."],
        hints=[
            "This is exactly what Python's set operators are for.",
            "`a - b` is the difference, `a & b` the intersection; wrap the results in `sorted()`.",
            "train, serve = set(training), set(serving)\nreturn {'missing_in_serving': sorted(train - serve),\n"
            "        'unexpected_in_serving': sorted(serve - train), 'shared': sorted(train & serve)}",
        ],
        solution='''
def compare_feature_sets(training_features: list[str], serving_features: list[str]) -> dict[str, list[str]]:
    """Report differences between training and serving feature lists."""
    train = set(training_features)
    serve = set(serving_features)
    return {
        "missing_in_serving": sorted(train - serve),
        "unexpected_in_serving": sorted(serve - train),
        "shared": sorted(train & serve),
    }
''',
        explanation="Converting to sets removes duplicates and makes differences O(n); sorting gives deterministic output.",
        edge_cases=["Identical lists → only `shared` is non-empty.", "Both empty → three empty lists.", "Duplicates collapse."],
        mistakes=["Returning sets (unordered, not lists).", "Nested loops over both lists (O(n·m)).", "Omitting keys whose list is empty."],
        tests=[
            T((["age", "income", "tenure"], ["age", "tenure", "device"]),
              {"missing_in_serving": ["income"], "unexpected_in_serving": ["device"], "shared": ["age", "tenure"]},
              example=True, note="typical skew"),
            T(([], []), {"missing_in_serving": [], "unexpected_in_serving": [], "shared": []}, example=True,
              note="boundary: both empty"),
            T((["b", "a", "a"], ["a", "b"]), {"missing_in_serving": [], "unexpected_in_serving": [], "shared": ["a", "b"]},
              note="duplicates and identical sets"),
            T((["Age"], ["age"]), {"missing_in_serving": ["Age"], "unexpected_in_serving": ["age"], "shared": []},
              note="case sensitive"),
            T((["x"], []), {"missing_in_serving": ["x"], "unexpected_in_serving": [], "shared": []}, note="nothing served"),
        ],
        complexity="O(n log n) because of sorting; the set operations themselves are O(n) on average.",
        extension="Also flag features present in both lists but with different dtypes, given two dicts name → dtype.",
    ),
    # ------------------------------------------------------------------------ Q014
    Question(
        id=14,
        title="Build an Inverted Tag Index",
        difficulty="Beginner",
        category=DS,
        roles=[AIE, DS_ROLE, PYDEV],
        skills=["dict of lists", "setdefault / defaultdict", "nested loops", "sorting"],
        minutes=15,
        scenario=(
            "A small document search service stores tags per document. To answer \"which "
            "documents have tag X?\" quickly, it needs the reverse mapping: tag → documents."
        ),
        problem=(
            "Write `invert_index(doc_tags)` that turns `{doc_id: [tags]}` into `{tag: [doc_ids]}` "
            "with each list of doc IDs sorted and free of duplicates."
        ),
        signature="def invert_index(doc_tags: dict[str, list[str]]) -> dict[str, list[str]]:",
        entry="invert_index",
        inputs=["`doc_tags`: dict mapping document id (str) to a list of tags (str). A tag may repeat within one document; a list may be empty."],
        output="A `dict[str, list[str]]`. Key order does not matter; each value list is sorted ascending.",
        constraints=[
            "A document appears at most once under each tag.",
            "Documents with no tags contribute nothing.",
            "Do not modify the input.",
        ],
        criteria=["Every (doc, tag) pair appears exactly once.", "Doc ID lists are sorted.", "Empty input → `{}`."],
        hints=[
            "Visit every (document, tag) pair and file the document under the tag.",
            "Collect into a `dict[str, set]` to remove duplicates, then convert each set to a sorted list.",
            "index = {}\nfor doc, tags in doc_tags.items():\n    for tag in tags:\n"
            "        index.setdefault(tag, set()).add(doc)\nreturn {tag: sorted(docs) for tag, docs in index.items()}",
        ],
        solution='''
def invert_index(doc_tags: dict[str, list[str]]) -> dict[str, list[str]]:
    """Map each tag to the sorted list of documents that carry it."""
    index: dict[str, set[str]] = {}
    for doc_id, tags in doc_tags.items():
        for tag in tags:
            index.setdefault(tag, set()).add(doc_id)
    return {tag: sorted(doc_ids) for tag, doc_ids in index.items()}
''',
        explanation="Sets deduplicate repeated tags inside a document; sorting at the end gives deterministic output.",
        edge_cases=["Empty dict → `{}`.", "Document with `[]` tags is ignored.", "Repeated tag within one document counted once."],
        mistakes=["Appending to lists without deduplication.", "Forgetting to sort, producing order-dependent output."],
        tests=[
            T(({"doc1": ["python", "ml"], "doc2": ["ml"], "doc3": ["python", "python", "sql"]},),
              {"python": ["doc1", "doc3"], "ml": ["doc1", "doc2"], "sql": ["doc3"]}, example=True,
              note="duplicate tag inside doc3"),
            T(({},), {}, example=True, note="boundary: empty"),
            T(({"d1": []},), {}, note="document without tags"),
            T(({"b": ["x"], "a": ["x"]},), {"x": ["a", "b"]}, note="sorted doc ids"),
        ],
        complexity="O(P + Σ k log k) where P is the number of (doc, tag) pairs and k the docs per tag.",
        extension="Add `search(index, tags, mode='all'|'any')` returning documents with all/any of the tags.",
    ),
    # ------------------------------------------------------------------------ Q015
    Question(
        id=15,
        title="Filter High-Confidence Predictions",
        difficulty="Beginner",
        category=FUNC,
        roles=[MLE, AIE, DS_ROLE],
        skills=["list comprehensions", "default arguments", "validation"],
        minutes=10,
        scenario=(
            "A content-moderation model returns a label and a confidence score for each item. "
            "Only predictions above a confidence threshold are auto-applied; the rest go to a "
            "human reviewer."
        ),
        problem=(
            "Write `high_confidence_labels(predictions, threshold=0.8)` that returns the labels of "
            "predictions whose score is greater than or equal to the threshold, in input order."
        ),
        signature="def high_confidence_labels(predictions: list[dict], threshold: float = 0.8) -> list[str]:",
        entry="high_confidence_labels",
        inputs=[
            "`predictions`: list of dicts with keys `\"label\"` (str) and `\"score\"` (float in [0, 1]).",
            "`threshold`: float in [0, 1], default 0.8.",
        ],
        output="A `list[str]` of labels, preserving input order (duplicates kept).",
        constraints=["`score >= threshold` qualifies.", "Raise `ValueError` if threshold is outside [0, 1].", "Do not modify the input."],
        criteria=["Uses `>=`.", "Preserves order.", "Validates the threshold."],
        hints=[
            "This is a filter + transform – a classic job for a comprehension.",
            "`[p['label'] for p in predictions if p['score'] >= threshold]`.",
            "validate threshold\nreturn [p['label'] for p in predictions if p['score'] >= threshold]",
        ],
        solution='''
def high_confidence_labels(predictions: list[dict], threshold: float = 0.8) -> list[str]:
    """Labels of predictions whose score is at least ``threshold``."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return [p["label"] for p in predictions if p["score"] >= threshold]
''',
        explanation="The comprehension expresses filter-then-project in one readable line; validation guards against thresholds like 80.",
        edge_cases=["Score equal to threshold qualifies.", "Empty input → `[]`.", "`threshold=0` keeps everything."],
        mistakes=["Using `>` instead of `>=`.", "Returning the dicts instead of labels.", "Accepting `threshold=80`."],
        tests=[
            T(([{"label": "spam", "score": 0.93}, {"label": "ok", "score": 0.41}, {"label": "abuse", "score": 0.8}],),
              ["spam", "abuse"], example=True, note="default threshold, boundary 0.8 included"),
            T(([{"label": "ok", "score": 0.5}], 0.5), ["ok"], note="custom threshold equal to score"),
            T(([],), [], example=True, note="boundary: empty"),
            T(([{"label": "a", "score": 0.1}, {"label": "a", "score": 0.0}], 0.0), ["a", "a"], note="threshold 0 keeps all"),
            T(([], 80), raises="ValueError", note="threshold out of range"),
        ],
        complexity="O(n) time, O(k) space for k kept labels.",
        extension="Return a tuple `(auto_applied, needs_review)` splitting the predictions in one pass.",
    ),
    # ------------------------------------------------------------------------ Q016
    Question(
        id=16,
        title="Structured Log Line Builder",
        difficulty="Beginner",
        category=FUNC,
        roles=[PYDEV, AIE, DE],
        skills=["*args", "**kwargs", "str.join", "sorted dict items", "f-strings"],
        minutes=15,
        scenario=(
            "Your team wants consistent, grep-friendly log lines. Instead of formatting strings "
            "by hand everywhere, you will build a tiny helper that accepts free-text parts and "
            "key/value fields."
        ),
        problem=(
            "Write `build_log_line(level, *parts, **fields)` that returns a line of the form "
            "`[LEVEL] part1 part2 | key1=value1 key2=value2`."
        ),
        signature="def build_log_line(level: str, *parts, **fields) -> str:",
        entry="build_log_line",
        inputs=[
            "`level`: one of `debug`, `info`, `warn`, `error` in any letter case.",
            "`*parts`: zero or more values of any type, converted with `str()`.",
            "`**fields`: zero or more keyword arguments of any value type.",
        ],
        output="A `str`.",
        constraints=[
            "Start with the upper-cased level in square brackets, e.g. `[INFO]`.",
            "If there are parts, append a space and the parts joined by single spaces.",
            "If there are fields, append ` | ` and `key=value` pairs sorted by key, joined by spaces.",
            "Raise `ValueError` for any other level.",
        ],
        criteria=["Correct format with and without parts/fields.", "Fields sorted by key.", "Invalid level raises `ValueError`."],
        hints=[
            "Build the line from up to three pieces and only add the pieces that exist.",
            "`' '.join(str(p) for p in parts)` and `sorted(fields.items())` do most of the work.",
            "check level\nline = f'[{level.upper()}]'\nif parts: line += ' ' + joined parts\n"
            "if fields: line += ' | ' + ' '.join(f'{k}={v}' for k, v in sorted(fields.items()))\nreturn line",
        ],
        solution='''
VALID_LEVELS = {"DEBUG", "INFO", "WARN", "ERROR"}


def build_log_line(level: str, *parts, **fields) -> str:
    """Format a structured, grep-friendly log line."""
    upper = level.upper()
    if upper not in VALID_LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {sorted(VALID_LEVELS)}")
    line = f"[{upper}]"
    if parts:
        line += " " + " ".join(str(part) for part in parts)
    if fields:
        line += " | " + " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
    return line
''',
        explanation="`*parts` arrives as a tuple and `**fields` as a dict; sorting the field items makes the output deterministic.",
        edge_cases=["No parts and no fields → just `[LEVEL]`.", "Only fields → `[LEVEL] | k=v`.", "Non-string parts are converted with `str()`."],
        mistakes=["Leaving a trailing space when there are no parts.", "Relying on keyword order instead of sorting.", "Joining non-string parts without `str()`."],
        tests=[
            T(("info", "model", "loaded"), "[INFO] model loaded | env=prod version=3",
              kwargs={"version": 3, "env": "prod"}, example=True, note="parts and fields"),
            T(("warn",), "[WARN]", example=True, note="boundary: level only"),
            T(("error",), "[ERROR] | latency_ms=812", kwargs={"latency_ms": 812}, note="fields only"),
            T(("Debug", "batch", 7, "done"), "[DEBUG] batch 7 done", note="non-string part"),
            T(("fatal", "x"), raises="ValueError", note="unknown level"),
        ],
        complexity="O(p + f log f) for p parts and f fields.",
        extension="Escape field values that contain spaces by quoting them, e.g. `msg=\"disk full\"`.",
    ),
    # ------------------------------------------------------------------------ Q017
    Question(
        id=17,
        title="Leaderboard: Top-N Models by Score",
        difficulty="Beginner",
        category=ALG,
        roles=[DS_ROLE, MLE, AAI],
        skills=["sorted() with key", "tuple keys for tie-breaking", "slicing"],
        minutes=10,
        scenario=(
            "An internal leaderboard ranks candidate models by validation accuracy. Ties are "
            "common, so the ranking must be deterministic: higher score first, then model name "
            "alphabetically."
        ),
        problem="Write `top_n(scores, n)` that returns the `n` best `(model_name, score)` pairs.",
        signature="def top_n(scores: dict[str, float], n: int) -> list[tuple[str, float]]:",
        entry="top_n",
        inputs=["`scores`: dict mapping model name to score.", "`n`: number of entries to return, `int` ≥ 0."],
        output="A list of `(name, score)` tuples, best first, length `min(n, len(scores))`.",
        constraints=[
            "Sort by score descending, then name ascending.",
            "If `n` exceeds the number of models, return all of them.",
            "Raise `ValueError` if `n` is negative.",
        ],
        criteria=["Correct order including ties.", "Handles `n = 0` and `n` larger than the input.", "Rejects negative `n`."],
        hints=[
            "A sort key can be a tuple; Python compares tuples element by element.",
            "Negate the score so that ascending sort order gives the highest score first: `key=lambda kv: (-kv[1], kv[0])`.",
            "validate n\nranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))\nreturn ranked[:n]",
        ],
        solution='''
def top_n(scores: dict[str, float], n: int) -> list[tuple[str, float]]:
    """Return the n best (name, score) pairs; ties broken by name."""
    if n < 0:
        raise ValueError("n must be >= 0")
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:n]
''',
        explanation="A composite key `(-score, name)` encodes both ordering rules in one stable sort; slicing handles `n` larger than the list.",
        edge_cases=["`n = 0` → `[]`.", "`n` larger than the dict → all models.", "Equal scores ordered by name."],
        mistakes=["`sorted(..., reverse=True)` with key `(score, name)` sorts names descending too.", "Using `heapq.nlargest` without handling ties."],
        tests=[
            T(({"bert": 0.91, "xgb": 0.88, "lr": 0.91, "rf": 0.85}, 3), [("bert", 0.91), ("lr", 0.91), ("xgb", 0.88)],
              example=True, note="tie broken by name"),
            T(({"a": 1.0}, 5), [("a", 1.0)], example=True, note="boundary: n larger than input"),
            T(({"a": 1.0, "b": 2.0}, 0), [], note="n = 0"),
            T(({}, 3), [], note="empty dict"),
            T(({"a": 1.0}, -1), raises="ValueError", note="negative n"),
        ],
        complexity="O(m log m) for m models (full sort). `heapq.nsmallest(n, ..., key=...)` gives O(m log n).",
        extension="Return dense ranks too, so tied models share a rank: `[(1, 'bert', 0.91), (1, 'lr', 0.91), (2, 'xgb', 0.88)]`.",
    ),
    # ------------------------------------------------------------------------ Q018
    Question(
        id=18,
        title="Read a Pipeline Config File",
        difficulty="Beginner",
        category=FILE,
        roles=[DE, PYDEV, MLE],
        skills=["open() with a context manager", "encoding", "iterating file lines", "FileNotFoundError"],
        minutes=10,
        scenario=(
            "A data pipeline reads simple text config files. Lines beginning with `#` are "
            "comments, blank lines are for readability, and stray whitespace is common."
        ),
        problem=(
            "Write `read_config_lines(path)` that returns the meaningful lines of a UTF-8 text "
            "file: each line stripped of surrounding whitespace, skipping blank lines and lines "
            "whose stripped text starts with `#`."
        ),
        signature="def read_config_lines(path: str) -> list[str]:",
        entry="read_config_lines",
        inputs=["`path`: path to a UTF-8 text file. The file may be empty or may not exist."],
        output="A `list[str]` in file order.",
        constraints=[
            "Use a `with` block so the file is always closed.",
            "Only full-line comments are removed; `#` in the middle of a line is kept.",
            "A missing file must raise `FileNotFoundError` (do not swallow it).",
        ],
        criteria=["Comment, blank and whitespace handling are correct.", "Empty file → `[]`.", "Missing file → `FileNotFoundError`."],
        hints=[
            "Iterate over the file object line by line.",
            "Strip each line first; then decide whether to skip it with `not line` or `line.startswith('#')`.",
            "result = []\nwith open(path, encoding='utf-8') as f:\n    for raw in f:\n        line = raw.strip()\n"
            "        if line and not line.startswith('#'): result.append(line)\nreturn result",
        ],
        solution='''
def read_config_lines(path: str) -> list[str]:
    """Return stripped, non-blank, non-comment lines from a config file."""
    lines = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines
''',
        explanation="The `with` block closes the file even if an error occurs; stripping first means indented comments are also recognised.",
        edge_cases=["Indented comment `   # x` is skipped.", "Empty file → `[]`.", "Missing path → `FileNotFoundError` propagates."],
        mistakes=["`open(path).readlines()` without closing the file.", "Checking `startswith('#')` before stripping.", "Catching all exceptions and returning `[]`, hiding misconfiguration."],
        tests=[
            T((File("pipeline.cfg", "# pipeline config\nsource=s3://bucket/raw\n\n  batch_size=500  \n# end\n"),),
              ["source=s3://bucket/raw", "batch_size=500"], example=True, note="comments and blank lines"),
            T((File("empty.cfg", ""),), [], example=True, note="boundary: empty file"),
            T((File("inline.cfg", "url=http://x/#anchor\n   # indented comment\n"),), ["url=http://x/#anchor"],
              note="inline # is kept"),
            T((TmpPath("missing.cfg"),), raises="FileNotFoundError", note="missing file"),
            T((DataFile("pipeline_config.txt"),),
              ["source=s3://example-bucket/raw/events/", "destination=warehouse.analytics.events",
               "batch_size=500", "max_retries=3", "timeout_seconds=30", "validate_schema=true"],
              note="sample dataset file"),
        ],
        complexity="O(n) time in the file size; O(k) space for the kept lines.",
        extension="Parse the kept lines into a dict and report the line number of any line that is not `key=value`.",
    ),
    # ------------------------------------------------------------------------ Q019
    Question(
        id=19,
        title="Fix the Off-by-One in last_n",
        difficulty="Beginner",
        category=DBG,
        roles=[PYDEV, DS_ROLE, MLE],
        skills=["negative slicing", "boundary testing", "debugging"],
        minutes=10,
        scenario=(
            "A dashboard shows the last *n* inference results. A teammate wrote:\n\n"
            "```python\ndef last_n(items, n):\n    return items[-n:]\n```\n\n"
            "It works in their demo, but users report that asking for the last **0** items shows "
            "*everything*, and negative values behave strangely."
        ),
        problem=(
            "Write a corrected `last_n(items, n)` that returns the last `n` items of a list as a "
            "new list and handles the boundary values that the original version gets wrong."
        ),
        signature="def last_n(items: list, n: int) -> list:",
        entry="last_n",
        inputs=["`items`: a list (may be empty).", "`n`: an `int`."],
        output="A new list with the last `min(n, len(items))` items in their original order.",
        constraints=[
            "`n = 0` → `[]`.",
            "`n` larger than the list → a copy of the whole list.",
            "Raise `ValueError` if `n < 0`.",
            "Do not modify `items`.",
        ],
        criteria=["`last_n(x, 0) == []`.", "Normal and oversized `n` work.", "Negative `n` raises `ValueError`."],
        hints=[
            "Try the original function with `n = 0` in a REPL. What does `-0` equal?",
            "`items[-0:]` is `items[0:]`. Compute a non-negative start index instead: `len(items) - n`, clamped at 0.",
            "if n < 0: raise ValueError\nstart = max(len(items) - n, 0)\nreturn items[start:]",
        ],
        solution='''
def last_n(items: list, n: int) -> list:
    """Return the last n items (as a new list)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    start = max(len(items) - n, 0)
    return items[start:]
''',
        explanation="Because `-0 == 0`, `items[-0:]` returns the whole list. Using an explicit non-negative start index avoids the trap.",
        edge_cases=["`n = 0` → `[]`.", "Empty list → `[]`.", "`n > len(items)` → full copy."],
        mistakes=["Special-casing only `n == 0` but still allowing negative `n`.", "Returning `items` itself instead of a new list."],
        tests=[
            T(([1, 2, 3, 4, 5], 2), [4, 5], example=True, note="normal case"),
            T(([1, 2, 3], 0), [], example=True, note="boundary: the original bug"),
            T(([1, 2], 5), [1, 2], note="n larger than list"),
            T(([], 3), [], note="empty list"),
            T(([1, 2, 3], 3), [1, 2, 3], note="n equal to length"),
            T(([1], -1), raises="ValueError", note="negative n"),
        ],
        complexity="O(n) time and space for the copied slice.",
        extension="Write a pytest test that would have caught the original bug.",
    ),
    # ------------------------------------------------------------------------ Q020
    Question(
        id=20,
        title="Summarize Latency Measurements with Gaps",
        difficulty="Beginner",
        category=DATA,
        roles=[MLE, DS_ROLE, AIE],
        skills=["filtering None", "min/max/sum", "round()", "empty-input handling"],
        minutes=15,
        scenario=(
            "A load test records the latency of each request in milliseconds. Requests that "
            "timed out were recorded as `None`. The report needs basic statistics over the "
            "requests that completed."
        ),
        problem=(
            "Write `summarize_latencies(values)` that returns the count, minimum, maximum and mean "
            "of the non-`None` values."
        ),
        signature="def summarize_latencies(values: list) -> dict:",
        entry="summarize_latencies",
        inputs=["`values`: list of non-negative ints/floats or `None`. May be empty."],
        output=(
            "`{'count': int, 'min': number|None, 'max': number|None, 'mean': float|None}`. "
            "`mean` is rounded to 2 decimals."
        ),
        constraints=[
            "Skip `None` values; keep `0` (it is a real measurement).",
            "If no values remain, return count 0 and `None` for min, max and mean.",
            "Do not modify the input.",
        ],
        criteria=["Correct statistics.", "Zero is not dropped.", "Empty/all-None input handled."],
        hints=[
            "First build a clean list, then compute statistics on it.",
            "`[v for v in values if v is not None]` keeps zeros; check for an empty list before calling `min()`.",
            "clean = [v for v in values if v is not None]\nif not clean: return {...None...}\n"
            "return {'count': len(clean), 'min': min(clean), 'max': max(clean), 'mean': round(sum(clean)/len(clean), 2)}",
        ],
        solution='''
def summarize_latencies(values: list) -> dict:
    """Count, min, max and mean of the non-None measurements."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": round(sum(clean) / len(clean), 2),
    }
''',
        explanation="Filtering with `is not None` keeps zero latencies; the early return avoids `min()` of an empty sequence and division by zero.",
        edge_cases=["All `None` → zeros/None result.", "`0` latency is counted.", "Single value → min = max = mean."],
        mistakes=["Filtering with `if v`, which drops 0.", "Calling `min([])` (ValueError).", "Dividing by `len(values)` including the gaps."],
        tests=[
            T(([120, None, 80, 250.5, None],), {"count": 3, "min": 80, "max": 250.5, "mean": 150.17},
              example=True, note="typical data"),
            T(([None, None],), {"count": 0, "min": None, "max": None, "mean": None}, example=True,
              note="boundary: nothing completed"),
            T(([0, None, 10],), {"count": 2, "min": 0, "max": 10, "mean": 5.0}, note="zero is kept"),
            T(([42],), {"count": 1, "min": 42, "max": 42, "mean": 42.0}, note="single value"),
            T(([],), {"count": 0, "min": None, "max": None, "mean": None}, note="empty list"),
        ],
        complexity="O(n) time and O(n) space for the filtered list (O(1) with a single running loop).",
        extension="Add the median and p95 (nearest-rank) to the summary.",
    ),
]
