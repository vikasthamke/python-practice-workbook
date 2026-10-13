"""Intermediate questions Q021–Q038: practical programming and data handling."""

import itertools
import time

from ..schema import (
    AAI, AIE, ALG, DE, DS, DS_ROLE, FUNC, FUND, MLE, PYDEV, SAMPLE_DATA_DIR, STR,
    CT, Question, T,
)


class FakeAPI:
    """Deterministic stand-in for a paginated HTTP API used by Q021's tests."""

    def __init__(self, pages: dict):
        self.pages = pages

    def __call__(self, cursor):
        page = self.pages[cursor]
        return {"items": list(page["items"]), "next_cursor": page["next_cursor"]}

    def __eq__(self, other):
        return isinstance(other, FakeAPI) and self.pages == other.pages

    def __repr__(self):
        return f"FakeAPI({len(self.pages)} pages)"


THREE_PAGES = FakeAPI({
    None: {"items": [1, 2], "next_cursor": "p2"},
    "p2": {"items": [3], "next_cursor": "p3"},
    "p3": {"items": [4, 5], "next_cursor": None},
})


# ------------------------------------------------------------------ custom tests
def _q021_counts_calls(impl, tmp):
    calls = []

    def fetch(cursor):
        calls.append(cursor)
        return THREE_PAGES(cursor)

    assert impl(fetch) == [1, 2, 3, 4, 5], "items from all pages should be collected in order"
    assert calls == [None, "p2", "p3"], f"expected cursors [None, 'p2', 'p3'] to be requested, got {calls}"


def _q024_sample_file(impl, tmp):
    lines = (SAMPLE_DATA_DIR / "application_logs.txt").read_text(encoding="utf-8").splitlines()
    parsed = [impl(line) for line in lines]
    valid = [p for p in parsed if p is not None]
    assert len(lines) == 40, "sample file should have 40 lines"
    assert len(valid) == 34, f"expected 34 parseable lines in application_logs.txt, got {len(valid)}"
    errors = sum(1 for p in valid if p["level"] == "ERROR")
    assert errors == 9, f"expected 9 ERROR lines, got {errors}"


def _q032_call_time_validation(impl, tmp):
    try:
        impl([1, 2, 3], 0)
    except ValueError:
        return
    raise AssertionError("batched([1, 2, 3], 0) must raise ValueError immediately, "
                         "before iteration starts")


def _q032_generator_input(impl, tmp):
    got = list(impl((ch for ch in "abcde"), 2))
    assert got == [["a", "b"], ["c", "d"], ["e"]], f"expected [['a', 'b'], ['c', 'd'], ['e']], got {got}"


def _q032_lazy(impl, tmp):
    gen = impl(itertools.count(), 2)
    assert iter(gen) is gen, "batched should return an iterator/generator, not a list"
    assert next(gen) == [0, 1] and next(gen) == [2, 3], "batches of an infinite iterator must be produced lazily"


def _make_flaky(failures, exc_type=ConnectionError, value="ok"):
    state = {"calls": 0}

    def flaky(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] <= failures:
            raise exc_type(f"failure {state['calls']}")
        return (value, args, kwargs)

    return flaky, state


def _q033_recovers(impl, tmp):
    flaky, state = _make_flaky(2)
    wrapped = impl(max_attempts=3)(flaky)
    assert wrapped(1, key="v") == ("ok", (1,), {"key": "v"}), "should return the result and pass args/kwargs through"
    assert state["calls"] == 3, f"expected 3 calls, got {state['calls']}"


def _q033_gives_up(impl, tmp):
    flaky, state = _make_flaky(10)
    wrapped = impl(max_attempts=2)(flaky)
    try:
        wrapped()
    except ConnectionError as exc:
        assert str(exc) == "failure 2", "the exception from the LAST attempt should be re-raised"
    else:
        raise AssertionError("expected the last ConnectionError to be re-raised")
    assert state["calls"] == 2, f"expected exactly 2 attempts, got {state['calls']}"


def _q033_non_retryable(impl, tmp):
    flaky, state = _make_flaky(10, exc_type=ValueError)
    wrapped = impl(max_attempts=5, exceptions=(ConnectionError, TimeoutError))(flaky)
    try:
        wrapped()
    except ValueError:
        pass
    else:
        raise AssertionError("non-retryable exceptions must propagate")
    assert state["calls"] == 1, "non-retryable exceptions must not be retried"


def _q033_wraps(impl, tmp):
    def fetch_embeddings():
        """Fetch embeddings from the API."""
        return 1

    wrapped = impl(max_attempts=2)(fetch_embeddings)
    assert wrapped.__name__ == "fetch_embeddings", "use functools.wraps to keep __name__"
    assert wrapped.__doc__ == "Fetch embeddings from the API.", "use functools.wraps to keep __doc__"


def _q033_validation(impl, tmp):
    try:
        impl(max_attempts=0)
    except ValueError:
        return
    raise AssertionError("retry(max_attempts=0) should raise ValueError")


def _q034_basic(impl, tmp):
    avg = impl(3)
    got = [avg(x) for x in [10, 20, 30, 40]]
    assert got == [10.0, 15.0, 20.0, 30.0], f"expected [10.0, 15.0, 20.0, 30.0], got {got}"


def _q034_independent(impl, tmp):
    a, b = impl(2), impl(2)
    a(100)
    assert b(1) == 1.0, "two moving averages must not share state"
    assert a(0) == 50.0


def _q034_window_one(impl, tmp):
    avg = impl(1)
    assert [avg(x) for x in [5, -3, 8]] == [5.0, -3.0, 8.0]


def _q034_invalid(impl, tmp):
    try:
        impl(0)
    except ValueError:
        return
    raise AssertionError("make_moving_average(0) should raise ValueError")


def _q034_fast(impl, tmp):
    avg = impl(10_000)
    start = time.perf_counter()
    last = None
    for i in range(30_000):
        last = avg(i)
    elapsed = time.perf_counter() - start
    assert last == sum(range(20_000, 30_000)) / 10_000, f"wrong average, got {last}"
    assert elapsed < 0.5, (f"30,000 updates with window 10,000 took {elapsed:.2f}s; each update "
                           "should be O(1) (keep a running total)")


def _q036_fast(impl, tmp):
    stamps = list(range(0, 400_000, 2))
    start = time.perf_counter()
    for q in range(5_000):
        assert impl(stamps, q * 10, q * 10 + 9) == 5
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"5,000 queries took {elapsed:.2f}s; use binary search (bisect)"


def _q038_fast(impl, tmp):
    values = [(i * 37) % 101 - 50 for i in range(200_000)]
    k = 1_000
    prefix = [0, *itertools.accumulate(values)]
    best = max(range(len(values) - k + 1), key=lambda s: (prefix[s + k] - prefix[s], -s))
    expected = (prefix[best + k] - prefix[best], best)
    start = time.perf_counter()
    got = impl(values, k)
    elapsed = time.perf_counter() - start
    assert got == expected, f"expected {expected}, got {got}"
    assert elapsed < 0.5, f"took {elapsed:.2f}s; slide the window in O(n) instead of re-summing"


QUESTIONS = [
    # ------------------------------------------------------------------------ Q021
    Question(
        id=21,
        title="Collect All Pages from a Cursor-Paginated API",
        difficulty="Intermediate",
        category=FUND,
        roles=[DE, AIE, PYDEV],
        skills=["while loops", "loop termination", "defensive limits", "callables as parameters"],
        minutes=20,
        scenario=(
            "A vendor API returns search results one page at a time. Each response contains the "
            "page's `items` and a `next_cursor` to request the following page; the last page has "
            "`next_cursor = None`. A buggy deployment once returned the same cursor forever, "
            "hanging the ingestion job overnight."
        ),
        problem=(
            "Write `fetch_all(fetch_page, max_pages=100)` that calls `fetch_page(cursor)` "
            "repeatedly – starting with `cursor=None` – and returns all items from all pages in "
            "order. Protect the loop against runaway pagination."
        ),
        signature="def fetch_all(fetch_page, max_pages: int = 100) -> list:",
        entry="fetch_all",
        inputs=[
            "`fetch_page`: a function taking a cursor (`None` or `str`) and returning "
            "`{'items': list, 'next_cursor': str | None}`.",
            "`max_pages`: maximum number of pages you may request, `int` ≥ 1.",
        ],
        output="A `list` containing every item of every page, in page order.",
        constraints=[
            "The first call is `fetch_page(None)`; each later call uses the previous `next_cursor`.",
            "Stop when `next_cursor` is `None`. Pages may have empty `items`.",
            "Raise `RuntimeError` if a cursor is returned that was already requested (a loop).",
            "Raise `RuntimeError` if more than `max_pages` pages would be needed.",
            "Raise `ValueError` if `max_pages < 1`.",
        ],
        criteria=[
            "Collects all items in order and requests each cursor exactly once.",
            "Exactly `max_pages` pages is allowed; one more is an error.",
            "Detects repeated cursors.",
        ],
        hints=[
            "You do not know the number of pages in advance – which loop fits that?",
            "Keep the current cursor, a counter of pages fetched and a set of cursors already "
            "requested. Check the limits *before* requesting another page.",
            "validate max_pages\nitems, seen, cursor, pages = [], set(), None, 0\nwhile True:\n"
            "    if pages == max_pages: raise RuntimeError\n    seen.add(cursor); page = fetch_page(cursor); pages += 1\n"
            "    items.extend(page['items']); cursor = page['next_cursor']\n    if cursor is None: return items\n"
            "    if cursor in seen: raise RuntimeError",
        ],
        solution='''
def fetch_all(fetch_page, max_pages: int = 100) -> list:
    """Follow next_cursor links until the last page and return all items."""
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    items: list = []
    seen: set = set()
    cursor = None
    pages = 0
    while True:
        if pages == max_pages:
            raise RuntimeError(f"pagination did not finish within {max_pages} pages")
        seen.add(cursor)
        page = fetch_page(cursor)
        pages += 1
        items.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            return items
        if cursor in seen:
            raise RuntimeError(f"cursor {cursor!r} was returned twice; aborting to avoid a loop")
''',
        explanation=(
            "A `while True` loop with explicit exits models 'repeat until the server says stop'. "
            "Two independent guards – a page budget and a set of seen cursors – turn a silent "
            "infinite loop into a clear, early error."
        ),
        edge_cases=[
            "Single page with `next_cursor=None` → one request.",
            "Empty pages in the middle are skipped over naturally by `extend`.",
            "Exactly `max_pages` pages succeed; the check happens before each request.",
        ],
        mistakes=[
            "`while cursor:` – the first cursor is `None`, so the loop never runs.",
            "Checking the page limit after fetching, which requests one page too many.",
            "Using `append(page['items'])`, creating a list of lists.",
        ],
        tests=[
            T((THREE_PAGES,), [1, 2, 3, 4, 5], note="three pages"),
            CT(_q021_counts_calls, note="each cursor is requested exactly once, in order"),
            T((FakeAPI({None: {"items": [], "next_cursor": None}}),), [], note="single empty page"),
            T((FakeAPI({None: {"items": [], "next_cursor": "b"}, "b": {"items": ["x"], "next_cursor": None}}),),
              ["x"], note="empty first page"),
            T((THREE_PAGES, 3), [1, 2, 3, 4, 5], note="boundary: exactly max_pages pages"),
            T((THREE_PAGES, 2), raises="RuntimeError", note="more pages than allowed"),
            T((FakeAPI({None: {"items": [1], "next_cursor": "a"}, "a": {"items": [2], "next_cursor": "a"}}),),
              raises="RuntimeError", note="repeated cursor"),
            T((THREE_PAGES, 0), raises="ValueError", note="invalid max_pages"),
        ],
        examples=['''
>>> pages = {None: {"items": [1, 2], "next_cursor": "p2"},
...          "p2": {"items": [3], "next_cursor": None}}
>>> fetch_all(lambda cursor: pages[cursor])
[1, 2, 3]
>>> fetch_all(lambda cursor: pages[cursor], max_pages=1)
Traceback (most recent call last):
    ...
RuntimeError: pagination did not finish within 1 pages
'''],
        complexity="O(P + N) time for P pages and N items; O(N + P) space for the items and the seen cursors.",
        extension="Turn it into a generator `iter_items(fetch_page)` that yields items as pages arrive.",
    ),
    # ------------------------------------------------------------------------ Q022
    Question(
        id=22,
        title="Evaluate a Feature-Flag Rollout",
        difficulty="Intermediate",
        category=FUND,
        roles=[PYDEV, AAI, MLE],
        skills=["rule ordering", "optional dict keys", "deterministic hashing", "boolean logic"],
        minutes=20,
        scenario=(
            "A new recommendation model is released behind a feature flag. Product managers can "
            "allow-list or deny-list specific users, restrict the flag to some countries, and roll "
            "it out to a percentage of users. The same user must always get the same answer."
        ),
        problem=(
            "Write `is_feature_enabled(flag, user)` that applies the rules below **in order** and "
            "returns `True` or `False`."
        ),
        signature="def is_feature_enabled(flag: dict, user: dict) -> bool:",
        entry="is_feature_enabled",
        inputs=[
            "`flag`: `{'enabled': bool}` plus optional keys `'deny'` (list of user ids), "
            "`'allow'` (list of user ids), `'countries'` (list of country codes) and "
            "`'rollout_percent'` (int 0–100, default 100).",
            "`user`: `{'id': str, 'country': str}`.",
        ],
        output="A `bool`.",
        constraints=[
            "1. If `flag['enabled']` is false → `False`.",
            "2. If the user id is in `deny` → `False`.",
            "3. If the user id is in `allow` → `True` (skips the remaining rules).",
            "4. If `countries` is present and the user's country is not in it → `False`.",
            "5. Otherwise compute `bucket = sum(ord(c) for c in user['id']) % 100` and return "
            "`bucket < rollout_percent`.",
            "Raise `ValueError` if `rollout_percent` is outside 0–100.",
        ],
        criteria=[
            "Rules are applied in the stated priority order.",
            "Missing optional keys behave as 'no restriction' (rollout defaults to 100).",
            "Bucketing is deterministic and uses a strict `<`.",
        ],
        hints=[
            "Write the five rules as a sequence of early `return` statements.",
            "Use `flag.get('deny', [])` for optional lists, and `'countries' in flag` to know "
            "whether a country restriction exists at all.",
            "validate rollout = flag.get('rollout_percent', 100)\nif not flag['enabled']: return False\n"
            "if uid in flag.get('deny', []): return False\nif uid in flag.get('allow', []): return True\n"
            "if 'countries' in flag and country not in flag['countries']: return False\n"
            "return sum(map(ord, uid)) % 100 < rollout",
        ],
        solution='''
def is_feature_enabled(flag: dict, user: dict) -> bool:
    """Decide whether a feature flag is on for a user."""
    rollout = flag.get("rollout_percent", 100)
    if not 0 <= rollout <= 100:
        raise ValueError(f"rollout_percent must be within 0-100, got {rollout}")
    user_id = user["id"]
    if not flag["enabled"]:
        return False
    if user_id in flag.get("deny", []):
        return False
    if user_id in flag.get("allow", []):
        return True
    if "countries" in flag and user["country"] not in flag["countries"]:
        return False
    bucket = sum(ord(ch) for ch in user_id) % 100
    return bucket < rollout
''',
        explanation=(
            "Early returns make the precedence explicit and easy to audit. `dict.get` with a "
            "default treats missing optional keys as 'no restriction'. The bucket only depends "
            "on the user id, so each user consistently lands on the same side of the rollout."
        ),
        edge_cases=[
            "Deny beats allow when a user is on both lists.",
            "Allow-listed users bypass the country and rollout rules.",
            "`rollout_percent` 0 disables everyone (except allow-list); 100 enables everyone.",
        ],
        mistakes=[
            "Checking allow before deny.",
            "Treating a missing `countries` key as 'no countries allowed'.",
            "Using Python's `hash()`, which is randomized per process for strings.",
        ],
        tests=[
            T(({"enabled": True, "countries": ["DE", "FR"], "rollout_percent": 50}, {"id": "alice", "country": "DE"}),
              True, example=True, note="bucket('alice') = 10 < 50"),
            T(({"enabled": True, "rollout_percent": 66}, {"id": "u1", "country": "US"}), False, example=True,
              note="boundary: bucket('u1') = 66 is not < 66"),
            T(({"enabled": True, "rollout_percent": 67}, {"id": "u1", "country": "US"}), True, note="bucket 66 < 67"),
            T(({"enabled": False, "allow": ["u1"]}, {"id": "u1", "country": "US"}), False, note="disabled flag"),
            T(({"enabled": True, "deny": ["u1"], "allow": ["u1"]}, {"id": "u1", "country": "US"}), False,
              note="deny beats allow"),
            T(({"enabled": True, "allow": ["u1"], "countries": ["DE"], "rollout_percent": 0},
               {"id": "u1", "country": "US"}), True, note="allow-list bypasses other rules"),
            T(({"enabled": True, "countries": ["DE", "FR"]}, {"id": "alice", "country": "US"}), False,
              note="country not allowed"),
            T(({"enabled": True}, {"id": "bob", "country": "IN"}), True, note="defaults to 100%"),
            T(({"enabled": True, "rollout_percent": 0}, {"id": "bob", "country": "IN"}), False, note="0% rollout"),
            T(({"enabled": True, "rollout_percent": 120}, {"id": "bob", "country": "IN"}), raises="ValueError",
              note="invalid rollout"),
        ],
        complexity="O(len(id) + len(lists)) per call; convert lists to sets if they are large and reused.",
        extension="Return a reason string alongside the decision (e.g. `'denied'`, `'country'`, `'rollout'`) for debugging.",
    ),
    # ------------------------------------------------------------------------ Q023
    Question(
        id=23,
        title="Build a Confusion Matrix Without Aliasing Bugs",
        difficulty="Intermediate",
        category=FUND,
        roles=[DS_ROLE, MLE, AAI],
        skills=["mutable objects and aliasing", "nested lists", "zip", "index lookup with dicts"],
        minutes=20,
        scenario=(
            "An evaluation script builds a confusion matrix with `matrix = [[0] * n] * n`. The "
            "numbers it reports are nonsense: every row is identical. The reason is that all rows "
            "are the *same list object*."
        ),
        problem=(
            "Write `confusion_matrix(actual, predicted, labels)` returning an `n × n` list of "
            "lists where `matrix[i][j]` counts items whose actual label is `labels[i]` and "
            "predicted label is `labels[j]`."
        ),
        signature="def confusion_matrix(actual: list[str], predicted: list[str], labels: list[str]) -> list[list[int]]:",
        entry="confusion_matrix",
        inputs=[
            "`actual`, `predicted`: equal-length lists of label strings (may be empty).",
            "`labels`: list of distinct label strings that defines row/column order.",
        ],
        output="A list of `len(labels)` rows, each a list of `len(labels)` ints.",
        constraints=[
            "Rows are actual labels, columns are predicted labels, both in `labels` order.",
            "Each row must be an independent list object.",
            "Raise `ValueError` if the two lists differ in length or contain a label not in `labels`.",
        ],
        criteria=["Counts are correct.", "Rows are independent objects.", "Invalid input raises `ValueError`."],
        hints=[
            "Print `[[0] * 2] * 2` after setting one cell – what happens and why?",
            "Create rows with a comprehension (`[[0] * n for _ in range(n)]`) and map labels to "
            "indexes with a dict for O(1) lookup.",
            "check lengths\nindex = {label: i for i, label in enumerate(labels)}\n"
            "matrix = [[0] * n for _ in range(n)]\nfor a, p in zip(actual, predicted):\n"
            "    if a not in index or p not in index: raise ValueError\n    matrix[index[a]][index[p]] += 1",
        ],
        solution='''
def confusion_matrix(actual: list[str], predicted: list[str], labels: list[str]) -> list[list[int]]:
    """Count (actual, predicted) label pairs into an n x n matrix."""
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]
    for true_label, pred_label in zip(actual, predicted):
        if true_label not in index or pred_label not in index:
            raise ValueError(f"unknown label in pair ({true_label!r}, {pred_label!r})")
        matrix[index[true_label]][index[pred_label]] += 1
    return matrix
''',
        explanation=(
            "`[[0] * n] * n` copies a *reference* to one inner list n times. The comprehension "
            "evaluates `[0] * size` once per row, producing independent lists. A label → index "
            "dict avoids repeated `labels.index()` scans."
        ),
        edge_cases=["Empty inputs → an all-zero matrix.", "Labels never predicted → zero columns.", "Unknown labels raise `ValueError`."],
        mistakes=["`[[0] * n] * n` (shared rows).", "Swapping rows and columns.", "Using `labels.index()` inside the loop (O(n) each)."],
        tests=[
            T((["cat", "cat", "dog", "dog", "dog"], ["cat", "dog", "dog", "dog", "cat"], ["cat", "dog"]),
              [[1, 1], [1, 2]], example=True, note="binary case"),
            T(([], [], ["x", "y"]), [[0, 0], [0, 0]], example=True, note="boundary: empty inputs"),
            T((["a", "b", "c", "a"], ["a", "c", "c", "b"], ["a", "b", "c"]), [[1, 1, 0], [0, 0, 1], [0, 0, 1]],
              note="three classes"),
            CT(lambda impl, tmp: _assert_independent_rows(impl), note="rows are independent lists"),
            T((["a"], ["a", "b"], ["a", "b"]), raises="ValueError", note="length mismatch"),
            T((["a"], ["z"], ["a", "b"]), raises="ValueError", note="unknown label"),
        ],
        complexity="O(n² + m) time and O(n²) space for n labels and m items.",
        extension="Compute per-class precision and recall from the matrix.",
    ),
    # ------------------------------------------------------------------------ Q024
    Question(
        id=24,
        title="Parse Structured Application Log Lines",
        difficulty="Intermediate",
        category=STR,
        roles=[DE, PYDEV, AIE],
        skills=["regular expressions", "named groups", "fullmatch", "returning None for invalid input"],
        minutes=20,
        scenario=(
            "Services write logs such as "
            "`2024-05-01T12:00:03Z ERROR [payment-service] Timeout after 3000ms`. An incident "
            "dashboard needs those lines as structured records; corrupted or foreign lines must "
            "be skipped rather than crash the parser."
        ),
        problem=(
            "Write `parse_log_line(line)` that returns a dict with keys `timestamp`, `level`, "
            "`service` and `message`, or `None` if the line does not match the format exactly."
        ),
        signature="def parse_log_line(line: str) -> dict | None:",
        entry="parse_log_line",
        inputs=[
            "`line`: one log line, possibly with surrounding whitespace or a trailing newline.",
            "Format: `<timestamp> <LEVEL> [<service>] <message>` separated by single spaces.",
        ],
        output="`{'timestamp': str, 'level': str, 'service': str, 'message': str}` or `None`.",
        constraints=[
            "`timestamp` matches `YYYY-MM-DDTHH:MM:SSZ` (digits, literal `T` and `Z`).",
            "`level` is one of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (upper case).",
            "`service` consists of lower-case letters, digits and hyphens inside square brackets.",
            "`message` is the non-empty remainder of the line (it may contain brackets).",
            "Ignore surrounding whitespace; return `None` for anything else.",
        ],
        criteria=["Valid lines produce the correct dict.", "Messages containing `[...]` are kept intact.", "Malformed lines return `None`."],
        hints=[
            "A single regular expression that must match the *whole* line is the cleanest tool.",
            "Use `re.fullmatch` with named groups such as `(?P<level>DEBUG|INFO|...)` and call "
            "`.groupdict()` on the match.",
            "PATTERN = r'(?P<timestamp>\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z) "
            "(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) \\[(?P<service>[a-z0-9-]+)\\] (?P<message>.+)'\n"
            "m = re.fullmatch(PATTERN, line.strip())\nreturn m.groupdict() if m else None",
        ],
        solution='''
import re

LOG_PATTERN = re.compile(
    r"(?P<timestamp>\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z) "
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) "
    r"\\[(?P<service>[a-z0-9-]+)\\] "
    r"(?P<message>.+)"
)


def parse_log_line(line: str) -> dict | None:
    """Parse one structured log line, or return None if it is malformed."""
    match = LOG_PATTERN.fullmatch(line.strip())
    return match.groupdict() if match else None
''',
        explanation=(
            "`fullmatch` forces the pattern to describe the entire line, so partial matches "
            "never sneak through. Named groups document the structure and `groupdict()` builds "
            "the result directly. Compiling the pattern once at module level avoids recompiling "
            "it for every line."
        ),
        edge_cases=["Trailing newline is stripped.", "Brackets inside the message are part of the message.", "Missing message → `None`."],
        mistakes=["Using `re.match`, which accepts trailing garbage.", "Splitting on spaces, which breaks messages containing spaces.", "Using `.*` for the message, which accepts empty messages."],
        tests=[
            T(("2024-05-01T12:00:03Z ERROR [payment-service] Timeout after 3000ms",),
              {"timestamp": "2024-05-01T12:00:03Z", "level": "ERROR", "service": "payment-service",
               "message": "Timeout after 3000ms"}, example=True, note="typical line"),
            T(("garbage line",), None, example=True, note="boundary: malformed line"),
            T(("2024-05-01T12:00:04Z INFO [auth] User login ok\n",),
              {"timestamp": "2024-05-01T12:00:04Z", "level": "INFO", "service": "auth", "message": "User login ok"},
              note="trailing newline"),
            T(("2024-05-01T12:00:05Z WARNING [model-api] Retrying [attempt 2/3]",),
              {"timestamp": "2024-05-01T12:00:05Z", "level": "WARNING", "service": "model-api",
               "message": "Retrying [attempt 2/3]"}, note="brackets in message"),
            T(("",), None, note="empty line"),
            T(("2024-05-01T12:00:03Z VERBOSE [x] msg",), None, note="unknown level"),
            T(("2024-05-01 12:00:03 ERROR [x] msg",), None, note="wrong timestamp format"),
            T(("2024-05-01T12:00:05Z INFO [svc]",), None, note="missing message"),
            T(("2024-05-01T12:00:05Z INFO [Bad_Service] msg",), None, note="invalid service name"),
            CT(_q024_sample_file, note="data/sample_data/application_logs.txt: 34 valid lines, 9 errors"),
        ],
        complexity="O(L) per line for a line of length L.",
        extension="Also parse an optional trailing `key=value` section into a dict of extra fields.",
    ),
    # ------------------------------------------------------------------------ Q025
    Question(
        id=25,
        title="Top-K Tokens for a Text Report",
        difficulty="Intermediate",
        category=STR,
        roles=[AIE, DS_ROLE, AAI],
        skills=["regex tokenization", "Counter", "stopword filtering", "multi-key sorting"],
        minutes=20,
        scenario=(
            "Support tickets are summarized by their most frequent words before an analyst reads "
            "them. Words like *the* add noise, and contractions such as *don't* should stay one "
            "token."
        ),
        problem=(
            "Write `top_k_tokens(text, k, stopwords=())` that tokenizes the text, removes "
            "stopwords and returns the `k` most frequent tokens with their counts."
        ),
        signature="def top_k_tokens(text: str, k: int, stopwords=()) -> list[tuple[str, int]]:",
        entry="top_k_tokens",
        inputs=[
            "`text`: any string.",
            "`k`: number of tokens to return, `int` ≥ 0.",
            "`stopwords`: an iterable of words to ignore (any letter case).",
        ],
        output="A list of `(token, count)` tuples, most frequent first, length ≤ `k`.",
        constraints=[
            "Lower-case the text. A token is a run of `a–z`/`0–9` characters, optionally joined by "
            "single apostrophes inside the word (`don't`, `model's`). Regex: `[a-z0-9]+(?:'[a-z0-9]+)*`.",
            "Stopwords are compared case-insensitively.",
            "Sort by count descending, then token ascending.",
            "Raise `ValueError` if `k < 0`.",
        ],
        criteria=["Correct tokenization including apostrophes.", "Stopwords removed case-insensitively.", "Deterministic ordering for ties."],
        hints=[
            "Split the work into three steps: tokenize, filter, count-and-rank.",
            "`re.findall` gives the tokens; `collections.Counter` counts them. Use "
            "`sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))` for the ranking.",
            "tokens = re.findall(PATTERN, text.lower())\nstop = {w.lower() for w in stopwords}\n"
            "counts = Counter(t for t in tokens if t not in stop)\nreturn sorted(counts.items(), key=...)[:k]",
        ],
        solution='''
import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def top_k_tokens(text: str, k: int, stopwords=()) -> list[tuple[str, int]]:
    """Most frequent tokens (count desc, token asc), excluding stopwords."""
    if k < 0:
        raise ValueError("k must be >= 0")
    stop = {word.lower() for word in stopwords}
    counts = Counter(t for t in TOKEN_PATTERN.findall(text.lower()) if t not in stop)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:k]
''',
        explanation=(
            "The regex encodes the token definition precisely; a set makes stopword lookups O(1); "
            "sorting by `(-count, token)` gives a stable, deterministic tie-break that "
            "`Counter.most_common` does not guarantee."
        ),
        edge_cases=["Empty text → `[]`.", "Leading/trailing quotes are not part of tokens.", "`k = 0` → `[]`."],
        mistakes=["`text.split()` keeps punctuation (`timeout.` ≠ `timeout`).", "Relying on `most_common` for tie order.", "Comparing stopwords without lower-casing them."],
        tests=[
            T(("The model's accuracy improved. The model's latency did not improve; accuracy matters!", 3,
               {"the", "did", "not"}), [("accuracy", 2), ("model's", 2), ("improve", 1)], example=True,
              note="apostrophes, stopwords, ties"),
            T(("Error: timeout. ERROR: Timeout! error", 2), [("error", 3), ("timeout", 2)], example=True,
              note="case-insensitive counting"),
            T(("", 5), [], note="empty text"),
            T(("'quoted' words don't", 5), [("don't", 1), ("quoted", 1), ("words", 1)], note="edge apostrophes"),
            T(("The the THE cat", 5, ["The"]), [("cat", 1)], note="stopwords in any case"),
            T(("gpu0 gpu1 gpu0", 1), [("gpu0", 2)], note="digits in tokens"),
            T(("a b", 0), [], note="k = 0"),
            T(("a", -1), raises="ValueError", note="negative k"),
        ],
        complexity="O(n + u log u) for text length n and u unique tokens; `heapq.nsmallest` reduces ranking to O(u log k).",
        extension="Return bigram counts as well (pairs of consecutive non-stopword tokens).",
    ),
    # ------------------------------------------------------------------------ Q026
    Question(
        id=26,
        title="Render an LLM Prompt Template Safely",
        difficulty="Intermediate",
        category=STR,
        roles=[AIE, AAI, PYDEV],
        skills=["re.sub with a function", "KeyError messages", "avoiding recursive substitution"],
        minutes=20,
        scenario=(
            "Prompts for a summarization model are stored as templates such as "
            "`\"Summarize this {{ doc_type }} for a {{audience}}.\"`. User-provided values may "
            "themselves contain braces, so substitution must happen exactly once and a missing "
            "variable must fail loudly instead of sending a broken prompt."
        ),
        problem=(
            "Write `render_prompt(template, variables)` that replaces every `{{ name }}` "
            "placeholder with `str(variables[name])`."
        ),
        signature="def render_prompt(template: str, variables: dict) -> str:",
        entry="render_prompt",
        inputs=[
            "`template`: a string that may contain placeholders.",
            "`variables`: dict mapping names to values of any type.",
        ],
        output="The rendered `str`.",
        constraints=[
            "A placeholder is `{{`, optional spaces, a name matching `[A-Za-z_][A-Za-z0-9_]*`, optional spaces, `}}`.",
            "Anything that is not a valid placeholder (e.g. `{single}` or `{{ 1bad }}`) is left unchanged.",
            "Raise `KeyError` naming the variable if a placeholder has no value.",
            "Extra variables are ignored. Inserted values are **not** scanned for placeholders again.",
        ],
        criteria=["Whitespace inside braces is tolerated.", "Missing variable raises `KeyError`.", "Values containing `{{...}}` are inserted literally."],
        hints=[
            "Find placeholders with a regex and compute each replacement with a function.",
            "`re.sub(pattern, repl_function, template)` calls your function once per match and never "
            "re-scans inserted text.",
            "PATTERN = r'\\{\\{\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\}\\}'\n"
            "def repl(m):\n    name = m.group(1)\n    if name not in variables: raise KeyError(name)\n"
            "    return str(variables[name])\nreturn re.sub(PATTERN, repl, template)",
        ],
        solution='''
import re

PLACEHOLDER = re.compile(r"\\{\\{\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\}\\}")


def render_prompt(template: str, variables: dict) -> str:
    """Substitute {{ name }} placeholders exactly once."""

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in variables:
            raise KeyError(f"missing template variable: {name}")
        return str(variables[name])

    return PLACEHOLDER.sub(replace, template)
''',
        explanation=(
            "`re.sub` with a callback performs a single left-to-right pass, so inserted values are "
            "never interpreted as template syntax (preventing prompt-template injection). Raising "
            "`KeyError` keeps silent failures out of production prompts."
        ),
        edge_cases=["Same placeholder used twice.", "Value containing `{{b}}` stays literal.", "Invalid names are left untouched."],
        mistakes=["Calling `str.replace` in a loop over variables – inserted text can be replaced again.", "Using `str.format`, which breaks on literal braces such as JSON examples.", "Silently substituting an empty string for missing variables."],
        tests=[
            T(("Summarize this {{ doc_type }} for a {{audience}}.", {"doc_type": "report", "audience": "CFO", "unused": 1}),
              "Summarize this report for a CFO.", example=True, note="whitespace and extra variables"),
            T(("Hello {{name}}", {}), raises="KeyError", example=True, note="boundary: missing variable"),
            T(("No placeholders {here}", {}), "No placeholders {here}", note="single braces untouched"),
            T(("{{x}}{{x}}", {"x": 7}), "77", note="repeated placeholder, non-string value"),
            T(("{{a}}", {"a": "{{b}}", "b": "B"}), "{{b}}", note="no recursive substitution"),
            T(("{{ 1bad }} {{ok}}", {"ok": "yes"}), "{{ 1bad }} yes", note="invalid name left as-is"),
            T(("", {}), "", note="empty template"),
        ],
        complexity="O(n + output length) for a template of length n.",
        extension="Support a default value syntax `{{ name | default text }}`.",
    ),
    # ------------------------------------------------------------------------ Q027
    Question(
        id=27,
        title="Aggregate Completed Orders per Customer",
        difficulty="Intermediate",
        category=DS,
        roles=[DS_ROLE, DE, PYDEV],
        skills=["grouping with dicts", "nested dict updates", "filtering", "rounding"],
        minutes=20,
        scenario=(
            "Marketing wants a per-customer summary of completed orders to segment customers. "
            "Cancelled and pending orders must be excluded."
        ),
        problem=(
            "Write `group_orders_by_customer(orders)` that returns, for each customer with at "
            "least one completed order, the number of completed orders, their total amount and "
            "the average order amount."
        ),
        signature="def group_orders_by_customer(orders: list[dict]) -> dict[str, dict]:",
        entry="group_orders_by_customer",
        inputs=["`orders`: list of dicts with `order_id`, `customer_id`, `amount` (float ≥ 0) and `status` (str)."],
        output="`{customer_id: {'orders': int, 'total': float, 'average': float}}`.",
        constraints=[
            "Only orders with `status == 'completed'` count.",
            "Customers without completed orders are omitted.",
            "Round `total` and `average` to 2 decimals; compute the average from the unrounded total.",
            "Do not modify the input.",
        ],
        criteria=["Correct grouping and aggregates.", "Non-completed orders ignored.", "Empty input → `{}`."],
        hints=[
            "Accumulate count and sum per customer first; compute averages at the end.",
            "Use a dict of `[count, total]` pairs (or two dicts) while looping.",
            "acc = {}\nfor o in orders:\n    if o['status'] != 'completed': continue\n    c = acc.setdefault(o['customer_id'], [0, 0.0])\n"
            "    c[0] += 1; c[1] += o['amount']\nreturn {cid: {'orders': n, 'total': round(t, 2), 'average': round(t / n, 2)} ...}",
        ],
        solution='''
def group_orders_by_customer(orders: list[dict]) -> dict[str, dict]:
    """Order count, total and average amount per customer (completed orders only)."""
    totals: dict[str, list] = {}
    for order in orders:
        if order["status"] != "completed":
            continue
        entry = totals.setdefault(order["customer_id"], [0, 0.0])
        entry[0] += 1
        entry[1] += order["amount"]
    return {
        customer: {"orders": count, "total": round(total, 2), "average": round(total / count, 2)}
        for customer, (count, total) in totals.items()
    }
''',
        explanation="One pass accumulates the raw sums; derived metrics are computed once per customer, avoiding rounding drift.",
        edge_cases=["Only cancelled orders → `{}`.", "Single order → average equals total.", "Float sums such as 19.99 + 30.01 are rounded."],
        mistakes=["Averaging rounded values.", "Including customers with zero completed orders (division by zero).", "Counting cancelled orders."],
        tests=[
            T(([
                {"order_id": "o1", "customer_id": "c1", "amount": 19.99, "status": "completed"},
                {"order_id": "o2", "customer_id": "c2", "amount": 5.00, "status": "completed"},
                {"order_id": "o3", "customer_id": "c1", "amount": 30.01, "status": "completed"},
                {"order_id": "o4", "customer_id": "c3", "amount": 12.5, "status": "cancelled"},
                {"order_id": "o5", "customer_id": "c2", "amount": 7.30, "status": "completed"},
            ],), {"c1": {"orders": 2, "total": 50.0, "average": 25.0},
                  "c2": {"orders": 2, "total": 12.3, "average": 6.15}}, example=True, note="mixed statuses"),
            T(([],), {}, example=True, note="boundary: no orders"),
            T(([{"order_id": "o1", "customer_id": "c9", "amount": 3.0, "status": "pending"}],), {}, note="no completed orders"),
            T(([{"order_id": "o1", "customer_id": "c1", "amount": 10.0, "status": "completed"},
                {"order_id": "o2", "customer_id": "c1", "amount": 0.0, "status": "completed"},
                {"order_id": "o3", "customer_id": "c1", "amount": 5.0, "status": "completed"}],),
              {"c1": {"orders": 3, "total": 15.0, "average": 5.0}}, note="zero amount still counts"),
        ],
        complexity="O(n) time, O(c) space for c customers.",
        extension="Add `first_order_id` and `last_order_id` per customer based on input order.",
    ),
    # ------------------------------------------------------------------------ Q028
    Question(
        id=28,
        title="Flatten a Nested Configuration Dictionary",
        difficulty="Intermediate",
        category=DS,
        roles=[DE, MLE, PYDEV],
        skills=["recursion", "nested dicts", "key collisions", "isinstance"],
        minutes=25,
        scenario=(
            "Experiment trackers store hyper-parameters as flat key/value pairs, but your config "
            "files are nested YAML/JSON. You need to flatten `{'model': {'name': 'bert'}}` into "
            "`{'model.name': 'bert'}` before logging."
        ),
        problem=(
            "Write `flatten_dict(data, sep='.')` that flattens nested dictionaries into a single "
            "level, joining keys with `sep`."
        ),
        signature="def flatten_dict(data: dict, sep: str = \".\") -> dict:",
        entry="flatten_dict",
        inputs=["`data`: a dict with string keys whose values may be dicts (nested to any depth), lists or scalars.", "`sep`: separator string."],
        output="A new flat `dict`.",
        constraints=[
            "Only dicts are flattened; lists and other values are kept as-is.",
            "An empty nested dict is kept as a value: `{'a': {}}` → `{'a': {}}`.",
            "Raise `ValueError` if two different paths produce the same flat key (e.g. `{'a.b': 1, 'a': {'b': 2}}`).",
            "Do not modify the input.",
        ],
        criteria=["Arbitrary depth handled.", "Lists untouched.", "Collisions detected.", "Custom separator respected."],
        hints=[
            "Flattening a nested dict is the same problem one level down, with a key prefix.",
            "Write a helper `walk(node, prefix)` that recurses for non-empty dict values and "
            "assigns other values into the result, checking for existing keys.",
            "def walk(node, prefix):\n    for key, value in node.items():\n        flat = f'{prefix}{sep}{key}' if prefix else key\n"
            "        if isinstance(value, dict) and value: walk(value, flat)\n        else:\n"
            "            if flat in result: raise ValueError\n            result[flat] = value",
        ],
        solution='''
def flatten_dict(data: dict, sep: str = ".") -> dict:
    """Flatten nested dicts into {'a.b.c': value} form."""
    result: dict = {}

    def walk(node: dict, prefix: str) -> None:
        for key, value in node.items():
            flat_key = f"{prefix}{sep}{key}" if prefix else key
            if isinstance(value, dict) and value:
                walk(value, flat_key)
            else:
                if flat_key in result:
                    raise ValueError(f"duplicate flattened key: {flat_key!r}")
                result[flat_key] = value

    walk(data, "")
    return result
''',
        explanation=(
            "The recursive helper carries the key prefix down the tree and writes leaves into "
            "one shared result dict, which also makes collision detection a simple membership test."
        ),
        edge_cases=["`{}` → `{}`.", "Empty nested dict kept as `{}`.", "Collisions raise `ValueError`."],
        mistakes=["Flattening lists into `a.0`, `a.1` (not requested).", "Silently overwriting colliding keys.", "Dropping empty nested dicts."],
        tests=[
            T(({"model": {"name": "bert", "params": {"layers": 12, "heads": 8}}, "metrics": {"f1": 0.91},
                "tags": ["nlp", "prod"]},),
              {"model.name": "bert", "model.params.layers": 12, "model.params.heads": 8, "metrics.f1": 0.91,
               "tags": ["nlp", "prod"]}, example=True, note="nested config"),
            T(({"a": {}},), {"a": {}}, example=True, note="boundary: empty nested dict"),
            T(({},), {}, note="empty input"),
            T(({"a": {"b": {"c": 1}}}, "/"), {"a/b/c": 1}, note="custom separator"),
            T(({"a.b": 1, "a": {"b": 2}},), raises="ValueError", note="key collision"),
            T(({"x": None, "y": {"z": [1, {"k": 2}]}},), {"x": None, "y.z": [1, {"k": 2}]}, note="dicts inside lists untouched"),
        ],
        complexity="O(total number of keys) time; recursion depth equals nesting depth.",
        extension="Write `unflatten_dict(flat, sep='.')` and check `unflatten(flatten(d)) == d`.",
    ),
    # ------------------------------------------------------------------------ Q029
    Question(
        id=29,
        title="Count Events per User per Day with Tuple Keys",
        difficulty="Intermediate",
        category=DS,
        roles=[DS_ROLE, DE, MLE],
        skills=["tuple dictionary keys", "datetime parsing", "skipping invalid records"],
        minutes=20,
        scenario=(
            "To build a daily-activity feature, you count how many events each user produced on "
            "each calendar day. Some events from the mobile SDK arrive without a user id or with "
            "a malformed timestamp."
        ),
        problem=(
            "Write `count_by_user_day(events)` returning a dict keyed by `(user_id, 'YYYY-MM-DD')` "
            "with event counts."
        ),
        signature="def count_by_user_day(events: list[dict]) -> dict[tuple[str, str], int]:",
        entry="count_by_user_day",
        inputs=["`events`: list of dicts with optional keys `user_id` (str) and `timestamp` (ISO-8601 string such as `'2024-03-01T09:15:00Z'`)."],
        output="`dict[(user_id, date_string), int]`.",
        constraints=[
            "The date is the first 10 characters of the timestamp and must be a valid `YYYY-MM-DD` date.",
            "Skip events with a missing/`None`/empty `user_id`, a missing timestamp, or an invalid date.",
            "Do not modify the input.",
        ],
        criteria=["Correct counts keyed by tuples.", "Invalid events skipped without crashing."],
        hints=[
            "Tuples are hashable, so `(user, day)` can be a dict key directly.",
            "Validate the date with `datetime.strptime(ts[:10], '%Y-%m-%d')` inside `try/except ValueError`.",
            "for e in events:\n    user, ts = e.get('user_id'), e.get('timestamp')\n    if not user or not ts: continue\n"
            "    try: strptime(ts[:10], '%Y-%m-%d') except ValueError: continue\n    key = (user, ts[:10]); counts[key] = counts.get(key, 0) + 1",
        ],
        solution='''
from datetime import datetime


def count_by_user_day(events: list[dict]) -> dict[tuple[str, str], int]:
    """Count events per (user_id, calendar day)."""
    counts: dict[tuple[str, str], int] = {}
    for event in events:
        user_id = event.get("user_id")
        timestamp = event.get("timestamp")
        if not user_id or not timestamp:
            continue
        day = timestamp[:10]
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            continue
        key = (user_id, day)
        counts[key] = counts.get(key, 0) + 1
    return counts
''',
        explanation="Tuple keys express the composite grouping without string concatenation; `strptime` rejects impossible dates such as `2024-02-30`.",
        edge_cases=["Events right after midnight belong to the new day.", "Impossible dates (Feb 30) skipped.", "Missing user skipped."],
        mistakes=["Building keys like `user + '_' + day` (ambiguous if ids contain `_`).", "Indexing `event['user_id']` (KeyError).", "Trusting `ts[:10]` without validation."],
        tests=[
            T(([
                {"user_id": "u1", "timestamp": "2024-03-01T09:15:00Z"},
                {"user_id": "u1", "timestamp": "2024-03-01T17:40:00Z"},
                {"user_id": "u2", "timestamp": "2024-03-01T08:00:00Z"},
                {"user_id": "u1", "timestamp": "2024-03-02T00:05:00Z"},
                {"user_id": None, "timestamp": "2024-03-02T01:00:00Z"},
                {"user_id": "u2"},
            ],), {("u1", "2024-03-01"): 2, ("u2", "2024-03-01"): 1, ("u1", "2024-03-02"): 1}, example=True,
              note="valid and invalid events"),
            T(([],), {}, example=True, note="boundary: no events"),
            T(([{"user_id": "u1", "timestamp": "03/01/2024 10:00"}, {"user_id": "u1", "timestamp": "2024-02-30T10:00:00Z"}],),
              {}, note="malformed and impossible dates"),
            T(([{"user_id": "", "timestamp": "2024-03-01T10:00:00Z"}, {"user_id": "u3", "timestamp": "2024-03-01"}],),
              {("u3", "2024-03-01"): 1}, note="empty user id skipped; date-only timestamp ok"),
        ],
        complexity="O(n) time, O(k) space for k distinct (user, day) pairs.",
        extension="Convert timestamps with offsets such as `+05:30` to UTC before taking the date.",
    ),
    # ------------------------------------------------------------------------ Q030
    Question(
        id=30,
        title="Diff Two Snapshots of a Product Catalog",
        difficulty="Intermediate",
        category=DS,
        roles=[DE, DS_ROLE, PYDEV],
        skills=["dict indexing by key", "set operations", "duplicate detection", "dict equality"],
        minutes=20,
        scenario=(
            "Each night the product catalog is exported. A change-data job compares yesterday's "
            "and today's snapshots to emit added, removed and changed products to downstream "
            "systems."
        ),
        problem=(
            "Write `diff_records(old, new, key='id')` that returns which keys were added, removed "
            "or changed between two lists of records."
        ),
        signature="def diff_records(old: list[dict], new: list[dict], key: str = \"id\") -> dict[str, list]:",
        entry="diff_records",
        inputs=[
            "`old`, `new`: lists of dicts; every record contains `key`. Key values within one list are of one sortable type.",
            "`key`: name of the identifying field.",
        ],
        output="`{'added': [...], 'removed': [...], 'changed': [...]}` – each list of key values sorted ascending.",
        constraints=[
            "A record is *changed* if it exists in both snapshots and the two dicts are not equal "
            "(including added or removed fields).",
            "Raise `ValueError` if a key value appears twice within the same list.",
            "Do not modify the inputs.",
        ],
        criteria=["Added/removed/changed computed correctly.", "Sorted output.", "Duplicates rejected."],
        hints=[
            "Index each snapshot as `{key_value: record}` first.",
            "Then `new_keys - old_keys` gives added keys, the reverse gives removed ones, and the "
            "intersection is where you compare records.",
            "old_by = index(old); new_by = index(new)   # raise on duplicates\nadded = sorted(new_by.keys() - old_by.keys())\n"
            "removed = sorted(old_by.keys() - new_by.keys())\nchanged = sorted(k for k in old_by.keys() & new_by.keys() if old_by[k] != new_by[k])",
        ],
        solution='''
def _index(records: list[dict], key: str) -> dict:
    indexed = {}
    for record in records:
        value = record[key]
        if value in indexed:
            raise ValueError(f"duplicate key {value!r}")
        indexed[value] = record
    return indexed


def diff_records(old: list[dict], new: list[dict], key: str = "id") -> dict[str, list]:
    """Compare two snapshots and list added, removed and changed keys."""
    old_by_key = _index(old, key)
    new_by_key = _index(new, key)
    shared = old_by_key.keys() & new_by_key.keys()
    return {
        "added": sorted(new_by_key.keys() - old_by_key.keys()),
        "removed": sorted(old_by_key.keys() - new_by_key.keys()),
        "changed": sorted(k for k in shared if old_by_key[k] != new_by_key[k]),
    }
''',
        explanation="Indexing turns an O(n·m) nested comparison into O(n + m); dict key views support set operators directly.",
        edge_cases=["Identical snapshots → three empty lists.", "A new field counts as a change.", "Duplicate keys → `ValueError`."],
        mistakes=["Comparing by list position instead of key.", "Nested loops over both lists.", "Silently keeping the last duplicate."],
        tests=[
            T(([{"id": 1, "price": 9.99, "stock": 3}, {"id": 2, "price": 5.0, "stock": 0}, {"id": 3, "price": 1.5, "stock": 10}],
               [{"id": 1, "price": 9.99, "stock": 3}, {"id": 3, "price": 1.75, "stock": 10}, {"id": 4, "price": 2.0, "stock": 1}]),
              {"added": [4], "removed": [2], "changed": [3]}, example=True, note="all three kinds"),
            T(([], []), {"added": [], "removed": [], "changed": []}, example=True, note="boundary: empty snapshots"),
            T(([{"id": "a", "x": 1}], [{"id": "a", "x": 1, "y": None}]), {"added": [], "removed": [], "changed": ["a"]},
              note="new field counts as change"),
            T(([{"sku": "b"}, {"sku": "a"}], [{"sku": "c"}], "sku"), {"added": ["c"], "removed": ["a", "b"], "changed": []},
              note="custom key, sorted output"),
            T(([{"id": 1}, {"id": 1}], []), raises="ValueError", note="duplicate key"),
        ],
        complexity="O((n + m) log(n + m)) because of sorting; indexing and set operations are O(n + m).",
        extension="For changed records, also report which fields changed: `{'id': 3, 'fields': ['price']}`.",
    ),
    # ------------------------------------------------------------------------ Q031
    Question(
        id=31,
        title="Safely Read Values from Nested API Responses",
        difficulty="Intermediate",
        category=DS,
        roles=[AIE, DE, PYDEV],
        skills=["nested data traversal", "type checks", "None vs missing", "string parsing"],
        minutes=20,
        scenario=(
            "A search API returns deeply nested JSON. Writing `resp['data']['items'][0]['scores']"
            "['relevance']` crashes whenever any level is missing. You want a helper that takes a "
            "dotted path and returns a default instead of raising."
        ),
        problem=(
            "Write `get_path(data, path, default=None)` that follows a dot-separated path through "
            "dicts and lists and returns the value found, or `default` if the path does not exist."
        ),
        signature="def get_path(data, path: str, default=None):",
        entry="get_path",
        inputs=[
            "`data`: any JSON-like value (dicts, lists, scalars).",
            "`path`: segments separated by `.`; `''` means the data itself.",
            "`default`: returned when the path cannot be followed.",
        ],
        output="The value at the path (which may legitimately be `None`), or `default`.",
        constraints=[
            "On a dict, a segment is used as a string key.",
            "On a list, a segment must consist only of digits and is used as a non-negative index.",
            "Only dicts and lists are traversed; any other value (including strings) ends the path → `default`.",
            "If the value exists and is `None`, return `None` – not `default`.",
            "Never raise for missing keys, bad indices or wrong types.",
        ],
        criteria=["Correct traversal through dicts and lists.", "Missing paths return `default`.", "Existing `None` values are returned."],
        hints=[
            "Walk the path one segment at a time, keeping a `current` value.",
            "At each step check `isinstance(current, dict)` / `isinstance(current, list)` and "
            "whether the segment is usable (`key in current`, `segment.isdigit()` and in range).",
            "if path == '': return data\ncurrent = data\nfor seg in path.split('.'):\n    if dict and seg in current: current = current[seg]\n"
            "    elif list and seg.isdigit() and int(seg) < len(current): current = current[int(seg)]\n    else: return default\nreturn current",
        ],
        solution='''
def get_path(data, path: str, default=None):
    """Follow a dotted path through dicts/lists; return default if it is missing."""
    if path == "":
        return data
    current = data
    for segment in path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        elif isinstance(current, list) and segment.isdigit() and int(segment) < len(current):
            current = current[int(segment)]
        else:
            return default
    return current
''',
        explanation=(
            "Each step checks both the container type and whether the segment exists before "
            "indexing, so no exception can occur. Because existence is checked explicitly, a "
            "stored `None` is distinguishable from a missing key."
        ),
        edge_cases=["Index beyond list length → default.", "Digit keys on dicts are treated as string keys.", "Strings are not traversed."],
        mistakes=["Using `current.get(seg) or default`, which replaces stored `None`/`0` with the default.", "`try/except Exception` around everything, hiding real bugs.", "Allowing negative indices via `int('-1')`."],
        tests=[
            T(({"data": {"items": [{"id": "a1", "scores": {"relevance": 0.92}}, {"id": "a2", "scores": {}}]}},
               "data.items.0.scores.relevance"), 0.92, example=True, note="dicts and list index"),
            T(({"data": {"items": [{"id": "a1", "scores": {"relevance": 0.92}}, {"id": "a2", "scores": {}}]}},
               "data.items.1.scores.relevance", 0.0), 0.0, example=True, note="boundary: missing key → default"),
            T(({"data": {"items": [{"id": "a1"}]}}, "data.items.5.id"), None, note="index out of range"),
            T(({"meta": {"page": 1}}, "meta.page.value", "n/a"), "n/a", note="cannot traverse an int"),
            T(({"a": None}, "a", "d"), None, note="stored None is returned"),
            T(({"0": "zero"}, "0"), "zero", note="digit key on a dict"),
            T(({"s": "hello"}, "s.0", "d"), "d", note="strings are not traversed"),
            T(({"x": [10, 20]}, "x.-1", "d"), "d", note="negative index not supported"),
            T(({"k": 1}, ""), {"k": 1}, note="empty path returns the data"),
        ],
        complexity="O(p) for a path with p segments.",
        extension="Support a wildcard segment `*` that collects the value from every list element.",
    ),
    # ------------------------------------------------------------------------ Q032
    Question(
        id=32,
        title="Lazy Batching Generator",
        difficulty="Intermediate",
        category=FUNC,
        roles=[MLE, DE, AIE],
        skills=["generators", "itertools.islice", "lazy evaluation", "eager argument validation"],
        minutes=25,
        scenario=(
            "Embedding requests must be sent in batches of at most 64 texts. The input can be a "
            "huge generator that streams rows from a file, so the batching helper must never "
            "load everything into memory."
        ),
        problem=(
            "Write `batched(iterable, size)` that returns an iterator yielding lists of up to "
            "`size` consecutive items. Invalid sizes must be rejected **immediately when "
            "`batched` is called**, not later when iteration starts."
        ),
        signature="def batched(iterable, size: int):",
        entry="batched",
        inputs=["`iterable`: any iterable, possibly infinite.", "`size`: positive `int`."],
        output="An iterator of lists; every list has `size` items except possibly the last.",
        constraints=[
            "Must be lazy: work with infinite iterators and consume only what is needed.",
            "Raise `ValueError` at call time if `size < 1`.",
            "Never yield an empty list.",
        ],
        criteria=["Correct batches.", "Works on infinite iterators.", "Eager validation."],
        hints=[
            "A function containing `yield` doesn't run until the first `next()` – so where should the validation live?",
            "Validate in an ordinary outer function, then `return` an inner generator. Inside, "
            "`list(itertools.islice(iterator, size))` grabs the next batch.",
            "def batched(iterable, size):\n    if size < 1: raise ValueError\n    def gen():\n        it = iter(iterable)\n"
            "        while batch := list(islice(it, size)):\n            yield batch\n    return gen()",
        ],
        solution='''
from itertools import islice


def batched(iterable, size: int):
    """Yield lists of up to ``size`` items; validates ``size`` eagerly."""
    if size < 1:
        raise ValueError("size must be >= 1")

    def generate():
        iterator = iter(iterable)
        while batch := list(islice(iterator, size)):
            yield batch

    return generate()
''',
        explanation=(
            "Generator functions defer *all* code until iteration, including validation. Splitting "
            "into an eager outer function and a lazy inner generator gives immediate errors with "
            "lazy data processing. `islice` takes at most `size` items from the shared iterator."
        ),
        edge_cases=["Empty iterable → no batches.", "Length divisible by size → no partial batch.", "Infinite input works lazily."],
        mistakes=["Putting the `raise` inside the generator body (error appears late).", "Calling `list(iterable)` first (breaks on infinite input).", "Yielding a final empty list."],
        tests=[
            T((range(7), 3), [[0, 1, 2], [3, 4, 5], [6]], materialize=True, note="partial last batch"),
            T(([], 4), [], materialize=True, note="empty input"),
            T((["a", "b"], 2), [["a", "b"]], materialize=True, note="exact size"),
            CT(_q032_generator_input, note="generator input"),
            CT(_q032_call_time_validation, note="ValueError raised at call time"),
            CT(_q032_lazy, note="lazy on an infinite iterator"),
        ],
        examples=['''
>>> list(batched(range(7), 3))
[[0, 1, 2], [3, 4, 5], [6]]
>>> list(batched([], 4))
[]
>>> batched([1, 2], 0)
Traceback (most recent call last):
    ...
ValueError: size must be >= 1
'''],
        complexity="O(n) total time; O(size) memory at any moment.",
        extension="Add a `max_tokens` budget so a batch closes early once the summed `len(text)` would exceed it.",
    ),
    # ------------------------------------------------------------------------ Q033
    Question(
        id=33,
        title="A Retry Decorator for Flaky Calls",
        difficulty="Intermediate",
        category=FUNC,
        roles=[AIE, DE, PYDEV],
        skills=["decorators with arguments", "closures", "functools.wraps", "exception handling"],
        minutes=25,
        scenario=(
            "Calls to a vector database occasionally fail with `ConnectionError`. Wrapping every "
            "call site in a retry loop duplicates code, so the team wants a reusable decorator."
        ),
        problem=(
            "Write `retry(max_attempts=3, exceptions=(Exception,))`, a decorator factory. The "
            "decorated function is called up to `max_attempts` times while it raises one of the "
            "given exception types."
        ),
        signature="def retry(max_attempts: int = 3, exceptions: tuple = (Exception,)):",
        entry="retry",
        inputs=["`max_attempts`: total attempts including the first, `int` ≥ 1.", "`exceptions`: tuple of exception classes that should trigger a retry."],
        output="A decorator; the decorated function returns the wrapped function's first successful result.",
        constraints=[
            "Pass all positional and keyword arguments through unchanged.",
            "If all attempts fail, re-raise the exception from the **last** attempt.",
            "Exceptions not in `exceptions` propagate immediately without retrying.",
            "Preserve the wrapped function's `__name__` and `__doc__` (use `functools.wraps`).",
            "Raise `ValueError` when `retry()` is called with `max_attempts < 1`.",
            "No sleeping between attempts (keep it testable).",
        ],
        criteria=["Recovers from transient failures.", "Gives up after the limit.", "Does not retry other exceptions.", "Keeps metadata."],
        hints=[
            "There are three nested functions: the factory, the decorator, and the wrapper.",
            "In the wrapper, loop over `range(max_attempts)`, `return` on success, and catch only "
            "`exceptions`; after the final attempt, `raise` again.",
            "def retry(max_attempts=3, exceptions=(Exception,)):\n    validate\n    def decorator(func):\n        @wraps(func)\n"
            "        def wrapper(*args, **kwargs):\n            for attempt in range(1, max_attempts + 1):\n                try: return func(*args, **kwargs)\n"
            "                except exceptions:\n                    if attempt == max_attempts: raise\n        return wrapper\n    return decorator",
        ],
        solution='''
import functools


def retry(max_attempts: int = 3, exceptions: tuple = (Exception,)):
    """Retry the decorated function on the given exception types."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == max_attempts:
                        raise
        return wrapper

    return decorator
''',
        explanation=(
            "The factory captures configuration, the decorator captures the function, and the "
            "wrapper runs the loop. A bare `raise` inside the `except` block re-raises the current "
            "(last) exception with its original traceback."
        ),
        edge_cases=["`max_attempts=1` behaves like no retry.", "Non-retryable exceptions escape after one call.", "Return values of `None` are still successes."],
        mistakes=["Catching `Exception` regardless of the `exceptions` argument.", "Forgetting `functools.wraps`.", "Swallowing the final exception and returning `None`."],
        tests=[
            CT(_q033_recovers, note="succeeds on the third attempt"),
            CT(_q033_gives_up, note="re-raises the last exception after max_attempts"),
            CT(_q033_non_retryable, note="other exceptions are not retried"),
            CT(_q033_wraps, note="functools.wraps keeps metadata"),
            CT(_q033_validation, note="max_attempts < 1 rejected"),
        ],
        examples=['''
>>> calls = []
>>> @retry(max_attempts=3, exceptions=(ConnectionError,))
... def query():
...     calls.append(1)
...     if len(calls) < 3:
...         raise ConnectionError("temporary")
...     return "results"
>>> query()
'results'
>>> len(calls)
3
'''],
        complexity="O(max_attempts) calls in the worst case; O(1) extra space.",
        extension="Add exponential backoff with an injectable `sleep` function so tests stay fast.",
    ),
    # ------------------------------------------------------------------------ Q034
    Question(
        id=34,
        title="Moving Average with a Closure",
        difficulty="Intermediate",
        category=FUNC,
        roles=[MLE, DS_ROLE, AIE],
        skills=["closures", "nonlocal", "collections.deque", "O(1) running totals"],
        minutes=20,
        scenario=(
            "A training loop logs a smoothed loss: the average of the last *N* batch losses. "
            "The smoothing function is called once per batch for millions of batches, so each "
            "update must be constant time."
        ),
        problem=(
            "Write `make_moving_average(window)` that returns a function `add(value)`. Each call "
            "records `value` and returns the mean of the most recent `window` values (or of all "
            "values so far if fewer have been seen)."
        ),
        signature="def make_moving_average(window: int):",
        entry="make_moving_average",
        inputs=["`window`: positive `int`.", "The returned `add(value)` receives ints or floats."],
        output="`make_moving_average` returns a callable; `add(value)` returns a `float`.",
        constraints=[
            "Each returned function has its own independent state.",
            "Each update must be O(1) – do not re-sum the whole window.",
            "Raise `ValueError` when `window < 1`.",
        ],
        criteria=["Correct averages during warm-up and steady state.", "Independent instances.", "Fast updates."],
        hints=[
            "The inner function can remember data defined in the outer function (a closure).",
            "Keep a `deque(maxlen=window)` *and* a running total. Subtract the value that is about "
            "to fall out of the window before appending.",
            "values = deque(); total = 0\ndef add(x):\n    nonlocal total\n    if len(values) == window: total -= values.popleft()\n"
            "    values.append(x); total += x\n    return total / len(values)\nreturn add",
        ],
        solution='''
from collections import deque


def make_moving_average(window: int):
    """Return add(value) -> mean of the last ``window`` values."""
    if window < 1:
        raise ValueError("window must be >= 1")
    values: deque = deque()
    total = 0

    def add(value: float) -> float:
        nonlocal total
        if len(values) == window:
            total -= values.popleft()
        values.append(value)
        total += value
        return total / len(values)

    return add
''',
        explanation=(
            "The closure keeps `values` and `total` private to each returned function. "
            "`nonlocal` lets the inner function rebind `total`. Maintaining a running sum makes "
            "every update O(1) regardless of window size. (For very long float streams, "
            "periodically recomputing the sum limits accumulated rounding error.)"
        ),
        edge_cases=["Warm-up: fewer than `window` values.", "`window=1` returns the latest value.", "Separate instances do not interfere."],
        mistakes=["`sum(values) / len(values)` on every call – O(window).", "Forgetting `nonlocal` (UnboundLocalError).", "Module-level state shared by all instances."],
        tests=[
            CT(_q034_basic, note="warm-up and steady state"),
            CT(_q034_window_one, note="window of one"),
            CT(_q034_independent, note="independent instances"),
            CT(_q034_invalid, note="window < 1 rejected"),
            CT(_q034_fast, note="O(1) updates with a large window"),
        ],
        examples=['''
>>> avg = make_moving_average(3)
>>> [avg(x) for x in [10, 20, 30, 40]]
[10.0, 15.0, 20.0, 30.0]
'''],
        complexity="O(1) time per update; O(window) space.",
        extension="Return an object with `add`, `value` and `reset` methods instead of a bare function.",
    ),
    # ------------------------------------------------------------------------ Q035
    Question(
        id=35,
        title="Multi-Key Sort with Mixed Directions",
        difficulty="Intermediate",
        category=FUNC,
        roles=[DS_ROLE, PYDEV, DE],
        skills=["key functions", "lambda", "sort stability", "None handling"],
        minutes=25,
        scenario=(
            "A model-comparison table lets users sort by several columns, e.g. *score descending, "
            "then latency ascending*. Some models have not been scored yet (`None`) and must "
            "always appear at the bottom."
        ),
        problem=(
            "Write `sort_records(records, order_by)` that returns a new list sorted by the given "
            "`(field, direction)` pairs, where direction is `'asc'` or `'desc'`."
        ),
        signature="def sort_records(records: list[dict], order_by: list[tuple[str, str]]) -> list[dict]:",
        entry="sort_records",
        inputs=[
            "`records`: list of dicts. A field may be missing or `None`.",
            "`order_by`: list of `(field, 'asc'|'desc')`, most important first. May be empty.",
        ],
        output="A new sorted `list` of the same dict objects.",
        constraints=[
            "The first pair is the primary key, the second breaks ties, and so on.",
            "Missing/`None` values sort **last** for that field, regardless of direction.",
            "Records that tie on every key keep their original relative order (stable).",
            "Raise `ValueError` for a direction other than `'asc'`/`'desc'`.",
            "Do not modify the input list.",
        ],
        criteria=["Correct multi-key ordering.", "None values last in both directions.", "Stable ties.", "Validation."],
        hints=[
            "Python's sort is stable – sorting by the *least* important key first and the most "
            "important key last gives a multi-key sort.",
            "For each key (in reverse order), separate records with a value from those without, "
            "sort the first group with `reverse=(direction == 'desc')`, then append the missing ones.",
            "result = list(records)\nfor field, direction in reversed(order_by):\n    validate direction\n"
            "    present = [r for r in result if r.get(field) is not None]\n    missing = [r for r in result if r.get(field) is None]\n"
            "    present.sort(key=lambda r: r[field], reverse=direction == 'desc')\n    result = present + missing\nreturn result",
        ],
        solution='''
def sort_records(records: list[dict], order_by: list[tuple[str, str]]) -> list[dict]:
    """Stable multi-key sort; None/missing values always sort last."""
    for _, direction in order_by:
        if direction not in ("asc", "desc"):
            raise ValueError(f"direction must be 'asc' or 'desc', got {direction!r}")
    result = list(records)
    for field, direction in reversed(order_by):
        present = [r for r in result if r.get(field) is not None]
        missing = [r for r in result if r.get(field) is None]
        present.sort(key=lambda r: r[field], reverse=direction == "desc")
        result = present + missing
    return result
''',
        explanation=(
            "Each pass is stable, so earlier passes (less important keys) survive as tie-breakers. "
            "`reverse=True` in Python keeps equal elements in their original order, so stability "
            "holds for descending keys too. Splitting out `None` avoids comparing `None` with numbers."
        ),
        edge_cases=["Empty `order_by` → a copy in original order.", "All values `None` → original order.", "Ties on all keys keep input order."],
        mistakes=["Negating values for descending order (fails for strings).", "Sorting in place and mutating the caller's list.", "Comparing `None < 0.8` (TypeError)."],
        tests=[
            T(([{"name": "lr", "score": 0.81, "latency": 12}, {"name": "xgb", "score": 0.88, "latency": 40},
                {"name": "bert", "score": 0.88, "latency": 95}, {"name": "rf", "score": None, "latency": 30},
                {"name": "svm", "score": 0.81, "latency": 12}], [("score", "desc"), ("latency", "asc")]),
              [{"name": "xgb", "score": 0.88, "latency": 40}, {"name": "bert", "score": 0.88, "latency": 95},
               {"name": "lr", "score": 0.81, "latency": 12}, {"name": "svm", "score": 0.81, "latency": 12},
               {"name": "rf", "score": None, "latency": 30}], example=True, note="desc then asc; None last; stable tie"),
            T(([{"name": "lr", "latency": 12}, {"name": "svm", "latency": 12}, {"name": "rf", "latency": 30}],
               [("latency", "asc"), ("name", "desc")]),
              [{"name": "svm", "latency": 12}, {"name": "lr", "latency": 12}, {"name": "rf", "latency": 30}],
              note="string key descending"),
            T(([{"v": 2}, {"v": 1}], []), [{"v": 2}, {"v": 1}], example=True, note="boundary: no sort keys"),
            T(([{"v": None}, {"v": 1}, {}, {"v": 0}], [("v", "asc")]), [{"v": 0}, {"v": 1}, {"v": None}, {}],
              note="None and missing last for asc"),
            T(([{"v": None}, {"v": 1}, {"v": 3}], [("v", "desc")]), [{"v": 3}, {"v": 1}, {"v": None}],
              note="None last for desc"),
            T(([{"v": 1}], [("v", "up")]), raises="ValueError", note="invalid direction"),
        ],
        complexity="O(k · n log n) for k sort keys.",
        extension="Accept a custom key function per field, e.g. case-insensitive name sorting.",
    ),
    # ------------------------------------------------------------------------ Q036
    Question(
        id=36,
        title="Count Events in a Time Range with Binary Search",
        difficulty="Intermediate",
        category=ALG,
        roles=[DE, DS_ROLE, MLE],
        skills=["binary search", "bisect_left/bisect_right", "inclusive ranges", "complexity"],
        minutes=20,
        scenario=(
            "A monitoring service keeps a sorted list of request timestamps (epoch seconds) per "
            "endpoint and answers thousands of 'how many requests between t1 and t2?' queries per "
            "second. A linear scan per query is too slow."
        ),
        problem=(
            "Write `count_in_range(sorted_timestamps, start, end)` that returns how many "
            "timestamps `t` satisfy `start <= t <= end`, in O(log n) time."
        ),
        signature="def count_in_range(sorted_timestamps: list[int], start: int, end: int) -> int:",
        entry="count_in_range",
        inputs=["`sorted_timestamps`: list of ints sorted ascending (duplicates allowed).", "`start`, `end`: ints."],
        output="An `int` ≥ 0.",
        constraints=["Both bounds are inclusive.", "If `start > end`, return 0.", "Must run in O(log n) per call."],
        criteria=["Correct counts including duplicates at the bounds.", "Empty list and inverted ranges handled.", "Fast for many queries."],
        hints=[
            "Find the first index ≥ start and the first index > end.",
            "`bisect.bisect_left(a, start)` and `bisect.bisect_right(a, end)` give exactly those positions.",
            "if start > end: return 0\nreturn bisect_right(a, end) - bisect_left(a, start)",
        ],
        solution='''
from bisect import bisect_left, bisect_right


def count_in_range(sorted_timestamps: list[int], start: int, end: int) -> int:
    """Number of timestamps t with start <= t <= end (inclusive)."""
    if start > end:
        return 0
    return bisect_right(sorted_timestamps, end) - bisect_left(sorted_timestamps, start)
''',
        explanation=(
            "`bisect_left` returns the first position whose value is ≥ start; `bisect_right` the "
            "first position whose value is > end. Their difference is the size of the inclusive "
            "slice, found with two O(log n) searches."
        ),
        edge_cases=["Duplicates at both bounds are counted.", "Range outside the data → 0.", "`start > end` → 0."],
        mistakes=["Using `bisect_left` for the end bound (excludes values equal to end).", "Scanning the list with a comprehension (O(n)).", "Returning a negative number for inverted ranges."],
        tests=[
            T(([100, 105, 105, 110, 120, 150], 105, 120), 4, example=True, note="duplicates at lower bound"),
            T(([100, 105, 105, 110, 120, 150], 150, 150), 1, example=True, note="boundary: single-point range"),
            T(([100, 105, 105, 110, 120, 150], 0, 99), 0, note="before all data"),
            T(([100, 105, 105, 110, 120, 150], 106, 109), 0, note="gap between values"),
            T(([100, 105, 105, 110, 120, 150], 100, 150), 6, note="whole range"),
            T(([100, 105], 200, 100), 0, note="start > end"),
            T(([], 0, 10), 0, note="empty list"),
            CT(_q036_fast, note="5,000 queries on 200,000 timestamps"),
        ],
        complexity="O(log n) time per query, O(1) space.",
        extension="Support queries over a stream where new timestamps keep arriving (hint: `bisect.insort` or a deque with periodic trimming).",
    ),
    # ------------------------------------------------------------------------ Q037
    Question(
        id=37,
        title="Merge Two Sorted Event Streams",
        difficulty="Intermediate",
        category=ALG,
        roles=[DE, PYDEV, MLE],
        skills=["two-pointer technique", "stable merging", "loop invariants"],
        minutes=20,
        scenario=(
            "Two services each produce a time-ordered list of events. A replay tool needs a "
            "single list ordered by timestamp. When two events share a timestamp, events from the "
            "first service must come first so replays are deterministic."
        ),
        problem=(
            "Write `merge_sorted(a, b)` that merges two lists of event dicts, each already sorted "
            "by `'ts'`, into one list sorted by `'ts'` using the two-pointer technique."
        ),
        signature="def merge_sorted(a: list[dict], b: list[dict]) -> list[dict]:",
        entry="merge_sorted",
        inputs=["`a`, `b`: lists of dicts with a numeric `'ts'` key, each sorted ascending by `'ts'`. Either may be empty."],
        output="A new list containing all events from both inputs.",
        constraints=[
            "On equal `ts`, all tied events from `a` come before tied events from `b`.",
            "Relative order within each input is preserved.",
            "Implement the merge with two indices – the point of the exercise is the two-pointer "
            "loop (the checker can only verify the result, so hold yourself to this).",
            "Do not modify the inputs.",
        ],
        criteria=["Correct merged order.", "Tie rule respected.", "Linear time."],
        hints=[
            "Keep an index into each list and repeatedly take the smaller head.",
            "Use `<=` when comparing `a[i]['ts']` with `b[j]['ts']` so ties prefer `a`; when one "
            "list is exhausted, extend with the rest of the other.",
            "i = j = 0; out = []\nwhile i < len(a) and j < len(b):\n    if a[i]['ts'] <= b[j]['ts']: out.append(a[i]); i += 1\n"
            "    else: out.append(b[j]); j += 1\nout.extend(a[i:]); out.extend(b[j:])\nreturn out",
        ],
        solution='''
def merge_sorted(a: list[dict], b: list[dict]) -> list[dict]:
    """Merge two ts-sorted lists; on ties, events from ``a`` come first."""
    i = j = 0
    merged: list[dict] = []
    while i < len(a) and j < len(b):
        if a[i]["ts"] <= b[j]["ts"]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    merged.extend(a[i:])
    merged.extend(b[j:])
    return merged
''',
        explanation="Each step appends one element and advances one pointer, so the loop runs at most `len(a) + len(b)` times. `<=` encodes the tie-break rule.",
        edge_cases=["One list empty → copy of the other.", "All equal timestamps → all of `a`, then all of `b`.", "Leftover tail appended in one step."],
        mistakes=["Using `<` (ties prefer `b`).", "Forgetting to append the remaining tail.", "Popping from the front of lists (O(n) per pop)."],
        tests=[
            T(([{"ts": 1, "src": "a"}, {"ts": 4, "src": "a"}, {"ts": 4, "src": "a2"}],
               [{"ts": 2, "src": "b"}, {"ts": 4, "src": "b"}, {"ts": 9, "src": "b"}]),
              [{"ts": 1, "src": "a"}, {"ts": 2, "src": "b"}, {"ts": 4, "src": "a"}, {"ts": 4, "src": "a2"},
               {"ts": 4, "src": "b"}, {"ts": 9, "src": "b"}], example=True, note="ties prefer a"),
            T(([], [{"ts": 1}]), [{"ts": 1}], example=True, note="boundary: one side empty"),
            T(([], []), [], note="both empty"),
            T(([{"ts": 5}], [{"ts": 1}, {"ts": 2}]), [{"ts": 1}, {"ts": 2}, {"ts": 5}], note="b first"),
            T(([{"ts": 3, "s": "a"}], [{"ts": 3, "s": "b"}]), [{"ts": 3, "s": "a"}, {"ts": 3, "s": "b"}], note="single tie"),
        ],
        complexity="O(n + m) time and space.",
        extension="Generalize to k sorted lists with `heapq` while keeping the same tie rule (list index as tie-breaker).",
    ),
    # ------------------------------------------------------------------------ Q038
    Question(
        id=38,
        title="Busiest Window of Requests (Sliding Window)",
        difficulty="Intermediate",
        category=ALG,
        roles=[DE, MLE, PYDEV],
        skills=["sliding window", "running sums", "tie-breaking", "complexity"],
        minutes=20,
        scenario=(
            "Capacity planning needs the busiest stretch of `k` consecutive minutes in a day of "
            "per-minute request counts. Recomputing every window from scratch is O(n·k) and too "
            "slow for long histories."
        ),
        problem=(
            "Write `max_window_sum(values, k)` that returns `(max_sum, start_index)` for the "
            "contiguous window of length `k` with the largest sum. If several windows tie, return "
            "the earliest."
        ),
        signature="def max_window_sum(values: list[int], k: int) -> tuple[int, int]:",
        entry="max_window_sum",
        inputs=["`values`: list of ints (may be negative).", "`k`: window length."],
        output="A tuple `(max_sum, start_index)`.",
        constraints=["Raise `ValueError` if `k < 1` or `k > len(values)`.", "Earliest window wins ties.", "Must run in O(n)."],
        criteria=["Correct sum and index.", "Ties → earliest.", "Handles negatives.", "Linear time."],
        hints=[
            "Adjacent windows share k − 1 elements.",
            "Compute the first window's sum, then for each step add the entering value and "
            "subtract the leaving value. Only update the best when strictly greater.",
            "window = sum(values[:k]); best = (window, 0)\nfor i in range(k, len(values)):\n"
            "    window += values[i] - values[i - k]\n    if window > best[0]: best = (window, i - k + 1)\nreturn best",
        ],
        solution='''
def max_window_sum(values: list[int], k: int) -> tuple[int, int]:
    """Largest sum of k consecutive values and the earliest start index."""
    if k < 1 or k > len(values):
        raise ValueError("k must satisfy 1 <= k <= len(values)")
    window = sum(values[:k])
    best_sum, best_start = window, 0
    for i in range(k, len(values)):
        window += values[i] - values[i - k]
        if window > best_sum:
            best_sum, best_start = window, i - k + 1
    return best_sum, best_start
''',
        explanation="The running sum turns each window update into O(1); updating only on a strictly greater sum keeps the earliest window on ties.",
        edge_cases=["`k == len(values)` → one window.", "All negative values.", "Ties keep the first window."],
        mistakes=["`sum(values[i:i+k])` inside the loop (O(n·k)).", "Using `>=`, which returns the last tied window.", "Initializing best to 0, wrong for negative data."],
        tests=[
            T(([3, 1, 4, 1, 5, 9, 2, 6], 3), (17, 5), example=True, note="typical"),
            T(([5, 5, 5], 2), (10, 0), example=True, note="boundary: tie → earliest"),
            T(([-2, -1, -3], 1), (-1, 1), note="negative values"),
            T(([4], 1), (4, 0), note="single element"),
            T(([1, 2], 3), raises="ValueError", note="k larger than input"),
            T(([1, 2], 0), raises="ValueError", note="k = 0"),
            CT(_q038_fast, note="200,000 values with k = 1,000 in O(n)"),
        ],
        complexity="O(n) time, O(1) extra space.",
        extension="Return the top 3 non-overlapping windows.",
    ),
]


def _assert_independent_rows(impl):
    matrix = impl(["a"], ["a"], ["a", "b", "c"])
    assert len({id(row) for row in matrix}) == 3, "each row must be a separate list object"
    matrix[0][1] = 99
    assert matrix[1][1] == 0 and matrix[2][1] == 0, "changing one row must not change the others"
