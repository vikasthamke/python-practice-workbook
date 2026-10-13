"""Advanced questions Q056–Q070: complex transformations, algorithms and debugging."""

import gzip
import random
import time

from ..schema import (
    AAI, AIE, ALG, DE, DS, DS_ROLE, FILE, FUNC, FUND, MLE, OOP, PYDEV, STR,
    CT, DataFile, File, Question, T,
)

BASE = 1_714_560_000  # 2024-05-01T10:40:00Z, divisible by 60


# ------------------------------------------------------------------ custom tests
def _q061_lru(impl, tmp):
    calls = []

    @impl(maxsize=2)
    def square(x):
        calls.append(x)
        return x * x

    results = [square(v) for v in [1, 2, 1, 3, 2]]
    assert results == [1, 4, 1, 9, 4], f"wrong results {results}"
    assert calls == [1, 2, 3, 2], f"underlying calls should be [1, 2, 3, 2] (LRU eviction), got {calls}"
    info = square.cache_info()
    assert info == {"hits": 1, "misses": 4, "size": 2, "maxsize": 2}, f"unexpected cache_info {info}"


def _q061_none_and_kwargs(impl, tmp):
    calls = []

    @impl(maxsize=10)
    def lookup(key, scale=1):
        calls.append((key, scale))
        return None if key == "missing" else key * scale

    lookup("missing")
    lookup("missing")
    assert calls.count(("missing", 1)) == 1, "a cached None result must be a cache hit"
    lookup("ab", scale=2)
    lookup("ab", scale=2)
    lookup("ab")
    assert calls.count(("ab", 2)) == 1 and calls.count(("ab", 1)) == 1, "keyword arguments must be part of the key"


def _q061_exceptions_not_cached(impl, tmp):
    state = {"n": 0}

    @impl(maxsize=4)
    def flaky(x):
        state["n"] += 1
        if state["n"] == 1:
            raise ConnectionError("first call fails")
        return x

    try:
        flaky(5)
    except ConnectionError:
        pass
    assert flaky(5) == 5 and state["n"] == 2, "exceptions must not be cached"


def _q061_metadata_and_clear(impl, tmp):
    @impl(maxsize=3)
    def embed(text):
        """Return a fake embedding."""
        return len(text)

    embed("a")
    embed("a")
    assert embed.__name__ == "embed" and embed.__doc__ == "Return a fake embedding.", "use functools.wraps"
    embed.cache_clear()
    assert embed.cache_info() == {"hits": 0, "misses": 0, "size": 0, "maxsize": 3}, "cache_clear resets entries and stats"


def _q061_invalid(impl, tmp):
    try:
        impl(maxsize=0)
    except ValueError:
        return
    raise AssertionError("memoize(maxsize=0) should raise ValueError")


def _q062_lazy(impl, tmp):
    def source():
        yield f"{BASE} 200"
        yield f"{BASE + 5} 500"
        yield f"{BASE + 60} 200"  # first line of the next window
        raise AssertionError("the generator read further than needed – it must be lazy")

    it = impl(source(), 60)
    assert next(it) == (BASE, 0.5), "the first window should be yielded as soon as the next window starts"


def _q063_basic(impl, tmp):
    @impl
    def scale(x: float, factor: int = 2) -> float:
        return x * factor

    assert scale(1.5) == 3.0 and scale(2) == 4, "ints are acceptable where float is annotated"
    for bad_call in (lambda: scale("a"), lambda: scale(2, factor=True), lambda: scale(2, factor=1.5)):
        try:
            bad_call()
        except TypeError:
            continue
        raise AssertionError("wrongly typed argument was not rejected (note: bool is not accepted as int)")
    assert scale.__name__ == "scale", "use functools.wraps"


def _q063_message(impl, tmp):
    @impl
    def greet(name: str) -> str:
        return "hi " + name

    try:
        greet(42)
    except TypeError as exc:
        assert "name" in str(exc), f"the error message should mention the parameter name, got: {exc}"
    else:
        raise AssertionError("greet(42) should raise TypeError")


def _q063_optional_and_generics(impl, tmp):
    @impl
    def label(name: str | None = None) -> str:
        return name or "anon"

    assert label(None) == "anon" and label("x") == "x"
    try:
        label(3)
    except TypeError:
        pass
    else:
        raise AssertionError("str | None should reject an int")

    @impl
    def total(xs: list[int]) -> int:
        return sum(xs)

    assert total([1, 2]) == 3
    try:
        total((1, 2))
    except TypeError:
        pass
    else:
        raise AssertionError("list[int] should at least check that the argument is a list")


def _q063_return_and_varargs(impl, tmp):
    @impl
    def broken() -> int:
        return "not an int"

    try:
        broken()
    except TypeError:
        pass
    else:
        raise AssertionError("the return value should be checked against the return annotation")

    @impl
    def mean(*values: float) -> float:
        return sum(values) / len(values)

    assert mean(1, 2.5) == 1.75
    try:
        mean(1, "2")
    except TypeError:
        pass
    else:
        raise AssertionError("each *args value should be checked")

    @impl
    def untyped(a, b):
        return (a, b)

    assert untyped("x", 1) == ("x", 1), "unannotated parameters must not be checked"


def _q067_fast(impl, tmp):
    rng = random.Random(5)
    jobs = []
    for _ in range(50_000):
        start = rng.randint(0, 1_000_000)
        jobs.append((start, start + rng.randint(1, 5_000)))
    events = sorted([(s, 1) for s, _ in jobs] + [(e, -1) for _, e in jobs], key=lambda e: (e[0], e[1]))
    expected = running = 0
    for _, delta in events:
        running += delta
        expected = max(expected, running)
    start = time.perf_counter()
    got = impl(jobs)
    elapsed = time.perf_counter() - start
    assert got == expected, f"expected {expected}, got {got}"
    assert elapsed < 1.0, f"took {elapsed:.2f}s for 50,000 jobs – use a sweep line or heap (O(n log n))"


def _q068_sample(impl, tmp):
    from ..schema import SAMPLE_DATA_DIR

    result = impl(str(SAMPLE_DATA_DIR / "transactions.csv"), "customer_id", "amount")
    assert result["skipped_rows"] == 4, f"expected 4 skipped rows, got {result['skipped_rows']}"
    assert sum(g["count"] for g in result["groups"].values()) == 34, "34 rows should be aggregated"
    assert sorted(result["groups"]) == [f"c{i:03d}" for i in range(1, 9)], f"unexpected groups {sorted(result['groups'])}"


def _q069_order(impl, tmp):
    logs = tmp / "logs"
    logs.mkdir()
    (logs / "app.log").write_text("c1\nc2\n", encoding="utf-8")
    (logs / "app.log.1").write_text("b1\n", encoding="utf-8")
    (logs / "app.log.2.gz").write_bytes(gzip.compress(b"a1\na2\n"))
    (logs / "app.log.10").write_text("old\n", encoding="utf-8")
    (logs / "app.log.bak").write_text("ignore me\n", encoding="utf-8")
    (logs / "other.log").write_text("not this file\n", encoding="utf-8")
    result = impl(str(logs), "app.log")
    assert iter(result) is result, "iter_rotated_logs should return a generator/iterator"
    got = list(result)
    assert got == ["old", "a1", "a2", "b1", "c1", "c2"], f"expected oldest-first order, got {got}"


def _q069_no_current_file(impl, tmp):
    (tmp / "svc.log.1").write_text("only rotated\r\n", encoding="utf-8")
    got = list(impl(str(tmp), "svc.log"))
    assert got == ["only rotated"], f"a missing current file is fine and line endings are stripped; got {got}"


def _q069_duplicate(impl, tmp):
    (tmp / "app.log.2").write_text("x\n", encoding="utf-8")
    (tmp / "app.log.2.gz").write_bytes(gzip.compress(b"y\n"))
    try:
        list(impl(str(tmp), "app.log"))
    except ValueError:
        return
    raise AssertionError("both app.log.2 and app.log.2.gz exist – that should raise ValueError")


def _q069_missing_dir(impl, tmp):
    try:
        list(impl(str(tmp / "does-not-exist"), "app.log"))
    except FileNotFoundError:
        return
    raise AssertionError("a missing directory should raise FileNotFoundError")


def _q070_container(impl, tmp):
    s = impl("loss", [0.9, 0.7, 0.5, 0.4])
    assert len(s) == 4 and s[0] == 0.9 and s[-1] == 0.4 and list(s) == [0.9, 0.7, 0.5, 0.4]
    part = s[1:3]
    assert isinstance(part, impl) and part.name == "loss" and list(part) == [0.7, 0.5], "slices must return a MetricSeries"
    assert repr(part) == "MetricSeries('loss', [0.7, 0.5])", f"unexpected repr {part!r}"


def _q070_arithmetic(impl, tmp):
    a, b = impl("a", [1, 2]), impl("b", [10, 20])
    total = a + b
    assert total == impl("a+b", [11, 22]), f"elementwise addition failed: {total!r}"
    assert a + 1 == impl("a", [2, 3]) and 1 + a == impl("a", [2, 3]), "scalar addition on both sides"
    assert a * 3 == impl("a", [3, 6]) and 3 * a == impl("a", [3, 6]), "scalar multiplication on both sides"
    assert list(a) == [1, 2], "operators must not modify the operands"
    try:
        a + impl("c", [1, 2, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("adding series of different lengths should raise ValueError")
    try:
        a + "x"
    except TypeError:
        pass
    else:
        raise AssertionError("adding a string should raise TypeError (return NotImplemented)")


def _q070_sum(impl, tmp):
    a, b = impl("a", [1, 2]), impl("b", [10, 20])
    assert sum([a, b]) == impl("a+b", [11, 22]), "sum() starts from 0, so 0 + series must work (__radd__)"
    assert [x * 2 for x in a] == [2, 4], "iteration should yield the values"


def _q070_validation_and_copy(impl, tmp):
    values = [1.0, 2.0]
    s = impl("m", values)
    values.append(3.0)
    assert len(s) == 2, "the series must store its own copy of the values"
    for bad in (["x"], [True]):
        try:
            impl("m", bad)
        except TypeError:
            continue
        raise AssertionError(f"MetricSeries('m', {bad!r}) should raise TypeError")
    assert impl("m", [1, 2, 3]).mean() == 2.0
    assert impl("e", []).mean() is None, "mean() of an empty series should be None"
    assert impl("m", [1]) != impl("n", [1]), "series with different names are not equal"


# ------------------------------------------------------------------------ questions
JSON_FENCE_TEXT = 'Here is the result:\n```json\n{"label": "spam", "confidence": 0.97}\n```\nLet me know!'

TREE_ROWS = [
    {"id": "ml", "parent_id": None, "name": "Machine Learning"},
    {"id": "dl", "parent_id": "ml", "name": "Deep Learning"},
    {"id": "cv", "parent_id": "dl", "name": "Computer Vision"},
    {"id": "nlp", "parent_id": "dl", "name": "NLP"},
    {"id": "de", "parent_id": None, "name": "Data Engineering"},
    {"id": "cls", "parent_id": "ml", "name": "Classification"},
]
TREE = [
    {"id": "de", "name": "Data Engineering", "children": []},
    {"id": "ml", "name": "Machine Learning", "children": [
        {"id": "cls", "name": "Classification", "children": []},
        {"id": "dl", "name": "Deep Learning", "children": [
            {"id": "cv", "name": "Computer Vision", "children": []},
            {"id": "nlp", "name": "NLP", "children": []},
        ]},
    ]},
]

VECTORS = {"doc_a": [1, 0, 0], "doc_b": [0.9, 0.1, 0], "doc_c": [0, 1, 0], "doc_d": [0, 0, 0], "doc_e": [2, 0, 0]}

QUESTIONS = [
    # ------------------------------------------------------------------------ Q056
    Question(
        id=56,
        title="Simulate a Token-Bucket Rate Limiter",
        difficulty="Advanced",
        category=FUND,
        roles=[AIE, PYDEV, DE],
        skills=["stateful simulation", "floating-point state", "loop invariants", "validation"],
        minutes=30,
        scenario=(
            "Your LLM gateway protects an upstream API with a token bucket: the bucket holds up to "
            "`capacity` tokens, refills continuously at `refill_rate` tokens per second, and each "
            "request needs one whole token. Before changing the production settings you want to "
            "replay yesterday's request timestamps and see which requests would have been rejected."
        ),
        problem=(
            "Write `simulate_rate_limiter(request_times, capacity, refill_rate)` that returns, for "
            "each request, `True` if it is allowed and `False` if it is rejected."
        ),
        signature="def simulate_rate_limiter(request_times: list[float], capacity: int, refill_rate: float) -> list[bool]:",
        entry="simulate_rate_limiter",
        inputs=[
            "`request_times`: non-decreasing timestamps in seconds (ints or floats). May be empty.",
            "`capacity`: `int` ≥ 1. `refill_rate`: tokens per second, > 0.",
        ],
        output="A `list[bool]` with one entry per request.",
        constraints=[
            "The bucket is full (`capacity` tokens) at the time of the first request.",
            "Before each request at time `t`: `tokens = min(capacity, tokens + (t - last_t) * refill_rate)`.",
            "If `tokens >= 1`, allow the request and subtract 1; otherwise reject it (rejections cost nothing).",
            "Raise `ValueError` if times decrease, `capacity < 1` or `refill_rate <= 0`.",
        ],
        criteria=["Bursts up to capacity are allowed.", "Fractional refill accumulates across rejections.", "Refill is capped.", "Validation."],
        hints=[
            "You only need two pieces of state: the current token count and the time of the last request.",
            "Refill lazily at each request using the elapsed time, cap at capacity, then decide.",
            "tokens = capacity; last = times[0] if times else 0\nfor t in times:\n    if t < last: raise ValueError\n"
            "    tokens = min(capacity, tokens + (t - last) * rate); last = t\n    if tokens >= 1: tokens -= 1; out.append(True)\n    else: out.append(False)",
        ],
        solution='''
def simulate_rate_limiter(request_times: list[float], capacity: int, refill_rate: float) -> list[bool]:
    """Replay request timestamps through a token bucket."""
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    if refill_rate <= 0:
        raise ValueError("refill_rate must be > 0")
    decisions: list[bool] = []
    tokens = float(capacity)
    last_time = request_times[0] if request_times else 0
    for t in request_times:
        if t < last_time:
            raise ValueError(f"request times must be non-decreasing ({t} after {last_time})")
        tokens = min(capacity, tokens + (t - last_time) * refill_rate)
        last_time = t
        if tokens >= 1:
            tokens -= 1
            decisions.append(True)
        else:
            decisions.append(False)
    return decisions
''',
        explanation=(
            "Refilling lazily (only when a request arrives) is equivalent to continuous refilling "
            "and needs O(1) work per request. Capping with `min` prevents idle periods from "
            "building up unlimited burst capacity; rejected requests keep their fractional tokens."
        ),
        edge_cases=["Simultaneous requests drain the bucket.", "Long idle periods refill only up to capacity.", "Half tokens carry over to the next request."],
        mistakes=["Refilling by whole seconds only.", "Charging a token for rejected requests.", "Starting with an empty bucket."],
        tests=[
            T(([0, 0, 0, 0, 1, 2, 2.5, 10], 3, 1), [True, True, True, False, True, True, False, True], example=True,
              note="burst, refill, cap"),
            T(([0, 1, 2, 3, 4], 1, 0.5), [True, False, True, False, True], example=True, note="fractional refill"),
            T(([], 5, 1), [], note="no requests"),
            T(([100, 100.5], 1, 2), [True, True], note="first request at a non-zero time"),
            T(([5, 4], 2, 1), raises="ValueError", note="decreasing times"),
            T(([0], 0, 1), raises="ValueError", note="capacity 0"),
            T(([0], 1, 0), raises="ValueError", note="refill_rate 0"),
        ],
        complexity="O(n) time, O(n) space for the output.",
        extension="Return the time at which each rejected request *could* have been served if it waited.",
    ),
    # ------------------------------------------------------------------------ Q057
    Question(
        id=57,
        title="Extract a JSON Object from LLM Output",
        difficulty="Advanced",
        category=STR,
        roles=[AIE, AAI, PYDEV],
        skills=["regex with DOTALL", "json.JSONDecoder.raw_decode", "fallback strategies", "robust parsing"],
        minutes=30,
        scenario=(
            "You ask an LLM to answer with JSON, but replies arrive wrapped in prose: sometimes in "
            "a fenced ```json block, sometimes inline like `Sure! {\"label\": \"ham\"} Hope that "
            "helps.` Braces may also appear inside JSON strings or in non-JSON text."
        ),
        problem=(
            "Write `extract_json_block(text)` that returns the first JSON **object** (dict) found "
            "in the text, preferring fenced json blocks."
        ),
        signature="def extract_json_block(text: str) -> dict:",
        entry="extract_json_block",
        inputs=["`text`: arbitrary model output."],
        output="The parsed `dict`.",
        constraints=[
            "1. Look at fenced blocks that open with three backticks followed by `json` (any case) and "
            "close with three backticks. Return the first one whose content parses to a dict.",
            "2. Otherwise scan the whole text: for each `{` from left to right, try to decode a JSON "
            "value starting there; return the first that is a dict.",
            "3. If nothing qualifies, raise `ValueError`.",
            "Braces inside JSON strings must not confuse the parser; JSON arrays are not accepted.",
        ],
        criteria=["Fenced blocks preferred.", "Inline objects found.", "Strings containing braces handled.", "Clear failure."],
        hints=[
            "Don't count braces yourself – the json module can tell you where a valid value ends.",
            "`json.JSONDecoder().raw_decode(text, idx)` parses one value starting at `idx` and ignores "
            "whatever follows. For fences, use `re.finditer(r'```json\\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)`.",
            "for m in FENCE.finditer(text): try json.loads(m.group(1)) -> return if dict\n"
            "decoder = json.JSONDecoder()\nfor i, ch in enumerate(text):\n    if ch == '{':\n        try: obj, _ = decoder.raw_decode(text, i)\n"
            "        except json.JSONDecodeError: continue\n        if isinstance(obj, dict): return obj\nraise ValueError",
        ],
        solution='''
import json
import re

FENCE = re.compile(r"```json\\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_block(text: str) -> dict:
    """Return the first JSON object in LLM output (fenced json blocks first)."""
    for match in FENCE.finditer(text):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in text")
''',
        explanation=(
            "`raw_decode` uses the real JSON grammar, so braces and escaped quotes inside strings "
            "are handled correctly, and it stops at the end of the first value. Trying each `{` in "
            "order skips non-JSON text such as `{x}`. The non-greedy fence regex finds each block separately."
        ),
        edge_cases=["Invalid fenced block → fall back to scanning.", "`{x}` before the real object.", "Top-level arrays rejected.", "Uppercase `JSON` fence tag."],
        mistakes=["`text[text.find('{'):text.rfind('}')+1]` breaks with multiple objects or trailing braces.", "Counting braces without handling strings.", "Greedy `.*` in the fence regex spanning two blocks."],
        tests=[
            T((JSON_FENCE_TEXT,), {"label": "spam", "confidence": 0.97}, example=True, note="fenced block"),
            T(('Sure! {"label": "ham", "tags": ["a", "b"]} Hope that helps.',), {"label": "ham", "tags": ["a", "b"]},
              example=True, note="inline object"),
            T(('Answer: {"text": "use {braces} and \\"quotes\\"", "ok": true}',), {"text": 'use {braces} and "quotes"', "ok": True},
              note="braces and escaped quotes inside strings"),
            T(("Set {x} first, then: {\"a\": 1}",), {"a": 1}, note="non-JSON braces before the object"),
            T(('{"outer": {"inner": [1, {"k": "v"}]}} trailing }',), {"outer": {"inner": [1, {"k": "v"}]}}, note="nested object, trailing brace"),
            T(("```json\n{bad}\n```\nFallback {\"a\": 2}",), {"a": 2}, note="invalid fence falls back to scanning"),
            T(("```JSON\n{\"x\": 1}\n```",), {"x": 1}, note="uppercase fence tag"),
            T(("```json\n[1, 2]\n```",), raises="ValueError", note="array is not an object"),
            T(("No json here",), raises="ValueError", example=True, note="boundary: nothing to extract"),
        ],
        complexity="Worst case O(n²) when many `{` characters start invalid JSON; typical replies are O(n).",
        extension="Validate the extracted object against required keys and retry the LLM call with an error message when keys are missing.",
    ),
    # ------------------------------------------------------------------------ Q058
    Question(
        id=58,
        title="Redact PII from Log Text",
        difficulty="Advanced",
        category=STR,
        roles=[DE, AIE, PYDEV],
        skills=["regular expressions", "lookarounds", "re.subn", "ordering of substitutions"],
        minutes=35,
        scenario=(
            "Support transcripts are sent to an LLM for summarization. Compliance requires that "
            "e-mail addresses, IPv4 addresses and US-style phone numbers are replaced with "
            "placeholders first, and wants counts of what was redacted."
        ),
        problem=(
            "Write `redact_pii(text)` that replaces emails with `<EMAIL>`, IPv4 addresses with "
            "`<IP>` and phone numbers with `<PHONE>`, returning the new text and a dict of counts."
        ),
        signature="def redact_pii(text: str) -> tuple[str, dict[str, int]]:",
        entry="redact_pii",
        inputs=["`text`: any string."],
        output="`(redacted_text, {'email': n, 'ip': n, 'phone': n})` – all three keys always present.",
        constraints=[
            "Apply in this order: emails, then IPs, then phones.",
            "Email: `local@domain.tld` where local is `[A-Za-z0-9._%+-]+`, domain `[A-Za-z0-9.-]+` and tld 2+ letters.",
            "IPv4: four dot-separated numbers 0–255 (no leading `+`); not preceded by a digit or dot and not followed by a digit or a dot+digit (so `10.0.0.256` and `1.2.3.4.5` are not IPs, but a sentence-final `10.0.0.1.` is).",
            "Phone: `ddd-ddd-dddd`, `ddd.ddd.dddd`, `ddd ddd dddd` or `(ddd) ddd-dddd`, not touching other digits. Plain 10-digit runs are not phones.",
        ],
        criteria=["All three PII types replaced.", "Invalid IPs and digit runs untouched.", "Accurate counts."],
        hints=[
            "Write one compiled regex per PII type and apply them in order.",
            "`re.subn` returns `(new_text, count)`. Use lookarounds such as `(?<![\\d.])` and `(?!\\.?\\d)` "
            "to stop matches inside longer numbers. An octet is `25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d`.",
            "counts = {}\nfor name, pattern, token in [('email', EMAIL, '<EMAIL>'), ('ip', IPV4, '<IP>'), ('phone', PHONE, '<PHONE>')]:\n"
            "    text, counts[name] = pattern.subn(token, text)\nreturn text, counts",
        ],
        solution='''
import re

OCTET = r"(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)"
EMAIL = re.compile(r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b")
IPV4 = re.compile(rf"(?<![\\d.]){OCTET}(?:\\.{OCTET}){{3}}(?!\\.?\\d)")
PHONE = re.compile(r"(?<!\\d)(?:\\(\\d{3}\\) ?|\\d{3}[-. ])\\d{3}[-. ]\\d{4}(?!\\d)")


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails, IPv4 addresses and phone numbers with placeholders."""
    counts: dict[str, int] = {}
    for name, pattern, token in (("email", EMAIL, "<EMAIL>"), ("ip", IPV4, "<IP>"), ("phone", PHONE, "<PHONE>")):
        text, counts[name] = pattern.subn(token, text)
    return text, counts
''',
        explanation=(
            "Each pattern is anchored with lookarounds so it cannot match part of a longer number. "
            "Emails go first so their domains are never mistaken for IPs; `subn` gives the count "
            "for free. The octet alternation rejects values above 255."
        ),
        edge_cases=["`10.0.0.256` is not an IP.", "Sentence-final IP followed by a period is redacted.", "`1234567890` (no separators) is kept.", "`ops@10.0.0.1` keeps `ops@` and redacts the IP."],
        mistakes=["`\\d{1,3}(\\.\\d{1,3}){3}` matches 999.999.999.999.", "Redacting phones before IPs or emails, which corrupts them.", "Using `\\b` around IPs, which matches inside `1.2.3.4.5`."],
        tests=[
            T(("User jane.doe@example.com called from 192.168.1.20 and left 555-123-4567.",),
              ("User <EMAIL> called from <IP> and left <PHONE>.", {"email": 1, "ip": 1, "phone": 1}), example=True,
              note="one of each"),
            T(("No PII here.",), ("No PII here.", {"email": 0, "ip": 0, "phone": 0}), example=True, note="boundary: nothing to redact"),
            T(("Server 999.1.1.1 is invalid; 10.0.0.256 too; but 10.0.0.255 is fine.",),
              ("Server 999.1.1.1 is invalid; 10.0.0.256 too; but <IP> is fine.", {"email": 0, "ip": 1, "phone": 0}),
              note="octet range"),
            T(("Call (555) 987-6543 or 555.987.6543 or 555 987 6543; order id 1234567890 stays.",),
              ("Call <PHONE> or <PHONE> or <PHONE>; order id 1234567890 stays.", {"email": 0, "ip": 0, "phone": 3}),
              note="phone formats"),
            T(("a@x.io, B.Smith+ml@corp.co.uk",), ("<EMAIL>, <EMAIL>", {"email": 2, "ip": 0, "phone": 0}), note="two emails"),
            T(("ops@10.0.0.1 and version 1.2.3.4.5",), ("ops@<IP> and version 1.2.3.4.5", {"email": 0, "ip": 1, "phone": 0}),
              note="not an email; dotted version string kept"),
            T(("Gateway at 10.0.0.1.",), ("Gateway at <IP>.", {"email": 0, "ip": 1, "phone": 0}), note="IP at end of sentence"),
        ],
        complexity="O(n) per pattern for typical inputs (three linear passes).",
        extension="Replace each distinct value with a stable numbered token (`<EMAIL_1>`, `<EMAIL_2>`) so the LLM can still tell people apart.",
    ),
    # ------------------------------------------------------------------------ Q059
    Question(
        id=59,
        title="Build a Category Tree from Parent Links",
        difficulty="Advanced",
        category=DS,
        roles=[DE, PYDEV, DS_ROLE],
        skills=["adjacency maps", "recursion / iteration over trees", "cycle detection", "validation"],
        minutes=35,
        scenario=(
            "A product taxonomy is stored as flat rows `(id, parent_id, name)`. The front end needs "
            "a nested tree, and data errors (unknown parents, cycles, duplicate ids) must be "
            "caught before publishing."
        ),
        problem="Write `build_tree(rows)` that converts the flat rows into a sorted, nested list of root nodes.",
        signature="def build_tree(rows: list[dict]) -> list[dict]:",
        entry="build_tree",
        inputs=["`rows`: dicts with `id` (str), `parent_id` (str or `None` for roots) and `name` (str), in any order."],
        output="A list of root nodes `{'id', 'name', 'children': [...]}`, recursively; roots and children sorted by `id`.",
        constraints=[
            "Raise `ValueError` for duplicate ids, a `parent_id` that does not exist, or any cycle "
            "(including a node that is its own parent).",
            "Every row must appear exactly once in the tree.",
            "Do not modify the input.",
        ],
        criteria=["Correct nesting and ordering.", "All three error types detected.", "Order of input rows irrelevant."],
        hints=[
            "Index rows by id and group child ids by parent id first.",
            "Build nodes recursively starting from roots. Nodes in a cycle are never reachable from a "
            "root, so compare the number of nodes you built with the number of rows.",
            "by_id = {...} (check duplicates); children = defaultdict(list)\nfor row: check parent exists; children[parent].append(id)\n"
            "def build(id): return {'id', 'name', 'children': [build(c) for c in sorted(children[id])]}\n"
            "tree = [build(r) for r in sorted(children[None])]\nif built != len(rows): raise ValueError('cycle')",
        ],
        solution='''
def build_tree(rows: list[dict]) -> list[dict]:
    """Nest flat (id, parent_id, name) rows into a sorted tree."""
    by_id: dict[str, dict] = {}
    for row in rows:
        if row["id"] in by_id:
            raise ValueError(f"duplicate id {row['id']!r}")
        by_id[row["id"]] = row
    children: dict = {}
    for row in rows:
        parent = row["parent_id"]
        if parent is not None and parent not in by_id:
            raise ValueError(f"unknown parent {parent!r} for {row['id']!r}")
        children.setdefault(parent, []).append(row["id"])

    built = 0

    def build(node_id: str) -> dict:
        nonlocal built
        built += 1
        return {
            "id": node_id,
            "name": by_id[node_id]["name"],
            "children": [build(child) for child in sorted(children.get(node_id, []))],
        }

    tree = [build(root) for root in sorted(children.get(None, []))]
    if built != len(rows):
        raise ValueError("cycle detected: some nodes are not reachable from a root")
    return tree
''',
        explanation=(
            "The parent → children map turns tree construction into a simple recursive walk from "
            "the roots. Because a cycle has no path to a root, its nodes are never built – "
            "comparing counts detects cycles without extra bookkeeping (and the recursion cannot "
            "loop forever because it only follows edges from the roots)."
        ),
        edge_cases=["Children listed before parents.", "Self-parent row → cycle.", "Empty input → `[]`."],
        mistakes=["Following parent links upward without cycle protection (infinite loop).", "Forgetting to sort children.", "Silently dropping orphans."],
        tests=[
            T((TREE_ROWS,), TREE, example=True, note="two roots, three levels"),
            T(([],), [], example=True, note="boundary: no rows"),
            T(([{"id": "b", "parent_id": "a", "name": "B"}, {"id": "a", "parent_id": None, "name": "A"}],),
              [{"id": "a", "name": "A", "children": [{"id": "b", "name": "B", "children": []}]}], note="child before parent"),
            T(([{"id": "x", "parent_id": "missing", "name": "X"}],), raises="ValueError", note="unknown parent"),
            T(([{"id": "a", "parent_id": "b", "name": "A"}, {"id": "b", "parent_id": "a", "name": "B"},
                {"id": "r", "parent_id": None, "name": "R"}],), raises="ValueError", note="cycle"),
            T(([{"id": "a", "parent_id": "a", "name": "A"}],), raises="ValueError", note="self-parent"),
            T(([{"id": "a", "parent_id": None, "name": "A"}, {"id": "a", "parent_id": None, "name": "A2"}],),
              raises="ValueError", note="duplicate id"),
        ],
        complexity="O(n log n) time (sorting children), O(n) space; recursion depth equals tree depth.",
        extension="Return each node's full path (`ml/dl/cv`) and depth as well.",
    ),
    # ------------------------------------------------------------------------ Q060
    Question(
        id=60,
        title="Path-Based Diff of Two JSON Documents",
        difficulty="Advanced",
        category=DS,
        roles=[MLE, DE, PYDEV],
        skills=["recursive comparison", "path building", "type-aware equality", "sorting results"],
        minutes=40,
        scenario=(
            "Two training runs produced different results. Their config snapshots are large nested "
            "JSON documents, and you want a precise list of what differs, such as "
            "`params.lr changed 0.001 → 0.0005`."
        ),
        problem=(
            "Write `json_diff(a, b)` that returns every difference between two JSON-like values as "
            "`(path, kind, old, new)` tuples sorted by path."
        ),
        signature="def json_diff(a, b) -> list[tuple]:",
        entry="json_diff",
        inputs=["`a`, `b`: JSON-like values built from dicts (string keys), lists, strings, numbers, bools and `None`."],
        output="A list of `(path, kind, old, new)` tuples, sorted by `path`.",
        constraints=[
            "`kind` is `'added'` (old is `None`), `'removed'` (new is `None`) or `'changed'`.",
            "Paths: dict keys are joined with `.` (no leading dot at the root), list indexes use `[i]`, "
            "e.g. `params.lr`, `tags[2]`, `[0].a`. The root itself has path `''`.",
            "Dicts are compared key by key; lists index by index (extra items are added/removed).",
            "Values of different types are *changed* even if `==` (e.g. `1` vs `1.0`, `1` vs `True`).",
            "Do not modify the inputs.",
        ],
        criteria=["Recursive dict/list comparison.", "Correct path syntax.", "Type-aware equality.", "Sorted output."],
        hints=[
            "Write a recursive helper `walk(old, new, path)` that appends to a shared list.",
            "Recurse only when both values are dicts or both are lists; otherwise compare with "
            "`type(a) is not type(b) or a != b`.",
            "walk(a, b, path):\n    if both dicts: for key in union: child = f'{path}.{key}' if path else key; added/removed/recurse\n"
            "    elif both lists: for i in range(max len): child = f'{path}[{i}]'; added/removed/recurse\n"
            "    elif type differs or a != b: changes.append((path, 'changed', a, b))\nreturn sorted(changes, key=lambda c: c[0])",
        ],
        solution='''
def json_diff(a, b) -> list[tuple]:
    """List (path, kind, old, new) differences between two JSON-like values."""
    changes: list[tuple] = []

    def walk(old, new, path: str) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            for key in old.keys() | new.keys():
                child = f"{path}.{key}" if path else str(key)
                if key not in new:
                    changes.append((child, "removed", old[key], None))
                elif key not in old:
                    changes.append((child, "added", None, new[key]))
                else:
                    walk(old[key], new[key], child)
        elif isinstance(old, list) and isinstance(new, list):
            for i in range(max(len(old), len(new))):
                child = f"{path}[{i}]"
                if i >= len(new):
                    changes.append((child, "removed", old[i], None))
                elif i >= len(old):
                    changes.append((child, "added", None, new[i]))
                else:
                    walk(old[i], new[i], child)
        elif type(old) is not type(new) or old != new:
            changes.append((path, "changed", old, new))

    walk(a, b, "")
    return sorted(changes, key=lambda change: change[0])
''',
        explanation=(
            "Recursion mirrors the document structure and the path string is extended at each "
            "level. Checking `type(old) is not type(new)` catches JSON-meaningful differences that "
            "Python's `==` hides (`1 == 1.0 == True`). Sorting gives deterministic output."
        ),
        edge_cases=["Identical documents → `[]`.", "Dict replaced by list → one 'changed' entry at that path.", "Root scalars → path `''`."],
        mistakes=["Using `a == b` only (misses `1` vs `1.0`).", "Recursing into strings.", "Leading dots in root-level paths."],
        tests=[
            T(({"model": "bert", "params": {"lr": 0.001, "layers": 12}, "tags": ["nlp", "prod"], "owner": "ml-team"},
               {"model": "bert", "params": {"lr": 0.0005, "layers": 12, "dropout": 0.1}, "tags": ["nlp", "staging", "v2"]}),
              [("owner", "removed", "ml-team", None), ("params.dropout", "added", None, 0.1),
               ("params.lr", "changed", 0.001, 0.0005), ("tags[1]", "changed", "prod", "staging"),
               ("tags[2]", "added", None, "v2")], example=True, note="all change kinds"),
            T(({"a": [1, 2]}, {"a": [1, 2]}), [], example=True, note="boundary: identical"),
            T((1, 2), [("", "changed", 1, 2)], note="root scalars"),
            T(({"a": {"x": 1}}, {"a": [1]}), [("a", "changed", {"x": 1}, [1])], note="type change dict → list"),
            T(({"v": 1, "w": 1}, {"v": 1.0, "w": True}), [("v", "changed", 1, 1.0), ("w", "changed", 1, True)],
              note="int vs float vs bool"),
            T(([{"a": 1}], [{"a": 2}, {"b": 3}]), [("[0].a", "changed", 1, 2), ("[1]", "added", None, {"b": 3})], note="root list"),
            T(({"items": [{"id": 1, "q": 2}]}, {"items": [{"id": 1, "q": 3}]}), [("items[0].q", "changed", 2, 3)], note="nested list of dicts"),
        ],
        complexity="O(N) for N total elements, plus O(d log d) to sort d differences.",
        extension="Add an option to compare lists of dicts by an `id` field instead of by index.",
    ),
    # ------------------------------------------------------------------------ Q061
    Question(
        id=61,
        title="Memoize with LRU Eviction and Cache Statistics",
        difficulty="Advanced",
        category=FUNC,
        roles=[AIE, MLE, PYDEV],
        skills=["decorators", "OrderedDict", "hashable cache keys", "sentinels", "function attributes"],
        minutes=35,
        scenario=(
            "Computing an embedding costs money and ~200 ms. Many requests repeat the same text, so "
            "you cache results – but memory is limited, so only the most recently used entries may "
            "be kept. Operators also want hit/miss statistics."
        ),
        problem=(
            "Write a decorator factory `memoize(maxsize=128)` that caches results by arguments with "
            "least-recently-used eviction and attaches `cache_info()` and `cache_clear()` to the "
            "decorated function. Implement the cache yourself (don't wrap `functools.lru_cache`)."
        ),
        signature="def memoize(maxsize: int = 128):",
        entry="memoize",
        inputs=["`maxsize`: `int` ≥ 1.", "Decorated functions take hashable positional/keyword arguments."],
        output="A decorator. `f.cache_info()` returns `{'hits', 'misses', 'size', 'maxsize'}`; `f.cache_clear()` resets entries and statistics.",
        constraints=[
            "The key combines positional args and keyword args (`f(1)` and `f(x=1)` may be different keys).",
            "A hit marks the entry as most recently used; inserting beyond `maxsize` evicts the least recently used entry.",
            "`None` results are cached like any other value.",
            "Exceptions are not cached.",
            "Preserve `__name__`/`__doc__`; raise `ValueError` for `maxsize < 1`.",
        ],
        criteria=["Correct LRU behaviour.", "Accurate statistics.", "None cached, exceptions not.", "Metadata kept."],
        hints=[
            "An `OrderedDict` remembers insertion order and can move a key to the end in O(1).",
            "Build the key as `(args, tuple(sorted(kwargs.items())))`. Check membership with `in` – "
            "not `cache.get(key)`, which cannot distinguish a cached `None` from a miss.",
            "if key in cache: hits += 1; cache.move_to_end(key); return cache[key]\nmisses += 1; result = func(...)\n"
            "cache[key] = result\nif len(cache) > maxsize: cache.popitem(last=False)\nreturn result",
        ],
        solution='''
import functools
from collections import OrderedDict


def memoize(maxsize: int = 128):
    """LRU memoization decorator with cache_info() and cache_clear()."""
    if maxsize < 1:
        raise ValueError("maxsize must be >= 1")

    def decorator(func):
        cache: OrderedDict = OrderedDict()
        stats = {"hits": 0, "misses": 0}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                stats["hits"] += 1
                cache.move_to_end(key)
                return cache[key]
            stats["misses"] += 1
            result = func(*args, **kwargs)
            cache[key] = result
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        def cache_info() -> dict:
            return {"hits": stats["hits"], "misses": stats["misses"], "size": len(cache), "maxsize": maxsize}

        def cache_clear() -> None:
            cache.clear()
            stats.update(hits=0, misses=0)

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorator
''',
        explanation=(
            "The `OrderedDict` keeps entries in recency order: `move_to_end` on a hit and "
            "`popitem(last=False)` to evict the oldest give O(1) LRU operations. If `func` raises, "
            "nothing is stored. Functions are objects, so helper functions can be attached as attributes."
        ),
        edge_cases=["Cached `None` counts as a hit.", "Exceptions retried on the next call.", "`cache_clear` resets statistics too."],
        mistakes=["`if cache.get(key):` treats falsy results as misses.", "Evicting by insertion order instead of recency (FIFO).", "A module-level cache shared by all decorated functions."],
        tests=[
            CT(_q061_lru, note="LRU eviction order and statistics"),
            CT(_q061_none_and_kwargs, note="None results cached; kwargs are part of the key"),
            CT(_q061_exceptions_not_cached, note="exceptions are not cached"),
            CT(_q061_metadata_and_clear, note="functools.wraps and cache_clear"),
            CT(_q061_invalid, note="maxsize < 1 rejected"),
        ],
        examples=['''
>>> @memoize(maxsize=2)
... def square(x):
...     return x * x
>>> [square(v) for v in [1, 2, 1, 3, 2]]
[1, 4, 1, 9, 4]
>>> square.cache_info()
{'hits': 1, 'misses': 4, 'size': 2, 'maxsize': 2}
'''],
        complexity="O(1) average time per call (plus hashing the arguments); O(maxsize) memory.",
        extension="Add a `ttl_seconds` option with an injectable clock so entries expire.",
    ),
    # ------------------------------------------------------------------------ Q062
    Question(
        id=62,
        title="Streaming Error Rate per Time Window (Generator Pipeline)",
        difficulty="Advanced",
        category=FUNC,
        roles=[DE, AIE, MLE],
        skills=["generators", "lazy evaluation", "stateful streaming", "parsing with validation"],
        minutes=35,
        scenario=(
            "An alerting job tails access logs containing `<epoch_seconds> <status_code>` lines, "
            "ordered by time. It must report the 5xx error rate for each time window as soon as "
            "that window is complete – it cannot wait for the stream to end because the stream "
            "never ends."
        ),
        problem=(
            "Write a generator `error_rate_per_window(lines, window_seconds)` that yields "
            "`(window_start, error_rate)` for each window containing at least one request."
        ),
        signature="def error_rate_per_window(lines, window_seconds: int):",
        entry="error_rate_per_window",
        inputs=[
            "`lines`: any iterable of strings (possibly infinite), sorted by timestamp.",
            "`window_seconds`: window length, `int` ≥ 1.",
        ],
        output="An iterator of `(window_start: int, error_rate: float)` tuples in time order.",
        constraints=[
            "Window start = `ts // window_seconds * window_seconds`.",
            "An error is a status code ≥ 500. Error rate = errors / requests, rounded to 4 decimals.",
            "Skip lines that are not exactly two integers separated by whitespace (including blank lines).",
            "Yield a window as soon as a line from a later window arrives, and the final window when the input ends.",
            "Never read further than needed; never materialize the input.",
            "Raise `ValueError` if a timestamp belongs to an earlier window than the current one, or if `window_seconds < 1` "
            "(raising when iteration starts is acceptable).",
        ],
        criteria=["Correct windows and rates.", "Malformed lines skipped.", "Lazy: works on infinite streams.", "Validation."],
        hints=[
            "Keep the current window start and its counters; emit when the window changes.",
            "Use `yield` inside the loop when a new window begins, and once more after the loop "
            "for the last window (if any requests were seen).",
            "current = None; total = errors = 0\nfor line in lines:\n    parse or continue\n    window = ts // size * size\n"
            "    if current is not None and window != current:\n        if window < current: raise ValueError\n"
            "        yield current, round(errors / total, 4); total = errors = 0\n    current = window; total += 1; errors += status >= 500\n"
            "if current is not None: yield current, round(errors / total, 4)",
        ],
        solution='''
def error_rate_per_window(lines, window_seconds: int):
    """Lazily yield (window_start, 5xx error rate) for each non-empty window."""
    if window_seconds < 1:
        raise ValueError("window_seconds must be >= 1")
    current = None
    total = errors = 0
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            ts, status = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        window = ts // window_seconds * window_seconds
        if current is not None and window != current:
            if window < current:
                raise ValueError(f"timestamp {ts} is out of order")
            yield current, round(errors / total, 4)
            total = errors = 0
        current = window
        total += 1
        if status >= 500:
            errors += 1
    if current is not None:
        yield current, round(errors / total, 4)
''',
        explanation=(
            "The generator holds only the counters for the current window, so memory is O(1) no "
            "matter how long the stream is. A window is emitted exactly when the first line of a "
            "later window is read, which is the earliest moment it is known to be complete."
        ),
        edge_cases=["Gaps between windows produce no output for empty windows.", "Blank/malformed lines skipped.", "Last window emitted at end of input."],
        mistakes=["`list(lines)` or sorting the input (breaks on infinite streams).", "Forgetting the final window.", "Counting 4xx as errors."],
        tests=[
            T(([f"{BASE} 200", f"{BASE + 10} 500", f"{BASE + 59} 200", f"{BASE + 60} 503", f"{BASE + 61} 200", "bad line", "",
                f"{BASE + 240} 404", f"{BASE + 241} 500"], 60),
              [(BASE, 0.3333), (BASE + 60, 0.5), (BASE + 240, 0.5)], materialize=True, note="three windows, malformed lines"),
            T(([], 60), [], materialize=True, note="empty input"),
            T((["x y", "12 abc"], 60), [], materialize=True, note="only malformed lines"),
            T(([f"{BASE + 60} 200", f"{BASE} 500"], 60), raises="ValueError", materialize=True, note="out-of-order window"),
            T(([f"{BASE} 200"], 0), raises="ValueError", materialize=True, note="window_seconds < 1"),
            CT(_q062_lazy, note="lazy: does not read past the next window"),
        ],
        examples=[f'''
>>> lines = ["{BASE} 200", "{BASE + 10} 500", "{BASE + 60} 503", "bad line", "{BASE + 61} 200"]
>>> list(error_rate_per_window(lines, 60))
[({BASE}, 0.5), ({BASE + 60}, 0.5)]
'''],
        complexity="O(n) time over the stream, O(1) memory.",
        extension="Emit empty windows too (rate 0.0 with count 0) so charts show gaps explicitly.",
    ),
    # ------------------------------------------------------------------------ Q063
    Question(
        id=63,
        title="Runtime Type-Checking Decorator from Annotations",
        difficulty="Advanced",
        category=FUNC,
        roles=[PYDEV, AIE, MLE],
        skills=["inspect.signature", "typing.get_type_hints", "get_origin/get_args", "decorators", "*args handling"],
        minutes=45,
        scenario=(
            "A feature-serving API is called from several teams' notebooks, and wrongly typed "
            "arguments (strings instead of floats, `True` instead of a count) cause confusing "
            "errors deep inside the code. You want a decorator that validates arguments against "
            "the function's type hints at the boundary."
        ),
        problem=(
            "Write a decorator `validate_types` that checks every bound argument (and the return "
            "value) against its annotation and raises `TypeError` on a mismatch."
        ),
        signature="def validate_types(func):",
        entry="validate_types",
        inputs=["`func`: a function whose parameters may be annotated with classes, `X | None`, `Optional[X]` or generics such as `list[int]`."],
        output="A wrapped function with the same name/docstring that behaves like `func` when the types are valid.",
        constraints=[
            "Plain classes: `isinstance` check, except that `bool` is **not** accepted for `int` or `float`, and `int` **is** accepted for `float`.",
            "Unions (`A | B`, `Optional[A]`): valid if any member matches; `None` matches `NoneType`.",
            "Generics (`list[int]`, `dict[str, float]`): check only the outer type (`list`, `dict`).",
            "`*args: T` checks every positional extra; `**kwargs: T` every keyword extra.",
            "Unannotated parameters and unsupported annotations (e.g. `typing.Any`) are not checked.",
            "The error message must name the parameter (or `return`).",
        ],
        criteria=["Correct acceptance and rejection rules.", "Unions and generics.", "Varargs and return checks.", "Metadata preserved."],
        hints=[
            "`inspect.signature(func).bind(*args, **kwargs)` maps arguments to parameter names; "
            "`typing.get_type_hints(func)` resolves annotations.",
            "Write `_matches(value, annotation)`: unwrap unions with `typing.get_origin`/`get_args` "
            "(origin is `typing.Union` or `types.UnionType`), reduce generics to their origin, then "
            "apply the bool/int/float rules.",
            "hints = get_type_hints(func); sig = signature(func)\nwrapper: bound = sig.bind(...)\nfor name, value in bound.arguments.items():\n"
            "    kind = sig.parameters[name].kind\n    values = value if VAR_POSITIONAL else value.values() if VAR_KEYWORD else [value]\n"
            "    check each against hints[name]\nresult = func(...); check 'return'",
        ],
        solution='''
import functools
import inspect
import types
import typing


def _matches(value, annotation) -> bool:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        return any(_matches(value, member) for member in typing.get_args(annotation))
    if origin is not None:
        annotation = origin
    if annotation is None or annotation is type(None):
        return value is None
    if not isinstance(annotation, type):
        return True  # unsupported annotation (e.g. typing.Any): do not check
    if isinstance(value, bool) and annotation in (int, float):
        return False
    if annotation is float:
        return isinstance(value, (int, float))
    return isinstance(value, annotation)


def validate_types(func):
    """Check arguments and return value against the function's type hints."""
    hints = typing.get_type_hints(func)
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            if name not in hints:
                continue
            kind = signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_POSITIONAL:
                values = value
            elif kind is inspect.Parameter.VAR_KEYWORD:
                values = value.values()
            else:
                values = [value]
            for item in values:
                if not _matches(item, hints[name]):
                    raise TypeError(f"argument {name!r} expected {hints[name]!r}, got {type(item).__name__}")
        result = func(*args, **kwargs)
        if "return" in hints and not _matches(result, hints["return"]):
            raise TypeError(f"return value expected {hints['return']!r}, got {type(result).__name__}")
        return result

    return wrapper
''',
        explanation=(
            "`signature.bind` reproduces Python's own argument matching, so positional, keyword "
            "and default arguments all end up under their parameter names. Resolving hints once at "
            "decoration time keeps each call cheap. `get_origin` exposes both union forms and "
            "generic aliases in a uniform way."
        ),
        edge_cases=["`True` rejected for `int`.", "`2` accepted for `float`.", "`str | None` accepts `None`.", "Unannotated params skipped."],
        mistakes=["Reading `func.__annotations__` directly (string annotations stay strings).", "`isinstance(x, list[int])` (TypeError).", "Ignoring `*args` values."],
        tests=[
            CT(_q063_basic, note="plain classes; bool/int/float rules; wraps"),
            CT(_q063_message, note="message names the parameter"),
            CT(_q063_optional_and_generics, note="unions and generic aliases"),
            CT(_q063_return_and_varargs, note="return value, *args, unannotated params"),
        ],
        examples=['''
>>> @validate_types
... def scale(x: float, factor: int = 2) -> float:
...     return x * factor
>>> scale(1.5)
3.0
>>> scale("1.5")
Traceback (most recent call last):
    ...
TypeError: argument 'x' expected <class 'float'>, got str
'''],
        complexity="O(number of arguments) per call after a one-time O(parameters) setup.",
        extension="Validate the element types of `list[int]` and `dict[str, float]` recursively.",
    ),
    # ------------------------------------------------------------------------ Q064
    Question(
        id=64,
        title="Top-K Nearest Neighbours by Cosine Similarity",
        difficulty="Advanced",
        category=ALG,
        roles=[AIE, AAI, MLE],
        skills=["vector math in pure Python", "heapq.nsmallest", "tie-breaking", "input validation"],
        minutes=35,
        scenario=(
            "A small retrieval-augmented generation (RAG) prototype stores document embeddings in "
            "a dict. Before adopting a vector database, you implement exact nearest-neighbour "
            "search to have a correct baseline for testing."
        ),
        problem=(
            "Write `nearest_neighbors(query, vectors, k)` that returns the `k` most similar "
            "documents by cosine similarity, highest first."
        ),
        signature="def nearest_neighbors(query: list[float], vectors: dict[str, list[float]], k: int) -> list[tuple[str, float]]:",
        entry="nearest_neighbors",
        inputs=["`query`: list of numbers.", "`vectors`: dict doc_id → list of numbers (same dimension as the query).", "`k`: `int` ≥ 0."],
        output="List of `(doc_id, similarity)` with similarity rounded to 4 decimals, length ≤ `k`.",
        constraints=[
            "cosine(a, b) = dot(a, b) / (|a| · |b|).",
            "Rank by the unrounded similarity descending, then doc_id ascending.",
            "Skip documents whose vector has zero length (all zeros).",
            "Raise `ValueError` for a zero query vector, mismatched dimensions or negative `k`.",
        ],
        criteria=["Correct similarities and ordering.", "Zero vectors skipped.", "Validation."],
        hints=[
            "Precompute the query norm once; then compute one similarity per document.",
            "`math.hypot(*v)` or `math.sqrt(sum(x*x for x in v))` gives the norm; "
            "`heapq.nsmallest(k, items, key=lambda it: (-sim, doc_id))` selects the top k without a full sort.",
            "validate\nq_norm = norm(query)\nscored = []\nfor doc, v in vectors.items():\n    check dims; n = norm(v); if n == 0: continue\n"
            "    scored.append((doc, dot(query, v) / (q_norm * n)))\nbest = nsmallest(k, scored, key=lambda s: (-s[1], s[0]))\nreturn [(d, round(s, 4)) for d, s in best]",
        ],
        solution='''
import heapq
import math


def nearest_neighbors(query: list[float], vectors: dict[str, list[float]], k: int) -> list[tuple[str, float]]:
    """Exact top-k cosine similarity search."""
    if k < 0:
        raise ValueError("k must be >= 0")
    query_norm = math.sqrt(sum(x * x for x in query))
    if query_norm == 0:
        raise ValueError("query vector must not be all zeros")
    scored = []
    for doc_id, vector in vectors.items():
        if len(vector) != len(query):
            raise ValueError(f"dimension mismatch for {doc_id!r}: {len(vector)} != {len(query)}")
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            continue
        dot = sum(a * b for a, b in zip(query, vector))
        scored.append((doc_id, dot / (query_norm * norm)))
    best = heapq.nsmallest(k, scored, key=lambda item: (-item[1], item[0]))
    return [(doc_id, round(score, 4)) for doc_id, score in best]
''',
        explanation=(
            "The query norm is computed once. `heapq.nsmallest` with the key `(-similarity, id)` "
            "returns the k best results in sorted order in O(n log k). Rounding happens only for "
            "the output, so ranking uses full precision."
        ),
        edge_cases=["Ties (e.g. parallel vectors) ordered by id.", "Negative similarities rank last.", "`k` larger than the number of documents."],
        mistakes=["Ranking by rounded values.", "Dividing by zero for empty vectors.", "Using Euclidean distance instead of cosine."],
        tests=[
            T(([1, 0, 0], VECTORS, 3), [("doc_a", 1.0), ("doc_e", 1.0), ("doc_b", 0.9939)], example=True,
              note="tie by id, zero vector skipped"),
            T(([1, 0, 0], VECTORS, 0), [], example=True, note="boundary: k = 0"),
            T(([1, 0, 0], VECTORS, 10), [("doc_a", 1.0), ("doc_e", 1.0), ("doc_b", 0.9939), ("doc_c", 0.0)], note="k larger than n"),
            T(([1, 0], {"x": [-1, 0], "y": [0, 1]}, 2), [("y", 0.0), ("x", -1.0)], note="negative similarity"),
            T(([1, 0], {"x": [1, 0, 0]}, 1), raises="ValueError", note="dimension mismatch"),
            T(([0, 0], {"x": [1, 0]}, 1), raises="ValueError", note="zero query"),
            T(([1, 0], {"x": [1, 0]}, -1), raises="ValueError", note="negative k"),
        ],
        complexity="O(n·d + n log k) time for n documents of dimension d; O(n) space.",
        extension="Pre-normalize all document vectors once so each query needs only dot products.",
    ),
    # ------------------------------------------------------------------------ Q065
    Question(
        id=65,
        title="Deterministic Execution Order for Pipeline Tasks",
        difficulty="Advanced",
        category=ALG,
        roles=[DE, MLE, PYDEV],
        skills=["topological sort (Kahn's algorithm)", "heapq for deterministic ordering", "cycle detection"],
        minutes=35,
        scenario=(
            "An ML pipeline defines tasks and their dependencies (`train` depends on `clean` and "
            "`features`). The scheduler must run tasks in an order that respects every dependency, "
            "be reproducible run to run, and refuse to start if the dependencies contain a cycle."
        ),
        problem=(
            "Write `execution_order(dependencies)` that returns a valid execution order. Whenever "
            "several tasks are ready, run the alphabetically smallest first."
        ),
        signature="def execution_order(dependencies: dict[str, list[str]]) -> list[str]:",
        entry="execution_order",
        inputs=["`dependencies`: task → list of tasks it depends on. Tasks that appear only as dependencies are tasks too. Lists may contain duplicates."],
        output="A list containing every task exactly once.",
        constraints=[
            "Every task appears after all of its dependencies.",
            "Among ready tasks, always choose the alphabetically smallest.",
            "Raise `ValueError` if there is a cycle (including a task depending on itself).",
            "Do not modify the input.",
        ],
        criteria=["Valid topological order.", "Deterministic tie-breaking.", "Cycle detection."],
        hints=[
            "Count how many unfinished dependencies each task has (its in-degree).",
            "Kahn's algorithm: start with all tasks of in-degree 0 in a min-heap; pop one, append it, "
            "and decrement the in-degree of tasks that depend on it. If you finish with tasks left "
            "over, there is a cycle.",
            "deps = {t: set(d) for ...} incl. tasks only seen as deps\ndependents = reverse map; indegree = {t: len(deps[t])}\n"
            "heap = [t for t, d in indegree.items() if d == 0]; heapify\nwhile heap: t = heappop; order.append(t); for d in dependents[t]: indegree[d] -= 1; push if 0\n"
            "if len(order) != len(tasks): raise ValueError",
        ],
        solution='''
import heapq


def execution_order(dependencies: dict[str, list[str]]) -> list[str]:
    """Topological order of tasks; ties broken alphabetically (Kahn's algorithm)."""
    requires: dict[str, set[str]] = {}
    for task, deps in dependencies.items():
        requires.setdefault(task, set()).update(deps)
        for dep in deps:
            requires.setdefault(dep, set())
    dependents: dict[str, list[str]] = {task: [] for task in requires}
    for task, deps in requires.items():
        for dep in deps:
            dependents[dep].append(task)
    remaining = {task: len(deps) for task, deps in requires.items()}
    ready = [task for task, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        task = heapq.heappop(ready)
        order.append(task)
        for dependent in dependents[task]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(order) != len(requires):
        stuck = sorted(task for task, count in remaining.items() if count > 0)
        raise ValueError(f"dependency cycle among tasks: {stuck}")
    return order
''',
        explanation=(
            "Kahn's algorithm only releases a task when all its prerequisites are done. Using a "
            "heap for the ready set makes the choice among ready tasks deterministic. Tasks in a "
            "cycle never reach in-degree 0, so a short result signals a cycle."
        ),
        edge_cases=["Tasks only mentioned as dependencies.", "Duplicate dependencies (sets).", "Self-dependency → cycle."],
        mistakes=["DFS without cycle detection (infinite recursion).", "Using a plain list/queue for the ready set (order depends on dict order).", "Forgetting tasks that have no entry of their own."],
        tests=[
            T(({"train": ["clean", "features"], "features": ["clean"], "clean": ["extract"], "evaluate": ["train"],
                "report": ["evaluate", "features"]},), ["extract", "clean", "features", "train", "evaluate", "report"],
              example=True, note="ML pipeline"),
            T(({"b": [], "a": [], "c": ["a"]},), ["a", "b", "c"], example=True, note="boundary: alphabetical tie-break"),
            T(({"z": [], "y": ["z"], "a": []},), ["a", "z", "y"], note="ready set changes over time"),
            T(({},), [], note="no tasks"),
            T(({"b": ["a", "a"]},), ["a", "b"], note="duplicate dependency, implicit task"),
            T(({"a": ["b"], "b": ["a"]},), raises="ValueError", note="cycle"),
            T(({"a": ["a"]},), raises="ValueError", note="self-dependency"),
        ],
        complexity="O((V + E) log V) time with the heap; O(V + E) space.",
        extension="Return execution *levels* – groups of tasks that can run in parallel.",
    ),
    # ------------------------------------------------------------------------ Q066
    Question(
        id=66,
        title="Fuzzy-Match User Input with Edit Distance",
        difficulty="Advanced",
        category=ALG,
        roles=[AIE, DS_ROLE, PYDEV],
        skills=["dynamic programming", "Levenshtein distance", "space optimization", "tie-breaking"],
        minutes=40,
        scenario=(
            "A command-line tool for your ML platform accepts library names. Users type `pyhton` "
            "or `tensorflw`; instead of failing, the tool should suggest the closest known name "
            "when it is close enough."
        ),
        problem=(
            "Write `fuzzy_match(query, candidates, max_distance)` that returns the candidate with "
            "the smallest case-insensitive Levenshtein distance to `query`, or `None` if no "
            "candidate is within `max_distance`. Implement the distance yourself."
        ),
        signature="def fuzzy_match(query: str, candidates: list[str], max_distance: int) -> str | None:",
        entry="fuzzy_match",
        inputs=["`query`: str.", "`candidates`: list of str (may be empty).", "`max_distance`: `int` ≥ 0."],
        output="The best candidate exactly as written in `candidates`, or `None`.",
        constraints=[
            "Levenshtein distance = minimum number of single-character insertions, deletions or substitutions.",
            "Compare lower-cased strings.",
            "Ties: the earliest candidate in the list wins.",
            "Raise `ValueError` if `max_distance < 0`.",
        ],
        criteria=["Correct distances.", "Threshold respected.", "Tie-breaking.", "Original casing returned."],
        hints=[
            "Distance between prefixes can be built from distances of shorter prefixes.",
            "Keep only the previous DP row: `curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (a[i-1] != b[j-1]))`.",
            "def lev(a, b):\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        curr = [i]\n"
            "        for j, cb in enumerate(b, 1): curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))\n        prev = curr\n    return prev[-1]\n"
            "best = None; best_d = max_distance + 1\nfor c in candidates: d = lev(q, c.lower()); if d < best_d: best, best_d = c, d",
        ],
        solution='''
def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,                    # deletion
                current[j - 1] + 1,                 # insertion
                previous[j - 1] + (char_a != char_b),  # substitution
            ))
        previous = current
    return previous[-1]


def fuzzy_match(query: str, candidates: list[str], max_distance: int) -> str | None:
    """Closest candidate within max_distance (case-insensitive), or None."""
    if max_distance < 0:
        raise ValueError("max_distance must be >= 0")
    target = query.lower()
    best, best_distance = None, max_distance + 1
    for candidate in candidates:
        distance = _levenshtein(target, candidate.lower())
        if distance < best_distance:
            best, best_distance = candidate, distance
    return best
''',
        explanation=(
            "The DP table entry (i, j) is the distance between the first i characters of `a` and "
            "the first j of `b`; each row depends only on the previous one, so O(min) memory "
            "suffices. Starting `best_distance` at `max_distance + 1` and using strict `<` "
            "implements both the threshold and the earliest-wins tie rule."
        ),
        edge_cases=["Empty query → distance equals candidate length.", "No candidates → `None`.", "Transpositions count as 2 edits."],
        mistakes=["Comparing case-sensitively.", "Using `<=` (later candidates win ties).", "Allocating the full n×m table for long strings (works, but wasteful)."],
        tests=[
            T(("pyhton", ["python", "java", "pytorch"], 2), "python", example=True, note="transposition = 2 edits"),
            T(("xyz", ["python"], 2), None, example=True, note="boundary: nothing close enough"),
            T(("SKLearn", ["scikit-learn", "sklearn", "keras"], 1), "sklearn", note="case-insensitive exact"),
            T(("tensorflw", ["TensorFlow", "tensorboard"], 1), "TensorFlow", note="original casing returned"),
            T(("", ["ab", "a"], 1), "a", note="empty query"),
            T(("cat", ["bat", "cot"], 1), "bat", note="tie → earliest"),
            T(("cat", [], 3), None, note="no candidates"),
            T(("x", ["x"], -1), raises="ValueError", note="negative threshold"),
        ],
        complexity="O(Σ |q|·|c|) time over candidates; O(|c|) space per comparison.",
        extension="Stop computing a row early once its minimum exceeds `max_distance` (banded DP).",
    ),
    # ------------------------------------------------------------------------ Q067
    Question(
        id=67,
        title="Minimum GPU Workers for a Job Schedule",
        difficulty="Advanced",
        category=ALG,
        roles=[MLE, DE, PYDEV],
        skills=["sweep-line algorithm", "event sorting with tie rules", "heaps", "complexity"],
        minutes=30,
        scenario=(
            "Training jobs are scheduled with fixed start and end times. Each job needs one GPU "
            "worker for its whole duration. To size the cluster you need the maximum number of "
            "jobs running at the same time."
        ),
        problem="Write `min_workers(jobs)` returning the minimum number of workers that can run all jobs.",
        signature="def min_workers(jobs: list[tuple[int, int]]) -> int:",
        entry="min_workers",
        inputs=["`jobs`: list of `(start, end)` pairs with integer times. May be empty."],
        output="An `int` ≥ 0.",
        constraints=[
            "Jobs are half-open `[start, end)`: a job ending at `t` frees its worker for a job starting at `t`.",
            "Raise `ValueError` if any job has `start >= end`.",
            "Must handle 50,000 jobs in well under a second (O(n log n)).",
        ],
        criteria=["Correct peak concurrency.", "Touching jobs share a worker.", "Efficient."],
        hints=[
            "Only start and end moments change the number of running jobs.",
            "Create +1 events for starts and −1 events for ends; sort by time with ends before starts "
            "at equal times, then track the running sum. (A min-heap of end times also works.)",
            "events = [(s, +1) ...] + [(e, -1) ...]\nevents.sort()  # (t, -1) sorts before (t, +1)\nrunning = best = 0\n"
            "for _, delta in events: running += delta; best = max(best, running)\nreturn best",
        ],
        solution='''
def min_workers(jobs: list[tuple[int, int]]) -> int:
    """Peak number of overlapping half-open jobs."""
    events = []
    for start, end in jobs:
        if start >= end:
            raise ValueError(f"job ({start}, {end}) must have start < end")
        events.append((start, 1))
        events.append((end, -1))
    events.sort()  # at equal times, -1 (end) sorts before +1 (start)
    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return peak
''',
        explanation=(
            "The number of busy workers only changes at event times. Sorting `(time, delta)` "
            "tuples places ends (−1) before starts (+1) at the same time, which implements the "
            "half-open rule; the running sum's maximum is the answer."
        ),
        edge_cases=["Touching jobs reuse one worker.", "Fully nested jobs all overlap.", "No jobs → 0."],
        mistakes=["Processing starts before ends at equal times.", "Checking every pair of jobs (O(n²)).", "Treating jobs as closed intervals."],
        tests=[
            T(([(0, 30), (5, 10), (15, 20)],), 2, example=True, note="classic"),
            T(([(1, 5), (5, 10)],), 1, example=True, note="boundary: touching jobs"),
            T(([],), 0, note="no jobs"),
            T(([(0, 10), (1, 9), (2, 8), (3, 7)],), 4, note="nested"),
            T(([(9, 12), (1, 4), (2, 6), (5, 8)],), 2, note="unsorted input"),
            T(([(3, 3)],), raises="ValueError", note="zero-length job"),
            CT(_q067_fast, note="50,000 jobs in under a second"),
        ],
        complexity="O(n log n) time for sorting the 2n events; O(n) space.",
        extension="Also return an assignment of each job to a worker id (heap of free workers).",
    ),
    # ------------------------------------------------------------------------ Q068
    Question(
        id=68,
        title="Stream-Aggregate a Large CSV by Group",
        difficulty="Advanced",
        category=FILE,
        roles=[DE, DS_ROLE, PYDEV],
        skills=["streaming file processing", "csv.DictReader", "constant-memory aggregation", "data validation"],
        minutes=35,
        scenario=(
            "A partner sends a multi-gigabyte CSV every night. Loading it into memory crashes the "
            "worker, but the report only needs per-group statistics. Rows with unusable values "
            "should be counted, not crash the job."
        ),
        problem=(
            "Write `aggregate_csv_stream(path, group_col, value_col)` that reads the file row by "
            "row and returns count, sum, min, max and mean of `value_col` per value of `group_col`."
        ),
        signature="def aggregate_csv_stream(path: str, group_col: str, value_col: str) -> dict:",
        entry="aggregate_csv_stream",
        inputs=["`path`: UTF-8 CSV with a header row.", "`group_col`, `value_col`: column names."],
        output=(
            "`{'groups': {group: {'count': int, 'sum': float, 'min': float, 'max': float, 'mean': float}}, "
            "'skipped_rows': int}`; `sum` and `mean` rounded to 2 decimals."
        ),
        constraints=[
            "Process one row at a time; memory must grow with the number of groups, not rows.",
            "Group values are stripped; a row is skipped if the group is empty/missing or the value "
            "is missing or does not parse as a finite float (negative values are allowed).",
            "Raise `ValueError` if either column is missing from the header (including an empty file).",
            "Blank lines are ignored entirely.",
        ],
        criteria=["Correct per-group statistics.", "Bad rows counted.", "Header validation.", "Streaming."],
        hints=[
            "Iterate over `csv.DictReader` directly – it reads lazily.",
            "Keep `[count, total, min, max]` per group in a dict; compute `mean` only at the end.",
            "with open(...) as f:\n    reader = DictReader(f); validate fieldnames\n    for row in reader:\n        group = (row.get(group_col) or '').strip()\n"
            "        try: value = float(row.get(value_col)) except (TypeError, ValueError): skipped += 1; continue\n"
            "        if not group or not isfinite(value): skipped += 1; continue\n        update stats",
        ],
        solution='''
import csv
import math


def aggregate_csv_stream(path: str, group_col: str, value_col: str) -> dict:
    """Per-group count/sum/min/max/mean computed in a single streaming pass."""
    stats: dict[str, list] = {}
    skipped = 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        for column in (group_col, value_col):
            if column not in header:
                raise ValueError(f"column {column!r} not found in header")
        for row in reader:
            group = (row.get(group_col) or "").strip()
            try:
                value = float(row.get(value_col))
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not group or not math.isfinite(value):
                skipped += 1
                continue
            entry = stats.get(group)
            if entry is None:
                stats[group] = [1, value, value, value]
            else:
                entry[0] += 1
                entry[1] += value
                entry[2] = min(entry[2], value)
                entry[3] = max(entry[3], value)
    groups = {
        group: {"count": n, "sum": round(total, 2), "min": low, "max": high, "mean": round(total / n, 2)}
        for group, (n, total, low, high) in stats.items()
    }
    return {"groups": groups, "skipped_rows": skipped}
''',
        explanation=(
            "`DictReader` yields one row at a time, so only the per-group accumulators stay in "
            "memory. `float(None)` raises `TypeError` for truncated rows, and `math.isfinite` "
            "rejects `nan`/`inf`, so all bad values follow the same 'skip and count' path."
        ),
        edge_cases=["Header-only file → no groups, 0 skipped.", "`nan` and `inf` skipped.", "Negative values are valid here."],
        mistakes=["`list(reader)` or `readlines()` (loads everything).", "Storing every value to compute min/max later.", "Crashing on truncated rows (`None` values)."],
        tests=[
            T((File("sales.csv", "region,revenue,notes\nEU,100.5,a\nUS,200,b\nEU,abc,c\n,50,d\n\nEU,49.5,e\nUS,-20,f\nAPAC,nan,g\n"),
               "region", "revenue"),
              {"groups": {"EU": {"count": 2, "sum": 150.0, "min": 49.5, "max": 100.5, "mean": 75.0},
                          "US": {"count": 2, "sum": 180.0, "min": -20.0, "max": 200.0, "mean": 90.0}}, "skipped_rows": 3},
              example=True, note="valid, invalid and blank rows"),
            T((File("header.csv", "region,revenue\n"), "region", "revenue"), {"groups": {}, "skipped_rows": 0},
              example=True, note="boundary: header only"),
            T((File("short.csv", "g,v\nA,1\nB\n A ,2\n"), "g", "v"),
              {"groups": {"A": {"count": 2, "sum": 3.0, "min": 1.0, "max": 2.0, "mean": 1.5}}, "skipped_rows": 1},
              note="truncated row skipped, group stripped"),
            T((File("x.csv", "a,b\n1,2\n"), "region", "b"), raises="ValueError", note="missing column"),
            T((File("empty.csv", ""), "a", "b"), raises="ValueError", note="empty file"),
            CT(_q068_sample, note="data/sample_data/transactions.csv grouped by customer"),
        ],
        complexity="O(n) time, O(g) memory for g groups.",
        extension="Add an approximate median per group using a fixed-size reservoir sample.",
    ),
    # ------------------------------------------------------------------------ Q069
    Question(
        id=69,
        title="Read Rotated (and Gzipped) Log Files in Order",
        difficulty="Advanced",
        category=FILE,
        roles=[DE, PYDEV, AIE],
        skills=["pathlib globbing", "numeric sorting of file names", "gzip", "generators", "validation"],
        minutes=35,
        scenario=(
            "A log rotation tool keeps `app.log` (current), `app.log.1` (previous), `app.log.2.gz`, "
            "… where larger numbers are older and some files are compressed. Incident analysis "
            "needs every line in chronological order, without decompressing everything to disk."
        ),
        problem=(
            "Write a generator `iter_rotated_logs(directory, base_name)` that yields every line of "
            "the rotated files from oldest to newest, then the current file."
        ),
        signature="def iter_rotated_logs(directory: str, base_name: str):",
        entry="iter_rotated_logs",
        inputs=["`directory`: folder path.", "`base_name`: current log file name, e.g. `'app.log'`."],
        output="An iterator of lines (strings without the trailing `\\n` or `\\r\\n`).",
        constraints=[
            "Rotated files are `<base>.<N>` or `<base>.<N>.gz` with a positive integer `N`; read them in "
            "**descending numeric** order (`.10` is older than `.2`), then `<base>` if it exists.",
            "Ignore other files (e.g. `app.log.bak`, `other.log`).",
            "Raise `ValueError` if both `<base>.N` and `<base>.N.gz` exist; `FileNotFoundError` if the directory does not exist.",
            "Decode as UTF-8 and stream lazily (do not read whole files into memory).",
        ],
        criteria=["Correct chronological order.", "Gzip support.", "Irrelevant files ignored.", "Lazy generator."],
        hints=[
            "Collect `(N, path)` pairs first, then sort numerically.",
            "Use a regex like `re.escape(base) + r'\\.(\\d+)(\\.gz)?'` with `fullmatch` on each file "
            "name; open `.gz` files with `gzip.open(path, 'rt', encoding='utf-8')`.",
            "rotated = {}\nfor p in Path(directory).iterdir(): m = PATTERN.fullmatch(p.name); if m: check duplicate N; rotated[int(m[1])] = p\n"
            "paths = [rotated[n] for n in sorted(rotated, reverse=True)] + ([current] if current.exists() else [])\n"
            "for p in paths: with opener(p) as f: for line in f: yield line.rstrip('\\r\\n')",
        ],
        solution='''
import gzip
import re
from pathlib import Path


def iter_rotated_logs(directory: str, base_name: str):
    """Yield lines from base.N(.gz) files (oldest first), then from base."""
    folder = Path(directory)
    pattern = re.compile(re.escape(base_name) + r"\\.(\\d+)(\\.gz)?")
    rotated: dict[int, Path] = {}
    for path in folder.iterdir():
        match = pattern.fullmatch(path.name)
        if not match or int(match.group(1)) < 1:
            continue
        number = int(match.group(1))
        if number in rotated:
            raise ValueError(f"both {rotated[number].name} and {path.name} exist")
        rotated[number] = path
    paths = [rotated[n] for n in sorted(rotated, reverse=True)]
    current = folder / base_name
    if current.exists():
        paths.append(current)
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield line.rstrip("\\r\\n")
''',
        explanation=(
            "Sorting by the parsed integer avoids the string-sort trap (`'10' < '2'`). Choosing "
            "the opener per file lets plain and gzip files share one loop, and yielding line by "
            "line keeps memory constant. The function is a generator, so nothing happens until "
            "the caller iterates."
        ),
        edge_cases=["No current file (just rotated).", "Windows line endings stripped.", "Duplicate `N` in plain and gzip form."],
        mistakes=["Sorting file names as strings.", "Using `glob('app.log*')` and accidentally including `app.log.bak`.", "Reading compressed bytes without text mode."],
        tests=[
            CT(_q069_order, note="numeric order, gzip, ignored files"),
            CT(_q069_no_current_file, note="only rotated files; CRLF stripped"),
            CT(_q069_duplicate, note="duplicate rotation number"),
            CT(_q069_missing_dir, note="missing directory"),
            T((DataFile("rotated_logs"), "app.log"),
              ["2024-05-01T08:00:00Z INFO [gateway] service started", "2024-05-01T08:05:00Z WARNING [gateway] slow upstream 1840ms",
               "2024-05-02T08:00:00Z INFO [gateway] daily restart", "2024-05-02T09:12:00Z ERROR [gateway] upstream timeout",
               "2024-05-03T08:00:00Z INFO [gateway] daily restart", "2024-05-03T10:30:00Z INFO [gateway] config reloaded",
               "2024-05-04T08:00:00Z INFO [gateway] daily restart", "2024-05-04T11:45:00Z ERROR [gateway] connection reset"],
              materialize=True, example=True, note="data/sample_data/rotated_logs"),
        ],
        complexity="O(total lines) time, O(f log f) to sort f files, O(1) memory per line.",
        extension="Accept a `since` timestamp and skip whole files whose last line is older than it.",
    ),
    # ------------------------------------------------------------------------ Q070
    Question(
        id=70,
        title="MetricSeries: a Container Class with Operators",
        difficulty="Advanced",
        category=OOP,
        roles=[MLE, DS_ROLE, PYDEV],
        skills=["__len__/__iter__/__getitem__", "slicing", "__add__/__radd__/__mul__", "NotImplemented", "__eq__/__repr__"],
        minutes=40,
        scenario=(
            "Training dashboards manipulate per-epoch metric curves: slice the last 10 epochs, add "
            "two loss components, scale by a constant. You want a small, well-behaved container "
            "class instead of passing raw lists around."
        ),
        problem=(
            "Implement `MetricSeries(name, values)` supporting `len`, iteration, indexing and "
            "slicing, `+` and `*` with scalars and other series, equality, `repr` and `mean()`."
        ),
        signature="class MetricSeries:",
        entry="MetricSeries",
        starter='''
class MetricSeries:
    """A named, immutable-by-convention sequence of metric values."""

    def __init__(self, name: str, values):
        # TODO
        pass
''',
        inputs=["`name`: str.", "`values`: iterable of ints/floats (`bool` and other types are invalid)."],
        output="A `MetricSeries` object with the behaviour below.",
        constraints=[
            "Store a private copy of the values; raise `TypeError` for non-numeric values or `bool`.",
            "`s[i]` → a value; `s[a:b]` → a new `MetricSeries` with the same name.",
            "`s + t` (same length) → elementwise sum named `'<s.name>+<t.name>'`; different lengths → `ValueError`.",
            "`s + c`, `c + s`, `s * c`, `c * s` for numbers `c` → elementwise, name unchanged.",
            "Unsupported operand types → `TypeError` (return `NotImplemented`). Operands are never modified.",
            "`==` compares name and values; `repr(s)` → `MetricSeries('loss', [0.7, 0.5])`; `mean()` → `None` when empty.",
        ],
        criteria=["Container protocol.", "Operators incl. reflected ones.", "Correct NotImplemented use.", "Encapsulation."],
        hints=[
            "Python calls `__radd__`/`__rmul__` on the right operand when the left one returns `NotImplemented` (e.g. `1 + s`).",
            "In `__getitem__`, check `isinstance(index, slice)`. Factor scalar-vs-series logic into a helper `_combine(other, op, name)`.",
            "__add__(other): if MetricSeries: check len; return MetricSeries(f'{a}+{b}', [x + y ...])\n"
            "elif number (not bool): return MetricSeries(name, [x + other ...])\nelse: return NotImplemented\n__radd__ = __add__ for scalars",
        ],
        solution='''
from numbers import Real


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


class MetricSeries:
    """A named sequence of metric values supporting arithmetic."""

    def __init__(self, name: str, values):
        values = list(values)
        for value in values:
            if not _is_number(value):
                raise TypeError(f"metric values must be numbers, got {value!r}")
        self.name = name
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MetricSeries(self.name, self._values[index])
        return self._values[index]

    def __eq__(self, other) -> bool:
        if not isinstance(other, MetricSeries):
            return NotImplemented
        return self.name == other.name and self._values == other._values

    def __repr__(self) -> str:
        return f"MetricSeries({self.name!r}, {self._values!r})"

    def _combine(self, other, op, symbol: str):
        if isinstance(other, MetricSeries):
            if len(other) != len(self):
                raise ValueError(f"length mismatch: {len(self)} vs {len(other)}")
            return MetricSeries(f"{self.name}{symbol}{other.name}",
                                [op(a, b) for a, b in zip(self._values, other._values)])
        if _is_number(other):
            return MetricSeries(self.name, [op(a, other) for a in self._values])
        return NotImplemented

    def __add__(self, other):
        return self._combine(other, lambda a, b: a + b, "+")

    def __mul__(self, other):
        return self._combine(other, lambda a, b: a * b, "*")

    def __radd__(self, other):
        return self + other if _is_number(other) else NotImplemented

    def __rmul__(self, other):
        return self * other if _is_number(other) else NotImplemented

    def mean(self) -> float | None:
        return sum(self._values) / len(self._values) if self._values else None
''',
        explanation=(
            "Implementing the sequence protocol (`__len__`, `__iter__`, `__getitem__`) makes the "
            "class work with `list()`, `for`, `sum()` and slicing. Returning `NotImplemented` (not "
            "raising) lets Python try the reflected method and finally raise a standard "
            "`TypeError`. New objects are always returned, so operands never change."
        ),
        edge_cases=["Negative indexes and slices.", "`1 + s` uses `__radd__`.", "Empty series → `mean()` is `None`."],
        mistakes=["Storing the caller's list directly (aliasing).", "Returning a plain list from slices.", "Raising `TypeError` inside `__add__` instead of returning `NotImplemented`."],
        tests=[
            CT(_q070_container, note="len, iteration, indexing, slicing, repr"),
            CT(_q070_arithmetic, note="+ and * with series and scalars; NotImplemented"),
            CT(_q070_validation_and_copy, note="validation, private copy, mean, equality"),
            CT(_q070_sum, note="works with sum() and iteration"),
        ],
        examples=['''
>>> loss = MetricSeries("loss", [0.9, 0.7, 0.5, 0.4])
>>> len(loss), loss[0], loss[-1]
(4, 0.9, 0.4)
>>> loss[1:3]
MetricSeries('loss', [0.7, 0.5])
>>> MetricSeries("a", [1, 2]) + MetricSeries("b", [10, 20])
MetricSeries('a+b', [11, 22])
>>> 2 * MetricSeries("a", [1, 2])
MetricSeries('a', [2, 4])
'''],
        complexity="O(n) for arithmetic and slicing; O(1) for indexing and `len`.",
        extension="Add `rolling_mean(window)` returning a new series, and make the class hashable by storing a tuple.",
    ),
]
