"""Advanced questions Q071–Q085: complex transformations, algorithms and debugging."""

import json
import random
import statistics
import time
import tracemalloc
from bisect import bisect_right

from ..schema import (
    AAI, AIE, DATA, DBG, DE, DS_ROLE, INT, MLE, OOP, PERF, PYDEV, SAMPLE_DATA_DIR,
    CT, Question, T,
)


def _load(name: str):
    return json.loads((SAMPLE_DATA_DIR / name).read_text(encoding="utf-8"))


def _raises(exc_type, func, message):
    try:
        func()
    except exc_type:
        return
    raise AssertionError(message)


# ------------------------------------------------------------------ Q071 tests
def _q071_range(impl, tmp):
    v = impl({"type": "range", "min": 0, "max": 1})
    assert v.validate(0.5) == [] and v.validate(0) == [] and v.validate(1) == [], "bounds are inclusive"
    assert v.validate(1.5) == ["must be <= 1"], f"got {v.validate(1.5)}"
    assert v.validate(-0.1) == ["must be >= 0"], f"got {v.validate(-0.1)}"
    assert v.validate("x") == ["expected a number"] and v.validate(True) == ["expected a number"]
    assert impl({"type": "range", "min": 10}).validate(10**9) == [], "a missing max means no upper bound"


def _q071_regex_one_of(impl, tmp):
    r = impl({"type": "regex", "pattern": "[a-z]+-v[0-9]+"})
    assert r.validate("bert-v2") == []
    assert r.validate("Bert-v2") == ["must match pattern '[a-z]+-v[0-9]+'"], f"got {r.validate('Bert-v2')}"
    assert r.validate("bert-v2-extra") != [], "the whole string must match (fullmatch)"
    assert r.validate(42) == ["expected a string"]
    o = impl({"type": "one_of", "choices": ["cpu", "gpu"]})
    assert o.validate("gpu") == [] and o.validate("tpu") == ["must be one of ['cpu', 'gpu']"]


def _q071_composite(impl, tmp):
    v = impl({"type": "all", "validators": [{"type": "range", "min": 0}, {"type": "one_of", "choices": [1, 2, 3]}]})
    assert v.validate(2) == []
    assert v.validate(5) == ["must be one of [1, 2, 3]"]
    assert v.validate(-1) == ["must be >= 0", "must be one of [1, 2, 3]"], "'all' must run every child and keep their order"
    opt = impl({"type": "optional", "validator": {"type": "regex", "pattern": "[A-Z]{2}"}})
    assert opt.validate(None) == [] and opt.validate("US") == [] and opt.validate("us") == ["must match pattern '[A-Z]{2}'"]
    nested = impl({"type": "optional", "validator": {"type": "all", "validators": [{"type": "range", "max": 5}]}})
    assert nested.validate(9) == ["must be <= 5"], "validators must compose recursively"


def _q071_bad_specs(impl, tmp):
    for spec in ({"type": "email"}, {"type": "regex"}, {"type": "one_of"}, {"min": 1},
                 {"type": "all", "validators": [{"type": "nope"}]}):
        _raises(ValueError, lambda spec=spec: impl(spec), f"build_validator({spec!r}) should raise ValueError")


# ------------------------------------------------------------------ Q072 tests
def _q072_order_and_wildcard(impl, tmp):
    bus, calls = impl(), []
    bus.subscribe("*", lambda t, p: calls.append(("wild", t)))
    bus.subscribe("model.deployed", lambda t, p: calls.append(("a", p["version"])))
    bus.subscribe("model.deployed", lambda t, p: calls.append(("b", p["version"])))
    result = bus.publish("model.deployed", {"version": 3})
    assert calls == [("a", 3), ("b", 3), ("wild", "model.deployed")], f"unexpected call order {calls}"
    assert result == {"delivered": 3, "errors": []}, f"unexpected publish result {result}"
    assert bus.publish("nobody.listens", {}) == {"delivered": 1, "errors": []}, "wildcard handlers receive every event"
    assert bus.subscriber_count("model.deployed") == 2


def _q072_errors_isolated(impl, tmp):
    bus, calls = impl(), []

    def broken(event_type, payload):
        raise RuntimeError("boom")

    bus.subscribe("x", broken)
    bus.subscribe("x", lambda t, p: calls.append("after"))
    result = bus.publish("x", None)
    assert calls == ["after"], "a failing handler must not stop the others"
    assert result == {"delivered": 1, "errors": [("broken", "boom")]}, f"unexpected result {result}"


def _q072_unsubscribe(impl, tmp):
    bus, calls = impl(), []
    handler = lambda t, p: calls.append(p)  # noqa: E731
    unsubscribe = bus.subscribe("e", handler)
    again = bus.subscribe("e", handler)
    assert bus.subscriber_count("e") == 1, "subscribing the same handler twice to one event is ignored"
    bus.publish("e", 1)
    unsubscribe()
    unsubscribe()  # second call is harmless
    again()
    bus.publish("e", 2)
    assert calls == [1] and bus.subscriber_count("e") == 0


def _q072_unsubscribe_during_publish(impl, tmp):
    bus, calls = impl(), []
    holder = {}

    def once(event_type, payload):
        calls.append("once")
        holder["unsub"]()

    holder["unsub"] = bus.subscribe("tick", once)
    bus.subscribe("tick", lambda t, p: calls.append("other"))
    bus.publish("tick", None)
    bus.publish("tick", None)
    assert calls == ["once", "other", "other"], f"iterate over a snapshot of handlers; got {calls}"
    _raises(TypeError, lambda: bus.subscribe("tick", "not callable"), "non-callable handlers should raise TypeError")


# ------------------------------------------------------------------ Q073 tests
def _median_ok_a(values):
    return statistics.median(values)


def _median_ok_b(values):
    if not values:
        raise ValueError("empty")
    data = sorted(values)
    mid = len(data) // 2
    return float(data[mid]) if len(data) % 2 else (data[mid - 1] + data[mid]) / 2


def _mutant_unsorted(values):
    if not values:
        raise ValueError("empty")
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _mutant_upper_middle(values):
    if not values:
        raise ValueError("empty")
    return sorted(values)[len(values) // 2]


def _mutant_empty_zero(values):
    if not values:
        return 0.0
    return statistics.median(values)


def _mutant_in_place(values):
    if not values:
        raise ValueError("empty")
    values.sort()
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _mutant_floor_div(values):
    if not values:
        raise ValueError("empty")
    data = sorted(values)
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else (data[mid - 1] + data[mid]) // 2


def _mutant_mean(values):
    if not values:
        raise ValueError("empty")
    return sum(values) / len(values)


def _mutant_off_by_one(values):
    if not values:
        raise ValueError("empty")
    data = sorted(values)
    mid = len(data) // 2
    return data[mid - 1] if len(data) % 2 else (data[mid - 1] + data[mid]) / 2


def _accepts(name, median):
    def test(impl, tmp):
        try:
            impl(median)
        except Exception as exc:
            raise AssertionError(f"your checker rejected a CORRECT implementation ({name}): "
                                 f"{type(exc).__name__}: {exc}") from None
    return test


def _rejects(description, mutant):
    def test(impl, tmp):
        try:
            impl(mutant)
        except Exception:
            return
        raise AssertionError(f"your checker accepted a broken median that {description}")
    return test


# ------------------------------------------------------------------ Q074 tests
def _q074_values(impl, tmp):
    checks = impl({"latency_ms": 500, "error_rate": 0.01, "cpu": 0.9})
    assert sorted(checks) == ["cpu", "error_rate", "latency_ms"]
    results = (checks["latency_ms"](450), checks["latency_ms"](650), checks["error_rate"](0.02),
               checks["error_rate"](0.01), checks["cpu"](0.95))
    assert results == (True, False, False, True, False), (f"got {results}; every check seems to use the "
                                                          "same limit – the classic late-binding closure bug")


def _q074_names_and_snapshot(impl, tmp):
    thresholds = {"latency_ms": 500}
    checks = impl(thresholds)
    thresholds["latency_ms"] = 1
    assert checks["latency_ms"](450) is True, "checks must capture the limits at creation time"
    assert checks["latency_ms"].__name__ == "check_latency_ms", f"__name__ should be 'check_latency_ms', got {checks['latency_ms'].__name__!r}"


def _q074_many(impl, tmp):
    checks = impl({f"m{i}": i for i in range(100)})
    wrong = [i for i in range(100) if not (checks[f"m{i}"](i) and not checks[f"m{i}"](i + 1))]
    assert not wrong, f"checks for metrics {wrong[:5]}... do not use their own limits"


def _q074_validation(impl, tmp):
    assert impl({}) == {}
    _raises(TypeError, lambda: impl({"x": "500"}), "non-numeric thresholds should raise TypeError")
    _raises(TypeError, lambda: impl({"x": True}), "bool thresholds should raise TypeError")


# ------------------------------------------------------------------ Q075 tests
def _q075_in_place(impl, tmp):
    sessions = {"s1": 100.0, "s2": 940.0, "s3": 400.0, "s4": 700.0}
    same = sessions
    removed = impl(sessions, 1000.0, 300.0)
    assert removed == ["s1", "s3"], f"expected ['s1', 's3'], got {removed}"
    assert same is sessions and sessions == {"s2": 940.0, "s4": 700.0}, f"the dict must be modified in place; now {sessions}"


# ------------------------------------------------------------------ Q076 tests
def _q076_sample(impl, tmp):
    events = _load("events.json")
    sessions = impl(events, 30)
    assert len(sessions) == 28, f"expected 28 sessions in events.json, got {len(sessions)}"
    assert sum(s["event_count"] for s in sessions) == len(events) == 80
    assert sessions[0]["user_id"] == "u01", "results are sorted by user then start time"


# ------------------------------------------------------------------ Q077 tests
def _q077_fast(impl, tmp):
    left = [{"id": i, "a": i} for i in range(20_000)]
    right = [{"id": i, "b": -i} for i in range(19_999, -1, -1)]
    start = time.perf_counter()
    out = impl(left, right, "id")
    elapsed = time.perf_counter() - start
    assert len(out) == 20_000 and out[123] == {"id": 123, "a": 123, "b": -123}
    assert elapsed < 1.0, f"took {elapsed:.2f}s – index the right side in a dict instead of scanning it per row"


def _q077_new_dicts(impl, tmp):
    left = [{"k": 1, "v": "x"}, {"k": 2, "v": "y"}]
    out = impl(left, [{"k": 1, "w": 1}], "k")
    out[0]["v"] = "changed"
    out[1]["v"] = "changed"
    assert left == [{"k": 1, "v": "x"}, {"k": 2, "v": "y"}], "output rows must be new dicts, even for unmatched rows"


# ------------------------------------------------------------------ Q078 tests
def _q078_sample(impl, tmp):
    flagged = impl(_load("model_metrics.json"), 3.0)
    assert [(r["metric"], r["ts"]) for r in flagged] == [
        ("latency_p95_ms", "2024-05-01T13:00:00Z"), ("throughput_rps", "2024-05-02T02:00:00Z"),
    ], f"unexpected anomalies: {[(r['metric'], r['ts'], r['z_score']) for r in flagged]}"


# ------------------------------------------------------------------ Q079 tests
def _q079_fast(impl, tmp):
    rng = random.Random(3)
    values = [rng.randint(0, 10_000_000) for _ in range(50_000)]
    ordered = sorted(values)
    expected = sum(bisect_right(ordered, v + 50, i + 1) - (i + 1) for i, v in enumerate(ordered))
    start = time.perf_counter()
    got = impl(values, 50)
    elapsed = time.perf_counter() - start
    assert got == expected, f"expected {expected}, got {got}"
    assert elapsed < 1.0, f"took {elapsed:.2f}s for 50,000 values – sort once and use two pointers"


# ------------------------------------------------------------------ Q080 tests
def _q080_generator(impl, tmp):
    stream = ({"id": i, "score": (i * 7919) % 1000} for i in range(5_000))
    got = impl(stream, 2, "score")
    assert got == [{"id": 321, "score": 999}, {"id": 1321, "score": 999}], f"got {got}"


def _q080_memory(impl, tmp):
    def stream():
        for i in range(200_000):
            yield {"id": i, "payload": "x" * 50, "score": (i * 7919) % 100_003}

    tracemalloc.start()
    try:
        result = impl(stream(), 5, "score")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert [r["score"] for r in result] == [100_002] * 2 + [100_001] * 2 + [100_000], f"wrong result {[r['score'] for r in result]}"
    assert peak < 2_000_000, f"peak memory {peak / 1e6:.1f} MB – keep only k items (a bounded heap), not the whole stream"


# ------------------------------------------------------------------ Q081 tests
class FakeProfiles:
    """Callable profile service used in Q081 tests (raises KeyError for unknown users)."""

    def __init__(self, profiles: dict):
        self.profiles = profiles

    def __call__(self, user_id):
        return dict(self.profiles[user_id])

    def __eq__(self, other):
        return isinstance(other, FakeProfiles) and self.profiles == other.profiles

    def __repr__(self):
        return f"FakeProfiles({len(self.profiles)} users)"


PROFILES = FakeProfiles({"u1": {"country": "IN", "tier": "pro"}, "u2": {"country": "DE", "tier": "free"}})


def _q081_calls_once(impl, tmp):
    calls = []

    def lookup(user_id):
        calls.append(user_id)
        return PROFILES(user_id)

    events = [{"user_id": u, "n": i} for i, u in enumerate(["u1", "u2", "u1", "ghost", "u2", "ghost", "u1"])]
    out = impl(events, lookup)
    assert sorted(calls) == ["ghost", "u1", "u2"], f"lookup must be called once per distinct user (incl. unknown ones); calls={calls}"
    assert [e["user_country"] for e in out] == ["IN", "DE", "IN", None, "DE", None, "IN"]


def _q081_other_errors(impl, tmp):
    def lookup(user_id):
        raise ConnectionError("profile service down")

    _raises(ConnectionError, lambda: impl([{"user_id": "u1"}], lookup), "errors other than KeyError must propagate")


# ------------------------------------------------------------------ data-backed tests
def _q083_sample(impl, tmp):
    report = impl(_load("prediction_logs.json"))
    assert report["accuracy"] == 0.775, f"expected accuracy 0.775, got {report['accuracy']}"
    assert sorted(report["per_class"]) == ["billing", "bug", "feature_request", "other"]
    assert report["per_class"]["bug"]["support"] == 40


def _q084_sample(impl, tmp):
    requests = []
    for line in (SAMPLE_DATA_DIR / "inference_requests.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            requests.append(record)
    report = impl(requests)
    assert sorted(report) == ["/v1/classify", "/v1/embed", "/v1/generate"]
    assert sum(r["count"] for r in report.values()) == 48
    gen = report["/v1/generate"]
    assert gen["p50"] <= gen["p95"] <= gen["p99"], "percentiles must be non-decreasing"


def _q085_sample(impl, tmp):
    summary = impl(_load("retry_logs.json"))
    assert summary["requests"] == 30, f"expected 30 requests, got {summary['requests']}"
    assert summary["gaps"] == ["req-017"], f"expected gaps ['req-017'], got {summary['gaps']}"
    assert summary["succeeded_first_try"] + summary["succeeded_after_retry"] + summary["failed"] == 30


LLM_SCHEMA = {
    "type": "object",
    "required": ["sentiment", "confidence", "entities"],
    "additional": False,
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "entities": {"type": "array", "items": {
            "type": "object", "required": ["name", "type"],
            "properties": {"name": {"type": "string", "max_length": 50},
                           "type": {"type": "string", "enum": ["person", "org", "product"]}},
        }},
        "summary": {"type": "string", "max_length": 200},
    },
}

RETRY_RECORDS = [
    {"request_id": "r1", "attempt": 1, "status": "success"},
    {"request_id": "r2", "attempt": 1, "status": "retryable_error"},
    {"request_id": "r2", "attempt": 2, "status": "success"},
    {"request_id": "r3", "attempt": 1, "status": "retryable_error"},
    {"request_id": "r3", "attempt": 2, "status": "retryable_error"},
    {"request_id": "r3", "attempt": 3, "status": "fatal_error"},
    {"request_id": "r4", "attempt": 2, "status": "success"},
    {"request_id": "r4", "attempt": 1, "status": "retryable_error"},
    {"request_id": "r4", "attempt": 1, "status": "retryable_error"},
    {"request_id": "r5", "attempt": 1, "status": "retryable_error"},
    {"request_id": "r5", "attempt": 3, "status": "success"},
]

SESSION_EVENTS = [
    {"user_id": "u1", "timestamp": "2024-05-01T10:10:00Z"},
    {"user_id": "u1", "timestamp": "2024-05-01T09:00:00Z"},
    {"user_id": "u2", "timestamp": "2024-05-01T09:05:00Z"},
    {"user_id": "u1", "timestamp": "2024-05-01T09:39:00Z"},
    {"user_id": "u1", "timestamp": "2024-05-01T10:40:00Z"},
    {"user_id": "u1", "timestamp": "2024-05-01T09:10:00Z"},
]

QUESTIONS = [
    # ------------------------------------------------------------------------ Q071
    Question(
        id=71,
        title="Composable Field Validators (Polymorphism and Composition)",
        difficulty="Advanced",
        category=OOP,
        roles=[DE, PYDEV, AIE],
        skills=["class hierarchies", "polymorphism", "composite pattern", "factory functions", "recursion"],
        minutes=40,
        scenario=(
            "A data-contract service lets teams describe validation rules as JSON, e.g. "
            "`{'type': 'range', 'min': 0, 'max': 1}` for a probability. Rules can be combined "
            "(`all`) or made optional. The service turns each spec into a validator object once and "
            "reuses it for millions of values."
        ),
        problem=(
            "Write `build_validator(spec)` that returns an object with a `validate(value) -> list[str]` "
            "method (empty list = valid). Implement it with a small class hierarchy: one class per rule "
            "type, sharing a common interface."
        ),
        signature="def build_validator(spec: dict):",
        entry="build_validator",
        inputs=[
            "`{'type': 'range', 'min': number (optional), 'max': number (optional)}`",
            "`{'type': 'regex', 'pattern': str}`",
            "`{'type': 'one_of', 'choices': list}`",
            "`{'type': 'all', 'validators': [spec, ...]}`",
            "`{'type': 'optional', 'validator': spec}`",
        ],
        output="A validator object whose `validate(value)` returns a list of error strings.",
        constraints=[
            "range: non-numbers and bools → `['expected a number']`; below min → `'must be >= {min}'`; above max → `'must be <= {max}'` (inclusive bounds).",
            "regex: non-strings → `['expected a string']`; no full match → `\"must match pattern '{pattern}'\"` (pattern shown with `repr`).",
            "one_of: value not in choices → `'must be one of {choices!r}'`.",
            "all: run every child validator and concatenate their errors in order.",
            "optional: `None` is valid; anything else is passed to the inner validator.",
            "Raise `ValueError` from `build_validator` for an unknown type or a missing required key (also inside nested specs).",
        ],
        criteria=["Exact error messages.", "Recursive composition.", "Validation of specs at build time.", "Validators are reusable."],
        hints=[
            "Give every rule class the same method, `validate(value) -> list[str]`; composite rules hold other validators.",
            "Use a dict mapping type names to builder functions/classes, and let `build_validator` call itself for nested specs. "
            "Convert `KeyError` for missing keys into `ValueError`.",
            "class Range: __init__(min, max); validate(v): type check, bound checks\nclass AllOf: __init__(children); validate(v): sum of child errors\n"
            "class Optional_: validate(v): [] if v is None else inner.validate(v)\nbuild_validator: kind = spec.get('type'); dispatch; recurse for children",
        ],
        solution='''
import re


class Validator:
    """Common interface: validate(value) -> list of error messages."""

    def validate(self, value) -> list[str]:
        raise NotImplementedError


class RangeValidator(Validator):
    def __init__(self, minimum=None, maximum=None):
        self.minimum, self.maximum = minimum, maximum

    def validate(self, value) -> list[str]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return ["expected a number"]
        errors = []
        if self.minimum is not None and value < self.minimum:
            errors.append(f"must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            errors.append(f"must be <= {self.maximum}")
        return errors


class RegexValidator(Validator):
    def __init__(self, pattern: str):
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    def validate(self, value) -> list[str]:
        if not isinstance(value, str):
            return ["expected a string"]
        return [] if self._compiled.fullmatch(value) else [f"must match pattern {self.pattern!r}"]


class OneOfValidator(Validator):
    def __init__(self, choices: list):
        self.choices = list(choices)

    def validate(self, value) -> list[str]:
        return [] if value in self.choices else [f"must be one of {self.choices!r}"]


class AllOfValidator(Validator):
    def __init__(self, children: list[Validator]):
        self.children = children

    def validate(self, value) -> list[str]:
        return [error for child in self.children for error in child.validate(value)]


class OptionalValidator(Validator):
    def __init__(self, inner: Validator):
        self.inner = inner

    def validate(self, value) -> list[str]:
        return [] if value is None else self.inner.validate(value)


def build_validator(spec: dict) -> Validator:
    """Turn a JSON-style rule spec into a reusable validator object."""
    kind = spec.get("type")
    try:
        if kind == "range":
            return RangeValidator(spec.get("min"), spec.get("max"))
        if kind == "regex":
            return RegexValidator(spec["pattern"])
        if kind == "one_of":
            return OneOfValidator(spec["choices"])
        if kind == "all":
            return AllOfValidator([build_validator(child) for child in spec["validators"]])
        if kind == "optional":
            return OptionalValidator(build_validator(spec["validator"]))
    except KeyError as exc:
        raise ValueError(f"spec of type {kind!r} is missing key {exc.args[0]!r}") from exc
    raise ValueError(f"unknown validator type {kind!r}")
''',
        explanation=(
            "Each rule is a small class with one responsibility, and callers only rely on the "
            "shared `validate` method (polymorphism). `AllOfValidator` and `OptionalValidator` "
            "contain other validators (composition), so arbitrary nesting works without special "
            "cases. Compiling regexes once in `__init__` makes repeated validation cheap."
        ),
        edge_cases=["Inclusive bounds.", "`True` is not a number.", "Nested optional/all specs.", "Errors inside nested specs surface at build time."],
        mistakes=["Re-parsing the spec on every `validate` call.", "Stopping `all` at the first failing child.", "Using `re.match` (accepts trailing text)."],
        tests=[
            CT(_q071_range, note="range rules and messages"),
            CT(_q071_regex_one_of, note="regex and one_of rules"),
            CT(_q071_composite, note="all / optional composition"),
            CT(_q071_bad_specs, note="invalid specs raise ValueError"),
        ],
        examples=['''
>>> probability = build_validator({"type": "range", "min": 0, "max": 1})
>>> probability.validate(0.7), probability.validate(1.4), probability.validate("high")
([], ['must be <= 1'], ['expected a number'])
>>> device = build_validator({"type": "optional", "validator": {"type": "one_of", "choices": ["cpu", "gpu"]}})
>>> device.validate(None), device.validate("tpu")
([], ["must be one of ['cpu', 'gpu']"])
'''],
        complexity="Building is O(size of spec); validating is O(number of rules) plus regex cost.",
        extension="Add a `describe()` method to each class that renders the rule as human-readable text.",
    ),
    # ------------------------------------------------------------------------ Q072
    Question(
        id=72,
        title="In-Process Event Bus with Isolated Handler Failures",
        difficulty="Advanced",
        category=OOP,
        roles=[PYDEV, AIE, MLE],
        skills=["observer pattern", "closures returned from methods", "defensive iteration", "error isolation"],
        minutes=40,
        scenario=(
            "Your model-serving app emits events such as `model.deployed`. Several components "
            "(cache warmer, metrics, Slack notifier) react to them. A bug in one listener must not "
            "prevent the others from running, and components need to unsubscribe cleanly."
        ),
        problem="Implement a class `EventBus` with `subscribe`, `publish` and `subscriber_count`.",
        signature="class EventBus:",
        entry="EventBus",
        starter='''
class EventBus:
    """Publish/subscribe dispatcher for in-process events."""

    def __init__(self):
        # TODO
        pass

    def subscribe(self, event_type: str, handler):
        # TODO: return an unsubscribe function
        pass

    def publish(self, event_type: str, payload) -> dict:
        # TODO
        pass

    def subscriber_count(self, event_type: str) -> int:
        # TODO
        pass
''',
        inputs=[
            "`subscribe(event_type, handler)`: `handler(event_type, payload)` is any callable; `event_type` may be `'*'` (all events).",
            "`publish(event_type, payload)`: payload is any object.",
        ],
        output=(
            "`subscribe` returns an *unsubscribe* function. `publish` returns "
            "`{'delivered': int, 'errors': [(handler_name, message), ...]}`."
        ),
        constraints=[
            "Handlers for the specific type run first (subscription order), then `'*'` handlers (subscription order).",
            "Subscribing the same handler to the same type twice has no extra effect.",
            "If a handler raises an `Exception`, record `(handler.__name__, str(exc))` and continue; `delivered` counts handlers that returned normally.",
            "Calling an unsubscribe function more than once is harmless. Handlers may unsubscribe during `publish` without breaking the current dispatch.",
            "`subscriber_count(event_type)` counts handlers for that exact type. Non-callable handlers → `TypeError`.",
        ],
        criteria=["Order and wildcard semantics.", "Error isolation and reporting.", "Unsubscribe works, including during publish."],
        hints=[
            "Store a `dict[str, list[handler]]`. The unsubscribe function can be a closure over the event type and handler.",
            "In `publish`, iterate over a *copy* of the handler lists so modifications during dispatch are safe. Wrap each call in `try/except Exception`.",
            "subscribe: validate; lst = self._handlers.setdefault(t, []); if h not in lst: lst.append(h)\n    def unsubscribe(): if h in lst: lst.remove(h)\n    return unsubscribe\n"
            "publish: for h in [*specific, *wildcard]: try: h(t, p); delivered += 1 except Exception as e: errors.append((name, str(e)))",
        ],
        solution='''
class EventBus:
    """Publish/subscribe dispatcher for in-process events."""

    def __init__(self):
        self._handlers: dict[str, list] = {}

    def subscribe(self, event_type: str, handler):
        if not callable(handler):
            raise TypeError("handler must be callable")
        handlers = self._handlers.setdefault(event_type, [])
        if handler not in handlers:
            handlers.append(handler)

        def unsubscribe() -> None:
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    def publish(self, event_type: str, payload) -> dict:
        targets = list(self._handlers.get(event_type, []))
        if event_type != "*":
            targets += list(self._handlers.get("*", []))
        delivered = 0
        errors = []
        for handler in targets:
            try:
                handler(event_type, payload)
            except Exception as exc:
                errors.append((getattr(handler, "__name__", repr(handler)), str(exc)))
            else:
                delivered += 1
        return {"delivered": delivered, "errors": errors}

    def subscriber_count(self, event_type: str) -> int:
        return len(self._handlers.get(event_type, []))
''',
        explanation=(
            "Returning a closure from `subscribe` gives callers a handle that knows exactly what "
            "to remove. Copying the handler lists before dispatch means a handler that "
            "unsubscribes (or subscribes) mid-publish cannot disturb the iteration in progress. "
            "Catching `Exception` per handler isolates failures while still reporting them."
        ),
        edge_cases=["Publishing with no handlers.", "Handler unsubscribing itself.", "Double unsubscribe.", "Wildcard-only delivery."],
        mistakes=["Iterating the live list while handlers remove themselves (skips handlers).", "Letting one exception abort the publish.", "Catching `BaseException` (swallows KeyboardInterrupt)."],
        tests=[
            CT(_q072_order_and_wildcard, note="ordering and wildcard handlers"),
            CT(_q072_errors_isolated, note="failing handler isolated and reported"),
            CT(_q072_unsubscribe, note="idempotent subscribe/unsubscribe"),
            CT(_q072_unsubscribe_during_publish, note="unsubscribe during publish; TypeError"),
        ],
        examples=['''
>>> bus = EventBus()
>>> log = []
>>> stop = bus.subscribe("model.deployed", lambda t, p: log.append(p["version"]))
>>> bus.publish("model.deployed", {"version": 3})
{'delivered': 1, 'errors': []}
>>> stop()
>>> bus.publish("model.deployed", {"version": 4})
{'delivered': 0, 'errors': []}
>>> log
[3]
'''],
        complexity="O(h) per publish for h handlers; O(h) per subscribe/unsubscribe (list membership).",
        extension="Add asynchronous delivery with a `queue.Queue` and a worker thread.",
    ),
    # ------------------------------------------------------------------------ Q073
    Question(
        id=73,
        title="Write Tests That Catch Broken Implementations",
        difficulty="Advanced",
        category=DBG,
        roles=[PYDEV, DS_ROLE, MLE],
        skills=["unit testing", "test design", "edge cases", "mutation testing mindset"],
        minutes=30,
        scenario=(
            "Your team's statistics helpers were rewritten for speed, and several subtle bugs slipped "
            "through review because the tests only checked `median([1, 2, 3]) == 2`. This time you "
            "write the tests. Your test function will be run against a correct implementation and "
            "against seven deliberately broken ones (\"mutants\")."
        ),
        problem=(
            "Write `check_median(median)` that exercises a `median(values)` implementation with "
            "assertions. It must return normally for any correct implementation and raise "
            "(preferably `AssertionError`) for each broken one."
        ),
        signature="def check_median(median) -> None:",
        entry="check_median",
        starter='''
def check_median(median) -> None:
    """Raise AssertionError if ``median`` is not a correct implementation."""
    # TODO: call median(...) on carefully chosen inputs and assert the results
    pass
''',
        inputs=[
            "`median`: a function under test. A correct `median(values)`:",
            "• returns the middle value of the **sorted** values for odd lengths;",
            "• returns the mean of the two middle values for even lengths (e.g. `2.5`);",
            "• raises `ValueError` for an empty list;",
            "• does not modify its input and accepts negative numbers.",
        ],
        output="`None` for correct implementations; an exception for broken ones.",
        constraints=[
            "Compare results with `==` (a correct implementation may return `2` or `2.0` for odd lengths).",
            "Your checker must accept **every** correct implementation – do not test incidental details such as the exception message.",
            "The mutants include bugs in sorting, even-length handling, empty input, input mutation, integer division, "
            "off-by-one indexing, and confusing the median with the mean.",
        ],
        criteria=["Accepts two different correct implementations.", "Rejects all seven mutants."],
        hints=[
            "For every rule in the specification, ask: which input would expose an implementation that ignores this rule?",
            "Use unsorted input, an even-length input whose answer is not an integer, skewed data where mean ≠ median, an "
            "empty list inside `try/except ValueError`, and check that the input list is unchanged after the call.",
            "assert median([3, 1, 2]) == 2\nassert median([4, 1, 3, 2]) == 2.5\nassert median([1, 2, 100]) == 2\nassert median([-5, -1, -3]) == -3\n"
            "data = [3, 1, 2]; median(data); assert data == [3, 1, 2]\ntry: median([]) except ValueError: pass else: raise AssertionError",
        ],
        solution='''
def check_median(median) -> None:
    """Raise AssertionError if ``median`` is not a correct implementation."""
    assert median([3, 1, 2]) == 2, "odd length with unsorted input"
    assert median([5]) == 5, "single value"
    assert median([4, 1, 3, 2]) == 2.5, "even length averages the two middle values"
    assert median([1, 2, 100]) == 2, "skewed data: median is not the mean"
    assert median([-5, -1, -3]) == -3, "negative numbers"
    data = [3, 1, 2]
    median(data)
    assert data == [3, 1, 2], "median must not modify its input"
    try:
        median([])
    except ValueError:
        pass
    else:
        raise AssertionError("median([]) must raise ValueError")
''',
        explanation=(
            "Each assertion targets one rule of the specification, and its input is chosen so that "
            "an implementation breaking that rule produces a *different* answer: unsorted input "
            "catches a missing sort, `2.5` catches integer division and upper-middle bugs, "
            "`[1, 2, 100]` separates median from mean, and the copy check catches in-place sorting. "
            "Asserting only on documented behaviour keeps the checker from rejecting valid variants."
        ),
        edge_cases=["Symmetric data hides the mean/median bug.", "Results compared with `==`, so int vs float both pass.", "Empty list must raise."],
        mistakes=["Only testing already-sorted, symmetric inputs.", "Asserting `type(result) is float` (rejects `statistics.median`).", "Asserting exact exception messages."],
        tests=[
            CT(_accepts("statistics.median", _median_ok_a), note="accepts statistics.median"),
            CT(_accepts("hand-written sort-based median", _median_ok_b), note="accepts a hand-written correct median"),
            CT(_rejects("does not sort the input", _mutant_unsorted), note="rejects: no sorting"),
            CT(_rejects("returns the upper middle value for even lengths", _mutant_upper_middle), note="rejects: even length"),
            CT(_rejects("returns 0.0 for an empty list instead of raising", _mutant_empty_zero), note="rejects: empty input"),
            CT(_rejects("sorts the caller's list in place", _mutant_in_place), note="rejects: input mutation"),
            CT(_rejects("uses // for the even-length average", _mutant_floor_div), note="rejects: integer division"),
            CT(_rejects("returns the mean", _mutant_mean), note="rejects: mean instead of median"),
            CT(_rejects("has an off-by-one index for odd lengths", _mutant_off_by_one), note="rejects: off-by-one"),
        ],
        examples=['''
>>> import statistics
>>> check_median(statistics.median) is None
True
>>> check_median(lambda values: sum(values) / len(values))
Traceback (most recent call last):
    ...
AssertionError: skewed data: median is not the mean
'''],
        complexity="Constant number of calls to `median`.",
        extension="Rewrite your checker as a parametrized pytest test (`@pytest.mark.parametrize`) with descriptive ids.",
    ),
    # ------------------------------------------------------------------------ Q074
    Question(
        id=74,
        title="Fix the Late-Binding Closure Bug",
        difficulty="Advanced",
        category=DBG,
        roles=[PYDEV, MLE, AIE],
        skills=["closures and late binding", "default-argument binding / factory functions", "function attributes"],
        minutes=25,
        scenario=(
            "An alerting module builds one check function per metric:\n\n"
            "```python\ndef make_threshold_checks(thresholds):\n    checks = {}\n"
            "    for name, limit in thresholds.items():\n        checks[name] = lambda value: value <= limit\n"
            "    return checks\n```\n\n"
            "Every check behaves as if it had the *last* metric's limit, so latency alerts fire at "
            "0.9 ms. The closures look up `limit` when they are *called*, not when they are created."
        ),
        problem=(
            "Write a corrected `make_threshold_checks(thresholds)` returning `{name: check}` where "
            "`check(value)` returns `value <= limit` for that metric's own limit."
        ),
        signature="def make_threshold_checks(thresholds: dict[str, float]) -> dict:",
        entry="make_threshold_checks",
        inputs=["`thresholds`: dict metric name → numeric limit (int or float, not bool)."],
        output="A dict metric name → check function. Each function's `__name__` is `'check_<name>'`.",
        constraints=[
            "Each check uses the limit of its own metric, captured at creation time (later changes to the input dict have no effect).",
            "Raise `TypeError` for non-numeric (or bool) limits.",
            "Empty input → `{}`.",
        ],
        criteria=["Each closure has its own limit.", "Snapshot semantics.", "Function names set.", "Validation."],
        hints=[
            "A closure captures *variables*, not values. All lambdas in the loop share the same `limit` variable.",
            "Create each check inside a helper function (a factory) so each call gets its own scope – or bind the value with a default argument.",
            "def _make(name, limit):\n    def check(value): return value <= limit\n    check.__name__ = f'check_{name}'\n    return check\n"
            "return {name: _make(name, limit) for name, limit in thresholds.items()}",
        ],
        solution='''
def _make_check(name: str, limit: float):
    def check(value: float) -> bool:
        return value <= limit

    check.__name__ = f"check_{name}"
    return check


def make_threshold_checks(thresholds: dict[str, float]) -> dict:
    """One independent `value <= limit` check per metric."""
    checks = {}
    for name, limit in thresholds.items():
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            raise TypeError(f"limit for {name!r} must be a number, got {limit!r}")
        checks[name] = _make_check(name, limit)
    return checks
''',
        explanation=(
            "Each call to `_make_check` creates a new scope with its own `limit`, so every returned "
            "function closes over a different variable. (The `lambda value, limit=limit: ...` "
            "trick also works, but exposes `limit` as an overridable parameter.)"
        ),
        edge_cases=["Changing the input dict after creation.", "Integer and float limits.", "Empty mapping."],
        mistakes=["Keeping the lambda and only renaming the loop variable.", "Using `functools.partial` but forgetting `__name__` (partials have no `__name__`).", "Accepting `True` as a limit."],
        tests=[
            CT(_q074_values, note="each check uses its own limit"),
            CT(_q074_names_and_snapshot, note="limits captured at creation; __name__"),
            CT(_q074_validation, note="empty input and TypeError"),
            CT(_q074_many, note="100 generated metrics each keep their own limit"),
        ],
        examples=['''
>>> checks = make_threshold_checks({"latency_ms": 500, "error_rate": 0.01})
>>> checks["latency_ms"](450), checks["error_rate"](0.02)
(True, False)
>>> checks["latency_ms"].__name__
'check_latency_ms'
'''],
        complexity="O(n) to build n checks; O(1) per check call.",
        extension="Support both upper and lower limits: `{'cpu': (None, 0.9), 'free_disk_gb': (20, None)}`.",
    ),
    # ------------------------------------------------------------------------ Q075
    Question(
        id=75,
        title="Remove Expired Sessions Without Breaking Iteration",
        difficulty="Advanced",
        category=DBG,
        roles=[PYDEV, DE, AIE],
        skills=["mutation during iteration", "in-place updates", "boundary conditions", "API contracts"],
        minutes=20,
        scenario=(
            "A chat service keeps `session_id → last_seen` (epoch seconds) in a dict shared by "
            "several components, so expired sessions must be removed **in place**. The current code "
            "crashes intermittently:\n\n"
            "```python\nfor sid, seen in sessions.items():\n    if now - seen >= ttl:\n        del sessions[sid]\n```\n\n"
            "`RuntimeError: dictionary changed size during iteration` – and it also expires sessions exactly at the TTL, which product says should still be alive."
        ),
        problem=(
            "Write `drop_stale(sessions, now, ttl)` that deletes, in place, every session whose "
            "age `now - last_seen` is **strictly greater** than `ttl`, and returns the removed ids sorted."
        ),
        signature="def drop_stale(sessions: dict[str, float], now: float, ttl: float) -> list[str]:",
        entry="drop_stale",
        mutates_input=True,
        inputs=["`sessions`: dict id → last-seen timestamp.", "`now`: current timestamp.", "`ttl`: seconds, ≥ 0."],
        output="A sorted `list[str]` of removed session ids. The dict itself is modified (same object).",
        constraints=[
            "Modify the given dict in place – do not rebind or return a new dict.",
            "Age exactly equal to `ttl` is still alive.",
            "Raise `ValueError` if `ttl < 0`.",
        ],
        criteria=["No RuntimeError.", "Correct in-place result.", "Boundary handled.", "Sorted return value."],
        hints=[
            "You cannot change a dict's size while iterating over it – so iterate over something else.",
            "First collect the expired keys (e.g. with a list comprehension over `sessions.items()`), then delete them in a second loop.",
            "if ttl < 0: raise ValueError\nexpired = [sid for sid, seen in sessions.items() if now - seen > ttl]\nfor sid in expired: del sessions[sid]\nreturn sorted(expired)",
        ],
        solution='''
def drop_stale(sessions: dict[str, float], now: float, ttl: float) -> list[str]:
    """Delete sessions older than ttl (in place); return their ids sorted."""
    if ttl < 0:
        raise ValueError("ttl must be >= 0")
    expired = [session_id for session_id, last_seen in sessions.items() if now - last_seen > ttl]
    for session_id in expired:
        del sessions[session_id]
    return sorted(expired)
''',
        explanation=(
            "Collecting keys first means the dict is not modified while being iterated. Deleting "
            "afterwards keeps the same dict object, so other components holding a reference see "
            "the update. Using `>` encodes 'exactly at the TTL is still alive'."
        ),
        edge_cases=["Age == ttl kept.", "Nothing expired → `[]`.", "Empty dict."],
        mistakes=["`sessions = {k: v ...}` – rebinds a local name; the caller's dict is unchanged.", "Iterating `sessions.items()` while deleting.", "Using `>=`."],
        tests=[
            T(({"a": 0.0, "b": 50.0, "c": 100.0}, 100.0, 50.0), ["a"], example=True, note="age == ttl is kept"),
            T(({}, 10.0, 5.0), [], example=True, note="boundary: empty"),
            T(({"x": 99.0}, 100.0, 5.0), [], note="nothing expired"),
            T(({"z": 1.0, "y": 2.0}, 100.0, 5.0), ["y", "z"], note="sorted ids"),
            CT(_q075_in_place, note="dict modified in place"),
            T(({"a": 1.0}, 2.0, -1.0), raises="ValueError", note="negative ttl"),
        ],
        complexity="O(n + k log k) time for n sessions and k expired; O(k) extra space.",
        extension="Make it thread-safe with a `threading.Lock` shared by the readers and the cleaner.",
    ),
    # ------------------------------------------------------------------------ Q076
    Question(
        id=76,
        title="Sessionize Clickstream Events",
        difficulty="Advanced",
        category=DATA,
        roles=[DS_ROLE, DE, MLE],
        skills=["grouping and sorting", "datetime arithmetic", "gap-based segmentation", "output formatting"],
        minutes=40,
        scenario=(
            "Product analytics defines a *session* as a run of a user's events where consecutive "
            "events are at most 30 minutes apart. Session counts and lengths feed an engagement "
            "model. Events arrive unsorted from several devices."
        ),
        problem=(
            "Write `sessionize(events, gap_minutes=30)` that groups each user's events into sessions "
            "and returns one summary per session."
        ),
        signature="def sessionize(events: list[dict], gap_minutes: int = 30) -> list[dict]:",
        entry="sessionize",
        inputs=["`events`: dicts with `user_id` (str) and `timestamp` (`YYYY-MM-DDTHH:MM:SSZ`), in any order.", "`gap_minutes`: positive int."],
        output=(
            "A list of `{'user_id', 'session_start', 'session_end', 'event_count'}` dicts (timestamps in the input format), "
            "sorted by `user_id`, then `session_start`."
        ),
        constraints=[
            "A gap of exactly `gap_minutes` keeps the events in the same session; a larger gap starts a new one.",
            "Raise `ValueError` for malformed timestamps or `gap_minutes <= 0`.",
            "Do not modify the input.",
        ],
        criteria=["Correct session boundaries.", "Unsorted input handled.", "Sorted output.", "Validation."],
        hints=[
            "Sessionization is easy once each user's timestamps are sorted.",
            "Group parsed datetimes by user, sort each list, then walk it and start a new session when "
            "`current - previous > timedelta(minutes=gap)`.",
            "by_user = defaultdict(list) of datetimes\nfor user in sorted(by_user):\n    times = sorted(...); start = prev = times[0]; count = 1\n"
            "    for t in times[1:]:\n        if t - prev > gap: emit(start, prev, count); start, count = t, 0\n        count += 1; prev = t\n    emit(start, prev, count)",
        ],
        solution='''
from datetime import datetime, timedelta

FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def sessionize(events: list[dict], gap_minutes: int = 30) -> list[dict]:
    """Split each user's events into sessions separated by gaps > gap_minutes."""
    if gap_minutes <= 0:
        raise ValueError("gap_minutes must be > 0")
    gap = timedelta(minutes=gap_minutes)
    by_user: dict[str, list[datetime]] = {}
    for event in events:
        by_user.setdefault(event["user_id"], []).append(datetime.strptime(event["timestamp"], FORMAT))

    sessions = []
    for user_id in sorted(by_user):
        times = sorted(by_user[user_id])
        start = previous = times[0]
        count = 0
        for moment in times:
            if moment - previous > gap:
                sessions.append(_summary(user_id, start, previous, count))
                start, count = moment, 0
            count += 1
            previous = moment
        sessions.append(_summary(user_id, start, previous, count))
    return sessions


def _summary(user_id: str, start: datetime, end: datetime, count: int) -> dict:
    return {"user_id": user_id, "session_start": start.strftime(FORMAT),
            "session_end": end.strftime(FORMAT), "event_count": count}
''',
        explanation=(
            "Parsing to `datetime` makes gap arithmetic exact and validates the format. Sorting per "
            "user turns segmentation into one linear pass: compare each event with the previous one "
            "and close the session when the gap is too large. Iterating users in sorted order gives "
            "the required output order for free."
        ),
        edge_cases=["Gap exactly equal to the limit stays in one session.", "Single-event sessions (start == end).", "Unsorted input."],
        mistakes=["Comparing timestamp strings arithmetically.", "Measuring gaps from the session start instead of the previous event.", "Forgetting to emit the last session."],
        tests=[
            T((SESSION_EVENTS, 30),
              [{"user_id": "u1", "session_start": "2024-05-01T09:00:00Z", "session_end": "2024-05-01T09:39:00Z", "event_count": 3},
               {"user_id": "u1", "session_start": "2024-05-01T10:10:00Z", "session_end": "2024-05-01T10:40:00Z", "event_count": 2},
               {"user_id": "u2", "session_start": "2024-05-01T09:05:00Z", "session_end": "2024-05-01T09:05:00Z", "event_count": 1}],
              example=True, note="gap of 31 min splits; gap of exactly 30 does not"),
            T(([], 30), [], example=True, note="boundary: no events"),
            T((SESSION_EVENTS, 120),
              [{"user_id": "u1", "session_start": "2024-05-01T09:00:00Z", "session_end": "2024-05-01T10:40:00Z", "event_count": 5},
               {"user_id": "u2", "session_start": "2024-05-01T09:05:00Z", "session_end": "2024-05-01T09:05:00Z", "event_count": 1}],
              note="larger gap merges sessions"),
            T(([{"user_id": "u", "timestamp": "2024-05-01 09:00"}], 30), raises="ValueError", note="malformed timestamp"),
            T((SESSION_EVENTS, 0), raises="ValueError", note="gap must be positive"),
            CT(_q076_sample, note="data/sample_data/events.json"),
        ],
        complexity="O(n log n) time for sorting, O(n) space.",
        extension="Add `duration_seconds` and the list of distinct event types per session.",
    ),
    # ------------------------------------------------------------------------ Q077
    Question(
        id=77,
        title="Hash-Based Left Join of Two Record Lists",
        difficulty="Advanced",
        category=DATA,
        roles=[DE, DS_ROLE, PYDEV],
        skills=["hash join", "one-to-many joins", "column conflict handling", "None-key semantics"],
        minutes=35,
        scenario=(
            "You need to attach subscription plans to user records inside a lightweight ETL script "
            "that cannot depend on a database or pandas. A user may have several plans, some users "
            "have none, and both tables contain a `name` column."
        ),
        problem=(
            "Write `left_join(left, right, on)` that returns every left row combined with each "
            "matching right row (SQL `LEFT JOIN` semantics)."
        ),
        signature="def left_join(left: list[dict], right: list[dict], on: str) -> list[dict]:",
        entry="left_join",
        inputs=["`left`, `right`: lists of dicts.", "`on`: the join key field name."],
        output="A list of new dicts.",
        constraints=[
            "Output order: left rows in order; for each, its matches in right order.",
            "A matched row contains all left fields plus the right fields; a right field (other than `on`) whose name already exists in the left row is renamed `<field>_right`.",
            "A left row with no match appears once, unchanged (as a copy) and without right fields.",
            "Missing or `None` keys never match (left rows with them are kept unmatched; right rows are ignored).",
            "Must run in O(n + m + output) time – index the right side.",
            "Do not modify the inputs; every output row is a new dict.",
        ],
        criteria=["One-to-many matches.", "Unmatched rows kept.", "Conflicts renamed.", "Fast and non-mutating."],
        hints=[
            "Scanning `right` for each left row is O(n·m).",
            "Build `index = {key: [right rows]}` first, then for each left row look up its matches and merge field by field.",
            "index = defaultdict(list); for r in right: if r.get(on) is not None: index[r[on]].append(r)\nfor l in left:\n    matches = index.get(l.get(on), []) if l.get(on) is not None else []\n"
            "    if not matches: out.append(dict(l)); continue\n    for r in matches: row = dict(l); for k, v in r.items(): if k == on: continue; row[k + '_right' if k in l else k] = v",
        ],
        solution='''
def left_join(left: list[dict], right: list[dict], on: str) -> list[dict]:
    """SQL-style LEFT JOIN of two lists of dicts on one key."""
    index: dict = {}
    for row in right:
        key = row.get(on)
        if key is not None:
            index.setdefault(key, []).append(row)

    joined = []
    for row in left:
        key = row.get(on)
        matches = index.get(key, []) if key is not None else []
        if not matches:
            joined.append(dict(row))
            continue
        for match in matches:
            combined = dict(row)
            for field, value in match.items():
                if field == on:
                    continue
                combined[f"{field}_right" if field in row else field] = value
            joined.append(combined)
    return joined
''',
        explanation=(
            "A hash index on the right side turns each lookup into O(1) on average, so the join is "
            "linear in the input plus output size – the same idea databases use for hash joins. "
            "Building each output row with `dict(row)` guarantees new objects."
        ),
        edge_cases=["One-to-many matches.", "Right rows with unmatched keys are ignored.", "`None` keys never match."],
        mistakes=["Nested loops (O(n·m)).", "Overwriting left fields with right fields silently.", "Treating `None == None` as a match."],
        tests=[
            T(([{"user_id": 1, "name": "Asha"}, {"user_id": 2, "name": "Ben"}, {"user_id": 3, "name": "Chen"}],
               [{"user_id": 1, "plan": "pro", "name": "A. R."}, {"user_id": 1, "plan": "team"}, {"user_id": 3, "plan": "free"},
                {"user_id": 4, "plan": "pro"}], "user_id"),
              [{"user_id": 1, "name": "Asha", "plan": "pro", "name_right": "A. R."}, {"user_id": 1, "name": "Asha", "plan": "team"},
               {"user_id": 2, "name": "Ben"}, {"user_id": 3, "name": "Chen", "plan": "free"}], example=True,
              note="one-to-many, unmatched, conflict"),
            T(([{"k": 1}], [], "k"), [{"k": 1}], example=True, note="boundary: empty right side"),
            T(([{"k": None, "v": 1}, {"v": 2}], [{"k": None, "w": 2}], "k"), [{"k": None, "v": 1}, {"v": 2}], note="None/missing keys never match"),
            T(([], [{"k": 1}], "k"), [], note="empty left side"),
            CT(_q077_new_dicts, note="output rows are new dicts"),
            CT(_q077_fast, note="20,000 x 20,000 rows in under a second"),
        ],
        complexity="O(n + m + output size) time; O(m) extra space for the index.",
        extension="Add `how='inner'|'left'|'outer'` and handle unmatched right rows for the outer join.",
    ),
    # ------------------------------------------------------------------------ Q078
    Question(
        id=78,
        title="Flag Metric Anomalies with Per-Group Z-Scores",
        difficulty="Advanced",
        category=DATA,
        roles=[DS_ROLE, MLE, AIE],
        skills=["grouping", "statistics module", "two-pass algorithms", "degenerate cases"],
        minutes=30,
        scenario=(
            "A monitoring job checks daily metric snapshots for unusual values. Each metric has its "
            "own scale, so anomalies are defined relative to that metric's own mean and spread."
        ),
        problem=(
            "Write `flag_anomalies(records, threshold=3.0)` returning the records whose value is more "
            "than `threshold` population standard deviations away from their metric's mean."
        ),
        signature="def flag_anomalies(records: list[dict], threshold: float = 3.0) -> list[dict]:",
        entry="flag_anomalies",
        inputs=["`records`: dicts with `metric` (str), `value` (number or `None`) and `ts` (str).", "`threshold`: positive number."],
        output="Copies of the flagged records with an added `z_score` (rounded to 2 decimals), in input order.",
        constraints=[
            "Statistics are computed per metric over its non-`None` values: mean and **population** standard deviation (`statistics.pstdev`).",
            "z = (value − mean) / stdev; flag when `abs(z) > threshold`.",
            "Metrics with fewer than 3 values or a standard deviation of 0 never produce anomalies.",
            "`None` values are ignored. Raise `ValueError` if `threshold <= 0`. Do not modify the input.",
        ],
        criteria=["Per-group statistics.", "Correct z-scores and threshold.", "Degenerate groups handled.", "Input order kept."],
        hints=[
            "Two passes: first compute statistics per metric, then evaluate each record.",
            "Collect values per metric in a dict of lists, compute `(mean, pstdev)` for qualifying groups, and in the second pass build `{**record, 'z_score': round(z, 2)}`.",
            "values = group non-None values by metric\nstats = {m: (fmean(v), pstdev(v)) for m, v in values.items() if len(v) >= 3}\n"
            "for r in records: skip None; m, s = stats.get(...) or skip; if s == 0: skip; z = (r['value'] - m) / s; if abs(z) > threshold: append copy",
        ],
        solution='''
import statistics


def flag_anomalies(records: list[dict], threshold: float = 3.0) -> list[dict]:
    """Records more than ``threshold`` population std-devs from their metric's mean."""
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    values: dict[str, list[float]] = {}
    for record in records:
        if record["value"] is not None:
            values.setdefault(record["metric"], []).append(record["value"])
    stats = {
        metric: (statistics.fmean(vals), statistics.pstdev(vals))
        for metric, vals in values.items()
        if len(vals) >= 3
    }
    flagged = []
    for record in records:
        if record["value"] is None or record["metric"] not in stats:
            continue
        mean, stdev = stats[record["metric"]]
        if stdev == 0:
            continue
        z = (record["value"] - mean) / stdev
        if abs(z) > threshold:
            flagged.append({**record, "z_score": round(z, 2)})
    return flagged
''',
        explanation=(
            "The first pass aggregates per metric so the second pass can score each record in O(1). "
            "Guarding against tiny groups and zero spread avoids division by zero and meaningless "
            "z-scores. Note that with population stdev a single outlier among n points can never "
            "exceed z = √(n−1), so thresholds must suit the group size."
        ),
        edge_cases=["Constant metric (stdev 0).", "Fewer than 3 values.", "Negative z-scores are anomalies too."],
        mistakes=["Using one global mean for all metrics.", "Sample stdev when the spec says population.", "Dividing by zero for constant metrics."],
        tests=[
            T(([{"metric": "latency", "value": v, "ts": f"t{i}"} for i, v in enumerate([100, 102, 98, 101, 99, 100, 250])]
               + [{"metric": "error_rate", "value": 0.01, "ts": f"e{i}"} for i in range(3)]
               + [{"metric": "cpu", "value": None, "ts": "c0"}], 2.0),
              [{"metric": "latency", "value": 250, "ts": "t6", "z_score": 2.45}], example=True,
              note="outlier flagged; constant metric ignored"),
            T(([], 3.0), [], example=True, note="boundary: no records"),
            T(([{"metric": "m", "value": 1, "ts": "a"}, {"metric": "m", "value": 100, "ts": "b"}], 0.5), [], note="fewer than 3 values"),
            T(([{"metric": "m", "value": v, "ts": str(i)} for i, v in enumerate([10, 10, 10, 10, 10, 10, 10, 10, 10, -40])], 2.5),
              [{"metric": "m", "value": -40, "ts": "9", "z_score": -3.0}], note="negative z-score"),
            T(([{"metric": "m", "value": 1, "ts": "a"}], 0), raises="ValueError", note="threshold must be positive"),
            CT(_q078_sample, note="data/sample_data/model_metrics.json"),
        ],
        complexity="O(n) time, O(n) space for the grouped values.",
        extension="Use a robust score instead: median and median absolute deviation (MAD).",
    ),
    # ------------------------------------------------------------------------ Q079
    Question(
        id=79,
        title="Count Near-Duplicate Amounts in O(n log n)",
        difficulty="Advanced",
        category=PERF,
        roles=[DS_ROLE, DE, PYDEV],
        skills=["sorting to enable two pointers", "counting without enumerating", "benchmarking assumptions"],
        minutes=30,
        scenario=(
            "A fraud heuristic counts pairs of transactions whose amounts (in cents) differ by at "
            "most a tolerance. The prototype compares every pair:\n\n"
            "```python\ncount = 0\nfor i in range(len(values)):\n    for j in range(i + 1, len(values)):\n"
            "        if abs(values[i] - values[j]) <= tol:\n            count += 1\n```\n\n"
            "It takes minutes on a day of data."
        ),
        problem=(
            "Write `count_pairs_within_tolerance(values, tolerance)` returning the number of index "
            "pairs `i < j` with `|values[i] - values[j]| <= tolerance`, in O(n log n)."
        ),
        signature="def count_pairs_within_tolerance(values: list[int], tolerance: int) -> int:",
        entry="count_pairs_within_tolerance",
        inputs=["`values`: list of ints (amounts in cents; duplicates allowed).", "`tolerance`: `int` ≥ 0."],
        output="An `int`.",
        constraints=[
            "Same result as the quadratic version.",
            "50,000 values must take well under a second.",
            "Raise `ValueError` for a negative tolerance. Do not modify the input.",
        ],
        criteria=["Correct counts including duplicates.", "O(n log n).", "Validation."],
        hints=[
            "Pair counts don't depend on order – so you may sort a copy.",
            "In sorted order, for each right index `j` keep a left pointer `i` advanced while `v[j] - v[i] > tol`; then `j - i` values pair with `v[j]`.",
            "data = sorted(values); count = i = 0\nfor j, v in enumerate(data):\n    while v - data[i] > tolerance: i += 1\n    count += j - i\nreturn count",
        ],
        solution='''
def count_pairs_within_tolerance(values: list[int], tolerance: int) -> int:
    """Number of pairs whose difference is at most ``tolerance`` (sorting + two pointers)."""
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    data = sorted(values)
    count = left = 0
    for right, value in enumerate(data):
        while value - data[left] > tolerance:
            left += 1
        count += right - left
    return count
''',
        explanation=(
            "After sorting, the values within tolerance of `data[right]` (to its left) form a "
            "contiguous window `[left, right)`. Both pointers only move forward, so the scan is "
            "O(n) and sorting dominates at O(n log n). Counting `right - left` avoids "
            "materializing the (possibly quadratic) list of pairs. Using integer cents avoids "
            "float comparisons such as `1.05 - 1.0 <= 0.05` being false."
        ),
        edge_cases=["All values equal: n·(n−1)/2 pairs.", "Tolerance 0 counts exact duplicates.", "Empty input → 0."],
        mistakes=["Sorting the caller's list in place.", "Building every pair (memory blow-up).", "Using floats for money."],
        tests=[
            T(([100, 105, 200, 102], 5), 3, example=True, note="three close pairs"),
            T(([5, 5, 5], 0), 3, example=True, note="boundary: tolerance 0 with duplicates"),
            T(([], 3), 0, note="empty"),
            T(([1, 10, 20], 5), 0, note="no pairs"),
            T(([30, 10, 20], 10), 2, note="unsorted input"),
            T(([1], -1), raises="ValueError", note="negative tolerance"),
            CT(_q079_fast, note="50,000 values in under a second"),
        ],
        complexity="O(n log n) time (sort) + O(n) scan; O(n) space for the sorted copy.",
        extension="Benchmark both versions with `timeit` at n = 1k, 2k, 4k, 8k and show the quadratic vs n log n growth.",
    ),
    # ------------------------------------------------------------------------ Q080
    Question(
        id=80,
        title="Top-K Records from a Stream with Bounded Memory",
        difficulty="Advanced",
        category=PERF,
        roles=[DE, MLE, AIE],
        skills=["heapq", "bounded heaps", "stable tie-breaking", "streaming", "tracemalloc"],
        minutes=35,
        scenario=(
            "An evaluation job scores millions of generated answers and keeps only the 20 best for "
            "human review. Scores arrive as a stream; collecting them in a list and sorting would "
            "need gigabytes of memory."
        ),
        problem=(
            "Write `top_k_stream(records, k, field)` that returns the `k` records with the largest "
            "`record[field]` while holding at most `k` records in memory."
        ),
        signature="def top_k_stream(records, k: int, field: str) -> list[dict]:",
        entry="top_k_stream",
        inputs=["`records`: any iterable of dicts (often a generator).", "`k`: `int` ≥ 0.", "`field`: name of the numeric score field."],
        output="A list of at most `k` records, highest score first; among equal scores, earlier records first.",
        constraints=[
            "Skip records where the field is missing or `None`.",
            "Single pass over the input; memory O(k), not O(n).",
            "Raise `ValueError` for `k < 0`.",
        ],
        criteria=["Correct top-k with stable ties.", "Works on generators.", "Bounded memory."],
        hints=[
            "A *min*-heap of size k keeps the k largest items seen so far: its root is the weakest one, which a better item replaces.",
            "Push tuples `(score, -arrival_index, record)` – the unique index means records are never compared and makes later arrivals 'smaller' on ties. Use `heapq.heappushpop`/`heapreplace` once the heap is full.",
            "heap = []\nfor i, r in enumerate(records):\n    v = r.get(field); if v is None: continue\n    item = (v, -i, r)\n"
            "    if len(heap) < k: heappush(heap, item)\n    elif item > heap[0]: heapreplace(heap, item)\nreturn [r for _, _, r in sorted(heap, reverse=True)]",
        ],
        solution='''
import heapq


def top_k_stream(records, k: int, field: str) -> list[dict]:
    """k records with the largest ``field`` using an O(k) min-heap."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return []
    heap: list[tuple] = []
    for index, record in enumerate(records):
        value = record.get(field)
        if value is None:
            continue
        item = (value, -index, record)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [record for _, _, record in sorted(heap, reverse=True)]
''',
        explanation=(
            "The heap always holds the k best items so far, with the weakest at `heap[0]`, so each "
            "new item costs O(log k) and memory stays O(k). Encoding arrival order as `-index` "
            "makes earlier records win ties and guarantees tuple comparison never reaches the dicts."
        ),
        edge_cases=["`k = 0` → `[]` without consuming memory.", "Fewer valid records than k.", "Ties resolved by arrival order."],
        mistakes=["`sorted(records)[:k]` (O(n) memory).", "Tuples `(score, record)` – comparing dicts raises TypeError on ties.", "A max-heap of everything."],
        tests=[
            T(([{"id": "a", "score": 0.5}, {"id": "b", "score": 0.9}, {"id": "c", "score": 0.7}, {"id": "d", "score": 0.9},
                {"id": "e"}, {"id": "f", "score": None}, {"id": "g", "score": 0.1}], 3, "score"),
              [{"id": "b", "score": 0.9}, {"id": "d", "score": 0.9}, {"id": "c", "score": 0.7}], example=True,
              note="ties keep arrival order; missing scores skipped"),
            T(([{"id": "a", "score": 1}], 0, "score"), [], example=True, note="boundary: k = 0"),
            T(([{"id": "a", "score": 1}, {"id": "b", "score": 3}], 5, "score"), [{"id": "b", "score": 3}, {"id": "a", "score": 1}],
              note="k larger than input"),
            T(([{"id": "a", "score": 1}], -1, "score"), raises="ValueError", note="negative k"),
            CT(_q080_generator, note="generator input"),
            CT(_q080_memory, note="200,000 streamed records with < 2 MB peak memory"),
        ],
        complexity="O(n log k) time, O(k) memory.",
        extension="Return the top k *per group* (e.g. per prompt template) with one heap per group.",
    ),
    # ------------------------------------------------------------------------ Q081
    Question(
        id=81,
        title="Enrich Events with an Expensive Lookup (Avoid Repeated Calls)",
        difficulty="Advanced",
        category=PERF,
        roles=[DE, AIE, PYDEV],
        skills=["caching within a call", "negative caching", "exception semantics", "non-mutation"],
        minutes=25,
        scenario=(
            "Each clickstream event must be enriched with the user's country and tier from a "
            "profile service. A call costs ~50 ms, and a batch of 100,000 events contains only a few "
            "thousand distinct users. Unknown users make the service raise `KeyError`."
        ),
        problem=(
            "Write `enrich_events(events, lookup)` that returns new event dicts with `user_country` "
            "and `user_tier` added, calling `lookup` at most once per distinct `user_id`."
        ),
        signature="def enrich_events(events: list[dict], lookup) -> list[dict]:",
        entry="enrich_events",
        inputs=[
            "`events`: dicts containing `user_id`.",
            "`lookup(user_id)`: returns `{'country': str, 'tier': str}` or raises `KeyError` for unknown users.",
        ],
        output="New dicts: the original fields plus `user_country` and `user_tier` (`None` for unknown users).",
        constraints=[
            "Call `lookup` at most once per distinct user id – **including** unknown ids (cache the miss).",
            "Exceptions other than `KeyError` must propagate.",
            "Preserve event order; do not modify the input events.",
        ],
        criteria=["One lookup per distinct id.", "Unknown users → None fields.", "Other errors propagate.", "Non-mutating."],
        hints=[
            "Keep a dict `user_id → profile or None` for the duration of the call.",
            "Check the cache with `in` (not `.get`), because `None` is a legitimate cached value meaning 'unknown user'.",
            "cache = {}\nfor e in events:\n    uid = e['user_id']\n    if uid not in cache:\n        try: cache[uid] = lookup(uid)\n        except KeyError: cache[uid] = None\n"
            "    p = cache[uid]\n    out.append({**e, 'user_country': p and p['country'], 'user_tier': p and p['tier']})",
        ],
        solution='''
def enrich_events(events: list[dict], lookup) -> list[dict]:
    """Add profile fields to events, calling ``lookup`` once per distinct user."""
    profiles: dict = {}
    enriched = []
    for event in events:
        user_id = event["user_id"]
        if user_id not in profiles:
            try:
                profiles[user_id] = lookup(user_id)
            except KeyError:
                profiles[user_id] = None
        profile = profiles[user_id]
        enriched.append({
            **event,
            "user_country": profile["country"] if profile else None,
            "user_tier": profile["tier"] if profile else None,
        })
    return enriched
''',
        explanation=(
            "A per-call dict cache reduces the cost from one call per event to one per distinct "
            "user. Caching the *miss* as `None` (negative caching) prevents repeated calls for "
            "unknown users. Catching only `KeyError` keeps outages visible."
        ),
        edge_cases=["Repeated unknown users.", "Lookup raising ConnectionError.", "Empty event list."],
        mistakes=["`if cache.get(uid):` re-queries unknown users every time.", "`except Exception` hides service outages.", "`event.update(...)` mutates the caller's events."],
        tests=[
            T(([{"user_id": "u1", "page": "home"}, {"user_id": "ghost", "page": "docs"}], PROFILES),
              [{"user_id": "u1", "page": "home", "user_country": "IN", "user_tier": "pro"},
               {"user_id": "ghost", "page": "docs", "user_country": None, "user_tier": None}], example=True,
              note="known and unknown users"),
            T(([], PROFILES), [], example=True, note="boundary: no events"),
            CT(_q081_calls_once, note="lookup called once per distinct user"),
            CT(_q081_other_errors, note="non-KeyError exceptions propagate"),
        ],
        complexity="O(n) time plus u lookups for u distinct users; O(u) cache memory.",
        extension="Batch the misses: collect unknown ids first and call a `lookup_many(ids)` once.",
    ),
    # ------------------------------------------------------------------------ Q082
    Question(
        id=82,
        title="Validate Structured LLM Output Against a Schema",
        difficulty="Advanced",
        category=INT,
        roles=[AIE, AAI, PYDEV],
        skills=["recursive validation", "path-tracking error messages", "type dispatch", "JSON-like data"],
        minutes=45,
        scenario=(
            "An LLM extracts sentiment, confidence and entities from support tickets as JSON. "
            "Models occasionally invent fields, return confidence 1.2 or use unknown entity types. "
            "Before storing results you validate them against a small schema and log precise errors."
        ),
        problem=(
            "Write `validate_llm_output(payload, schema)` that returns a sorted list of error "
            "messages describing every problem (empty list = valid)."
        ),
        signature="def validate_llm_output(payload, schema: dict) -> list[str]:",
        entry="validate_llm_output",
        inputs=[
            "`schema` nodes have a `type` of `object`, `array`, `string`, `number`, `integer` or `boolean`.",
            "object: `properties` (name → schema), optional `required` (list) and `additional` (bool, default `True`).",
            "array: `items` (schema), optional `min_items`. string: optional `enum`, `max_length`. number/integer: optional `minimum`, `maximum`.",
        ],
        output="`list[str]` sorted alphabetically. Each message is `'<path>: <problem>'`.",
        constraints=[
            "Paths start at `$`; object fields append `.name`, array items `[i]`.",
            "Type errors: `'<path>: expected <type>'` (bools are not numbers/integers; ints are numbers) – do not check deeper.",
            "Missing required: `'<path>.<name>: is required'`; unexpected field with `additional: False`: `'<path>.<name>: is not allowed'`.",
            "enum: `'<path>: must be one of <enum!r>'`; max_length: `'<path>: longer than <n> characters'`; "
            "minimum/maximum: `'<path>: must be >= <m>'` / `'<path>: must be <= <m>'`; min_items: `'<path>: must have at least <n> items'`.",
            "Raise `ValueError` for an unknown schema type.",
        ],
        criteria=["All rule types.", "Exact paths and messages.", "Recursive arrays/objects.", "Sorted output."],
        hints=[
            "Write `walk(value, schema, path, errors)` that checks the type first, then the type-specific rules, then recurses.",
            "Use a dict of type checkers: `{'string': lambda v: isinstance(v, str), 'number': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool), ...}`.",
            "walk(v, s, path):\n    if not CHECK[s['type']](v): errors.append(f'{path}: expected {t}'); return\n    object: required/additional/recurse into properties present\n"
            "    array: min_items; recurse items with f'{path}[{i}]'\n    string: enum, max_length\n    number/integer: minimum/maximum, enum",
        ],
        solution='''
TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


def validate_llm_output(payload, schema: dict) -> list[str]:
    """Validate a JSON-like payload against a minimal schema; return sorted errors."""
    errors: list[str] = []

    def walk(value, node: dict, path: str) -> None:
        kind = node.get("type")
        if kind not in TYPE_CHECKS:
            raise ValueError(f"unknown schema type {kind!r} at {path}")
        if not TYPE_CHECKS[kind](value):
            errors.append(f"{path}: expected {kind}")
            return
        if "enum" in node and value not in node["enum"]:
            errors.append(f"{path}: must be one of {node['enum']!r}")
        if kind == "object":
            properties = node.get("properties", {})
            for name in node.get("required", []):
                if name not in value:
                    errors.append(f"{path}.{name}: is required")
            for name, item in value.items():
                if name in properties:
                    walk(item, properties[name], f"{path}.{name}")
                elif node.get("additional", True) is False:
                    errors.append(f"{path}.{name}: is not allowed")
        elif kind == "array":
            if "min_items" in node and len(value) < node["min_items"]:
                errors.append(f"{path}: must have at least {node['min_items']} items")
            if "items" in node:
                for index, item in enumerate(value):
                    walk(item, node["items"], f"{path}[{index}]")
        elif kind == "string":
            if "max_length" in node and len(value) > node["max_length"]:
                errors.append(f"{path}: longer than {node['max_length']} characters")
        elif kind in ("number", "integer"):
            if "minimum" in node and value < node["minimum"]:
                errors.append(f"{path}: must be >= {node['minimum']}")
            if "maximum" in node and value > node["maximum"]:
                errors.append(f"{path}: must be <= {node['maximum']}")

    walk(payload, schema, "$")
    return sorted(errors)
''',
        explanation=(
            "A single recursive function mirrors the schema's structure. Checking the type first "
            "and returning early prevents cascades of meaningless follow-up errors. Collecting "
            "every error (instead of raising on the first) gives the LLM-retry loop complete "
            "feedback in one round."
        ),
        edge_cases=["Root payload of the wrong type.", "`True` for a number field.", "Errors inside array items.", "Optional fields absent."],
        mistakes=["Stopping at the first error.", "`isinstance(True, int)` treating bools as numbers.", "Recursing into a value after a type error."],
        tests=[
            T(({"sentiment": "positive", "confidence": 0.93, "entities": [{"name": "Acme", "type": "org"}], "summary": "Happy customer."},
               LLM_SCHEMA), [], example=True, note="valid payload"),
            T(({"sentiment": "mixed", "confidence": 1.2, "entities": [{"name": "Acme", "type": "org"}, {"type": "city"}], "extra": True},
               LLM_SCHEMA),
              ["$.confidence: must be <= 1", "$.entities[1].name: is required",
               "$.entities[1].type: must be one of ['person', 'org', 'product']", "$.extra: is not allowed",
               "$.sentiment: must be one of ['positive', 'negative', 'neutral']"], example=True, note="many errors, sorted"),
            T(([1], LLM_SCHEMA), ["$: expected object"], note="boundary: root type error"),
            T(({"sentiment": "neutral", "confidence": True, "entities": "Acme"}, LLM_SCHEMA),
              ["$.confidence: expected number", "$.entities: expected array"], note="bool is not a number; no deeper checks"),
            T(({"tags": [], "n": 3.5}, {"type": "object", "properties": {"tags": {"type": "array", "min_items": 1, "items": {"type": "string"}},
                                                                          "n": {"type": "integer"}}}),
              ["$.n: expected integer", "$.tags: must have at least 1 items"], note="min_items and integer"),
            T(({"s": "x" * 6}, {"type": "object", "properties": {"s": {"type": "string", "max_length": 5}}}),
              ["$.s: longer than 5 characters"], note="max_length"),
            T(({}, {"type": "uuid"}), raises="ValueError", note="unknown schema type"),
        ],
        complexity="O(size of payload) time; recursion depth equals nesting depth.",
        extension="Return structured errors `{'path', 'code', 'message'}` and build an automatic 'repair' prompt for the LLM.",
    ),
    # ------------------------------------------------------------------------ Q083
    Question(
        id=83,
        title="Per-Class Precision, Recall and F1 from Prediction Logs",
        difficulty="Advanced",
        category=INT,
        roles=[DS_ROLE, MLE, AAI],
        skills=["counting with dicts", "safe division", "macro averaging", "rounding at the boundary"],
        minutes=35,
        scenario=(
            "A ticket-routing classifier logs `{label, prediction}` pairs in production. You want an "
            "evaluation report without installing scikit-learn on the logging host. The formulas are "
            "given below – the task is implementing them carefully in Python."
        ),
        problem="Write `classification_report(records)` that returns per-class metrics, macro averages and accuracy.",
        signature="def classification_report(records: list[dict]) -> dict:",
        entry="classification_report",
        inputs=["`records`: dicts with `label` and `prediction` (strings; either may be `None`)."],
        output=(
            "`{'per_class': {cls: {'precision', 'recall', 'f1', 'support'}}, 'macro_avg': {'precision', 'recall', 'f1'}, 'accuracy': float}` "
            "– all floats rounded to 4 decimals."
        ),
        constraints=[
            "Skip records whose label or prediction is `None`. Raise `ValueError` if no records remain.",
            "Classes = every value seen as a label **or** prediction.",
            "precision = TP/(TP+FP), recall = TP/(TP+FN), f1 = 2PR/(P+R); any 0/0 is 0.0. support = number of true labels.",
            "macro_avg = unweighted mean of the **unrounded** per-class values; accuracy = correct / total.",
        ],
        criteria=["Correct confusion counts.", "Zero-division handled.", "Macro averages from unrounded values.", "Rounding only at the end."],
        hints=[
            "Count TP, FP and FN per class in one pass, then compute the ratios.",
            "For each record: if correct → TP[label] += 1; otherwise FP[prediction] += 1 and FN[label] += 1. Write a `_safe_div(a, b)` helper.",
            "for r in valid: if r['label'] == r['prediction']: tp[...] += 1 else: fp[pred] += 1; fn[label] += 1\nfor c in classes: p = safe(tp, tp+fp); r = safe(tp, tp+fn); f = safe(2*p*r, p+r)\n"
            "macro = mean of raw values; round everything at the end",
        ],
        solution='''
def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_report(records: list[dict]) -> dict:
    """Per-class precision/recall/F1/support, macro averages and accuracy."""
    pairs = [(r["label"], r["prediction"]) for r in records
             if r.get("label") is not None and r.get("prediction") is not None]
    if not pairs:
        raise ValueError("no records with both label and prediction")
    classes = sorted({label for label, _ in pairs} | {pred for _, pred in pairs})
    tp = dict.fromkeys(classes, 0)
    fp = dict.fromkeys(classes, 0)
    fn = dict.fromkeys(classes, 0)
    support = dict.fromkeys(classes, 0)
    for label, prediction in pairs:
        support[label] += 1
        if label == prediction:
            tp[label] += 1
        else:
            fp[prediction] += 1
            fn[label] += 1

    raw = {}
    for cls in classes:
        precision = _safe_div(tp[cls], tp[cls] + fp[cls])
        recall = _safe_div(tp[cls], tp[cls] + fn[cls])
        raw[cls] = (precision, recall, _safe_div(2 * precision * recall, precision + recall))

    per_class = {
        cls: {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": support[cls]}
        for cls, (p, r, f) in raw.items()
    }
    n = len(classes)
    macro = {
        "precision": round(sum(v[0] for v in raw.values()) / n, 4),
        "recall": round(sum(v[1] for v in raw.values()) / n, 4),
        "f1": round(sum(v[2] for v in raw.values()) / n, 4),
    }
    accuracy = round(sum(tp.values()) / len(pairs), 4)
    return {"per_class": per_class, "macro_avg": macro, "accuracy": accuracy}
''',
        explanation=(
            "A single pass fills TP/FP/FN/support counters; every misclassification is both a false "
            "positive for the predicted class and a false negative for the true class. Averaging "
            "unrounded values and rounding only in the output avoids accumulating rounding error."
        ),
        edge_cases=["Class that is only predicted (support 0).", "Class never predicted (precision 0/0 → 0).", "All-None records → ValueError."],
        mistakes=["Ignoring classes that appear only as predictions.", "Macro-averaging the rounded values.", "ZeroDivisionError for unseen classes."],
        tests=[
            T(([{"label": l, "prediction": p} for l, p in [("cat", "cat"), ("cat", "dog"), ("dog", "dog"), ("dog", "dog"),
                                                            ("bird", "cat"), ("bird", "bird"), ("dog", "bird")]],),
              {"per_class": {"bird": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2},
                             "cat": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2},
                             "dog": {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667, "support": 3}},
               "macro_avg": {"precision": 0.5556, "recall": 0.5556, "f1": 0.5556}, "accuracy": 0.5714}, example=True,
              note="three classes"),
            T(([{"label": "a", "prediction": "b"}, {"label": None, "prediction": "a"}],),
              {"per_class": {"a": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
                             "b": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}},
               "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "accuracy": 0.0}, example=True,
              note="boundary: zero divisions, None skipped"),
            T(([{"label": "x", "prediction": "x"}],),
              {"per_class": {"x": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}},
               "macro_avg": {"precision": 1.0, "recall": 1.0, "f1": 1.0}, "accuracy": 1.0}, note="perfect"),
            T(([{"label": None, "prediction": None}],), raises="ValueError", note="no usable records"),
            CT(_q083_sample, note="data/sample_data/prediction_logs.json"),
        ],
        complexity="O(n + c) time, O(c) space for c classes.",
        extension="Add weighted averages (weighted by support) and a printable text table.",
    ),
    # ------------------------------------------------------------------------ Q084
    Question(
        id=84,
        title="Latency SLO Report per Endpoint (Nearest-Rank Percentiles)",
        difficulty="Advanced",
        category=INT,
        roles=[AIE, MLE, DE],
        skills=["grouping", "nearest-rank percentiles", "math.ceil", "handling missing measurements"],
        minutes=30,
        scenario=(
            "The inference platform's SLO says p95 latency must stay below a threshold per "
            "endpoint, and the 5xx rate below 1%. You compute a daily report from request logs. "
            "Some requests timed out before a latency could be recorded."
        ),
        problem=(
            "Write `latency_report(requests)` that returns, per endpoint, the request count, 5xx "
            "error rate and nearest-rank p50/p95/p99 latency."
        ),
        signature="def latency_report(requests: list[dict]) -> dict[str, dict]:",
        entry="latency_report",
        inputs=["`requests`: dicts with `endpoint` (str), `status` (int) and `latency_ms` (number or `None`)."],
        output="`{endpoint: {'count': int, 'error_rate': float, 'p50', 'p95', 'p99'}}`.",
        constraints=[
            "`count` includes every request; `error_rate` = share with `status >= 500`, rounded to 4 decimals.",
            "Percentiles use only non-`None` latencies, nearest-rank method: sort, `rank = ceil(p / 100 * n)`, value = `sorted[rank - 1]`.",
            "If an endpoint has no latencies, its percentiles are `None`.",
            "Do not modify the input.",
        ],
        criteria=["Correct grouping and counts.", "Nearest-rank percentiles.", "Missing latencies handled."],
        hints=[
            "Group per endpoint, keeping the count, the error count and a list of latencies.",
            "Sort each latency list once and index it three times with `math.ceil(p / 100 * n) - 1`.",
            "for r in requests: g = groups.setdefault(ep, {'count': 0, 'errors': 0, 'lat': []}); update\n"
            "def nearest(sorted_vals, p): return sorted_vals[ceil(p / 100 * len(sorted_vals)) - 1]",
        ],
        solution='''
import math


def _nearest_rank(sorted_values: list[float], p: float):
    if not sorted_values:
        return None
    rank = math.ceil(p / 100 * len(sorted_values))
    return sorted_values[max(rank, 1) - 1]


def latency_report(requests: list[dict]) -> dict[str, dict]:
    """Per-endpoint count, 5xx rate and nearest-rank p50/p95/p99."""
    groups: dict[str, dict] = {}
    for request in requests:
        group = groups.setdefault(request["endpoint"], {"count": 0, "errors": 0, "latencies": []})
        group["count"] += 1
        if request["status"] >= 500:
            group["errors"] += 1
        if request.get("latency_ms") is not None:
            group["latencies"].append(request["latency_ms"])
    report = {}
    for endpoint, group in groups.items():
        latencies = sorted(group["latencies"])
        report[endpoint] = {
            "count": group["count"],
            "error_rate": round(group["errors"] / group["count"], 4),
            "p50": _nearest_rank(latencies, 50),
            "p95": _nearest_rank(latencies, 95),
            "p99": _nearest_rank(latencies, 99),
        }
    return report
''',
        explanation=(
            "Nearest-rank percentiles always return an observed value, which is what many SLO "
            "tools report. Sorting once per endpoint serves all three percentiles. Missing "
            "latencies still count as requests (and possibly errors) but cannot contribute a time."
        ),
        edge_cases=["All latencies missing → `None` percentiles.", "Single latency → all percentiles equal it.", "4xx is not an error."],
        mistakes=["Dropping requests with missing latency from the count.", "Using linear interpolation (different method).", "Off-by-one: `sorted[rank]`."],
        tests=[
            T(([{"endpoint": "/v1/classify", "status": 500 if i in (3, 7) else 200, "latency_ms": i} for i in range(1, 21)]
               + [{"endpoint": "/v1/embed", "status": 200, "latency_ms": 30}, {"endpoint": "/v1/embed", "status": 503, "latency_ms": None},
                  {"endpoint": "/v1/embed", "status": 200, "latency_ms": 10}],),
              {"/v1/classify": {"count": 20, "error_rate": 0.1, "p50": 10, "p95": 19, "p99": 20},
               "/v1/embed": {"count": 3, "error_rate": 0.3333, "p50": 10, "p95": 30, "p99": 30}}, example=True,
              note="two endpoints; missing latency counted"),
            T(([],), {}, example=True, note="boundary: no requests"),
            T(([{"endpoint": "/x", "status": 504, "latency_ms": None}],),
              {"/x": {"count": 1, "error_rate": 1.0, "p50": None, "p95": None, "p99": None}}, note="no latencies"),
            T(([{"endpoint": "/x", "status": 404, "latency_ms": 7}],),
              {"/x": {"count": 1, "error_rate": 0.0, "p50": 7, "p95": 7, "p99": 7}}, note="4xx is not an error"),
            CT(_q084_sample, note="data/sample_data/inference_requests.jsonl"),
        ],
        complexity="O(n log n) time (sorting per endpoint), O(n) space.",
        extension="Flag endpoints breaching an SLO dict such as `{'/v1/generate': {'p95': 2000, 'error_rate': 0.01}}`.",
    ),
    # ------------------------------------------------------------------------ Q085
    Question(
        id=85,
        title="Reconstruct Request Outcomes from Retry Logs",
        difficulty="Advanced",
        category=INT,
        roles=[AIE, DE, PYDEV],
        skills=["grouping by id", "de-duplication by composite key", "out-of-order data", "gap detection"],
        minutes=40,
        scenario=(
            "An LLM client retries failed calls. Each attempt is logged separately, logs arrive out "
            "of order, the shipper sometimes duplicates a line, and occasionally an attempt is lost. "
            "Reliability engineers want to know how often retries rescue requests."
        ),
        problem="Write `summarize_retries(records)` that reconstructs each request's attempts and reports the outcomes.",
        signature="def summarize_retries(records: list[dict]) -> dict:",
        entry="summarize_retries",
        inputs=["`records`: dicts with `request_id` (str), `attempt` (int ≥ 1) and `status` (`'success'`, `'retryable_error'` or `'fatal_error'`), in any order."],
        output=(
            "`{'requests', 'succeeded_first_try', 'succeeded_after_retry', 'failed', 'total_attempts', 'retry_rate', 'max_attempts', 'gaps'}`"
        ),
        constraints=[
            "Duplicate `(request_id, attempt)` records count once (keep the first).",
            "A request's outcome is the status of its **highest** attempt number.",
            "`succeeded_first_try`: highest attempt is 1 and succeeded; `succeeded_after_retry`: highest attempt > 1 and succeeded; `failed`: final status is not success.",
            "`total_attempts`: number of distinct attempt records; `retry_rate`: share of requests whose highest attempt > 1 (rounded to 4; 0.0 when empty).",
            "`max_attempts`: largest attempt number seen (0 when empty); `gaps`: sorted request ids whose attempt numbers are not exactly 1..highest.",
        ],
        criteria=["Correct outcome classification.", "Duplicates removed.", "Out-of-order data.", "Gap detection."],
        hints=[
            "Build `{request_id: {attempt: status}}` first – a dict per request removes duplicates naturally.",
            "Use `if attempt not in attempts: attempts[attempt] = status` to keep the first occurrence; then `max(attempts)` gives the final attempt and `set(attempts) != set(range(1, last + 1))` detects gaps.",
            "by_request = {}\nfor r in records: by_request.setdefault(rid, {}).setdefault(attempt, status)\nfor rid, attempts in by_request.items():\n"
            "    last = max(attempts); final = attempts[last]; classify; total += len(attempts); gaps if keys != 1..last",
        ],
        solution='''
def summarize_retries(records: list[dict]) -> dict:
    """Outcome statistics from per-attempt retry logs."""
    by_request: dict[str, dict[int, str]] = {}
    for record in records:
        attempts = by_request.setdefault(record["request_id"], {})
        attempts.setdefault(record["attempt"], record["status"])

    summary = {"requests": len(by_request), "succeeded_first_try": 0, "succeeded_after_retry": 0,
               "failed": 0, "total_attempts": 0, "retry_rate": 0.0, "max_attempts": 0, "gaps": []}
    retried = 0
    for request_id, attempts in by_request.items():
        last = max(attempts)
        summary["total_attempts"] += len(attempts)
        summary["max_attempts"] = max(summary["max_attempts"], last)
        if last > 1:
            retried += 1
        if attempts[last] != "success":
            summary["failed"] += 1
        elif last == 1:
            summary["succeeded_first_try"] += 1
        else:
            summary["succeeded_after_retry"] += 1
        if set(attempts) != set(range(1, last + 1)):
            summary["gaps"].append(request_id)
    if by_request:
        summary["retry_rate"] = round(retried / len(by_request), 4)
    summary["gaps"].sort()
    return summary
''',
        explanation=(
            "Nesting dicts by request and attempt both groups the records and de-duplicates them "
            "(`setdefault` keeps the first status). Once each request's attempts are known, the "
            "final outcome, retry status and gaps are simple set and max operations."
        ),
        edge_cases=["Duplicate log lines.", "Attempt 2 logged before attempt 1.", "Missing attempt numbers.", "Empty input."],
        mistakes=["Using the *last record seen* instead of the highest attempt.", "Counting duplicates in `total_attempts`.", "Dividing by zero for empty input."],
        tests=[
            T((RETRY_RECORDS,),
              {"requests": 5, "succeeded_first_try": 1, "succeeded_after_retry": 3, "failed": 1, "total_attempts": 10,
               "retry_rate": 0.8, "max_attempts": 3, "gaps": ["r5"]}, example=True, note="duplicates, out of order, gap"),
            T(([],), {"requests": 0, "succeeded_first_try": 0, "succeeded_after_retry": 0, "failed": 0, "total_attempts": 0,
                      "retry_rate": 0.0, "max_attempts": 0, "gaps": []}, example=True, note="boundary: no records"),
            T(([{"request_id": "a", "attempt": 1, "status": "fatal_error"}],),
              {"requests": 1, "succeeded_first_try": 0, "succeeded_after_retry": 0, "failed": 1, "total_attempts": 1,
               "retry_rate": 0.0, "max_attempts": 1, "gaps": []}, note="fatal on first try"),
            T(([{"request_id": "b", "attempt": 2, "status": "success"}],),
              {"requests": 1, "succeeded_first_try": 0, "succeeded_after_retry": 1, "failed": 0, "total_attempts": 1,
               "retry_rate": 1.0, "max_attempts": 2, "gaps": ["b"]}, note="first attempt missing"),
            CT(_q085_sample, note="data/sample_data/retry_logs.json"),
        ],
        complexity="O(n) time and space (plus sorting the gap list).",
        extension="Report the distribution of attempts needed for successful requests, e.g. `{1: 120, 2: 30, 3: 4}`.",
    ),
]
