"""Expert questions Q086–Q100: engineering-quality solutions and realistic challenges."""

import json
import random
import string
import threading
import time
import tracemalloc

from ..schema import (
    AAI, AIE, ALG, DATA, DBG, DE, DS_ROLE, FILE, FUNC, INT, MLE, OOP, PERF, PYDEV,
    SAMPLE_DATA_DIR, CT, File, Question, T, TmpPath,
)


def _raises(exc_type, func, message):
    try:
        func()
    except exc_type:
        return
    raise AssertionError(message)


# ------------------------------------------------------------------ Q086 tests
def _q086_order_and_errors(impl, tmp):
    def work(x):
        time.sleep(random.Random(x).uniform(0, 0.01))
        if x % 4 == 3:
            raise ValueError(f"bad item {x}")
        return x * 10

    got = impl(work, range(10), max_workers=3)
    expected = [{"ok": False, "error": f"ValueError: bad item {x}"} if x % 4 == 3 else {"ok": True, "value": x * 10}
                for x in range(10)]
    assert got == expected, f"results must be in input order with errors captured; got {got}"


def _q086_concurrency_limit(impl, tmp):
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def work(x):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return x

    impl(work, range(12), max_workers=4)
    assert state["peak"] <= 4, f"{state['peak']} calls ran at once; max_workers is 4"
    assert state["peak"] >= 2, "calls do not seem to run concurrently"


def _q086_speedup(impl, tmp):
    start = time.perf_counter()
    impl(lambda x: time.sleep(0.05) or x, range(20), max_workers=5)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.6, f"20 x 50 ms tasks with 5 workers took {elapsed:.2f}s (serial would be 1.0s)"


def _q086_bounded_submission(impl, tmp):
    lock = threading.Lock()
    state = {"consumed": 0, "finished": 0, "worst": 0}

    def source():
        for i in range(60):
            with lock:
                state["consumed"] += 1
                state["worst"] = max(state["worst"], state["consumed"] - state["finished"])
            yield i

    def work(x):
        time.sleep(0.005)
        with lock:
            state["finished"] += 1
        return x

    results = impl(work, source(), max_workers=2)
    assert [r["value"] for r in results] == list(range(60))
    assert state["worst"] <= 2 * 2 + 1, (f"up to {state['worst']} items were pulled from the input before finishing; "
                                         "keep at most 2 * max_workers tasks in flight")


def _q086_edge(impl, tmp):
    assert impl(lambda x: x, [], max_workers=2) == []
    _raises(ValueError, lambda: impl(lambda x: x, [1], max_workers=0), "max_workers < 1 should raise ValueError")


# ------------------------------------------------------------------ Q087 tests
def _q087_scale(impl, tmp):
    rng = random.Random(9)
    vocab = ["".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(5000)]
    docs = {}
    planted = []
    for i in range(1500):
        docs[f"d{i:04d}"] = " ".join(rng.choices(vocab, k=30))
    for i in range(50):
        words = docs[f"d{i:04d}"].split()
        words[15] = "changed"
        docs[f"z{i:04d}"] = " ".join(words)
        planted.append([f"d{i:04d}", f"z{i:04d}"])
    start = time.perf_counter()
    clusters = impl(docs, 0.7)
    elapsed = time.perf_counter() - start
    assert clusters == planted, "expected exactly the 50 planted near-duplicate pairs"
    assert elapsed < 3.0, f"took {elapsed:.2f}s for 1,550 documents – use an inverted index to find candidate pairs"


# ------------------------------------------------------------------ Q088 tests
def _q088_sequence(impl, tmp):
    log, state = tmp / "app.log", tmp / "state" / "app.offset.json"

    def append(text):
        with open(log, "ab") as handle:
            handle.write(text.encode("utf-8"))

    append("a\nb\n")
    assert impl(str(log), str(state)) == ["a", "b"]
    append("c\npar")
    assert impl(str(log), str(state)) == ["c"], "a partial last line must wait for its newline"
    append("tial\r\n")
    assert impl(str(log), str(state)) == ["partial"], "the completed line is returned once; \\r\\n is stripped"
    assert impl(str(log), str(state)) == [], "no new data → no lines"
    append("café ☕\n")
    assert impl(str(log), str(state)) == ["café ☕"], "offsets must be byte offsets (UTF-8)"
    log.write_bytes(b"new\n")  # rotated/truncated: smaller than the saved offset
    assert impl(str(log), str(state)) == ["new"], "a file smaller than the saved offset was rotated – start from 0"


def _q088_state_file(impl, tmp):
    log, state = tmp / "x.log", tmp / "x.state"
    log.write_bytes(b"one\ntwo\n")
    impl(str(log), str(state))
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved == {"offset": 8}, f"state file should contain {{'offset': 8}}, got {saved}"
    assert sorted(p.name for p in tmp.iterdir()) == ["x.log", "x.state"], "write the state atomically; no temp files left"


def _q088_corrupt_state(impl, tmp):
    log, state = tmp / "y.log", tmp / "y.state"
    log.write_bytes(b"one\n")
    for content in ("not json", '{"offset": "7"}', '{"offset": -1}', "[]"):
        state.write_text(content, encoding="utf-8")
        _raises(ValueError, lambda: impl(str(log), str(state)), f"corrupt state {content!r} should raise ValueError")


# ------------------------------------------------------------------ Q089 tests
def _q089_model(Typed):
    class ModelConfig:
        learning_rate = Typed(float, min_value=0.0, max_value=1.0)
        epochs = Typed(int, min_value=1)
        name = Typed(str)
        notes = Typed(str, nullable=True)

        def __init__(self, name, learning_rate, epochs, notes=None):
            self.name, self.learning_rate, self.epochs, self.notes = name, learning_rate, epochs, notes

    return ModelConfig


def _q089_basic(impl, tmp):
    Config = _q089_model(impl)
    a = Config("bert", 0.001, 3)
    b = Config("gpt", 1, 10, notes="baseline")
    assert (a.name, a.learning_rate, a.epochs, a.notes) == ("bert", 0.001, 3, None)
    assert (b.name, b.learning_rate, b.epochs, b.notes) == ("gpt", 1, 10, "baseline"), \
        "each instance must keep its own values (don't store values on the descriptor)"
    assert isinstance(Config.learning_rate, impl), "class-level access should return the descriptor itself"


def _q089_type_errors(impl, tmp):
    Config = _q089_model(impl)
    c = Config("bert", 0.1, 3)
    for attr, value in (("epochs", 2.5), ("epochs", True), ("learning_rate", "0.1"), ("name", None), ("learning_rate", False)):
        try:
            setattr(c, attr, value)
        except TypeError as exc:
            assert attr in str(exc), f"the TypeError message should name the attribute {attr!r}: {exc}"
        else:
            raise AssertionError(f"setting {attr} = {value!r} should raise TypeError")
    assert (c.epochs, c.learning_rate, c.name) == (3, 0.1, "bert"), "rejected assignments must not change the value"


def _q089_ranges(impl, tmp):
    Config = _q089_model(impl)
    c = Config("bert", 0.1, 3)
    c.learning_rate = 0.0
    c.learning_rate = 1.0
    for attr, value in (("learning_rate", 1.5), ("learning_rate", -0.01), ("epochs", 0)):
        try:
            setattr(c, attr, value)
        except ValueError as exc:
            assert attr in str(exc), f"the ValueError message should name the attribute {attr!r}: {exc}"
        else:
            raise AssertionError(f"setting {attr} = {value!r} should raise ValueError")


def _q089_unset(impl, tmp):
    class Partial:
        size = impl(int)

    p = Partial()
    _raises(AttributeError, lambda: p.size, "reading an attribute that was never set should raise AttributeError")
    p.size = 4
    assert p.size == 4


# ------------------------------------------------------------------ Q091 tests
def _q091_fast(impl, tmp):
    rng = random.Random(4)
    features = [{"entity_id": f"e{i % 10}", "ts": rng.randint(0, 1_000_000), "features": {"v": i}} for i in range(50_000)]
    labels = [{"entity_id": f"e{i % 10}", "ts": rng.randint(0, 1_000_000), "label": i % 2} for i in range(50_000)]
    start = time.perf_counter()
    out = impl(labels, features)
    elapsed = time.perf_counter() - start
    by_entity = {}
    for f in features:
        by_entity.setdefault(f["entity_id"], []).append(f)
    probe = labels[777]
    candidates = [f for f in by_entity[probe["entity_id"]] if f["ts"] <= probe["ts"]]
    expected_ts = max((f["ts"] for f in candidates), default=None)
    assert out[777]["feature_ts"] == expected_ts, "wrong as-of match on the large input"
    assert elapsed < 2.0, f"took {elapsed:.2f}s – sort features per entity once and use bisect"


# ------------------------------------------------------------------ Q093 tests
def _write_lines(path, lines):
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _q093_merge(impl, tmp):
    a, b, c = tmp / "a.log", tmp / "b.log", tmp / "c.log"
    _write_lines(a, ["2024-05-01T10:00:00Z a1", "2024-05-01T10:05:00Z a2", "", "2024-05-01T10:09:00Z a3"])
    _write_lines(b, ["2024-05-01T10:01:00Z b1", "2024-05-01T10:05:00Z b2"])
    c.write_text("", encoding="utf-8")
    out = tmp / "merged" / "all.log"
    n = impl([str(a), str(b), str(c)], str(out))
    assert n == 5, f"expected 5 lines written, got {n}"
    assert out.read_text(encoding="utf-8").splitlines() == [
        "2024-05-01T10:00:00Z a1", "2024-05-01T10:01:00Z b1", "2024-05-01T10:05:00Z a2",
        "2024-05-01T10:05:00Z b2", "2024-05-01T10:09:00Z a3"], "wrong merge order (ties keep file order)"


def _q093_unsorted(impl, tmp):
    a = tmp / "a.log"
    _write_lines(a, ["2024-05-01T10:05:00Z late", "2024-05-01T10:00:00Z early"])
    out = tmp / "out.log"
    _raises(ValueError, lambda: impl([str(a)], str(out)), "an unsorted input file should raise ValueError")
    assert not out.exists(), "on failure the output file must not be created"
    assert sorted(p.name for p in tmp.iterdir()) == ["a.log"], "no temporary files may be left behind"


def _q093_memory(impl, tmp):
    paths = []
    for f in range(3):
        path = tmp / f"part{f}.log"
        with open(path, "w", encoding="utf-8") as handle:
            for i in range(60_000):
                handle.write(f"{i * 3 + f:012d} event payload {'x' * 30}\n")
        paths.append(str(path))
    tracemalloc.start()
    try:
        n = impl(paths, str(tmp / "merged.log"))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert n == 180_000
    assert peak < 3_000_000, f"peak memory {peak / 1e6:.1f} MB – stream the files (heapq.merge) instead of loading them"


# ------------------------------------------------------------------ Q094 tests
def _q094_properties(impl, tmp):
    stream = (f"item-{i}" for i in range(1000))
    sample = impl(stream, 10, seed=1)
    assert len(sample) == 10 and len(set(sample)) == 10, "sample must contain k distinct items"
    indexes = [int(s.split("-")[1]) for s in sample]
    assert indexes == sorted(indexes), "return the sample in stream order"
    assert impl(range(5), 10, seed=3) == [0, 1, 2, 3, 4], "fewer than k items → all of them"
    assert impl(range(100), 5, seed=42) == impl(range(100), 5, seed=42), "same seed must give the same sample"
    assert impl(range(10), 0, seed=1) == []
    _raises(ValueError, lambda: impl(range(10), -1), "negative k should raise ValueError")


def _q094_uniform(impl, tmp):
    counts = [0] * 10
    for trial in range(20_000):
        for item in impl(range(10), 2, seed=trial):
            counts[item] += 1
    assert all(abs(c - 4000) < 350 for c in counts), f"sample is not uniform: counts per item {counts} (expected ~4000 each)"


def _q094_memory(impl, tmp):
    def stream():
        for i in range(300_000):
            yield ("row", i, "x" * 40)

    tracemalloc.start()
    try:
        sample = impl(stream(), 20, seed=7)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(sample) == 20
    assert peak < 1_000_000, f"peak memory {peak / 1e6:.1f} MB – keep only k items, never the whole stream"


# ------------------------------------------------------------------ Q095 tests
def _q095_fast(impl, tmp):
    rng = random.Random(8)
    tx = [{"user": f"u{rng.randint(0, 19)}", "ts": rng.randint(0, 200_000), "amount": rng.randint(1, 500)} for _ in range(100_000)]
    start = time.perf_counter()
    got = impl(tx, 50_000)
    elapsed = time.perf_counter() - start
    probe = tx[4242]
    expected = round(sum(t["amount"] for t in tx if t["user"] == probe["user"] and probe["ts"] - 50_000 < t["ts"] <= probe["ts"]), 2)
    assert got[4242] == expected, f"wrong rolling sum for transaction 4242: expected {expected}, got {got[4242]}"
    assert elapsed < 2.0, f"took {elapsed:.2f}s – sort per user and slide a window instead of rescanning"


# ------------------------------------------------------------------ Q096 tests
def _make_model(bad="BAD", transient_failures=0, wrong_length=False):
    state = {"calls": 0, "batches": []}

    def predict_batch(inputs):
        state["calls"] += 1
        state["batches"].append(list(inputs))
        if state["calls"] <= transient_failures:
            raise TimeoutError("upstream timeout")
        if bad in inputs:
            raise ValueError("bad input")
        if wrong_length:
            return [len(x) for x in inputs][:-1]
        return [len(x) for x in inputs]

    return predict_batch, state


ITEMS = [{"id": f"x{i}", "input": "ok" * i} for i in range(1, 9)]


def _q096_success(impl, tmp):
    predict, state = _make_model()
    out = impl(ITEMS[:5], predict, batch_size=2)
    assert out == {"predictions": {f"x{i}": 2 * i for i in range(1, 6)}, "failed": {}, "calls": 3}, f"got {out}"
    assert state["batches"][0] == ["ok", "okok"], "batches must follow input order"


def _q096_isolation(impl, tmp):
    items = [dict(item) for item in ITEMS]
    items[4]["input"] = "BAD"
    predict, state = _make_model()
    out = impl(items, predict, batch_size=4)
    assert out["failed"] == {"x5": "ValueError: bad input"}, f"only the bad item should fail; got {out['failed']}"
    assert sorted(out["predictions"]) == sorted(f"x{i}" for i in range(1, 9) if i != 5)
    assert out["calls"] == 6 == state["calls"], f"expected 6 calls with halving (4, 4 → 2, 2 → 1, 1), got {state['calls']}"


def _q096_transient(impl, tmp):
    predict, state = _make_model(transient_failures=2)
    out = impl(ITEMS[:2], predict, batch_size=2, max_retries=2)
    assert out["failed"] == {} and out["calls"] == 3, f"two transient failures then success should take 3 calls; got {out}"
    predict, state = _make_model(transient_failures=100)
    out = impl(ITEMS[:3], predict, batch_size=2, max_retries=1)
    assert out["calls"] == 4, f"each batch gets 1 + max_retries attempts; expected 4 calls, got {out['calls']}"
    assert out["failed"] == {f"x{i}": "TimeoutError: upstream timeout" for i in (1, 2, 3)}, f"got {out['failed']}"


def _q096_contract(impl, tmp):
    predict, _ = _make_model(wrong_length=True)
    _raises(RuntimeError, lambda: impl(ITEMS[:3], predict, batch_size=3), "a wrong number of predictions should raise RuntimeError")
    predict, _ = _make_model()
    _raises(ValueError, lambda: impl([{"id": "a", "input": "x"}, {"id": "a", "input": "y"}], predict, batch_size=2),
            "duplicate ids should raise ValueError")
    _raises(ValueError, lambda: impl(ITEMS, predict, batch_size=0), "batch_size < 1 should raise ValueError")
    _raises(ValueError, lambda: impl(ITEMS, predict, batch_size=2, max_retries=-1), "max_retries < 0 should raise ValueError")
    assert impl([], predict, batch_size=2) == {"predictions": {}, "failed": {}, "calls": 0}


# ------------------------------------------------------------------ Q097 tests
def _q097_sample(impl, tmp):
    records = json.loads((SAMPLE_DATA_DIR / "eval_results.json").read_text(encoding="utf-8"))
    report = impl(records, "v1", "v2", "language", min_support=30, tolerance=0.02)
    verdicts = {name: s["verdict"] for name, s in report["slices"].items()}
    assert verdicts == {"de": "insufficient_data", "en": "improvement", "es": "neutral", "hi": "regression"}, f"got {verdicts}"
    assert report["unpaired"] == 6, f"expected 6 unpaired records, got {report['unpaired']}"


# ------------------------------------------------------------------ data
ASOF_FEATURES = [
    {"entity_id": "u1", "ts": 100, "features": {"spend_7d": 10}},
    {"entity_id": "u1", "ts": 200, "features": {"spend_7d": 25}},
    {"entity_id": "u2", "ts": 150, "features": {"spend_7d": 3}},
    {"entity_id": "u1", "ts": 200, "features": {"spend_7d": 30}},
    {"entity_id": "u1", "ts": 300, "features": {"spend_7d": 50}},
]
ASOF_LABELS = [
    {"entity_id": "u1", "ts": 250, "label": 1},
    {"entity_id": "u1", "ts": 99, "label": 0},
    {"entity_id": "u2", "ts": 150, "label": 0},
    {"entity_id": "u3", "ts": 500, "label": 1},
    {"entity_id": "u1", "ts": 300, "label": 1},
]

SCD_CURRENT = [
    {"customer_id": "c1", "attributes": {"tier": "free", "city": "Pune"}, "valid_from": "2024-01-01", "valid_to": None, "is_current": True},
    {"customer_id": "c2", "attributes": {"tier": "free"}, "valid_from": "2024-01-01", "valid_to": "2024-03-01", "is_current": False},
    {"customer_id": "c2", "attributes": {"tier": "pro"}, "valid_from": "2024-03-01", "valid_to": None, "is_current": True},
    {"customer_id": "c3", "attributes": {"tier": "pro"}, "valid_from": "2024-02-01", "valid_to": None, "is_current": True},
]
SCD_UPDATES = [
    {"customer_id": "c4", "attributes": {"tier": "free"}},
    {"customer_id": "c1", "attributes": {"tier": "pro", "city": "Pune"}},
    {"customer_id": "c2", "attributes": {"tier": "pro"}},
]

LATEST_RECORDS = [
    {"id": "sku-1", "price": 10, "updated_at": "2024-05-01T09:00:00Z"},
    {"id": "sku-1", "price": 12, "updated_at": "2024-05-01T10:30:00+02:00"},
    {"id": "sku-2", "price": 5, "updated_at": "2024-05-01T08:00:00+00:00"},
    {"id": "sku-2", "price": 6, "updated_at": "2024-05-01T08:00:00Z"},
    {"id": "sku-3", "price": 1},
    {"id": "sku-0", "price": 7, "updated_at": "2024-04-30T23:59:59-05:00"},
]

STREAM_EVENTS = [
    {"seq": 3, "type": "tool_call_delta", "call_id": "call_1", "arguments": '"Pune", "unit": "C"}'},
    {"seq": 0, "type": "text_delta", "text": "Let me check "},
    {"seq": 2, "type": "tool_call_delta", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": '},
    {"seq": 1, "type": "text_delta", "text": "the weather."},
    {"seq": 5, "type": "done", "finish_reason": "tool_calls"},
    {"seq": 3, "type": "tool_call_delta", "call_id": "call_1", "arguments": '"Pune", "unit": "C"}'},
    {"seq": 4, "type": "tool_call_delta", "call_id": "call_2", "name": "get_time", "arguments": '{"tz": "IST"}'},
]

CONTEXT_DOCS = [
    {"id": "d1", "text": "alpha beta gamma delta", "score": 0.5},
    {"id": "d2", "text": "one two three four five six seven eight", "score": 0.9},
    {"id": "d3", "text": "Alpha  beta gamma delta", "score": 0.7},
    {"id": "d4", "text": "short text", "score": 0.7},
    {"id": "d5", "text": "x " * 30, "score": 0.95},
]

CHUNK_TEXT = ("RAG systems retrieve context. They split documents into chunks. Each chunk should be "
              "self-contained! Overlap preserves continuity between chunks? Yes.")


def _eval(example, model, language, correct):
    return {"example_id": example, "model": model, "language": language, "correct": correct}


EVAL_RECORDS = (
    [_eval(f"en{i}", "v1", "en", i < 2) for i in range(4)] + [_eval(f"en{i}", "v2", "en", i < 3) for i in range(4)]
    + [_eval(f"hi{i}", "v1", "hi", i < 3) for i in range(4)] + [_eval(f"hi{i}", "v2", "hi", i < 1) for i in range(4)]
    + [_eval("de0", "v1", "de", True), _eval("de0", "v2", "de", False)]
    + [_eval("es0", "v1", "es", True), _eval("es0", "v2", "es", True), _eval("es1", "v1", "es", False),
       _eval("es1", "v2", "es", False), _eval("es2", "v2", "es", True)]
)

QUESTIONS = [
    # ------------------------------------------------------------------------ Q086
    Question(
        id=86,
        title="Bounded Parallel Map with Ordered Results and Error Capture",
        difficulty="Expert",
        category=FUNC,
        roles=[AIE, DE, MLE],
        skills=["concurrent.futures", "back-pressure", "result ordering", "error capture", "thread safety"],
        minutes=60,
        scenario=(
            "You need to call an external embedding API for 2 million documents read lazily from "
            "storage. Calls are I/O-bound, so threads help, but the provider allows only N concurrent "
            "requests and the input is far too large to submit as futures all at once. A few documents "
            "will fail and must not abort the whole job."
        ),
        problem=(
            "Write `parallel_map(func, items, max_workers=4)` that applies `func` to every item using "
            "at most `max_workers` threads, keeps no more than `2 * max_workers` tasks in flight, and "
            "returns one result record per item **in input order**."
        ),
        signature="def parallel_map(func, items, max_workers: int = 4) -> list[dict]:",
        entry="parallel_map",
        inputs=["`func`: callable taking one item.", "`items`: any finite iterable (possibly a lazy generator).", "`max_workers`: `int` ≥ 1."],
        output="A list where element *i* is `{'ok': True, 'value': result}` or `{'ok': False, 'error': '<ExceptionType>: <message>'}` for item *i*.",
        constraints=[
            "At most `max_workers` calls run simultaneously.",
            "Pull items from `items` lazily: never have more than `2 * max_workers` submitted-but-unfinished tasks.",
            "An `Exception` raised by `func` is captured in that item's record; other items continue.",
            "Results are ordered like the input regardless of completion order.",
            "Raise `ValueError` if `max_workers < 1`; an empty input returns `[]`.",
        ],
        criteria=["Ordered results.", "Errors captured per item.", "Concurrency bounded.", "Back-pressure on the input.", "Real speed-up for I/O-bound work."],
        hints=[
            "`ThreadPoolExecutor(max_workers=...)` bounds concurrency; the remaining problem is bounding how many futures you create.",
            "Keep a dict `future → index`. Before submitting a new item, if the dict is full, block on "
            "`concurrent.futures.wait(pending, return_when=FIRST_COMPLETED)` and record the finished ones.",
            "with ThreadPoolExecutor(max_workers) as pool:\n    for i, item in enumerate(items):\n        if len(pending) >= 2 * max_workers: done, _ = wait(pending, return_when=FIRST_COMPLETED); collect(done)\n"
            "        pending[pool.submit(func, item)] = i\n    while pending: collect(wait(pending, return_when=FIRST_COMPLETED)[0])\nreturn [results[i] for i in range(len(results))]",
        ],
        solution='''
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def parallel_map(func, items, max_workers: int = 4) -> list[dict]:
    """Apply func concurrently with bounded in-flight work; results in input order."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    results: dict[int, dict] = {}
    pending: dict = {}

    def collect(done) -> None:
        for future in done:
            index = pending.pop(future)
            try:
                results[index] = {"ok": True, "value": future.result()}
            except Exception as exc:
                results[index] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for index, item in enumerate(items):
            if len(pending) >= 2 * max_workers:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)
            pending[pool.submit(func, item)] = index
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            collect(done)
    return [results[i] for i in range(len(results))]
''',
        explanation=(
            "The executor limits *running* calls; the `pending` dict limits *submitted* calls, which "
            "creates back-pressure on the input iterator so memory stays bounded. Indexing results by "
            "input position restores order cheaply. `future.result()` re-raises the worker's "
            "exception in the main thread, where it is converted into an error record. Threads suit "
            "I/O-bound work; CPU-bound pure-Python work would need processes."
        ),
        edge_cases=["Empty input.", "Every item failing.", "Lazy generator input.", "max_workers = 1 (serial)."],
        mistakes=["`list(pool.map(func, items))` – one exception aborts everything and the input is consumed eagerly.",
                  "Submitting every item up front (millions of futures in memory).", "Appending results in completion order."],
        tests=[
            CT(_q086_order_and_errors, note="input order; errors captured"),
            CT(_q086_concurrency_limit, note="never more than max_workers at once"),
            CT(_q086_speedup, note="runs I/O-bound work concurrently"),
            CT(_q086_bounded_submission, note="at most 2 * max_workers tasks in flight"),
            CT(_q086_edge, note="empty input; invalid max_workers"),
        ],
        examples=['''
>>> def risky(x):
...     if x == 2:
...         raise ValueError("bad item")
...     return x * x
>>> parallel_map(risky, [1, 2, 3], max_workers=2)
[{'ok': True, 'value': 1}, {'ok': False, 'error': 'ValueError: bad item'}, {'ok': True, 'value': 9}]
'''],
        complexity="O(n) scheduling work; O(max_workers) in-flight tasks plus O(n) results.",
        extension="Add per-item retries with exponential backoff for exceptions listed in a `retry_on` tuple.",
    ),
    # ------------------------------------------------------------------------ Q087
    Question(
        id=87,
        title="Cluster Near-Duplicate Documents (Shingles + Union-Find)",
        difficulty="Expert",
        category=ALG,
        roles=[AIE, DS_ROLE, DE],
        skills=["set similarity (Jaccard)", "inverted index for candidate generation", "union-find", "deterministic output"],
        minutes=60,
        scenario=(
            "Before building a RAG index you de-duplicate a scraped knowledge base: many pages are "
            "copies with tiny edits. Comparing every pair of documents is quadratic, so you only "
            "compare documents that share at least one word *shingle*, then group transitive matches."
        ),
        problem=(
            "Write `cluster_near_duplicates(docs, threshold, shingle_size=3)` that returns groups of "
            "document ids whose shingle-set Jaccard similarity links them (directly or transitively)."
        ),
        signature="def cluster_near_duplicates(docs: dict[str, str], threshold: float, shingle_size: int = 3) -> list[list[str]]:",
        entry="cluster_near_duplicates",
        inputs=["`docs`: doc id → text.", "`threshold`: float in (0, 1].", "`shingle_size`: `int` ≥ 1."],
        output="A list of clusters (each a sorted list of ≥ 2 ids), sorted by their first id.",
        constraints=[
            "Tokens: `re.findall(r'[a-z0-9]+', text.lower())`. Shingles: the set of tuples of `shingle_size` consecutive tokens; "
            "a document with 1..shingle_size−1 tokens has the single shingle of all its tokens; no tokens → no shingles.",
            "Jaccard(A, B) = |A ∩ B| / |A ∪ B|. Documents are linked when Jaccard ≥ `threshold`. Documents without shingles are never linked.",
            "Clusters are connected components of the link graph (transitive).",
            "Must handle ~1,500 documents in well under 3 seconds (avoid comparing every pair).",
            "Raise `ValueError` for a threshold outside (0, 1] or `shingle_size < 1`.",
        ],
        criteria=["Correct similarity and threshold.", "Transitive clustering.", "Deterministic output format.", "Sub-quadratic candidate generation."],
        hints=[
            "Two documents with no shingle in common have Jaccard 0 – never compare them.",
            "Build an inverted index `shingle → set(doc ids)`, generate candidate pairs from each posting list, compute Jaccard only "
            "for those pairs, and merge linked docs with a union-find (disjoint set) structure.",
            "shingles = {doc: make_shingles(text)}\nindex: shingle → [docs]\ncandidates = {(a, b) for docs in index.values() for a, b in combinations(sorted(docs), 2)}\n"
            "for a, b in candidates: if jaccard >= t: union(a, b)\ngroup ids by find(id); keep groups of size >= 2; sort",
        ],
        solution='''
import re
from itertools import combinations


def _shingles(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return set()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def cluster_near_duplicates(docs: dict[str, str], threshold: float, shingle_size: int = 3) -> list[list[str]]:
    """Group documents whose shingle Jaccard similarity >= threshold (transitively)."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if shingle_size < 1:
        raise ValueError("shingle_size must be >= 1")
    shingles = {doc_id: _shingles(text, shingle_size) for doc_id, text in docs.items()}

    index: dict[tuple, list[str]] = {}
    for doc_id, doc_shingles in shingles.items():
        for shingle in doc_shingles:
            index.setdefault(shingle, []).append(doc_id)
    candidates = set()
    for posting in index.values():
        candidates.update(combinations(sorted(posting), 2))

    parent = {doc_id: doc_id for doc_id in docs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for a, b in candidates:
        sa, sb = shingles[a], shingles[b]
        if len(sa & sb) / len(sa | sb) >= threshold:
            parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for doc_id in docs:
        groups.setdefault(find(doc_id), []).append(doc_id)
    clusters = [sorted(members) for members in groups.values() if len(members) >= 2]
    return sorted(clusters, key=lambda members: members[0])
''',
        explanation=(
            "The inverted index restricts comparisons to pairs sharing at least one shingle, which "
            "for natural text is a tiny fraction of all pairs. Union-find merges linked documents in "
            "near-constant amortized time and naturally yields transitive clusters. Sorting members "
            "and clusters makes the output independent of dict and set ordering."
        ),
        edge_cases=["Identical texts after normalization (J = 1).", "Very short documents.", "Empty documents never cluster.", "Transitive chains A~B~C with A≁C."],
        mistakes=["Comparing all n² pairs.", "Clustering only direct matches (missing transitivity).",
                  "Using raw `str.split()` so punctuation/case differences break matches."],
        tests=[
            T(({"a": "The quick brown fox jumps over the lazy dog", "b": "the quick brown fox jumps over the lazy dog!",
                "c": "The quick brown fox jumped over the lazy dog", "d": "Completely different text about model deployment pipelines",
                "e": "completely different text about model deployment pipelines today", "f": ""}, 0.8),
              [["a", "b"], ["d", "e"]], example=True, note="two duplicate groups"),
            T(({"a": "The quick brown fox jumps over the lazy dog", "b": "the quick brown fox jumps over the lazy dog!",
                "c": "The quick brown fox jumped over the lazy dog"}, 0.4), [["a", "b", "c"]], example=True,
              note="boundary: J(a, c) = 0.4 exactly"),
            T(({"x": "a b c d e f", "y": "a b c d e f g h", "z": "c d e f g h i j"}, 0.5), [["x", "y", "z"]],
              note="transitive: x~y (0.67), y~z (0.5), x≁z (0.25)"),
            T(({"p": "hi there", "q": "Hi, there!", "r": "bye"}, 1.0), [["p", "q"]], note="short documents"),
            T(({"only": "one document"}, 0.5), [], note="nothing to cluster"),
            T(({"a": "x"}, 0), raises="ValueError", note="threshold 0"),
            T(({"a": "x"}, 1.5), raises="ValueError", note="threshold > 1"),
            CT(_q087_scale, note="1,550 documents with 50 planted duplicates"),
        ],
        complexity="O(total shingles + candidate pairs × shingle-set size); far below O(n²) for natural text. Union-find is ~O(α(n)) per operation.",
        extension="Replace exact Jaccard with MinHash signatures + locality-sensitive hashing for millions of documents.",
    ),
    # ------------------------------------------------------------------------ Q088
    Question(
        id=88,
        title="Incremental Log Tailing with a Durable Checkpoint",
        difficulty="Expert",
        category=FILE,
        roles=[DE, PYDEV, AIE],
        skills=["binary file offsets", "partial-line handling", "atomic checkpoint writes", "rotation detection", "UTF-8"],
        minutes=60,
        scenario=(
            "A collector ships new lines from an inference server's log every minute. After a crash "
            "it must resume exactly where it stopped – no duplicates, no gaps – and the server may "
            "be in the middle of writing a line. Log rotation truncates the file."
        ),
        problem=(
            "Write `process_new_lines(log_path, state_path)` that returns the complete lines appended "
            "since the last call and records the new position in a JSON checkpoint file."
        ),
        signature="def process_new_lines(log_path: str, state_path: str) -> list[str]:",
        entry="process_new_lines",
        inputs=["`log_path`: UTF-8 log file (must exist).", "`state_path`: JSON checkpoint `{'offset': int}`; may not exist yet (parent folders may be missing)."],
        output="A list of new complete lines (without `\\n`/`\\r\\n`), oldest first.",
        constraints=[
            "Offsets are **byte** offsets. Read the log in binary mode starting at the saved offset (0 if no state file).",
            "Only lines terminated by `\\n` are returned; a trailing partial line is left for the next call.",
            "If the log is smaller than the saved offset, it was rotated/truncated: start again from 0.",
            "Write the new state as `{'offset': <int>}` atomically (temp file in the same folder + `os.replace`).",
            "Raise `ValueError` for a corrupt state file (invalid JSON, not an object, or `offset` not a non-negative int); "
            "`FileNotFoundError` if the log does not exist.",
        ],
        criteria=["Exactly-once line delivery across calls.", "Partial lines deferred.", "Rotation handled.", "Atomic, validated checkpoint."],
        hints=[
            "Text-mode `tell()` values are opaque; work with bytes and decode each complete line yourself.",
            "Read from the offset to the end, find the last `b'\\n'` with `rfind`, and only consume up to there. "
            "The new offset is `old_offset + index_of_last_newline + 1`.",
            "offset = load_state()\nsize = os.path.getsize(log); if size < offset: offset = 0\nwith open(log, 'rb') as f: f.seek(offset); data = f.read()\n"
            "end = data.rfind(b'\\n') + 1\nlines = [l.decode('utf-8').rstrip('\\r') for l in data[:end].split(b'\\n')[:-1]]\nsave_state(offset + end)  # atomic",
        ],
        solution='''
import json
import os
import tempfile
from pathlib import Path


def _load_offset(state_path: Path) -> int:
    if not state_path.exists():
        return 0
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt state file {state_path}") from exc
    offset = state.get("offset") if isinstance(state, dict) else None
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError(f"invalid offset in state file {state_path}: {offset!r}")
    return offset


def _save_offset(state_path: Path, offset: int) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=state_path.parent, prefix=f".{state_path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"offset": offset}, handle)
        os.replace(tmp, state_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def process_new_lines(log_path: str, state_path: str) -> list[str]:
    """Return complete lines appended since the last checkpoint and advance it."""
    log, state = Path(log_path), Path(state_path)
    offset = _load_offset(state)
    size = log.stat().st_size  # raises FileNotFoundError for a missing log
    if size < offset:
        offset = 0  # rotated or truncated
    with open(log, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    end = data.rfind(b"\\n") + 1
    lines = [raw.decode("utf-8").rstrip("\\r") for raw in data[:end].split(b"\\n")[:-1]]
    _save_offset(state, offset + end)
    return lines
''',
        explanation=(
            "Binary mode makes offsets plain byte counts, which survive restarts and multi-byte "
            "characters. Consuming only up to the last newline guarantees a half-written line is "
            "delivered exactly once, later, in full. Saving the checkpoint with a temp file and "
            "`os.replace` means a crash leaves either the old or the new checkpoint – never a "
            "corrupt one. (If the downstream processing can fail, save the checkpoint only after "
            "it succeeds – that is the at-least-once vs at-most-once trade-off.)"
        ),
        edge_cases=["No new data → `[]` and unchanged offset.", "Partial line completed later.", "Truncated/rotated log.", "Multi-byte UTF-8."],
        mistakes=["Text-mode `seek(offset)` with character counts.", "Returning the partial last line (and then losing its remainder).", "Writing the checkpoint in place (corrupt on crash)."],
        tests=[
            T((File("app.log", "line1\nline2\npart"), TmpPath("checkpoints/app.json")), ["line1", "line2"], example=True,
              note="first call: complete lines only"),
            T((File("empty.log", ""), TmpPath("state.json")), [], example=True, note="boundary: empty log"),
            CT(_q088_sequence, note="incremental calls, partial lines, UTF-8, rotation"),
            CT(_q088_state_file, note="state content and atomic write"),
            CT(_q088_corrupt_state, note="corrupt checkpoints rejected"),
            T((TmpPath("missing.log"), TmpPath("state.json")), raises="FileNotFoundError", note="missing log"),
        ],
        complexity="O(new bytes) time and memory per call (read in fixed-size blocks to bound memory further).",
        extension="Detect rotation even when the new file is *larger* than the old offset by also storing the file's inode (`os.stat().st_ino`).",
    ),
    # ------------------------------------------------------------------------ Q089
    Question(
        id=89,
        title="Validated Attributes with a Reusable Descriptor",
        difficulty="Expert",
        category=OOP,
        roles=[PYDEV, MLE, AIE],
        skills=["descriptor protocol (__get__/__set__/__set_name__)", "per-instance storage", "validation", "error messages"],
        minutes=50,
        scenario=(
            "Every config class in your ML platform repeats the same property boilerplate to "
            "validate types and ranges. Libraries such as dataclasses, attrs and Django fields solve "
            "this with *descriptors*: one reusable object that controls attribute access for many classes."
        ),
        problem=(
            "Implement a descriptor class `Typed(expected_type, *, min_value=None, max_value=None, "
            "nullable=False)` that validates every assignment to the attribute it manages."
        ),
        signature="class Typed:",
        entry="Typed",
        starter='''
class Typed:
    """Data descriptor that validates type and range on assignment."""

    def __init__(self, expected_type, *, min_value=None, max_value=None, nullable=False):
        # TODO
        pass

    def __set_name__(self, owner, name):
        # TODO
        pass

    def __get__(self, instance, owner=None):
        # TODO
        pass

    def __set__(self, instance, value):
        # TODO
        pass
''',
        inputs=[
            "`expected_type`: a class such as `int`, `float` or `str`.",
            "`min_value`/`max_value`: optional inclusive bounds. `nullable`: whether `None` is allowed.",
        ],
        output="A descriptor; classes declare `field = Typed(...)` as class attributes.",
        constraints=[
            "Values are stored **per instance** (e.g. in `instance.__dict__` under the attribute name from `__set_name__`).",
            "Wrong type → `TypeError`; out of range → `ValueError`; both messages include the attribute name.",
            "`bool` is rejected for `int`/`float`; `int` is accepted for `float` (stored unchanged).",
            "`None` is allowed only when `nullable=True` (no range check then).",
            "Accessing the attribute on the class returns the descriptor; reading an unset attribute on an instance raises `AttributeError`.",
            "A rejected assignment leaves the previous value unchanged.",
        ],
        criteria=["Per-instance storage.", "Type and range validation.", "Correct descriptor protocol details.", "Helpful messages."],
        hints=[
            "A *data* descriptor (one that defines `__set__`) takes priority over the instance `__dict__`, so you can store the value there under the same name.",
            "`__set_name__(self, owner, name)` tells the descriptor its attribute name. In `__get__`, return `self` when `instance is None`.",
            "__get__: if instance is None: return self; try: return instance.__dict__[self.name] except KeyError: raise AttributeError(self.name)\n"
            "__set__: if value is None: if not nullable: raise TypeError; else: type check (bool special case); range check\n"
            "         instance.__dict__[self.name] = value",
        ],
        solution='''
class Typed:
    """Data descriptor that validates type and range on assignment."""

    def __init__(self, expected_type, *, min_value=None, max_value=None, nullable=False):
        self.expected_type = expected_type
        self.min_value = min_value
        self.max_value = max_value
        self.nullable = nullable
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        try:
            return instance.__dict__[self.name]
        except KeyError:
            raise AttributeError(f"{self.name!r} has not been set") from None

    def _type_ok(self, value) -> bool:
        if isinstance(value, bool) and self.expected_type in (int, float):
            return False
        if self.expected_type is float:
            return isinstance(value, (int, float))
        return isinstance(value, self.expected_type)

    def __set__(self, instance, value):
        if value is None:
            if not self.nullable:
                raise TypeError(f"{self.name} must not be None")
        else:
            if not self._type_ok(value):
                raise TypeError(f"{self.name} must be {self.expected_type.__name__}, got {type(value).__name__}")
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"{self.name} must be >= {self.min_value}, got {value!r}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"{self.name} must be <= {self.max_value}, got {value!r}")
        instance.__dict__[self.name] = value
''',
        explanation=(
            "The descriptor object is shared by every instance of the class, so it must never hold "
            "instance values itself. Because `Typed` defines `__set__`, attribute lookup consults it "
            "before the instance dict, letting it store values in `instance.__dict__` under the same "
            "name. `__set_name__` (Python 3.6+) removes the need to repeat the name. Validation runs "
            "before storing, so failed assignments change nothing."
        ),
        edge_cases=["Two instances with different values.", "Class-level access.", "Unset attribute.", "`True` for an `int` field."],
        mistakes=["Storing `self.value = value` on the descriptor (shared by all instances).", "Returning `None` for unset attributes.",
                  "Forgetting `instance is None` in `__get__` (class access breaks)."],
        tests=[
            CT(_q089_basic, note="per-instance values; class access"),
            CT(_q089_type_errors, note="TypeError with attribute name; failed assignments change nothing"),
            CT(_q089_ranges, note="inclusive range checks"),
            CT(_q089_unset, note="unset attribute → AttributeError"),
        ],
        examples=['''
>>> class TrainingConfig:
...     learning_rate = Typed(float, min_value=0.0, max_value=1.0)
...     epochs = Typed(int, min_value=1)
>>> cfg = TrainingConfig()
>>> cfg.learning_rate = 0.01
>>> cfg.epochs = 0
Traceback (most recent call last):
    ...
ValueError: epochs must be >= 1, got 0
>>> cfg.epochs = "3"
Traceback (most recent call last):
    ...
TypeError: epochs must be int, got str
'''],
        complexity="O(1) per attribute access.",
        extension="Add a class decorator that auto-generates `__init__` from all `Typed` attributes (like a mini dataclass).",
    ),
    # ------------------------------------------------------------------------ Q090
    Question(
        id=90,
        title="Debug the 'Latest Record per Key' Function",
        difficulty="Expert",
        category=DBG,
        roles=[DE, PYDEV, DS_ROLE],
        skills=["timezone-aware datetimes", "string vs instant comparison", "hidden mutation", "tie semantics", "regression tests"],
        minutes=45,
        scenario=(
            "A CDC (change-data-capture) job keeps the newest version of each product:\n\n"
            "```python\ndef latest_by_key(records, key=\"id\"):\n    latest = {}\n    for r in records:\n"
            "        k = r[key]\n        if k not in latest or r[\"updated_at\"] > latest[k][\"updated_at\"]:\n"
            "            latest[k] = r\n            r[\"is_latest\"] = True\n    return list(latest.values())\n```\n\n"
            "Reported problems: prices from European warehouses (`+02:00` timestamps) win even when they are older; "
            "duplicate events with the same instant keep the *first* copy although the business rule is "
            "'last write wins'; callers find an unexpected `is_latest` field in their data; records without "
            "`updated_at` crash the job; and the output order changes between runs of the upstream system."
        ),
        problem=(
            "Write a corrected `latest_by_key(records, key='id')` that returns the newest record per key "
            "according to the rules below."
        ),
        signature="def latest_by_key(records: list[dict], key: str = \"id\") -> list[dict]:",
        entry="latest_by_key",
        inputs=["`records`: dicts containing `key` and usually `updated_at`, an ISO-8601 timestamp with a UTC offset (`Z` or `±HH:MM`)."],
        output="A list with one record per key, sorted by key.",
        constraints=[
            "Compare actual instants: parse timestamps as timezone-aware datetimes (`Z` means UTC).",
            "Timestamps without an offset, or unparseable ones, raise `ValueError`.",
            "Records with a missing or `None` `updated_at` are skipped (a key with no valid records does not appear).",
            "Equal instants: the record appearing **later** in the input wins.",
            "Never modify the input records (return them unchanged; returning the same dict objects is fine).",
        ],
        criteria=["Timezone-correct comparison.", "Last-write-wins ties.", "No mutation.", "Missing timestamps skipped.", "Deterministic order."],
        hints=[
            "List each reported symptom and find the line responsible – there are five separate bugs.",
            "`datetime.fromisoformat(ts.replace('Z', '+00:00'))` parses both forms (Python ≥ 3.7); check `dt.tzinfo is None` "
            "to reject naive timestamps. Use `>=` for 'later wins' and remove the mutation.",
            "best = {}\nfor r in records:\n    ts = r.get('updated_at'); if ts is None: continue\n    instant = parse(ts)  # ValueError if naive\n"
            "    k = r[key]\n    if k not in best or instant >= best[k][0]: best[k] = (instant, r)\nreturn [best[k][1] for k in sorted(best)]",
        ],
        solution='''
from datetime import datetime


def _parse_instant(text: str) -> datetime:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError(f"timestamp {text!r} has no UTC offset")
    return moment


def latest_by_key(records: list[dict], key: str = "id") -> list[dict]:
    """Newest record per key by actual instant; later input wins ties."""
    best: dict = {}
    for record in records:
        stamp = record.get("updated_at")
        if stamp is None:
            continue
        instant = _parse_instant(stamp)
        record_key = record[key]
        if record_key not in best or instant >= best[record_key][0]:
            best[record_key] = (instant, record)
    return [best[k][1] for k in sorted(best)]
''',
        explanation=(
            "ISO strings only sort chronologically when they share the same offset format, so the "
            "fix compares aware `datetime` objects (which compare by instant). `>=` implements "
            "last-write-wins; storing `(instant, record)` avoids re-parsing; dropping the "
            "`is_latest` assignment removes the side effect; sorting keys makes the output deterministic."
        ),
        edge_cases=["`Z` vs `+00:00` for the same instant.", "Negative offsets crossing midnight.", "Keys with only invalid records.", "Naive timestamps."],
        mistakes=["Fixing only the comparison operator.", "Stripping offsets instead of converting to UTC.", "`datetime.fromisoformat` with `Z` on Python < 3.11 (needs the replace)."],
        tests=[
            T((LATEST_RECORDS,),
              [{"id": "sku-0", "price": 7, "updated_at": "2024-04-30T23:59:59-05:00"},
               {"id": "sku-1", "price": 10, "updated_at": "2024-05-01T09:00:00Z"},
               {"id": "sku-2", "price": 6, "updated_at": "2024-05-01T08:00:00Z"}], example=True,
              note="offsets, tie → later wins, missing timestamp; input not modified"),
            T(([],), [], example=True, note="boundary: no records"),
            T(([{"sku": "b", "updated_at": "2024-01-01T00:00:00Z"}, {"sku": "a", "updated_at": "2024-01-01T00:00:00Z"}], "sku"),
              [{"sku": "a", "updated_at": "2024-01-01T00:00:00Z"}, {"sku": "b", "updated_at": "2024-01-01T00:00:00Z"}],
              note="custom key, sorted output"),
            T(([{"id": "x", "updated_at": "2024-05-01T09:00:00"}],), raises="ValueError", note="naive timestamp"),
            T(([{"id": "x", "updated_at": "yesterday"}],), raises="ValueError", note="unparseable timestamp"),
            T(([{"id": "x", "updated_at": None}, {"id": "x"}],), [], note="no valid timestamps"),
        ],
        complexity="O(n + k log k) for n records and k keys.",
        extension="Write a pytest regression test for each of the five original bugs.",
    ),
    # ------------------------------------------------------------------------ Q091
    Question(
        id=91,
        title="Point-in-Time Correct Feature Join (As-Of Join)",
        difficulty="Expert",
        category=DATA,
        roles=[MLE, DS_ROLE, DE],
        skills=["avoiding label leakage", "per-key sorting", "bisect", "staleness limits", "tie semantics"],
        minutes=50,
        scenario=(
            "Your churn model looked great offline and failed in production. The cause: training rows "
            "were joined with features computed *after* the label event (data leakage). The fix is an "
            "as-of join: each label gets the latest feature values known at the label's timestamp."
        ),
        problem=(
            "Write `asof_join(labels, features, max_staleness=None)` that attaches to each label the "
            "most recent feature row for the same entity with `feature_ts <= label_ts`."
        ),
        signature="def asof_join(labels: list[dict], features: list[dict], max_staleness: int | None = None) -> list[dict]:",
        entry="asof_join",
        inputs=[
            "`labels`: dicts with `entity_id`, `ts` (int seconds) and other fields.",
            "`features`: dicts with `entity_id`, `ts` and `features` (a dict). Unsorted.",
            "`max_staleness`: optional int ≥ 0.",
        ],
        output="For each label (input order): a copy of the label plus `features` (a copy of the matched dict, or `None`) and `feature_ts` (int or `None`).",
        constraints=[
            "Only feature rows with `ts <= label ts` may be used (equal timestamps allowed).",
            "If several feature rows share the winning timestamp, the one appearing later in `features` wins.",
            "With `max_staleness`, a match older than `label_ts - max_staleness` is discarded (`None`).",
            "Must handle 50,000 labels × 50,000 feature rows in about a second or two – sort once, then binary search.",
            "Do not modify the inputs. Raise `ValueError` for negative `max_staleness`.",
        ],
        criteria=["No future leakage.", "Correct tie-breaking.", "Staleness rule.", "Efficient."],
        hints=[
            "Group features by entity and sort each group by timestamp once.",
            "Use a *stable* sort so later duplicates stay later, then `bisect_right(timestamps, label_ts) - 1` finds the last row with `ts <= label_ts`.",
            "by_entity = {e: sorted(rows, key=ts)} (stable); ts_lists = {e: [r['ts'] ...]}\nfor label: i = bisect_right(ts_lists[e], label['ts']) - 1\n"
            "    match = rows[i] if i >= 0 else None; apply staleness; out.append({**label, 'features': dict(...) or None, 'feature_ts': ...})",
        ],
        solution='''
from bisect import bisect_right


def asof_join(labels: list[dict], features: list[dict], max_staleness: int | None = None) -> list[dict]:
    """Attach the latest feature row with ts <= label ts (no future leakage)."""
    if max_staleness is not None and max_staleness < 0:
        raise ValueError("max_staleness must be >= 0")
    by_entity: dict[str, list[dict]] = {}
    for row in features:
        by_entity.setdefault(row["entity_id"], []).append(row)
    timestamps = {}
    for entity, rows in by_entity.items():
        rows.sort(key=lambda r: r["ts"])  # stable: later duplicates stay later
        timestamps[entity] = [r["ts"] for r in rows]

    joined = []
    for label in labels:
        rows = by_entity.get(label["entity_id"], [])
        index = bisect_right(timestamps.get(label["entity_id"], []), label["ts"]) - 1
        match = rows[index] if index >= 0 else None
        if match is not None and max_staleness is not None and label["ts"] - match["ts"] > max_staleness:
            match = None
        joined.append({
            **label,
            "features": dict(match["features"]) if match else None,
            "feature_ts": match["ts"] if match else None,
        })
    return joined
''',
        explanation=(
            "Sorting each entity's rows once costs O(m log m); each label then needs one O(log m) "
            "binary search instead of a scan. `bisect_right` lands *after* all rows with equal "
            "timestamps, so `index - 1` is the last of them – and because Python's sort is stable, "
            "that is the one that appeared last in the input. Grouping copies references only, so "
            "the caller's list order is untouched."
        ),
        edge_cases=["Label before any feature → None.", "Exact timestamp match.", "Unknown entity.", "Duplicate feature timestamps."],
        mistakes=["Using `<` (excludes features computed exactly at label time) or nearest-in-either-direction (leakage).",
                  "Sorting the caller's `features` list in place.", "Scanning all features per label."],
        tests=[
            T((ASOF_LABELS, ASOF_FEATURES),
              [{"entity_id": "u1", "ts": 250, "label": 1, "features": {"spend_7d": 30}, "feature_ts": 200},
               {"entity_id": "u1", "ts": 99, "label": 0, "features": None, "feature_ts": None},
               {"entity_id": "u2", "ts": 150, "label": 0, "features": {"spend_7d": 3}, "feature_ts": 150},
               {"entity_id": "u3", "ts": 500, "label": 1, "features": None, "feature_ts": None},
               {"entity_id": "u1", "ts": 300, "label": 1, "features": {"spend_7d": 50}, "feature_ts": 300}],
              example=True, note="ties, exact matches, no history, unknown entity"),
            T(([{"entity_id": "u1", "ts": 290, "label": 1}, {"entity_id": "u1", "ts": 250, "label": 0}], ASOF_FEATURES, 60),
              [{"entity_id": "u1", "ts": 290, "label": 1, "features": None, "feature_ts": None},
               {"entity_id": "u1", "ts": 250, "label": 0, "features": {"spend_7d": 30}, "feature_ts": 200}],
              example=True, note="boundary: staleness 90 s > 60 s discarded, 50 s kept"),
            T(([], ASOF_FEATURES), [], note="no labels"),
            T(([{"entity_id": "u1", "ts": 1, "label": 0}], []), [{"entity_id": "u1", "ts": 1, "label": 0, "features": None, "feature_ts": None}],
              note="no features"),
            T((ASOF_LABELS, ASOF_FEATURES, -1), raises="ValueError", note="negative staleness"),
            CT(_q091_fast, note="50,000 x 50,000 rows"),
        ],
        complexity="O(m log m + n log m) time for m feature rows and n labels; O(m + n) space.",
        extension="Support several feature tables, each with its own staleness limit, in one pass.",
    ),
    # ------------------------------------------------------------------------ Q092
    Question(
        id=92,
        title="Slowly Changing Dimension (Type 2) Merge",
        difficulty="Expert",
        category=DATA,
        roles=[DE, DS_ROLE, PYDEV],
        skills=["history tables", "immutable updates", "date validation", "data-integrity checks", "deterministic ordering"],
        minutes=55,
        scenario=(
            "Customer attributes (tier, city) change over time, and analysts need to know what was "
            "true on any past date – for example to reconstruct training data. The warehouse stores a "
            "*Type 2 slowly changing dimension*: every change closes the current row and adds a new one."
        ),
        problem=(
            "Write `scd2_merge(current, updates, as_of)` that applies a snapshot of customer attributes "
            "taken on date `as_of` to the history table and returns the new table."
        ),
        signature="def scd2_merge(current: list[dict], updates: list[dict], as_of: str) -> list[dict]:",
        entry="scd2_merge",
        inputs=[
            "`current`: rows `{'customer_id', 'attributes': dict, 'valid_from': 'YYYY-MM-DD', 'valid_to': date or None, 'is_current': bool}`.",
            "`updates`: `{'customer_id', 'attributes': dict}` – the latest known attributes.",
            "`as_of`: `'YYYY-MM-DD'` date of the snapshot.",
        ],
        output="A new list of rows sorted by `(customer_id, valid_from)`.",
        constraints=[
            "New customer → add a row `valid_from=as_of, valid_to=None, is_current=True`.",
            "Changed attributes → close the current row (`valid_to=as_of, is_current=False`) and add a new current row.",
            "Unchanged attributes, and customers absent from `updates`, keep their rows untouched (no deletes).",
            "Raise `ValueError` for: an invalid `as_of` date; duplicate customer ids in `updates`; a customer with more than "
            "one current row; or `as_of` earlier than that customer's current `valid_from` (late data is not supported).",
            "Do not modify the inputs; every returned row is a new dict (attributes copied).",
        ],
        criteria=["Correct open/close of history rows.", "Unchanged data untouched.", "Integrity checks.", "Pure function."],
        hints=[
            "Index the current rows: `customer_id → current row`. Validate everything before changing anything.",
            "Copy every row first (`{**row, 'attributes': dict(row['attributes'])}`), then mutate only your copies. `date.fromisoformat` validates the date.",
            "rows = deep-ish copies; current_by_id = {...} (check duplicates)\nfor u in updates: row = current_by_id.get(id)\n"
            "    if row is None: append new\n    elif row['attributes'] != u['attributes']: check as_of >= valid_from; close row; append new\nreturn sorted(rows, key=...)",
        ],
        solution='''
from datetime import date


def scd2_merge(current: list[dict], updates: list[dict], as_of: str) -> list[dict]:
    """Apply an attribute snapshot to a Type 2 SCD history table."""
    try:
        snapshot_day = date.fromisoformat(as_of)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid as_of date {as_of!r}") from exc

    rows = [{**row, "attributes": dict(row["attributes"])} for row in current]
    open_rows: dict[str, dict] = {}
    for row in rows:
        if row["is_current"]:
            if row["customer_id"] in open_rows:
                raise ValueError(f"customer {row['customer_id']!r} has more than one current row")
            open_rows[row["customer_id"]] = row

    seen: set[str] = set()
    for update in updates:
        customer = update["customer_id"]
        if customer in seen:
            raise ValueError(f"duplicate update for customer {customer!r}")
        seen.add(customer)
        existing = open_rows.get(customer)
        if existing is not None and existing["attributes"] == update["attributes"]:
            continue
        if existing is not None:
            if snapshot_day < date.fromisoformat(existing["valid_from"]):
                raise ValueError(f"as_of {as_of} is before the current row of {customer!r}")
            existing["valid_to"] = as_of
            existing["is_current"] = False
        rows.append({"customer_id": customer, "attributes": dict(update["attributes"]),
                     "valid_from": as_of, "valid_to": None, "is_current": True})
    return sorted(rows, key=lambda row: (row["customer_id"], row["valid_from"]))
''',
        explanation=(
            "Working on copies keeps the function pure, which makes merges easy to test and retry. "
            "Indexing current rows turns each update into an O(1) lookup; integrity problems are "
            "detected before they can corrupt history. Closing the old row with `valid_to = as_of` "
            "and opening the new one with `valid_from = as_of` gives contiguous, non-overlapping intervals."
        ),
        edge_cases=["New customer.", "Unchanged attributes.", "Absent customers.", "Change on the same day as valid_from."],
        mistakes=["Mutating the caller's rows (`row['is_current'] = False`).", "Comparing date strings without validating them.",
                  "Deleting customers missing from the snapshot."],
        tests=[
            T((SCD_CURRENT, SCD_UPDATES, "2024-06-01"),
              [{"customer_id": "c1", "attributes": {"tier": "free", "city": "Pune"}, "valid_from": "2024-01-01", "valid_to": "2024-06-01", "is_current": False},
               {"customer_id": "c1", "attributes": {"tier": "pro", "city": "Pune"}, "valid_from": "2024-06-01", "valid_to": None, "is_current": True},
               {"customer_id": "c2", "attributes": {"tier": "free"}, "valid_from": "2024-01-01", "valid_to": "2024-03-01", "is_current": False},
               {"customer_id": "c2", "attributes": {"tier": "pro"}, "valid_from": "2024-03-01", "valid_to": None, "is_current": True},
               {"customer_id": "c3", "attributes": {"tier": "pro"}, "valid_from": "2024-02-01", "valid_to": None, "is_current": True},
               {"customer_id": "c4", "attributes": {"tier": "free"}, "valid_from": "2024-06-01", "valid_to": None, "is_current": True}],
              example=True, note="changed, unchanged, absent and new customers"),
            T(([], [], "2024-06-01"), [], example=True, note="boundary: empty table and snapshot"),
            T((SCD_CURRENT, [SCD_UPDATES[1], SCD_UPDATES[1]], "2024-06-01"), raises="ValueError", note="duplicate updates"),
            T((SCD_CURRENT, [SCD_UPDATES[1]], "2023-12-31"), raises="ValueError", note="as_of before current valid_from"),
            T((SCD_CURRENT, [], "2024-13-01"), raises="ValueError", note="invalid date"),
            T((SCD_CURRENT + [{**SCD_CURRENT[0], "valid_from": "2024-02-01"}], [], "2024-06-01"), raises="ValueError",
              note="two current rows"),
        ],
        complexity="O((n + u) log(n + u)) time for sorting; O(n + u) space.",
        extension="Support late-arriving updates by splitting the historical row that covers `as_of`.",
    ),
    # ------------------------------------------------------------------------ Q093
    Question(
        id=93,
        title="K-Way Merge of Large Sorted Log Files",
        difficulty="Expert",
        category=PERF,
        roles=[DE, PYDEV, MLE],
        skills=["heapq.merge", "streaming I/O", "input validation while streaming", "atomic output", "memory profiling"],
        minutes=50,
        scenario=(
            "Each inference node writes its own time-ordered log. To replay traffic you need one "
            "chronological file – but each node log is several gigabytes, so loading and sorting "
            "everything is impossible."
        ),
        problem=(
            "Write `merge_sorted_files(paths, out_path)` that merges the files by the timestamp at the "
            "start of each line and writes the result to `out_path`, using memory independent of file size."
        ),
        signature="def merge_sorted_files(paths: list[str], out_path: str) -> int:",
        entry="merge_sorted_files",
        inputs=["`paths`: UTF-8 files whose non-blank lines start with a sortable timestamp token (text before the first space).", "`out_path`: destination path (parent folders may be missing)."],
        output="Returns the number of lines written. Each output line ends with `\\n`.",
        constraints=[
            "Order by the first whitespace-separated token; equal tokens keep the order of `paths`, then file order.",
            "Skip blank lines. Strip trailing newline characters before writing.",
            "Raise `ValueError` if any file is not sorted by its timestamps (detect it while streaming).",
            "On any failure, `out_path` must not be created or changed and no temporary files remain (write to a temp file, then `os.replace`).",
            "Peak memory must stay small even for very large inputs (stream; never read whole files).",
        ],
        criteria=["Correct stable merge.", "Unsorted input detected.", "Atomic output.", "Constant memory per file."],
        hints=[
            "`heapq.merge(*iterables, key=...)` lazily merges already-sorted iterables and is stable across iterables.",
            "Wrap each file in a generator that strips/skips lines and raises `ValueError` if a key decreases. Write to "
            "`tempfile.mkstemp(dir=out_dir)` and `os.replace` at the end; delete the temp file in an `except` block.",
            "def lines(path): prev = None; for raw in open(path): line = raw.rstrip('\\r\\n'); if not line.strip(): continue; key = line.split(maxsplit=1)[0]; check; yield line\n"
            "with ExitStack/open files: for line in heapq.merge(*(lines(p) for p in paths), key=first_token): out.write(line + '\\n'); n += 1\nos.replace(tmp, out_path)",
        ],
        solution='''
import heapq
import os
import tempfile
from pathlib import Path


def _first_token(line: str) -> str:
    return line.split(maxsplit=1)[0]


def _sorted_lines(path: str):
    previous = None
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\\r\\n")
            if not line.strip():
                continue
            key = _first_token(line)
            if previous is not None and key < previous:
                raise ValueError(f"{path} is not sorted at line {number}: {key!r} < {previous!r}")
            previous = key
            yield line


def merge_sorted_files(paths: list[str], out_path: str) -> int:
    """Stream-merge timestamp-sorted files into out_path atomically; return lines written."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    written = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for line in heapq.merge(*(_sorted_lines(p) for p in paths), key=_first_token):
                out.write(line + "\\n")
                written += 1
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return written
''',
        explanation=(
            "`heapq.merge` keeps only the current head line of each file in a heap, so memory is "
            "O(k) lines for k files regardless of their size, and it breaks ties by iterable order "
            "(stable). Validating sortedness inside each generator catches bad input as it streams. "
            "The temp-file + `os.replace` pattern guarantees readers never see a partial merge."
        ),
        edge_cases=["Empty files.", "Blank lines.", "Equal timestamps across files.", "Unsorted file discovered mid-way."],
        mistakes=["`sorted(all_lines)` (O(total) memory).", "Writing directly to `out_path` (partial file on failure).", "Comparing whole lines instead of the timestamp token (ties ordered by message text)."],
        tests=[
            CT(_q093_merge, note="stable 3-way merge with blank lines and an empty file"),
            CT(_q093_unsorted, note="unsorted input: ValueError, no output, no temp files"),
            CT(_q093_memory, note="180,000 lines with < 3 MB peak memory"),
            T(([File("a.log", "2024-05-01T10:00:00Z a\n"), File("b.log", "2024-05-01T09:00:00Z b\n")], TmpPath("out/merged.log")), 2,
              example=True, note="returns the number of lines written"),
            T(([], TmpPath("empty.log")), 0, example=True, note="boundary: no input files"),
        ],
        complexity="O(N log k) time for N lines across k files; O(k) memory.",
        extension="Parse the timestamps to datetimes so files with different offset formats can be merged correctly.",
    ),
    # ------------------------------------------------------------------------ Q094
    Question(
        id=94,
        title="Reservoir Sampling from an Unbounded Stream",
        difficulty="Expert",
        category=PERF,
        roles=[DS_ROLE, MLE, DE],
        skills=["Algorithm R", "uniform sampling proofs", "seeded randomness", "single-pass algorithms", "O(k) memory"],
        minutes=45,
        scenario=(
            "You want 1,000 random production prompts per day for human evaluation, but the prompts "
            "arrive as a stream of unknown length and storing them all is not allowed. Every prompt "
            "must have the same probability of being selected, and the sample must be reproducible "
            "from a seed for audits."
        ),
        problem=(
            "Write `reservoir_sample(iterable, k, seed=None)` that returns a uniform random sample of "
            "`k` items from a single pass over the stream, using O(k) memory."
        ),
        signature="def reservoir_sample(iterable, k: int, seed: int | None = None) -> list:",
        entry="reservoir_sample",
        inputs=["`iterable`: any iterable, possibly a one-shot generator of unknown length.", "`k`: `int` ≥ 0.", "`seed`: seed for `random.Random`."],
        output="A list of `min(k, n)` items, returned in the order they appeared in the stream.",
        constraints=[
            "Every item must be selected with probability `k / n` (the test checks uniformity statistically).",
            "Single pass; do not call `len()` on the input; memory O(k).",
            "Use a local `random.Random(seed)` (never the global random state); the same seed must give the same sample.",
            "Raise `ValueError` for `k < 0`.",
        ],
        criteria=["Correct sample size and uniqueness.", "Uniform probabilities.", "Reproducible.", "Bounded memory."],
        hints=[
            "Keep the first k items. For the i-th item (0-based, i ≥ k), keep it with probability k / (i + 1), replacing a random slot.",
            "Algorithm R: `j = rng.randrange(i + 1)`; if `j < k`, replace slot `j`. Store `(position, item)` pairs so you can restore stream order at the end.",
            "rng = random.Random(seed); reservoir = []\nfor i, item in enumerate(iterable):\n    if i < k: reservoir.append((i, item))\n"
            "    else: j = rng.randrange(i + 1); if j < k: reservoir[j] = (i, item)\nreturn [item for _, item in sorted(reservoir)]",
        ],
        solution='''
import random


def reservoir_sample(iterable, k: int, seed: int | None = None) -> list:
    """Uniform sample of k items from a stream in one pass (Algorithm R)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    rng = random.Random(seed)
    reservoir: list[tuple[int, object]] = []
    if k == 0:
        return []
    for index, item in enumerate(iterable):
        if index < k:
            reservoir.append((index, item))
        else:
            slot = rng.randrange(index + 1)
            if slot < k:
                reservoir[slot] = (index, item)
    reservoir.sort(key=lambda pair: pair[0])
    return [item for _, item in reservoir]
''',
        explanation=(
            "By induction, after processing i + 1 items each has probability k / (i + 1) of being in "
            "the reservoir: a new item enters with that probability, and an existing item survives "
            "with probability 1 − (k/(i+1))·(1/k) = i/(i+1). Memory is the reservoir only. Sorting by "
            "the recorded position (an int) restores stream order without comparing the items themselves."
        ),
        edge_cases=["Stream shorter than k.", "k = 0.", "One-shot generator input.", "Same seed twice."],
        mistakes=["`random.sample(list(stream), k)` (O(n) memory).", "`rng.randint(0, i + 1)` off-by-one (biased).", "Seeding the global `random` module."],
        tests=[
            CT(_q094_properties, note="size, uniqueness, order, reproducibility, edge cases"),
            CT(_q094_uniform, note="uniform selection over 20,000 trials"),
            CT(_q094_memory, note="300,000 streamed items with < 1 MB peak memory"),
            T((range(3), 5, 1), [0, 1, 2], example=True, note="fewer items than k"),
            T((range(10), 0, 1), [], example=True, note="boundary: k = 0"),
        ],
        complexity="O(n) time, O(k) memory (plus O(k log k) to restore order).",
        extension="Implement weighted reservoir sampling (Efraimidis–Spirakis: keep the k largest `u ** (1/w)` keys).",
    ),
    # ------------------------------------------------------------------------ Q095
    Question(
        id=95,
        title="Rolling Per-User Spend Feature in O(n log n)",
        difficulty="Expert",
        category=PERF,
        roles=[MLE, DS_ROLE, DE],
        skills=["sliding windows per key", "index bookkeeping", "equal-timestamp semantics", "performance testing"],
        minutes=50,
        scenario=(
            "A fraud model uses the feature *spend by the same user in the last 24 hours, including "
            "this transaction*. The prototype rescans the user's history for every transaction "
            "(O(n²) per user) and takes hours on a month of data."
        ),
        problem=(
            "Write `rolling_sum_by_key(transactions, window_seconds)` returning, for each transaction "
            "in input order, the sum of that user's amounts with timestamps in `(ts - window_seconds, ts]`."
        ),
        signature="def rolling_sum_by_key(transactions: list[dict], window_seconds: int) -> list[float]:",
        entry="rolling_sum_by_key",
        inputs=["`transactions`: dicts with `user`, `ts` (int seconds) and `amount` (number), in any order.", "`window_seconds`: positive int."],
        output="A `list[float]` aligned with the input, each value rounded to 2 decimals.",
        constraints=[
            "The window is half-open: `ts - window_seconds < t <= ts`.",
            "All of the user's transactions with the **same** timestamp are included in each other's sums (regardless of input order).",
            "100,000 transactions must take well under 2 seconds.",
            "Raise `ValueError` for `window_seconds <= 0`. Do not modify the input.",
        ],
        criteria=["Correct window boundaries.", "Equal timestamps handled.", "Results aligned with input order.", "Efficient."],
        hints=[
            "Process each user separately on their transactions sorted by time, remembering each one's original index.",
            "Slide two pointers over the sorted list: advance the right pointer through *all* rows with the current timestamp, "
            "and the left pointer past rows with `t <= ts - window`. Maintain a running sum.",
            "group (ts, amount, index) by user; sort\nfor each user: left = right = 0; total = 0\n    for each group of equal ts: add all its amounts (right pointer);\n"
            "    remove rows with t <= ts - window (left pointer); assign round(total, 2) to every index in the group",
        ],
        solution='''
def rolling_sum_by_key(transactions: list[dict], window_seconds: int) -> list[float]:
    """Per-user trailing-window sum of amounts for every transaction (input order)."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    by_user: dict = {}
    for index, tx in enumerate(transactions):
        by_user.setdefault(tx["user"], []).append((tx["ts"], tx["amount"], index))

    result = [0.0] * len(transactions)
    for rows in by_user.values():
        rows.sort(key=lambda row: row[0])
        total = 0.0
        left = right = 0
        while right < len(rows):
            ts = rows[right][0]
            group_start = right
            while right < len(rows) and rows[right][0] == ts:
                total += rows[right][1]
                right += 1
            while rows[left][0] <= ts - window_seconds:
                total -= rows[left][1]
                left += 1
            for _, _, index in rows[group_start:right]:
                result[index] = round(total, 2)
    return result
''',
        explanation=(
            "Sorting each user's transactions makes the window a contiguous range that only moves "
            "forward, so each row is added and removed once: O(n log n) overall for the sorts. "
            "Advancing the right pointer over a whole group of equal timestamps before assigning "
            "results implements the 'same timestamp counts' rule; carrying the original index lets "
            "results be written back in input order."
        ),
        edge_cases=["Out-of-order input.", "Several transactions at the same second.", "A transaction exactly `window` seconds older is excluded."],
        mistakes=["Rescanning history per transaction (O(n²)).", "Including only *earlier* same-timestamp rows (order-dependent results).",
                  "Floating-point drift: rounding only at the end is fine for a feature, but use integer cents for accounting."],
        tests=[
            T(([{"user": "u1", "ts": 100, "amount": 10.0}, {"user": "u2", "ts": 100, "amount": 5.0},
                {"user": "u1", "ts": 150, "amount": 20.0}, {"user": "u1", "ts": 150, "amount": 1.0},
                {"user": "u1", "ts": 400, "amount": 7.0}, {"user": "u1", "ts": 50, "amount": 3.0}], 100),
              [13.0, 5.0, 31.0, 31.0, 7.0, 3.0], example=True, note="out of order; equal timestamps; window edge"),
            T(([], 60), [], example=True, note="boundary: no transactions"),
            T(([{"user": "a", "ts": 0, "amount": 1}, {"user": "a", "ts": 60, "amount": 2}], 60), [1.0, 2.0],
              note="exactly window seconds older is excluded"),
            T(([{"user": "a", "ts": 5, "amount": 1}], 0), raises="ValueError", note="invalid window"),
            CT(_q095_fast, note="100,000 transactions"),
        ],
        complexity="O(n log n) time (per-user sorts) and O(n) space.",
        extension="Also return the rolling transaction *count* and *distinct merchants* for the same window.",
    ),
    # ------------------------------------------------------------------------ Q096
    Question(
        id=96,
        title="Resilient Batch Inference with Retries and Bad-Item Isolation",
        difficulty="Expert",
        category=INT,
        roles=[AIE, MLE, AAI],
        skills=["batching", "retry policies", "recursive bisection", "error classification", "contract checks"],
        minutes=60,
        scenario=(
            "A nightly job scores 1M support tickets with a hosted model that accepts batches. Two "
            "things go wrong: the endpoint sometimes times out (retrying helps), and one malformed "
            "ticket makes the *whole batch* fail with a validation error. You must retry transient "
            "errors and isolate bad items without re-scoring good ones more than necessary."
        ),
        problem=(
            "Write `run_batch_inference(items, predict_batch, batch_size, max_retries=2, "
            "retryable=(TimeoutError, ConnectionError))` that returns predictions for every item that "
            "can be scored and a failure reason for every item that cannot."
        ),
        signature=("def run_batch_inference(items: list[dict], predict_batch, batch_size: int, max_retries: int = 2,\n"
                   "                        retryable: tuple = (TimeoutError, ConnectionError)) -> dict:"),
        entry="run_batch_inference",
        inputs=[
            "`items`: dicts with a unique `id` and an `input`.",
            "`predict_batch(list_of_inputs)`: returns a list of predictions of the same length, or raises.",
            "`batch_size` ≥ 1, `max_retries` ≥ 0, `retryable`: tuple of exception types.",
        ],
        output="`{'predictions': {id: prediction}, 'failed': {id: '<ExcType>: <message>'}, 'calls': <number of predict_batch calls>}`",
        constraints=[
            "Split `items` into consecutive batches of `batch_size` and process them in order.",
            "Retryable exception → retry the same batch, up to `max_retries` extra attempts; if it still fails, every item in that batch fails with that error.",
            "Any other `Exception` → if the batch has one item, it fails with that error; otherwise split it into halves "
            "(`[:n//2]`, `[n//2:]`) and process each half the same way (each with a fresh retry budget).",
            "If `predict_batch` returns a list of the wrong length, raise `RuntimeError` immediately (contract violation).",
            "Raise `ValueError` for duplicate ids, `batch_size < 1` or `max_retries < 0`.",
        ],
        criteria=["Transient errors retried.", "Bad items isolated by bisection.", "Accurate call count.", "Contract and input validation."],
        hints=[
            "Write one recursive helper `process(batch)` that handles a single batch completely (retry loop + possible split).",
            "Inside the helper: loop `for attempt in range(max_retries + 1)`, catch `retryable` separately from `Exception`, and recurse on the two halves for non-retryable errors.",
            "def process(batch):\n    for attempt in range(max_retries + 1):\n        calls += 1\n        try: preds = predict_batch([i['input'] for i in batch])\n"
            "        except retryable as e: last = e; continue\n        except Exception as e: if len(batch) == 1: fail; return; process(first half); process(second half); return\n"
            "        check length; record predictions; return\n    mark all failed with last",
        ],
        solution='''
def run_batch_inference(items: list[dict], predict_batch, batch_size: int, max_retries: int = 2,
                        retryable: tuple = (TimeoutError, ConnectionError)) -> dict:
    """Batch scoring with retries for transient errors and bisection for bad items."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("item ids must be unique")

    result = {"predictions": {}, "failed": {}, "calls": 0}

    def describe(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"

    def process(batch: list[dict]) -> None:
        last_error = None
        for _ in range(max_retries + 1):
            result["calls"] += 1
            try:
                predictions = predict_batch([item["input"] for item in batch])
            except retryable as exc:
                last_error = exc
                continue
            except Exception as exc:
                if len(batch) == 1:
                    result["failed"][batch[0]["id"]] = describe(exc)
                else:
                    middle = len(batch) // 2
                    process(batch[:middle])
                    process(batch[middle:])
                return
            if len(predictions) != len(batch):
                raise RuntimeError(f"predict_batch returned {len(predictions)} predictions for {len(batch)} inputs")
            for item, prediction in zip(batch, predictions):
                result["predictions"][item["id"]] = prediction
            return
        for item in batch:
            result["failed"][item["id"]] = describe(last_error)

    for start in range(0, len(items), batch_size):
        process(items[start:start + batch_size])
    return result
''',
        explanation=(
            "Classifying errors drives the policy: transient errors deserve a retry of the same batch, "
            "while deterministic input errors need bisection – each split halves the suspects, so a "
            "single bad item in a batch of b costs O(log b) extra calls instead of b. Checking the "
            "response length protects against silently misaligned predictions, which would be far "
            "worse than a crash."
        ),
        edge_cases=["Empty input → no calls.", "Bad item alone in its batch.", "Transient errors exhausting retries.", "Wrong-length responses."],
        mistakes=["Retrying deterministic validation errors (wastes calls, never succeeds).", "Scoring items one by one after any failure (b extra calls).",
                  "Zipping predictions without checking the length."],
        tests=[
            CT(_q096_success, note="happy path: ordered batches, call count"),
            CT(_q096_isolation, note="bad item isolated by bisection"),
            CT(_q096_transient, note="transient errors retried / exhausted"),
            CT(_q096_contract, note="wrong-length output, duplicate ids, invalid arguments, empty input"),
        ],
        examples=['''
>>> def predict_batch(texts):
...     if "" in texts:
...         raise ValueError("empty text")
...     return [len(t) for t in texts]
>>> items = [{"id": "a", "input": "hi"}, {"id": "b", "input": ""}, {"id": "c", "input": "hello"}]
>>> run_batch_inference(items, predict_batch, batch_size=3)
{'predictions': {'a': 2, 'c': 5}, 'failed': {'b': 'ValueError: empty text'}, 'calls': 5}
'''],
        complexity="Without failures: n / batch_size calls. Each bad item adds O(log batch_size) calls; transient failures add up to `max_retries` calls per batch.",
        extension="Add exponential backoff with an injectable `sleep`, and a circuit breaker that stops after N consecutive transient failures.",
    ),
    # ------------------------------------------------------------------------ Q097
    Question(
        id=97,
        title="Compare Two Model Versions on Paired Evaluation Slices",
        difficulty="Expert",
        category=INT,
        roles=[MLE, AAI, DS_ROLE],
        skills=["pairing records", "slice analysis", "decision rules with thresholds", "data validation"],
        minutes=50,
        scenario=(
            "Before promoting model v2, you compare it with v1 on the same evaluation set. Overall "
            "accuracy improved, but stakeholders worry about specific languages. Some examples were "
            "only evaluated by one model (a crashed run), and comparing unpaired examples would be misleading."
        ),
        problem=(
            "Write `compare_models(records, baseline, candidate, slice_key, min_support=20, tolerance=0.02)` "
            "that compares accuracy on paired examples overall and per slice and assigns a verdict to each."
        ),
        signature=("def compare_models(records: list[dict], baseline: str, candidate: str, slice_key: str,\n"
                   "                   min_support: int = 20, tolerance: float = 0.02) -> dict:"),
        entry="compare_models",
        inputs=["`records`: dicts with `example_id`, `model`, `correct` (bool) and the `slice_key` field (e.g. `language`).",
                "`baseline`/`candidate`: model names. Records of other models are ignored."],
        output=(
            "`{'overall': stats, 'slices': {slice_value: stats}, 'unpaired': int}` where stats = "
            "`{'support', 'baseline_acc', 'candidate_acc', 'delta', 'verdict'}`; accuracies and delta rounded to 4."
        ),
        constraints=[
            "Only examples evaluated by **both** models are used; `unpaired` counts baseline/candidate records whose example lacks the other model.",
            "The slice of an example is taken from its baseline record.",
            "`delta = candidate_acc - baseline_acc` computed from unrounded values.",
            "verdict: `'insufficient_data'` if support < `min_support`; else `'regression'` if delta < −tolerance; "
            "`'improvement'` if delta > tolerance; otherwise `'neutral'`.",
            "Raise `ValueError` for a duplicate `(example_id, model)` pair or if no example is paired.",
        ],
        criteria=["Correct pairing.", "Correct per-slice accuracies.", "Verdict thresholds.", "Validation."],
        hints=[
            "Index results as `{example_id: {model: record}}` first.",
            "Collect the paired examples, then aggregate `(n, baseline_correct, candidate_correct)` overall and per slice with one helper that builds the stats dict.",
            "by_example = {}; detect duplicates\npaired = [(b, c) for ex in by_example.values() if both models present]\nunpaired = number of single-model entries\n"
            "def stats(pairs): n; b_acc; c_acc; delta; verdict\nslices = group pairs by b[slice_key]",
        ],
        solution='''
def _stats(pairs: list[tuple[dict, dict]], min_support: int, tolerance: float) -> dict:
    n = len(pairs)
    baseline_acc = sum(b["correct"] for b, _ in pairs) / n
    candidate_acc = sum(c["correct"] for _, c in pairs) / n
    delta = candidate_acc - baseline_acc
    if n < min_support:
        verdict = "insufficient_data"
    elif delta < -tolerance:
        verdict = "regression"
    elif delta > tolerance:
        verdict = "improvement"
    else:
        verdict = "neutral"
    return {"support": n, "baseline_acc": round(baseline_acc, 4), "candidate_acc": round(candidate_acc, 4),
            "delta": round(delta, 4), "verdict": verdict}


def compare_models(records: list[dict], baseline: str, candidate: str, slice_key: str,
                   min_support: int = 20, tolerance: float = 0.02) -> dict:
    """Paired accuracy comparison of two models, overall and per slice."""
    by_example: dict[str, dict[str, dict]] = {}
    for record in records:
        if record["model"] not in (baseline, candidate):
            continue
        models = by_example.setdefault(record["example_id"], {})
        if record["model"] in models:
            raise ValueError(f"duplicate result for ({record['example_id']!r}, {record['model']!r})")
        models[record["model"]] = record

    pairs = []
    unpaired = 0
    for models in by_example.values():
        if baseline in models and candidate in models:
            pairs.append((models[baseline], models[candidate]))
        else:
            unpaired += len(models)
    if not pairs:
        raise ValueError("no examples were evaluated by both models")

    slices: dict = {}
    for pair in pairs:
        slices.setdefault(pair[0][slice_key], []).append(pair)
    return {
        "overall": _stats(pairs, min_support, tolerance),
        "slices": {name: _stats(group, min_support, tolerance) for name, group in slices.items()},
        "unpaired": unpaired,
    }
''',
        explanation=(
            "Pairing by example id guarantees both models are judged on identical inputs; unpaired "
            "records are reported instead of silently skewing the comparison. One `_stats` helper "
            "serves the overall and per-slice views, so the verdict rules are defined exactly once. "
            "The support threshold prevents over-reacting to tiny slices."
        ),
        edge_cases=["Examples evaluated by only one model.", "Tiny slices.", "Delta exactly ± tolerance → neutral.", "Records from a third model."],
        mistakes=["Comparing accuracies over different example sets.", "Rounding before computing delta.", "Using `<=` for regression (tolerance boundary)."],
        tests=[
            T((EVAL_RECORDS, "v1", "v2", "language", 2, 0.1),
              {"overall": {"support": 11, "baseline_acc": 0.6364, "candidate_acc": 0.4545, "delta": -0.1818, "verdict": "regression"},
               "slices": {"en": {"support": 4, "baseline_acc": 0.5, "candidate_acc": 0.75, "delta": 0.25, "verdict": "improvement"},
                          "hi": {"support": 4, "baseline_acc": 0.75, "candidate_acc": 0.25, "delta": -0.5, "verdict": "regression"},
                          "de": {"support": 1, "baseline_acc": 1.0, "candidate_acc": 0.0, "delta": -1.0, "verdict": "insufficient_data"},
                          "es": {"support": 2, "baseline_acc": 0.5, "candidate_acc": 0.5, "delta": 0.0, "verdict": "neutral"}},
               "unpaired": 1}, example=True, note="all four verdicts; one unpaired record; overall regression"),
            T(([{"example_id": "e1", "model": "v1", "lang": "en", "correct": True},
                {"example_id": "e1", "model": "v2", "lang": "en", "correct": True},
                {"example_id": "e1", "model": "v3", "lang": "en", "correct": False}], "v1", "v2", "lang", 1, 0.0),
              {"overall": {"support": 1, "baseline_acc": 1.0, "candidate_acc": 1.0, "delta": 0.0, "verdict": "neutral"},
               "slices": {"en": {"support": 1, "baseline_acc": 1.0, "candidate_acc": 1.0, "delta": 0.0, "verdict": "neutral"}},
               "unpaired": 0}, example=True, note="boundary: delta 0 with tolerance 0 is neutral; other models ignored"),
            T(([{"example_id": "e1", "model": "v1", "lang": "en", "correct": True},
                {"example_id": "e1", "model": "v1", "lang": "en", "correct": False}], "v1", "v2", "lang"),
              raises="ValueError", note="duplicate (example, model)"),
            T(([{"example_id": "e1", "model": "v1", "lang": "en", "correct": True}], "v1", "v2", "lang"),
              raises="ValueError", note="nothing paired"),
            CT(_q097_sample, note="data/sample_data/eval_results.json"),
        ],
        complexity="O(n) time and space.",
        extension="Add a paired bootstrap confidence interval for each delta and only flag regressions whose interval excludes 0.",
    ),
    # ------------------------------------------------------------------------ Q098
    Question(
        id=98,
        title="Assemble a Streamed LLM Response with Tool Calls",
        difficulty="Expert",
        category=INT,
        roles=[AIE, AAI, PYDEV],
        skills=["event reordering", "de-duplication", "incremental JSON fragments", "protocol validation", "state machines"],
        minutes=60,
        scenario=(
            "Your gateway receives a chat model's streamed response as numbered events relayed "
            "through a message queue. Events can arrive out of order and some are delivered twice. "
            "The response contains text and *tool calls* whose JSON arguments are split across "
            "several events. Before executing any tool you must reconstruct the full response and "
            "reject anything inconsistent."
        ),
        problem="Write `assemble_stream(events)` that validates the event sequence and returns the assembled message.",
        signature="def assemble_stream(events: list[dict]) -> dict:",
        entry="assemble_stream",
        inputs=[
            "Events have an int `seq` (starting at 0) and a `type`:",
            "`text_delta` with `text`; `tool_call_delta` with `call_id`, `arguments` (a JSON fragment) and – on any one delta of that call – `name`;",
            "`done` with `finish_reason`; `error` with `message`.",
        ],
        output="`{'text': str, 'tool_calls': [{'id', 'name', 'arguments': dict}], 'finish_reason': str | None, 'error': str | None}`",
        constraints=[
            "Process events in `seq` order. Exact duplicates are ignored; two different events with the same `seq` → `ValueError`.",
            "The sequence numbers must be exactly 0..max with no gaps → otherwise `ValueError`.",
            "Exactly one terminal event (`done` or `error`), and it must have the highest `seq` → otherwise `ValueError` (this includes an empty stream).",
            "Tool calls appear in order of their first delta; their argument fragments are concatenated and must parse to a JSON object; each call needs a name. Violations → `ValueError`.",
            "For an `error` terminal: return the text so far, `tool_calls = []`, `finish_reason = None`, `error = message`.",
        ],
        criteria=["Reordering and de-duplication.", "Gap and terminal validation.", "Tool-call assembly and parsing.", "Error streams."],
        hints=[
            "First normalise the input (dedupe by `seq`, validate), then assemble in a second pass – don't mix the two.",
            "Use a dict `seq → event` to detect conflicting duplicates, check `sorted(seqs) == list(range(len(seqs)))`, and keep tool calls in a dict (insertion order = first appearance).",
            "by_seq = {}; for e in events: if seq in by_seq and by_seq[seq] != e: raise; by_seq[seq] = e\nordered = [by_seq[i] for i in range(len(by_seq))] (check gaps)\n"
            "terminal checks\nfor e in ordered[:-1]: text += ...; calls.setdefault(id, {'name': None, 'args': []}) ...\nparse args with json.loads",
        ],
        solution='''
import json


def assemble_stream(events: list[dict]) -> dict:
    """Validate and assemble a streamed chat response (text + tool calls)."""
    by_seq: dict[int, dict] = {}
    for event in events:
        seq = event["seq"]
        if seq in by_seq and by_seq[seq] != event:
            raise ValueError(f"conflicting events with seq {seq}")
        by_seq[seq] = event
    if sorted(by_seq) != list(range(len(by_seq))):
        raise ValueError("missing events: sequence numbers are not contiguous from 0")
    ordered = [by_seq[i] for i in range(len(by_seq))]
    terminals = [e for e in ordered if e["type"] in ("done", "error")]
    if len(terminals) != 1 or ordered[-1]["type"] not in ("done", "error"):
        raise ValueError("stream must end with exactly one done/error event")

    text_parts: list[str] = []
    calls: dict[str, dict] = {}
    for event in ordered[:-1]:
        if event["type"] == "text_delta":
            text_parts.append(event["text"])
        elif event["type"] == "tool_call_delta":
            call = calls.setdefault(event["call_id"], {"name": None, "fragments": []})
            if event.get("name"):
                call["name"] = event["name"]
            call["fragments"].append(event.get("arguments", ""))
        else:
            raise ValueError(f"unknown event type {event['type']!r}")

    terminal = ordered[-1]
    if terminal["type"] == "error":
        return {"text": "".join(text_parts), "tool_calls": [], "finish_reason": None, "error": terminal["message"]}

    tool_calls = []
    for call_id, call in calls.items():
        if not call["name"]:
            raise ValueError(f"tool call {call_id!r} has no name")
        try:
            arguments = json.loads("".join(call["fragments"]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool call {call_id!r} has invalid JSON arguments") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"tool call {call_id!r} arguments must be a JSON object")
        tool_calls.append({"id": call_id, "name": call["name"], "arguments": arguments})
    return {"text": "".join(text_parts), "tool_calls": tool_calls,
            "finish_reason": terminal["finish_reason"], "error": None}
''',
        explanation=(
            "Normalising first (dedupe, gap check, terminal check) means the assembly loop can assume "
            "a clean, ordered stream. Argument fragments are only meaningful once concatenated, so "
            "they are buffered per call and parsed at the end; a dict preserves first-appearance "
            "order. Every inconsistency raises before any tool could be executed on partial data."
        ),
        edge_cases=["Duplicated event.", "Out-of-order fragments.", "Error terminal.", "Empty stream.", "Events after `done`."],
        mistakes=["Parsing each fragment separately.", "Appending events in arrival order.", "Treating a conflicting duplicate as harmless."],
        tests=[
            T((STREAM_EVENTS,),
              {"text": "Let me check the weather.",
               "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Pune", "unit": "C"}},
                              {"id": "call_2", "name": "get_time", "arguments": {"tz": "IST"}}],
               "finish_reason": "tool_calls", "error": None}, example=True, note="out of order with a duplicate"),
            T(([{"seq": 0, "type": "done", "finish_reason": "stop"}],),
              {"text": "", "tool_calls": [], "finish_reason": "stop", "error": None}, example=True, note="boundary: only a terminal event"),
            T(([{"seq": 0, "type": "text_delta", "text": "Partial"}, {"seq": 1, "type": "error", "message": "overloaded"}],),
              {"text": "Partial", "tool_calls": [], "finish_reason": None, "error": "overloaded"}, note="error terminal"),
            T(([{"seq": 0, "type": "text_delta", "text": "a"}, {"seq": 2, "type": "done", "finish_reason": "stop"}],),
              raises="ValueError", note="gap in seq"),
            T(([{"seq": 0, "type": "text_delta", "text": "a"}, {"seq": 0, "type": "text_delta", "text": "b"},
                {"seq": 1, "type": "done", "finish_reason": "stop"}],), raises="ValueError", note="conflicting duplicate"),
            T(([{"seq": 0, "type": "text_delta", "text": "a"}],), raises="ValueError", note="no terminal event"),
            T(([],), raises="ValueError", note="empty stream"),
            T(([{"seq": 0, "type": "done", "finish_reason": "stop"}, {"seq": 1, "type": "text_delta", "text": "late"}],),
              raises="ValueError", note="event after done"),
            T(([{"seq": 0, "type": "tool_call_delta", "call_id": "c", "name": "f", "arguments": '{"a": '},
                {"seq": 1, "type": "done", "finish_reason": "tool_calls"}],), raises="ValueError", note="incomplete JSON arguments"),
            T(([{"seq": 0, "type": "tool_call_delta", "call_id": "c", "arguments": "{}"},
                {"seq": 1, "type": "done", "finish_reason": "tool_calls"}],), raises="ValueError", note="tool call without name"),
        ],
        complexity="O(n log n) for ordering (O(n) with the contiguity check) plus O(total argument length) to parse.",
        extension="Make it incremental: a class that accepts events one at a time and exposes the text assembled so far.",
    ),
    # ------------------------------------------------------------------------ Q099
    Question(
        id=99,
        title="Pack Retrieved Context into a Token Budget",
        difficulty="Expert",
        category=INT,
        roles=[AIE, AAI, MLE],
        skills=["greedy selection with tie-breaking", "normalised de-duplication", "budget accounting", "order restoration"],
        minutes=45,
        scenario=(
            "Your RAG service retrieves ~50 passages per question, but the model's context window "
            "leaves room for only a fraction of them after reserving space for the answer. Retrieval "
            "also returns near-identical copies of the same passage. You need a deterministic packing "
            "step that keeps the most relevant unique passages and preserves their original order."
        ),
        problem="Write `pack_context(docs, budget, reserve=0)` that selects passages to fit the token budget and builds the prompt context.",
        signature="def pack_context(docs: list[dict], budget: int, reserve: int = 0) -> dict:",
        entry="pack_context",
        inputs=["`docs`: dicts with unique `id`, `text` and numeric `score` (higher is more relevant), in retrieval order.", "`budget`, `reserve`: ints."],
        output=(
            "`{'selected': [ids in original order], 'used_tokens': int, 'dropped': [sorted ids], 'prompt': str}` where `prompt` "
            "joins `'[<id>]\\n<text.strip()>'` for the selected docs with blank lines (`'\\n\\n'`)."
        ),
        constraints=[
            "A doc costs `len(text.split()) + 4` tokens (4 for the header and separator). Available tokens = `budget - reserve`.",
            "De-duplicate first: texts equal after lower-casing and collapsing whitespace are duplicates; keep the highest score (earlier doc on equal scores).",
            "Greedy: consider docs by score descending, then cost ascending, then id ascending; add a doc if it still fits, otherwise skip it and continue.",
            "Raise `ValueError` if `budget - reserve <= 0`, a score is not a number, or ids are not unique.",
        ],
        criteria=["Correct de-duplication.", "Deterministic greedy packing.", "Original order restored.", "Exact prompt format."],
        hints=[
            "Three phases: de-duplicate, choose, then render in the original order.",
            "Normalise with `' '.join(text.lower().split())`. Sort candidates with the key `(-score, cost, id)` and keep a running token total.",
            "keep = {} normalized → best doc\ncandidates = sorted(kept, key=lambda d: (-d['score'], cost(d), d['id']))\nfor d in candidates: if used + cost(d) <= available: chosen.add(d['id']); used += cost(d)\n"
            "selected = [d['id'] for d in docs if d['id'] in chosen]",
        ],
        solution='''
HEADER_TOKENS = 4


def _cost(doc: dict) -> int:
    return len(doc["text"].split()) + HEADER_TOKENS


def pack_context(docs: list[dict], budget: int, reserve: int = 0) -> dict:
    """Greedy, de-duplicated context packing that preserves retrieval order."""
    available = budget - reserve
    if available <= 0:
        raise ValueError("budget - reserve must be positive")
    ids = [doc["id"] for doc in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("document ids must be unique")
    for doc in docs:
        if isinstance(doc.get("score"), bool) or not isinstance(doc.get("score"), (int, float)):
            raise ValueError(f"document {doc['id']!r} has no numeric score")

    best: dict[str, dict] = {}
    for doc in docs:
        key = " ".join(doc["text"].lower().split())
        if key not in best or doc["score"] > best[key]["score"]:
            best[key] = doc
    candidates = sorted(best.values(), key=lambda d: (-d["score"], _cost(d), d["id"]))

    chosen: set[str] = set()
    used = 0
    for doc in candidates:
        if used + _cost(doc) <= available:
            chosen.add(doc["id"])
            used += _cost(doc)

    selected = [doc for doc in docs if doc["id"] in chosen]
    return {
        "selected": [doc["id"] for doc in selected],
        "used_tokens": used,
        "dropped": sorted(doc_id for doc_id in ids if doc_id not in chosen),
        "prompt": "\\n\\n".join(f"[{doc['id']}]\\n{doc['text'].strip()}" for doc in selected),
    }
''',
        explanation=(
            "De-duplicating before selection stops copies from wasting budget. A fully specified sort "
            "key makes the greedy choice deterministic; skipping (rather than stopping at) a doc that "
            "doesn't fit lets smaller relevant passages use leftover space. Re-emitting in retrieval "
            "order keeps documents in their natural reading sequence for the model."
        ),
        edge_cases=["Budget too small for anything → empty selection and prompt.", "Exact fit.", "Duplicates with equal scores."],
        mistakes=["Stopping at the first doc that doesn't fit.", "Counting characters instead of tokens.", "Emitting docs in score order (changes context flow)."],
        tests=[
            T((CONTEXT_DOCS, 30, 4),
              {"selected": ["d2", "d3", "d4"], "used_tokens": 26, "dropped": ["d1", "d5"],
               "prompt": "[d2]\none two three four five six seven eight\n\n[d3]\nAlpha  beta gamma delta\n\n[d4]\nshort text"},
              example=True, note="dedupe, skip-too-large, tie by cost, exact fit"),
            T((CONTEXT_DOCS, 5), {"selected": [], "used_tokens": 0, "dropped": ["d1", "d2", "d3", "d4", "d5"], "prompt": ""},
              example=True, note="boundary: nothing fits"),
            T(([{"id": "a", "text": "same text", "score": 0.5}, {"id": "b", "text": "SAME   text", "score": 0.5}], 100),
              {"selected": ["a"], "used_tokens": 6, "dropped": ["b"], "prompt": "[a]\nsame text"}, note="equal-score duplicates keep the earlier"),
            T((CONTEXT_DOCS, 10, 10), raises="ValueError", note="reserve consumes the budget"),
            T(([{"id": "a", "text": "x", "score": None}], 10), raises="ValueError", note="missing score"),
            T(([{"id": "a", "text": "x", "score": 1}, {"id": "a", "text": "y", "score": 2}], 10), raises="ValueError", note="duplicate ids"),
        ],
        complexity="O(n log n) time for sorting; O(n) space.",
        extension="Allow truncating the last passage to fill the remaining budget, ending at a sentence boundary.",
    ),
    # ------------------------------------------------------------------------ Q100
    Question(
        id=100,
        title="Sentence-Aware Text Chunking with Overlap for RAG",
        difficulty="Expert",
        category=INT,
        roles=[AIE, AAI, DE],
        skills=["regex sentence splitting", "greedy packing", "overlap with progress guarantees", "careful specification"],
        minutes=60,
        scenario=(
            "Your document ingestion pipeline splits knowledge-base articles into chunks before "
            "embedding them. Chunks that cut sentences in half retrieve poorly, and without overlap, "
            "facts that span a chunk boundary are lost. The chunker must also never loop forever on "
            "awkward inputs such as very long sentences."
        ),
        problem=(
            "Write `chunk_text(text, max_words, overlap_words)` that packs whole sentences into chunks of "
            "at most `max_words` words, repeating trailing sentences of the previous chunk as overlap."
        ),
        signature="def chunk_text(text: str, max_words: int, overlap_words: int) -> list[dict]:",
        entry="chunk_text",
        inputs=["`text`: any string.", "`max_words`: `int` ≥ 1.", "`overlap_words`: `int` with `0 <= overlap_words < max_words`."],
        output="A list of `{'index': i, 'text': str, 'word_count': int, 'unit_range': (start, end)}` – units `start..end-1` joined by single spaces.",
        constraints=[
            "Units: split the whitespace-normalised text with `re.split(r'(?<=[.!?])\\s+', ...)`; a sentence with more than `max_words` "
            "words becomes several units of `max_words` words (the last may be shorter). Empty text → `[]`.",
            "Chunk 0 starts at unit 0. Every chunk greedily adds the following units while the total stays ≤ `max_words`.",
            "Each later chunk starts with the longest run of trailing units of the previous chunk whose total is ≤ `overlap_words`; if that "
            "overlap plus the next new unit exceeds `max_words`, drop overlap units from the front until it fits (possibly all of them).",
            "Stop once the last unit has been included. Raise `ValueError` for invalid `max_words`/`overlap_words`.",
        ],
        criteria=["Sentence boundaries respected.", "Long sentences split.", "Overlap rule exact.", "Always terminates."],
        hints=[
            "Turn the text into a list of units with word counts first; then chunking is pure index arithmetic.",
            "Track `start` and `end` unit indexes. For the next chunk, walk backwards from `end` while the overlap fits, then "
            "move `start` forward while `overlap + words[end] > max_words`. The new chunk always contains unit `end`, which guarantees progress.",
            "units = make_units(); start = 0\nwhile True:\n    end = start; total = 0\n    while end < n and total + w[end] <= max_words: total += w[end]; end += 1\n"
            "    emit(start, end)\n    if end == n: break\n    s = end; ov = 0\n    while s > start and ov + w[s-1] <= overlap: s -= 1; ov += w[s]\n"
            "    while s < end and sum(w[s:end]) + w[end] > max_words: s += 1\n    start = s",
        ],
        solution='''
import re


def _units(text: str, max_words: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    units = []
    for sentence in re.split(r"(?<=[.!?])\\s+", normalized):
        words = sentence.split()
        for i in range(0, len(words), max_words):
            units.append(" ".join(words[i:i + max_words]))
    return units


def chunk_text(text: str, max_words: int, overlap_words: int) -> list[dict]:
    """Pack sentences into <= max_words chunks with sentence-level overlap."""
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap_words < max_words")
    units = _units(text, max_words)
    words = [len(unit.split()) for unit in units]
    chunks: list[dict] = []
    start = 0
    while units:
        end, total = start, 0
        while end < len(units) and total + words[end] <= max_words:
            total += words[end]
            end += 1
        chunks.append({"index": len(chunks), "text": " ".join(units[start:end]),
                       "word_count": total, "unit_range": (start, end)})
        if end == len(units):
            break
        next_start, overlap = end, 0
        while next_start > start and overlap + words[next_start - 1] <= overlap_words:
            next_start -= 1
            overlap += words[next_start]
        while next_start < end and overlap + words[end] > max_words:
            overlap -= words[next_start]
            next_start += 1
        start = next_start
    return chunks
''',
        explanation=(
            "Separating unit construction from packing turns the problem into index arithmetic on a "
            "list of word counts. Every chunk contains at least one unit it did not share with the "
            "previous chunk (`end` itself), and splitting long sentences guarantees every unit fits, "
            "so the loop always makes progress and terminates. Overlap is computed from whole units, "
            "so overlapping text is never cut mid-sentence."
        ),
        edge_cases=["Empty or whitespace-only text.", "A sentence longer than `max_words`.", "Overlap that cannot fit with the next unit.", "Single short text."],
        mistakes=["Splitting on '.' only (misses '!' and '?') or splitting decimals like 3.5 (the regex needs following whitespace).",
                  "Overlap computed in words that cuts sentences.", "Infinite loop when overlap + next unit never fits."],
        tests=[
            T((CHUNK_TEXT, 10, 5),
              [{"index": 0, "text": "RAG systems retrieve context. They split documents into chunks.", "word_count": 9, "unit_range": (0, 2)},
               {"index": 1, "text": "They split documents into chunks. Each chunk should be self-contained!", "word_count": 10, "unit_range": (1, 3)},
               {"index": 2, "text": "Each chunk should be self-contained! Overlap preserves continuity between chunks?", "word_count": 10,
                "unit_range": (2, 4)},
               {"index": 3, "text": "Overlap preserves continuity between chunks? Yes.", "word_count": 6, "unit_range": (3, 5)}],
              example=True, note="sentence packing with overlap"),
            T(("   ", 10, 2), [], example=True, note="boundary: empty text"),
            T(("one two three four five six seven eight nine ten eleven twelve. End.", 5, 0),
              [{"index": 0, "text": "one two three four five", "word_count": 5, "unit_range": (0, 1)},
               {"index": 1, "text": "six seven eight nine ten", "word_count": 5, "unit_range": (1, 2)},
               {"index": 2, "text": "eleven twelve. End.", "word_count": 3, "unit_range": (2, 4)}], note="long sentence split, no overlap"),
            T(("a b c d e. f g h i j. k l m n o.", 6, 5),
              [{"index": 0, "text": "a b c d e.", "word_count": 5, "unit_range": (0, 1)},
               {"index": 1, "text": "f g h i j.", "word_count": 5, "unit_range": (1, 2)},
               {"index": 2, "text": "k l m n o.", "word_count": 5, "unit_range": (2, 3)}], note="overlap dropped when it cannot fit"),
            T(("Version 3.5 is out. Upgrade now!", 20, 3),
              [{"index": 0, "text": "Version 3.5 is out. Upgrade now!", "word_count": 6, "unit_range": (0, 2)}], note="decimal point is not a sentence end"),
            T(("text", 5, 5), raises="ValueError", note="overlap must be < max_words"),
            T(("text", 0, 0), raises="ValueError", note="max_words must be >= 1"),
        ],
        complexity="O(n) time and space in the number of words (each unit is visited a constant number of times per chunk it appears in).",
        extension="Measure chunk size in model tokens using a pluggable `count_tokens(text)` function instead of words.",
    ),
]
