"""Intermediate questions Q039–Q055: practical programming and data handling."""

import json

from ..schema import (
    AAI, AIE, ALG, DATA, DBG, DE, DS_ROLE, FILE, INT, MLE, OOP, PERF, PYDEV,
    CT, DataFile, File, Question, T, TmpPath,
)

# ------------------------------------------------------------------ custom tests
TRANSACTIONS_CSV = (
    "transaction_id,customer_id,amount,currency,timestamp\n"
    "t1,c1,19.99,usd,2024-01-05T10:00:00Z\n"
    "t2,c2,abc,USD,2024-01-05T10:01:00Z\n"
    "t3,,5.00,EUR,2024-01-05T10:02:00Z\n"
    "t4,c3,-3.50,GBP,2024-01-05T10:03:00Z\n"
    "t5,c1,7.5,JPY,2024-01-05T10:04:00Z\n"
    " t6 , c4 , 100 , eur ,2024-01-05T10:05:00Z\n"
    "t7,c5,12.00\n"
    "t8,c6,inf,USD,2024-01-05T10:07:00Z\n"
)


def _q040_sample(impl, tmp):
    from ..schema import SAMPLE_DATA_DIR

    valid, invalid = impl(str(SAMPLE_DATA_DIR / "transactions.csv"))
    assert len(valid) == 30, f"expected 30 valid rows in transactions.csv, got {len(valid)}"
    assert invalid == [4, 9, 13, 18, 22, 27, 31, 35], f"unexpected invalid row numbers: {invalid}"
    assert all(isinstance(r["amount"], float) for r in valid), "amounts must be floats"


def _q041_sample(impl, tmp):
    from ..schema import SAMPLE_DATA_DIR

    records, bad = impl(str(SAMPLE_DATA_DIR / "inference_requests.jsonl"))
    assert len(records) == 48, f"expected 48 records, got {len(records)}"
    assert bad == [7, 19, 33], f"expected malformed lines [7, 19, 33], got {bad}"


def _expected_json(data) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _q042_creates_dirs(impl, tmp):
    target = tmp / "reports" / "2024" / "summary.json"
    data = {"b": 1, "a": [1, 2], "nested": {"z": None, "y": True}}
    assert impl(str(target), data) is None, "write_json_atomic should return None"
    assert target.read_text(encoding="utf-8") == _expected_json(data), "file content must be indent=2, sort_keys=True JSON plus a newline"


def _q042_overwrites(impl, tmp):
    target = tmp / "metrics.json"
    target.write_text("old", encoding="utf-8")
    impl(str(target), {"accuracy": 0.93})
    assert target.read_text(encoding="utf-8") == _expected_json({"accuracy": 0.93})
    assert sorted(p.name for p in tmp.iterdir()) == ["metrics.json"], "temporary files must not be left behind"


def _q042_failure_is_atomic(impl, tmp):
    target = tmp / "metrics.json"
    target.write_text("old", encoding="utf-8")
    try:
        impl(str(target), {"tags": {"not", "serializable"}})
    except TypeError:
        pass
    else:
        raise AssertionError("a non-serializable value should raise TypeError")
    assert target.read_text(encoding="utf-8") == "old", "the original file must be unchanged after a failed write"
    leftovers = sorted(p.name for p in tmp.iterdir())
    assert leftovers == ["metrics.json"], f"no temporary files may remain, found {leftovers}"


def _q042_unicode(impl, tmp):
    target = tmp / "names.json"
    impl(str(target), {"city": "Zürich", "emoji": "✅"})
    text = target.read_text(encoding="utf-8")
    assert "Zürich" in text and "✅" in text, "use ensure_ascii=False and UTF-8 encoding"


def _q043_cause_missing(impl, tmp):
    try:
        impl(str(tmp / "missing.json"))
    except Exception as exc:
        assert type(exc).__name__ == "ConfigError", f"expected ConfigError, got {type(exc).__name__}"
        assert isinstance(exc.__cause__, FileNotFoundError), "chain the original error with `raise ... from exc`"
        assert "missing.json" in str(exc), "the message should include the path"
    else:
        raise AssertionError("expected ConfigError for a missing file")


def _q043_cause_json(impl, tmp):
    path = tmp / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    try:
        impl(str(path))
    except Exception as exc:
        assert type(exc).__name__ == "ConfigError", f"expected ConfigError, got {type(exc).__name__}"
        assert type(exc.__cause__).__name__ == "JSONDecodeError", "chain the json.JSONDecodeError with `from`"
    else:
        raise AssertionError("expected ConfigError for invalid JSON")


def _q044_basic(impl, tmp):
    stats = impl()
    assert stats.count == 0 and stats.mean is None and stats.min is None and stats.max is None \
        and stats.variance is None, "an empty RunningStats should report count 0 and None statistics"
    for value in [4, 8, 6]:
        stats.add(value)
    assert (stats.count, stats.mean, stats.min, stats.max) == (3, 6.0, 4, 8)
    assert abs(stats.variance - 8 / 3) < 1e-9, f"population variance should be 2.6667, got {stats.variance}"


def _q044_types(impl, tmp):
    stats = impl()
    for bad in ["3", True, None]:
        try:
            stats.add(bad)
        except TypeError:
            continue
        raise AssertionError(f"add({bad!r}) should raise TypeError")
    assert stats.count == 0, "rejected values must not be counted"


def _q044_read_only(impl, tmp):
    stats = impl()
    stats.add(1)
    try:
        stats.count = 10
    except AttributeError:
        return
    raise AssertionError("count should be a read-only property")


def _q044_stable(impl, tmp):
    stats = impl()
    for value in [1e9 + 4, 1e9 + 7, 1e9 + 13, 1e9 + 16]:
        stats.add(value)
    assert abs(stats.variance - 22.5) < 1e-6, (f"variance should be 22.5, got {stats.variance}; the "
                                               "sum-of-squares formula loses precision – use Welford's update")


def _q044_independent(impl, tmp):
    a, b = impl(), impl()
    a.add(10)
    assert b.count == 0, "instances must not share state (avoid class attributes for data)"


def _q045_parse(impl, tmp):
    v = impl("v2.10.3")
    assert (v.major, v.minor, v.patch) == (2, 10, 3), "parse major, minor and patch as ints"
    assert str(v) == "2.10.3" and repr(v) == "Version('2.10.3')", f"unexpected str/repr: {str(v)!r}, {repr(v)!r}"


def _q045_invalid(impl, tmp):
    for bad in ["1.2", "1.2.3.4", "a.b.c", "1.-2.3", ""]:
        try:
            impl(bad)
        except ValueError:
            continue
        raise AssertionError(f"Version({bad!r}) should raise ValueError")


def _q045_ordering(impl, tmp):
    versions = [impl("1.10.0"), impl("1.2.3"), impl("v1.2.10"), impl("0.9.9")]
    assert [str(v) for v in sorted(versions)] == ["0.9.9", "1.2.3", "1.2.10", "1.10.0"], "compare numerically, not as strings"
    assert impl("1.2.3") <= impl("1.2.3") and impl("2.0.0") > impl("1.99.99")


def _q045_equality_hash(impl, tmp):
    assert impl("1.2.3") == impl("v1.2.3")
    assert len({impl("1.2.3"), impl("v1.2.3"), impl("1.2.4")}) == 2, "equal versions must hash equally"
    assert (impl("1.0.0") == "1.0.0") is False, "comparing with a non-Version should be False, not an error"
    try:
        _ = impl("1.0.0") < "2.0.0"
    except TypeError:
        pass
    else:
        raise AssertionError("ordering against a non-Version should raise TypeError (return NotImplemented)")


def _q045_bump(impl, tmp):
    v = impl("2.3.4")
    assert str(v.bump("major")) == "3.0.0" and str(v.bump("minor")) == "2.4.0" and str(v.bump("patch")) == "2.3.5"
    assert str(v) == "2.3.4", "bump must return a new Version and leave the original unchanged"
    try:
        v.bump("build")
    except ValueError:
        return
    raise AssertionError("bump('build') should raise ValueError")


def _q046_chain_and_run(impl, tmp):
    p = impl("clean")
    returned = p.add_step("strip", lambda r: {**r, "text": r["text"].strip()})
    assert returned is p, "add_step must return the pipeline itself for chaining"
    p.add_step("drop_empty", lambda r: r if r["text"] else None).add_step("upper", lambda r: {**r, "text": r["text"].upper()})
    assert p.steps == ["strip", "drop_empty", "upper"] and len(p) == 3
    out = p.run([{"text": " hi "}, {"text": "  "}, {"text": "ok"}])
    assert out == [{"text": "HI"}, {"text": "OK"}], f"unexpected output {out}"
    assert p.last_run_stats == {"input": 3, "output": 2, "dropped_by_step": {"strip": 0, "drop_empty": 1, "upper": 0}}, \
        f"unexpected stats {p.last_run_stats}"


def _q046_validation(impl, tmp):
    p = impl("v").add_step("a", lambda r: r)
    try:
        p.add_step("a", lambda r: r)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate step names should raise ValueError")
    try:
        p.add_step("b", "not callable")
    except TypeError:
        pass
    else:
        raise AssertionError("non-callable steps should raise TypeError")
    assert p.steps == ["a"], "failed add_step calls must not change the pipeline"


def _q046_empty_and_generators(impl, tmp):
    p = impl("noop")
    assert p.last_run_stats is None, "last_run_stats should be None before the first run"
    assert p.run(x for x in [1, 2]) == [1, 2], "an empty pipeline returns records unchanged; run must accept any iterable"
    assert p.last_run_stats == {"input": 2, "output": 2, "dropped_by_step": {}}


def _q046_independent(impl, tmp):
    a, b = impl("a"), impl("b")
    a.add_step("x", lambda r: r)
    assert b.steps == [], "pipelines must not share their step lists"


def _q047_defaults_not_shared(impl, tmp):
    first = impl("signup")
    second = impl("login")
    assert second == {"event": "login", "payload": {}, "tags": ["generated"]}, \
        f"second call returned {second}; default containers are being shared between calls"
    first["payload"]["x"] = 1
    first["tags"].append("y")
    assert impl("again") == {"event": "again", "payload": {}, "tags": ["generated"]}, "returned containers must be fresh objects"


def _q047_copies(impl, tmp):
    payload, tags = {"plan": "pro"}, ["web"]
    event = impl("upgrade", payload, tags)
    assert event["payload"] is not payload and event["tags"] is not tags, "return copies, not the caller's objects"


def _q048_independent(impl, tmp):
    users = [{"id": 1, "name": "Asha", "contact": {"email": "asha@example.com", "phone": "555-0101"}, "roles": ["admin"]}]
    result = impl(users)
    result[0]["roles"].append("owner")
    result[0]["contact"]["email"] = "changed"
    assert users[0]["roles"] == ["admin"] and users[0]["contact"]["email"] == "asha@example.com", \
        "the result shares nested objects with the input – use a deep copy"


def _q054_fast(impl, tmp):
    import time

    events = [f"u{i}" for i in range(30_000)]
    known = [f"u{i}" for i in range(0, 60_000, 2)]
    start = time.perf_counter()
    got = impl(events, known)
    elapsed = time.perf_counter() - start
    assert got == [f"u{i}" for i in range(1, 30_000, 2)], "wrong result on the large input"
    assert elapsed < 1.0, f"took {elapsed:.2f}s for 30,000 ids – membership tests on a list are O(n)"


def _q055_sample(impl, tmp):
    from ..schema import SAMPLE_DATA_DIR

    data = json.loads((SAMPLE_DATA_DIR / "token_usage.json").read_text(encoding="utf-8"))
    report = impl(data["records"], data["prices"])
    assert sorted(report) == ["embed-small", "gpt-large", "gpt-small", "local-llm"], f"unexpected models {sorted(report)}"
    assert report["local-llm"]["cost_usd"] is None, "models without a price must have cost_usd None"
    assert report["gpt-small"]["requests"] == 24, f"gpt-small should have 24 requests, got {report['gpt-small']['requests']}"
    assert sum(r["errors"] for r in report.values()) == 6, "there are 6 error records in the sample file"


PRICES = {"gpt-small": {"prompt": 0.0005, "completion": 0.0015}, "gpt-large": {"prompt": 0.01, "completion": 0.03}}

QUESTIONS = [
    # ------------------------------------------------------------------------ Q039
    Question(
        id=39,
        title="Merge Overlapping Maintenance Windows",
        difficulty="Intermediate",
        category=ALG,
        roles=[DE, PYDEV, MLE],
        skills=["sorting by start", "interval merging", "tuple handling", "validation"],
        minutes=20,
        scenario=(
            "Several teams schedule maintenance windows for a shared GPU cluster. The scheduler "
            "needs the combined blocked periods: overlapping or touching windows should become a "
            "single window."
        ),
        problem=(
            "Write `merge_intervals(intervals)` that merges all overlapping or touching closed "
            "intervals and returns them sorted by start."
        ),
        signature="def merge_intervals(intervals: list) -> list[tuple[int, int]]:",
        entry="merge_intervals",
        inputs=["`intervals`: list of `(start, end)` pairs (tuples or 2-item lists) of ints, in any order. May be empty."],
        output="A list of `(start, end)` **tuples**, sorted by start, with no overlaps.",
        constraints=[
            "Intervals are closed: `(1, 3)` and `(3, 5)` touch and merge into `(1, 5)`.",
            "Raise `ValueError` if any interval has `start > end`.",
            "Do not modify the input.",
        ],
        criteria=["Overlapping and touching intervals merged.", "Unsorted input handled.", "Nested intervals absorbed.", "Validation."],
        hints=[
            "After sorting by start, an interval can only overlap the most recently merged one.",
            "Iterate over sorted intervals; if `start <= last_end`, extend the last merged interval "
            "with `max(last_end, end)`, otherwise start a new one.",
            "validate\nmerged = []\nfor start, end in sorted(intervals):\n    if merged and start <= merged[-1][1]:\n"
            "        merged[-1] = (merged[-1][0], max(merged[-1][1], end))\n    else: merged.append((start, end))\nreturn merged",
        ],
        solution='''
def merge_intervals(intervals: list) -> list[tuple[int, int]]:
    """Merge overlapping or touching closed intervals."""
    for start, end in intervals:
        if start > end:
            raise ValueError(f"invalid interval ({start}, {end}): start > end")
    merged: list[tuple[int, int]] = []
    for start, end in sorted((s, e) for s, e in intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
''',
        explanation=(
            "Sorting by start guarantees that any interval overlapping the current one is "
            "adjacent in sorted order, so a single linear pass suffices. `max()` handles intervals "
            "fully contained in the previous one."
        ),
        edge_cases=["`[]` → `[]`.", "Touching intervals merge.", "An interval nested inside another is absorbed."],
        mistakes=["Forgetting to sort first.", "Replacing the end with the new end instead of `max(...)` (breaks nested intervals).", "Using `<` so touching intervals stay separate."],
        tests=[
            T(([(1, 3), (2, 6), (8, 10), (15, 18)],), [(1, 6), (8, 10), (15, 18)], example=True, note="classic"),
            T(([(1, 4), (4, 5)],), [(1, 5)], example=True, note="boundary: touching intervals"),
            T(([],), [], note="empty"),
            T(([(5, 7), (1, 2)],), [(1, 2), (5, 7)], note="unsorted, no overlap"),
            T(([(1, 10), (2, 3), (4, 5)],), [(1, 10)], note="nested intervals"),
            T(([[1, 2], [2, 3]],), [(1, 3)], note="lists in, tuples out"),
            T(([(3, 1)],), raises="ValueError", note="start > end"),
        ],
        complexity="O(n log n) time for sorting, O(n) space.",
        extension="Given the merged windows and a day range, return the free (unblocked) windows.",
    ),
    # ------------------------------------------------------------------------ Q040
    Question(
        id=40,
        title="Load and Validate a Transactions CSV",
        difficulty="Intermediate",
        category=FILE,
        roles=[DE, DS_ROLE, PYDEV],
        skills=["csv.DictReader", "type conversion", "row-level validation", "math.isfinite"],
        minutes=30,
        scenario=(
            "Finance exports transactions as CSV. Some rows are broken: non-numeric or negative "
            "amounts, unsupported currencies, missing customer ids, truncated rows. The pipeline "
            "must load the good rows and report the bad ones by row number for follow-up."
        ),
        problem=(
            "Write `load_transactions(path)` that returns `(valid_rows, invalid_row_numbers)`. "
            "Valid rows are converted to clean dicts; invalid rows are reported by their 1-based "
            "data-row number."
        ),
        signature="def load_transactions(path: str) -> tuple[list[dict], list[int]]:",
        entry="load_transactions",
        inputs=[
            "`path`: UTF-8 CSV file whose header contains at least `transaction_id, customer_id, "
            "amount, currency, timestamp` (other columns are ignored).",
        ],
        output=(
            "A tuple `(valid, invalid)`. `valid` is a list of dicts with exactly the keys "
            "`transaction_id`, `customer_id`, `amount` (float), `currency` (upper case) and "
            "`timestamp`; `invalid` is a list of data-row numbers (first data row = 1)."
        ),
        constraints=[
            "Strip surrounding whitespace from every field.",
            "A row is valid only if: ids and timestamp are non-empty; amount parses as a finite "
            "float > 0; currency (upper-cased) is one of `USD`, `EUR`, `GBP`.",
            "Rows with missing values (truncated rows) or extra values are invalid.",
            "Completely blank lines are skipped and do not count as rows.",
            "Raise `ValueError` if a required column is missing from the header; let "
            "`FileNotFoundError` propagate for a missing file.",
        ],
        criteria=["Valid rows correctly typed and cleaned.", "Each invalid row reported once with the right number.", "Header validation."],
        hints=[
            "`csv.DictReader` gives you a dict per row and `reader.fieldnames` for the header check.",
            "Write a helper `_parse_row(row) -> dict | None` and use `enumerate(reader, start=1)`. "
            "Truncated rows have `None` values; extra values appear under the key `None`.",
            "with open(path, newline='', encoding='utf-8') as f:\n    reader = csv.DictReader(f)\n    check required columns\n"
            "    for n, row in enumerate(reader, start=1):\n        rec = _parse_row(row)\n        (valid if rec else invalid).append(rec or n)",
        ],
        solution='''
import csv
import math

REQUIRED_COLUMNS = ("transaction_id", "customer_id", "amount", "currency", "timestamp")
CURRENCIES = {"USD", "EUR", "GBP"}


def _parse_row(row: dict) -> dict | None:
    if None in row:  # more values than header columns
        return None
    values = {}
    for column in REQUIRED_COLUMNS:
        raw = row.get(column)
        if raw is None:  # truncated row
            return None
        values[column] = raw.strip()
    if not values["transaction_id"] or not values["customer_id"] or not values["timestamp"]:
        return None
    try:
        amount = float(values["amount"])
    except ValueError:
        return None
    if not math.isfinite(amount) or amount <= 0:
        return None
    currency = values["currency"].upper()
    if currency not in CURRENCIES:
        return None
    return {**values, "amount": amount, "currency": currency}


def load_transactions(path: str) -> tuple[list[dict], list[int]]:
    """Return (valid rows, invalid data-row numbers) from a transactions CSV."""
    valid: list[dict] = []
    invalid: list[int] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        for row_number, row in enumerate(reader, start=1):
            record = _parse_row(row)
            if record is None:
                invalid.append(row_number)
            else:
                valid.append(record)
    return valid, invalid
''',
        explanation=(
            "Separating parsing (`_parse_row`) from file handling keeps each function small and "
            "testable. `float()` accepts `'inf'` and `'nan'`, so `math.isfinite` is needed; "
            "DictReader marks truncated rows with `None` values and extra values with a `None` key."
        ),
        edge_cases=["`inf`/`nan` amounts are invalid.", "Lower-case currency is normalized.", "Header-only file → `([], [])`.", "Empty file → `ValueError` (no header)."],
        mistakes=["Catching every exception per row, hiding programming errors.", "Counting the header as row 1.", "Opening without `newline=''` (the csv docs require it)."],
        tests=[
            T((File("tx.csv", TRANSACTIONS_CSV),),
              ([{"transaction_id": "t1", "customer_id": "c1", "amount": 19.99, "currency": "USD", "timestamp": "2024-01-05T10:00:00Z"},
                {"transaction_id": "t6", "customer_id": "c4", "amount": 100.0, "currency": "EUR", "timestamp": "2024-01-05T10:05:00Z"}],
               [2, 3, 4, 5, 7, 8]), example=True, note="every kind of invalid row"),
            T((File("header_only.csv", "transaction_id,customer_id,amount,currency,timestamp\n"),), ([], []),
              example=True, note="boundary: header only"),
            T((File("extra.csv", "transaction_id,customer_id,amount,currency,timestamp,channel\n"
                                 "t1,c1,5,GBP,2024-01-01T00:00:00Z,web\n\nt2,c2,6,GBP,2024-01-01T00:00:00Z,web,EXTRA\n"),),
              ([{"transaction_id": "t1", "customer_id": "c1", "amount": 5.0, "currency": "GBP", "timestamp": "2024-01-01T00:00:00Z"}], [2]),
              note="extra column ignored, blank line skipped, extra value invalid"),
            T((File("bad_header.csv", "transaction_id,amount\nt1,5\n"),), raises="ValueError", note="missing columns"),
            T((File("empty.csv", ""),), raises="ValueError", note="empty file has no header"),
            T((TmpPath("nope.csv"),), raises="FileNotFoundError", note="missing file"),
            CT(_q040_sample, note="data/sample_data/transactions.csv: 30 valid, 8 invalid rows"),
        ],
        complexity="O(n) time; O(v) memory for v valid rows.",
        extension="Return the reason for each invalid row, e.g. `{4: 'amount must be > 0'}`.",
    ),
    # ------------------------------------------------------------------------ Q041
    Question(
        id=41,
        title="Read JSON Lines with Malformed Records",
        difficulty="Intermediate",
        category=FILE,
        roles=[DE, AIE, MLE],
        skills=["JSON Lines", "json.loads", "JSONDecodeError", "line numbering"],
        minutes=20,
        scenario=(
            "An inference service logs one JSON object per line. Crashes occasionally leave "
            "truncated lines, and a buggy client once logged JSON arrays instead of objects. "
            "Analysis jobs must skip bad lines but report where they are."
        ),
        problem=(
            "Write `read_jsonl(path)` that returns `(records, malformed_line_numbers)` for a JSON "
            "Lines file."
        ),
        signature="def read_jsonl(path: str) -> tuple[list[dict], list[int]]:",
        entry="read_jsonl",
        inputs=["`path`: a UTF-8 text file with one JSON value per line. Blank lines may appear."],
        output="`(records, bad_lines)`: the dicts in file order, and the 1-based physical line numbers of malformed lines.",
        constraints=[
            "Line numbers count every physical line, including blank ones.",
            "Whitespace-only lines are ignored (neither records nor malformed).",
            "A line is malformed if it is not valid JSON **or** is valid JSON but not an object (dict).",
            "Let `FileNotFoundError` propagate.",
        ],
        criteria=["Records parsed in order.", "Malformed line numbers exact.", "Blank lines ignored but counted."],
        hints=[
            "Use `enumerate(file, start=1)` so you always know the physical line number.",
            "Wrap only `json.loads(line)` in `try/except json.JSONDecodeError`, then check `isinstance(value, dict)`.",
            "for n, line in enumerate(f, start=1):\n    if not line.strip(): continue\n    try: value = json.loads(line)\n"
            "    except json.JSONDecodeError: bad.append(n); continue\n    if isinstance(value, dict): records.append(value) else: bad.append(n)",
        ],
        solution='''
import json


def read_jsonl(path: str) -> tuple[list[dict], list[int]]:
    """Parse a JSON Lines file, collecting the line numbers of malformed lines."""
    records: list[dict] = []
    malformed: list[int] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                malformed.append(line_number)
    return records, malformed
''',
        explanation="Catching only `JSONDecodeError` (a `ValueError` subclass) keeps unrelated bugs visible; iterating the file object streams it line by line.",
        edge_cases=["Arrays/strings/numbers are valid JSON but not records.", "Blank lines still advance the line counter.", "Surrounding whitespace is fine for `json.loads`."],
        mistakes=["`except Exception` hides I/O errors.", "Reading the whole file with `json.load` (it is not one JSON document).", "Numbering only non-blank lines."],
        tests=[
            T((File("requests.jsonl", '{"id": 1, "latency_ms": 120}\n\n{"id": 2, "latency_ms": 95}\n{bad json}\n[1, 2]\n  {"id": 3}  \n'),),
              ([{"id": 1, "latency_ms": 120}, {"id": 2, "latency_ms": 95}, {"id": 3}], [4, 5]), example=True,
              note="blank, invalid and non-object lines"),
            T((File("empty.jsonl", ""),), ([], []), example=True, note="boundary: empty file"),
            T((File("truncated.jsonl", '{"id": 1}\n{"id": 2, "text": "unfinish'),), ([{"id": 1}], [2]), note="truncated last line"),
            T((File("scalars.jsonl", '42\n"text"\nnull\n'),), ([], [1, 2, 3]), note="JSON scalars are not records"),
            T((TmpPath("missing.jsonl"),), raises="FileNotFoundError", note="missing file"),
            CT(_q041_sample, note="data/sample_data/inference_requests.jsonl"),
        ],
        complexity="O(n) time in file size; O(r) memory for r records.",
        extension="Turn it into a generator that yields records and logs malformed lines, so huge files never sit in memory.",
    ),
    # ------------------------------------------------------------------------ Q042
    Question(
        id=42,
        title="Write a JSON Report Atomically",
        difficulty="Intermediate",
        category=FILE,
        roles=[DE, MLE, PYDEV],
        skills=["pathlib", "tempfile", "os.replace", "cleanup on failure", "json.dumps options"],
        minutes=25,
        scenario=(
            "A nightly job writes `metrics.json`, which a dashboard reads every minute. If the job "
            "crashes halfway through writing, the dashboard reads a truncated file and breaks. "
            "Writing to a temporary file and atomically renaming it fixes this."
        ),
        problem=(
            "Write `write_json_atomic(path, data)` that serializes `data` as pretty JSON and "
            "replaces `path` atomically, creating parent directories if needed."
        ),
        signature="def write_json_atomic(path: str, data) -> None:",
        entry="write_json_atomic",
        inputs=["`path`: destination file path (parent folders may not exist yet).", "`data`: a JSON-serializable value (or not – see rules)."],
        output=(
            "Returns `None`. The file contains `json.dumps(data, indent=2, sort_keys=True, "
            "ensure_ascii=False)` followed by a newline, encoded as UTF-8."
        ),
        constraints=[
            "Create missing parent directories.",
            "Write to a temporary file **in the same directory**, then `os.replace()` it onto `path`.",
            "If serialization fails (e.g. a `set` inside `data`), raise the `TypeError`, leave any "
            "existing file untouched, and leave no temporary files behind.",
        ],
        criteria=["Correct file content.", "Existing file replaced.", "Failure leaves the old file and no leftovers.", "Unicode preserved."],
        hints=[
            "Serialize first – if `json.dumps` fails you have not touched the disk yet.",
            "`tempfile.mkstemp(dir=parent)` creates a temp file on the same filesystem, which is what "
            "makes `os.replace` atomic. Clean it up in an `except` block if anything fails.",
            "text = json.dumps(...) + '\\n'\nparent.mkdir(parents=True, exist_ok=True)\nfd, tmp = mkstemp(dir=parent)\n"
            "try: write text via os.fdopen(fd, 'w', encoding='utf-8'); os.replace(tmp, path)\nexcept BaseException: remove tmp; raise",
        ],
        solution='''
import json
import os
import tempfile
from pathlib import Path


def write_json_atomic(path: str, data) -> None:
    """Write pretty JSON to ``path`` via a temp file + atomic rename."""
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
''',
        explanation=(
            "Serializing before touching the disk means the most common failure (non-serializable "
            "data) never creates a file. Writing to a temp file in the same directory and calling "
            "`os.replace` means readers see either the complete old file or the complete new one. "
            "`fsync` makes the content durable before the rename."
        ),
        edge_cases=["Parent directories created on demand.", "Existing file overwritten atomically.", "Non-serializable data → `TypeError`, nothing changed."],
        mistakes=["Writing directly to `path` (readers can see partial content).", "Creating the temp file in `/tmp` (rename across filesystems is not atomic and may fail).", "Leaving `.tmp` files behind after errors."],
        tests=[
            T((TmpPath("reports/summary.json"), {"b": 1, "a": [1, 2]}), None, example=True,
              note="returns None; file holds sorted, indented JSON"),
            CT(_q042_creates_dirs, note="creates parent folders and writes the exact format"),
            CT(_q042_overwrites, note="overwrites and leaves no temp files"),
            CT(_q042_failure_is_atomic, note="failure keeps the old file and cleans up"),
            CT(_q042_unicode, note="non-ASCII text preserved"),
        ],
        complexity="O(size of the JSON) time and memory.",
        extension="Keep the previous version as `metrics.json.bak` before replacing it.",
    ),
    # ------------------------------------------------------------------------ Q043
    Question(
        id=43,
        title="Load a Model Config with Clear, Chained Errors",
        difficulty="Intermediate",
        category=FILE,
        roles=[MLE, PYDEV, AIE],
        skills=["custom exceptions", "exception chaining (`raise ... from`)", "type validation", "defaults"],
        minutes=25,
        scenario=(
            "Training jobs read a JSON config. When it is missing or wrong, engineers currently see "
            "a raw `KeyError: 'batch_size'` deep in the training loop. You want one domain-specific "
            "exception type with a helpful message that still preserves the original cause."
        ),
        problem=(
            "Define `class ConfigError(Exception)` and write `load_config(path)` that loads and "
            "validates a JSON config, raising `ConfigError` for every problem."
        ),
        signature="def load_config(path: str) -> dict:",
        entry="load_config",
        starter='''
class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


def load_config(path: str) -> dict:
    # TODO: write your solution here
    pass
''',
        inputs=["`path`: path to a JSON file that should contain an object."],
        output="The config `dict` with all original keys plus `seed` defaulted to 42 when absent.",
        constraints=[
            "Missing file → `ConfigError` whose message contains the path, chained `from` the `FileNotFoundError`.",
            "Invalid JSON → `ConfigError` chained `from` the `json.JSONDecodeError`.",
            "Top-level value not an object → `ConfigError`.",
            "Required: `model_name` non-empty `str`; `batch_size` `int` > 0; `learning_rate` `int`/`float` > 0. "
            "`bool` is never accepted as a number.",
            "Optional `seed` must be an `int` (not `bool`) if present.",
            "Other keys are kept unchanged.",
        ],
        criteria=["Valid configs load with defaults.", "Every failure is a `ConfigError` with a useful message.", "Original exceptions are chained."],
        hints=[
            "Translate low-level exceptions at the boundary: catch them where they occur and raise your own.",
            "`raise ConfigError(f'config file not found: {path}') from exc` sets `__cause__`. Write a "
            "tiny helper to check 'is a real number' that rejects `bool`.",
            "try: text = open(...).read() except FileNotFoundError as exc: raise ConfigError(...) from exc\n"
            "try: data = json.loads(text) except json.JSONDecodeError as exc: raise ConfigError(...) from exc\n"
            "check dict, then each field\nreturn {'seed': 42, **data}",
        ],
        solution='''
import json


class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_config(path: str) -> dict:
    """Load and validate a training config, raising ConfigError on any problem."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid JSON: {exc.msg} (line {exc.lineno})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a JSON object, got {type(data).__name__}")

    name = data.get("model_name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("'model_name' is required and must be a non-empty string")
    batch_size = data.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ConfigError(f"'batch_size' must be a positive integer, got {batch_size!r}")
    learning_rate = data.get("learning_rate")
    if not _is_number(learning_rate) or learning_rate <= 0:
        raise ConfigError(f"'learning_rate' must be a positive number, got {learning_rate!r}")
    seed = data.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"'seed' must be an integer, got {seed!r}")
    return {**data, "seed": seed}
''',
        explanation=(
            "Each low-level failure is caught exactly where it can occur and re-raised as "
            "`ConfigError` with context; `from exc` keeps the original traceback for debugging. "
            "Field checks produce messages that name the offending key."
        ),
        edge_cases=["`batch_size: true` is rejected.", "`learning_rate: 1` (int) is accepted.", "Extra keys like `epochs` pass through."],
        mistakes=["Letting `KeyError`/`FileNotFoundError` escape.", "`raise ConfigError(...)` without `from`, losing the cause.", "Accepting `True` as a batch size."],
        tests=[
            T((File("cfg.json", '{"model_name": "churn-classifier", "batch_size": 64, "learning_rate": 0.001, "epochs": 20}'),),
              {"model_name": "churn-classifier", "batch_size": 64, "learning_rate": 0.001, "epochs": 20, "seed": 42},
              example=True, note="valid config, default seed"),
            T((File("cfg.json", '{"model_name": "m", "batch_size": 0, "learning_rate": 0.1}'),), raises="ConfigError",
              example=True, note="boundary: batch_size must be > 0"),
            T((File("cfg.json", '{"model_name": "m", "batch_size": 8, "learning_rate": 1, "seed": 7}'),),
              {"model_name": "m", "batch_size": 8, "learning_rate": 1, "seed": 7}, note="int learning rate, explicit seed"),
            T((File("cfg.json", '{"model_name": "m", "batch_size": true, "learning_rate": 0.1}'),), raises="ConfigError", note="bool batch size"),
            T((File("cfg.json", '{"batch_size": 8, "learning_rate": 0.1}'),), raises="ConfigError", note="missing model_name"),
            T((File("cfg.json", '{"model_name": "m", "batch_size": 8, "learning_rate": "fast"}'),), raises="ConfigError", note="wrong type"),
            T((File("cfg.json", '{"model_name": "m", "batch_size": 8, "learning_rate": 0.1, "seed": "x"}'),), raises="ConfigError", note="bad seed"),
            T((File("cfg.json", '[1, 2]'),), raises="ConfigError", note="top level is a list"),
            CT(_q043_cause_missing, note="missing file chained from FileNotFoundError"),
            CT(_q043_cause_json, note="invalid JSON chained from JSONDecodeError"),
            T((DataFile("model_config.json"),),
              {"model_name": "churn-classifier", "batch_size": 64, "learning_rate": 0.001, "epochs": 20,
               "features": ["tenure_months", "monthly_spend", "support_tickets"], "seed": 42},
              note="data/sample_data/model_config.json"),
        ],
        complexity="O(file size) time and memory.",
        extension="Collect *all* validation problems and raise one `ConfigError` listing every issue.",
    ),
    # ------------------------------------------------------------------------ Q044
    Question(
        id=44,
        title="RunningStats: an Online Statistics Class",
        difficulty="Intermediate",
        category=OOP,
        roles=[MLE, DS_ROLE, PYDEV],
        skills=["classes and instances", "read-only properties", "encapsulation", "Welford's algorithm"],
        minutes=30,
        scenario=(
            "A feature-monitoring service tracks the mean and variance of each numeric feature "
            "over a live stream. Values arrive one at a time and cannot be stored. Some features "
            "have large magnitudes (e.g. timestamps), where the textbook sum-of-squares formula "
            "loses precision."
        ),
        problem=(
            "Implement a class `RunningStats` with an `add(value)` method and read-only properties "
            "`count`, `mean`, `min`, `max` and `variance` (population variance)."
        ),
        signature="class RunningStats:",
        entry="RunningStats",
        starter='''
class RunningStats:
    """Online count, mean, min, max and population variance."""

    def __init__(self):
        # TODO: initialise private state
        pass

    def add(self, value: float) -> None:
        # TODO
        pass

    # TODO: add read-only properties: count, mean, min, max, variance
''',
        inputs=["`add(value)` receives ints or floats, one at a time. `bool` and other types are invalid."],
        output="Properties: `count` (int), `mean`, `min`, `max`, `variance` – the last four are `None` while empty.",
        constraints=[
            "Use O(1) memory: do not store the values.",
            "`variance` is the population variance (divide by n).",
            "Raise `TypeError` from `add` for non-numbers and `bool`; rejected values are not counted.",
            "Properties are read-only (assigning to them raises `AttributeError`).",
            "Instances must not share state.",
            "Results must be numerically stable for large values (Welford's method).",
        ],
        criteria=["Correct statistics.", "None while empty.", "Validation and read-only properties.", "Stable variance."],
        hints=[
            "Store a few running quantities in private attributes and expose them through `@property`.",
            "Welford's update: `delta = x - mean; mean += delta / n; m2 += delta * (x - mean)`. "
            "Variance is `m2 / n`.",
            "__init__: _count = 0; _mean = 0.0; _m2 = 0.0; _min = _max = None\nadd: validate; update count, mean, m2, min, max\n"
            "@property mean: return None if _count == 0 else _mean   (same pattern for the others)",
        ],
        solution='''
class RunningStats:
    """Online count, mean, min, max and population variance (Welford)."""

    def __init__(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = None
        self._max = None

    def add(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"value must be a number, got {type(value).__name__}")
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (value - self._mean)
        self._min = value if self._min is None else min(self._min, value)
        self._max = value if self._max is None else max(self._max, value)

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float | None:
        return self._mean if self._count else None

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def variance(self) -> float | None:
        return self._m2 / self._count if self._count else None

    def __repr__(self) -> str:
        return f"RunningStats(count={self._count}, mean={self.mean})"
''',
        explanation=(
            "Private attributes hold the state and properties expose it read-only. Welford's "
            "algorithm updates the mean and the sum of squared deviations incrementally, avoiding "
            "the catastrophic cancellation of `E[x²] − E[x]²` for large values."
        ),
        edge_cases=["Empty → `None` statistics.", "Single value → variance 0.", "Large offsets (1e9) keep precision."],
        mistakes=["Storing all values in a list.", "Using class attributes such as `values = []` (shared across instances).", "`sum(x**2)/n - mean**2` (unstable)."],
        tests=[
            CT(_q044_basic, note="statistics for [4, 8, 6] and empty state"),
            CT(_q044_types, note="TypeError for strings, bools and None"),
            CT(_q044_read_only, note="properties are read-only"),
            CT(_q044_stable, note="numerically stable variance"),
            CT(_q044_independent, note="instances are independent"),
        ],
        examples=['''
>>> stats = RunningStats()
>>> for value in [4, 8, 6]:
...     stats.add(value)
>>> stats.count, stats.mean, stats.min, stats.max
(3, 6.0, 4, 8)
>>> round(stats.variance, 4)
2.6667
>>> RunningStats().mean is None
True
'''],
        complexity="O(1) time per `add` and O(1) memory.",
        extension="Add `merge(other)` that combines two RunningStats objects (parallel Welford/Chan formula).",
    ),
    # ------------------------------------------------------------------------ Q045
    Question(
        id=45,
        title="A Comparable Semantic Version Class",
        difficulty="Intermediate",
        category=OOP,
        roles=[MLE, PYDEV, AIE],
        skills=["special methods", "functools.total_ordering", "__hash__/__eq__ contract", "NotImplemented"],
        minutes=30,
        scenario=(
            "A model registry stores versions such as `1.2.10` and `v1.10.0`. Sorting them as "
            "strings puts `1.10.0` before `1.2.3`, so the deploy script once rolled back to an old "
            "model. You will build a small value class that compares versions correctly."
        ),
        problem=(
            "Implement `Version(text)` that parses `MAJOR.MINOR.PATCH` (optional leading `v`), "
            "exposes `major`, `minor`, `patch`, supports equality, ordering, hashing, `str`, "
            "`repr`, and `bump(part)`."
        ),
        signature="class Version:",
        entry="Version",
        starter='''
class Version:
    """Semantic version MAJOR.MINOR.PATCH with numeric comparison."""

    def __init__(self, text: str):
        # TODO
        pass
''',
        inputs=["`text`: a string like `'1.4.2'` or `'v2.0.1'` (surrounding whitespace allowed)."],
        output=(
            "A `Version` object. `str(v)` → `'1.4.2'`; `repr(v)` → `\"Version('1.4.2')\"`; "
            "`v.bump(part)` returns a **new** Version."
        ),
        constraints=[
            "Exactly three non-negative integer parts are required; otherwise raise `ValueError`.",
            "Compare numerically: `(major, minor, patch)` tuples.",
            "`Version('1.2.3') == Version('v1.2.3')`; equal versions have equal hashes.",
            "Comparing with a non-Version: `==` returns `False`; `<` raises `TypeError` (return `NotImplemented`).",
            "`bump('major')` → `X+1.0.0`, `bump('minor')` → `X.Y+1.0`, `bump('patch')` → `X.Y.Z+1`; other parts → `ValueError`.",
        ],
        criteria=["Parsing and validation.", "Numeric ordering via special methods.", "Hash/eq consistency.", "Immutable-style bump."],
        hints=[
            "Most comparison logic can be delegated to tuples.",
            "Parse with `re.fullmatch(r'v?(\\d+)\\.(\\d+)\\.(\\d+)', text.strip())`. Implement `__eq__`, "
            "`__lt__` and `__hash__` using a `_key()` tuple, and decorate the class with `@functools.total_ordering`.",
            "__eq__/__lt__: if not isinstance(other, Version): return NotImplemented; compare self._key() with other._key()\n"
            "__hash__: return hash(self._key())\nbump: build a new Version from an f-string",
        ],
        solution='''
import re
from functools import total_ordering

_VERSION = re.compile(r"v?(\\d+)\\.(\\d+)\\.(\\d+)")


@total_ordering
class Version:
    """Semantic version MAJOR.MINOR.PATCH with numeric comparison."""

    def __init__(self, text: str):
        match = _VERSION.fullmatch(text.strip()) if isinstance(text, str) else None
        if not match:
            raise ValueError(f"invalid version string: {text!r}")
        self.major, self.minor, self.patch = (int(part) for part in match.groups())

    def _key(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() == other._key()

    def __lt__(self, other) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() < other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return f"Version('{self}')"

    def bump(self, part: str) -> "Version":
        if part == "major":
            return Version(f"{self.major + 1}.0.0")
        if part == "minor":
            return Version(f"{self.major}.{self.minor + 1}.0")
        if part == "patch":
            return Version(f"{self.major}.{self.minor}.{self.patch + 1}")
        raise ValueError(f"part must be 'major', 'minor' or 'patch', got {part!r}")
''',
        explanation=(
            "Tuples already compare lexicographically, so `_key()` gives correct numeric ordering. "
            "`total_ordering` derives `<=`, `>`, `>=` from `__eq__` and `__lt__`. Defining `__eq__` "
            "removes the default hash, so `__hash__` must be defined explicitly to keep instances "
            "usable in sets and dict keys."
        ),
        edge_cases=["`v` prefix and whitespace accepted.", "`1.2` or `1.2.3.4` rejected.", "Comparison with strings."],
        mistakes=["Comparing version strings directly.", "Raising `TypeError` from `__eq__` instead of returning `NotImplemented`.", "Defining `__eq__` without `__hash__` (instances become unhashable)."],
        tests=[
            CT(_q045_parse, note="parse parts, str and repr"),
            CT(_q045_invalid, note="invalid strings rejected"),
            CT(_q045_ordering, note="numeric ordering"),
            CT(_q045_equality_hash, note="equality, hashing, foreign types"),
            CT(_q045_bump, note="bump returns new versions"),
        ],
        examples=['''
>>> sorted([Version("1.10.0"), Version("1.2.3"), Version("v1.2.10")])
[Version('1.2.3'), Version('1.2.10'), Version('1.10.0')]
>>> str(Version("2.3.4").bump("minor"))
'2.4.0'
>>> Version("1.2.3") == Version("v1.2.3")
True
'''],
        complexity="O(1) per comparison; sorting n versions is O(n log n).",
        extension="Support pre-release tags (`1.0.0-rc.1`), which sort before the release.",
    ),
    # ------------------------------------------------------------------------ Q046
    Question(
        id=46,
        title="A Composable Record-Processing Pipeline Class",
        difficulty="Intermediate",
        category=OOP,
        roles=[DE, MLE, AIE],
        skills=["composition", "method chaining", "properties", "per-run statistics"],
        minutes=30,
        scenario=(
            "Data-cleaning code for a text classifier has grown into a long function. You want a "
            "small `Pipeline` object that holds named steps; each step receives a record and "
            "returns the transformed record, or `None` to drop it. Operators also want to know "
            "which step dropped how many records."
        ),
        problem=(
            "Implement `Pipeline(name)` with `add_step(name, func)` (chainable), a `steps` "
            "property, `len()`, `run(records)` and a `last_run_stats` attribute."
        ),
        signature="class Pipeline:",
        entry="Pipeline",
        starter='''
class Pipeline:
    """Ordered, named record-processing steps."""

    def __init__(self, name: str):
        # TODO
        pass

    def add_step(self, name: str, func) -> "Pipeline":
        # TODO
        pass

    def run(self, records) -> list:
        # TODO
        pass
''',
        inputs=[
            "`add_step(name, func)`: `name` is a str unique within the pipeline; `func` takes one record and returns a record or `None`.",
            "`run(records)`: any iterable of records (list, generator, ...).",
        ],
        output=(
            "`run` returns a list of records that passed all steps. `last_run_stats` is "
            "`{'input': int, 'output': int, 'dropped_by_step': {step_name: int}}` (or `None` before the first run)."
        ),
        constraints=[
            "`add_step` returns the pipeline itself; duplicate names → `ValueError`; non-callables → `TypeError`.",
            "`steps` is the list of step names in insertion order; `len(p)` is the number of steps.",
            "Steps run in insertion order; once a step returns `None`, later steps are skipped for that record.",
            "`dropped_by_step` contains every step name (0 if it dropped nothing).",
            "Pipelines must not share state.",
        ],
        criteria=["Chaining and validation.", "Correct output order.", "Accurate stats.", "Independent instances."],
        hints=[
            "Composition: the pipeline *has* a list of `(name, func)` pairs.",
            "In `run`, loop over records; inside, loop over steps and `break` when a step returns `None`. "
            "A `for ... else` tells you when no step dropped the record.",
            "for record in records:\n    total += 1; current = record\n    for name, func in self._steps:\n        current = func(current)\n"
            "        if current is None: dropped[name] += 1; break\n    else: output.append(current)",
        ],
        solution='''
class Pipeline:
    """Ordered, named record-processing steps."""

    def __init__(self, name: str):
        self.name = name
        self._steps: list[tuple[str, object]] = []
        self.last_run_stats: dict | None = None

    def add_step(self, name: str, func) -> "Pipeline":
        if not callable(func):
            raise TypeError(f"step {name!r} must be callable")
        if name in self.steps:
            raise ValueError(f"duplicate step name {name!r}")
        self._steps.append((name, func))
        return self

    @property
    def steps(self) -> list[str]:
        return [name for name, _ in self._steps]

    def __len__(self) -> int:
        return len(self._steps)

    def run(self, records) -> list:
        output = []
        dropped = {name: 0 for name, _ in self._steps}
        total = 0
        for record in records:
            total += 1
            current = record
            for name, func in self._steps:
                current = func(current)
                if current is None:
                    dropped[name] += 1
                    break
            else:
                output.append(current)
        self.last_run_stats = {"input": total, "output": len(output), "dropped_by_step": dropped}
        return output

    def __repr__(self) -> str:
        return f"Pipeline({self.name!r}, steps={self.steps})"
''',
        explanation=(
            "The class composes plain functions instead of requiring a class hierarchy of step "
            "types. Returning `self` enables fluent chaining; validation happens before mutation "
            "so failed calls leave the pipeline unchanged. `for ... else` runs the `else` branch "
            "only when the inner loop did not `break`."
        ),
        edge_cases=["Empty pipeline returns records unchanged.", "Generator input counted correctly.", "Failed `add_step` leaves steps unchanged."],
        mistakes=["Class-level `_steps = []` shared by all pipelines.", "Continuing to call later steps with `None`.", "Using `len(records)` (fails for generators)."],
        tests=[
            CT(_q046_chain_and_run, note="chaining, run output and stats"),
            CT(_q046_validation, note="duplicate and non-callable steps"),
            CT(_q046_empty_and_generators, note="empty pipeline, generator input"),
            CT(_q046_independent, note="instances are independent"),
        ],
        examples=['''
>>> p = (Pipeline("clean")
...      .add_step("strip", lambda r: r.strip())
...      .add_step("drop_empty", lambda r: r or None))
>>> p.run(["  hi ", "   ", "ok"])
['hi', 'ok']
>>> p.last_run_stats
{'input': 3, 'output': 2, 'dropped_by_step': {'strip': 0, 'drop_empty': 1}}
'''],
        complexity="O(n · s) for n records and s steps (plus the steps' own cost).",
        extension="Catch exceptions inside steps and record them per step instead of aborting the run.",
    ),
    # ------------------------------------------------------------------------ Q047
    Question(
        id=47,
        title="Fix the Mutable Default Argument Bug",
        difficulty="Intermediate",
        category=DBG,
        roles=[PYDEV, DE, AIE],
        skills=["mutable default arguments", "copying containers", "defensive programming"],
        minutes=15,
        scenario=(
            "An analytics SDK builds events with this helper:\n\n"
            "```python\ndef make_event(name, payload={}, tags=[]):\n    tags.append(\"generated\")\n"
            "    return {\"event\": name, \"payload\": payload, \"tags\": tags}\n```\n\n"
            "After a few calls, events carry `['generated', 'generated', 'generated']`, and "
            "callers find their own `tags` lists modified."
        ),
        problem=(
            "Write a corrected `make_event(name, payload=None, tags=None)` that returns "
            "`{'event': name, 'payload': <copy of payload>, 'tags': <copy of tags> + ['generated']}` "
            "without sharing state between calls or modifying the caller's objects."
        ),
        signature="def make_event(name: str, payload: dict | None = None, tags=None) -> dict:",
        entry="make_event",
        inputs=["`name`: non-empty event name.", "`payload`: dict or `None`.", "`tags`: list/tuple of strings or `None`."],
        output="`{'event': str, 'payload': dict, 'tags': list}`.",
        constraints=[
            "Use `None` as the default and create fresh containers inside the function.",
            "Never modify the caller's `payload` or `tags`; the returned containers are new objects.",
            "`tags` may be a tuple; the result's `tags` is always a list ending in `'generated'`.",
            "Raise `ValueError` for an empty `name`.",
        ],
        criteria=["No state shared between calls.", "Inputs untouched.", "Returned containers are copies."],
        hints=[
            "Default values are evaluated **once**, when the `def` statement runs.",
            "Use `None` defaults and build new objects: `dict(payload or {})`, `[*(tags or []), 'generated']`.",
            "if not name: raise ValueError\nreturn {'event': name, 'payload': dict(payload) if payload else {},\n"
            "        'tags': [*(tags or ()), 'generated']}",
        ],
        solution='''
def make_event(name: str, payload: dict | None = None, tags=None) -> dict:
    """Build an analytics event without sharing mutable state."""
    if not name:
        raise ValueError("name must be a non-empty string")
    return {
        "event": name,
        "payload": dict(payload) if payload is not None else {},
        "tags": [*(tags if tags is not None else ()), "generated"],
    }
''',
        explanation=(
            "Python evaluates default argument values once at definition time, so a `[]` default "
            "is one list shared by every call. `None` defaults plus fresh containers per call fix "
            "the leak, and copying the caller's containers avoids modifying them."
        ),
        edge_cases=["Repeated default calls stay independent.", "Tuple `tags` converted to list.", "Empty name rejected."],
        mistakes=["Keeping `tags=[]` but copying later (the default is still shared if ever mutated).", "`tags.append(...)` on the caller's list.", "Returning the caller's payload object."],
        tests=[
            T(("signup",), {"event": "signup", "payload": {}, "tags": ["generated"]}, example=True, note="defaults"),
            T(("signup", {"plan": "pro"}, ["web"]), {"event": "signup", "payload": {"plan": "pro"}, "tags": ["web", "generated"]},
              example=True, note="caller's objects must not be modified"),
            T(("x", None, ("a",)), {"event": "x", "payload": {}, "tags": ["a", "generated"]}, note="tuple tags"),
            CT(_q047_defaults_not_shared, note="no state shared between calls"),
            CT(_q047_copies, note="returns copies of the caller's containers"),
            T(("",), raises="ValueError", note="empty name"),
        ],
        complexity="O(p + t) to copy the payload and tags.",
        extension="Write a pytest test that fails on the original buggy version and passes on yours.",
    ),
    # ------------------------------------------------------------------------ Q048
    Question(
        id=48,
        title="Anonymize Users Without Corrupting the Source (Deep Copy)",
        difficulty="Intermediate",
        category=DBG,
        roles=[DE, DS_ROLE, PYDEV],
        skills=["shallow vs deep copy", "nested mutation bugs", "string partitioning"],
        minutes=20,
        scenario=(
            "Before sharing user data with an analytics vendor, a script anonymizes it. The "
            "original implementation used `user.copy()` and then edited `copy['contact']['email']`"
            " – which silently overwrote the emails in the production records as well."
        ),
        problem=(
            "Write `anonymize_users(users)` that returns anonymized copies of the user records "
            "and never modifies (or shares nested objects with) the input."
        ),
        signature="def anonymize_users(users: list[dict]) -> list[dict]:",
        entry="anonymize_users",
        inputs=[
            "`users`: list of dicts with `id`, `name`, an optional `contact` dict (with optional "
            "`email` and `phone`, possibly `None`) and other fields such as `roles` (list).",
        ],
        output="A new list of new dicts with the same structure.",
        constraints=[
            "`name` (if present) → `'REDACTED'`.",
            "`contact.email`: keep the first character of the local part, then `***@domain`, e.g. "
            "`asha.rao@example.com` → `a***@example.com`. `None` stays `None`; a value without `@` "
            "or with an empty local part → `'***'`.",
            "`contact.phone` (if present) → `None`.",
            "All other fields and keys are kept, but the result must not share any mutable objects with the input.",
        ],
        criteria=["Correct masking.", "Input unchanged.", "No shared nested objects."],
        hints=[
            "`dict.copy()` copies only the top level; nested dicts and lists are still shared.",
            "`copy.deepcopy(user)` gives a fully independent structure you can edit freely. Use "
            "`email.partition('@')` to split the address.",
            "for user in users:\n    clean = copy.deepcopy(user)\n    if 'name' in clean: clean['name'] = 'REDACTED'\n"
            "    contact = clean.get('contact')\n    if contact: mask email, set phone to None\n    result.append(clean)",
        ],
        solution='''
import copy


def _mask_email(email: str | None) -> str | None:
    if email is None:
        return None
    local, at, domain = email.partition("@")
    if not at or not local:
        return "***"
    return f"{local[0]}***@{domain}"


def anonymize_users(users: list[dict]) -> list[dict]:
    """Return anonymized deep copies of user records."""
    result = []
    for user in users:
        clean = copy.deepcopy(user)
        if "name" in clean:
            clean["name"] = "REDACTED"
        contact = clean.get("contact")
        if isinstance(contact, dict):
            if "email" in contact:
                contact["email"] = _mask_email(contact["email"])
            if "phone" in contact:
                contact["phone"] = None
        result.append(clean)
    return result
''',
        explanation=(
            "A shallow copy duplicates only the outer dict; `contact` and `roles` would still be "
            "the same objects as in the input. `copy.deepcopy` recursively copies everything, so "
            "edits to the result can never leak back."
        ),
        edge_cases=["User without `contact`.", "`email` is `None`.", "Malformed email `'no-at-sign'` → `'***'`."],
        mistakes=["`user.copy()` or `{**user}` followed by nested edits.", "Rebuilding dicts by hand and forgetting unknown keys.", "Crashing on `None` email."],
        tests=[
            T(([{"id": 1, "name": "Asha Rao", "contact": {"email": "asha.rao@example.com", "phone": "555-0101"}, "roles": ["admin"]},
                {"id": 2, "name": "Ben Ode", "contact": {"email": None, "phone": None}, "roles": []}],),
              [{"id": 1, "name": "REDACTED", "contact": {"email": "a***@example.com", "phone": None}, "roles": ["admin"]},
               {"id": 2, "name": "REDACTED", "contact": {"email": None, "phone": None}, "roles": []}],
              example=True, note="input must stay unchanged"),
            T(([],), [], example=True, note="boundary: no users"),
            T(([{"id": 3, "name": "C", "plan": "free"}],), [{"id": 3, "name": "REDACTED", "plan": "free"}], note="no contact"),
            T(([{"id": 4, "contact": {"email": "no-at-sign", "slack": "@c"}}],),
              [{"id": 4, "contact": {"email": "***", "slack": "@c"}}], note="malformed email, extra contact key"),
            CT(_q048_independent, note="no nested objects shared with the input"),
        ],
        complexity="O(total size of the records) time and space.",
        extension="Make the masking rules configurable: `{'contact.email': mask_email, 'name': 'REDACTED'}`.",
    ),
    # ------------------------------------------------------------------------ Q049
    Question(
        id=49,
        title="Fix Sorting of Mixed-Type Numeric IDs",
        difficulty="Intermediate",
        category=DBG,
        roles=[DE, PYDEV, DS_ROLE],
        skills=["type assumptions", "key functions", "explicit conversion", "error messages"],
        minutes=15,
        scenario=(
            "Records merged from two systems have ids as ints (`3`) or strings (`'007'`, `' 12'`). "
            "The old code `sorted(records, key=lambda r: r['id'])` crashes with `TypeError` on "
            "mixed types, and when all ids are strings it sorts `'10'` before `'9'`."
        ),
        problem="Write `sort_by_numeric_id(records)` that sorts records by the integer value of their `id`.",
        signature="def sort_by_numeric_id(records: list[dict]) -> list[dict]:",
        entry="sort_by_numeric_id",
        inputs=["`records`: list of dicts, each with an `id` that is an `int` or a string containing an integer (whitespace and leading zeros allowed)."],
        output="A new list sorted ascending by numeric id; records with equal ids keep their original order.",
        constraints=[
            "Raise `ValueError` mentioning the bad value if an id is not an integer or integer string "
            "(floats, `bool` and text such as `'abc'` are invalid).",
            "Do not modify the input.",
        ],
        criteria=["Numeric ordering across types.", "Stable for equal values.", "Clear error for invalid ids."],
        hints=[
            "Convert every id to the same type before comparing.",
            "Write `_to_int(value)` that accepts `int` (not `bool`) and strings parsable by `int()`; "
            "use it as the sort key.",
            "def _to_int(v):\n    if isinstance(v, bool): raise ValueError\n    if isinstance(v, int): return v\n"
            "    if isinstance(v, str): try: return int(v) except ValueError: pass\n    raise ValueError(...)\nreturn sorted(records, key=lambda r: _to_int(r['id']))",
        ],
        solution='''
def _to_int(value) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    raise ValueError(f"id {value!r} is not an integer")


def sort_by_numeric_id(records: list[dict]) -> list[dict]:
    """Sort records by the integer value of their id (stable)."""
    return sorted(records, key=lambda record: _to_int(record["id"]))
''',
        explanation=(
            "The key function normalizes every id to `int`, so comparisons are numeric and "
            "mixed types never meet. `int()` already ignores surrounding whitespace and leading "
            "zeros. `sorted` is stable, preserving the order of equal ids."
        ),
        edge_cases=["`' 007 '` → 7.", "Equal numeric ids keep input order.", "`3.0` and `True` are rejected."],
        mistakes=["`key=str` (lexicographic order).", "`key=int` crashes on whitespace? (no – but it accepts `True` and floats like `3.9` → 3).", "Silently dropping bad records."],
        tests=[
            T(([{"id": "10"}, {"id": "9"}, {"id": 3}, {"id": " 007 "}],), [{"id": 3}, {"id": " 007 "}, {"id": "9"}, {"id": "10"}],
              example=True, note="mixed types and string ordering"),
            T(([],), [], example=True, note="boundary: empty"),
            T(([{"id": "5", "v": "a"}, {"id": 5, "v": "b"}, {"id": "05", "v": "c"}],),
              [{"id": "5", "v": "a"}, {"id": 5, "v": "b"}, {"id": "05", "v": "c"}], note="stable for equal ids"),
            T(([{"id": "-2"}, {"id": 1}],), [{"id": "-2"}, {"id": 1}], note="negative ids"),
            T(([{"id": "abc"}],), raises="ValueError", note="non-numeric id"),
            T(([{"id": 3.0}],), raises="ValueError", note="float id"),
            T(([{"id": True}],), raises="ValueError", note="bool id"),
        ],
        complexity="O(n log n) time, O(n) space.",
        extension="Support ids like `'INV-0042'` by extracting the trailing number with a regex.",
    ),
    # ------------------------------------------------------------------------ Q050
    Question(
        id=50,
        title="Pivot Records into a Nested Summary Table",
        difficulty="Intermediate",
        category=DATA,
        roles=[DS_ROLE, DE, PYDEV],
        skills=["two-level grouping", "aggregation functions", "skipping incomplete rows"],
        minutes=25,
        scenario=(
            "A revenue report needs a region × month table. You do not want to pull in pandas for "
            "one small job, so you will implement the pivot with plain dictionaries."
        ),
        problem=(
            "Write `pivot(records, index, columns, values, agg='sum')` that returns "
            "`{index_value: {column_value: aggregate}}`."
        ),
        signature="def pivot(records: list[dict], index: str, columns: str, values: str, agg: str = \"sum\") -> dict:",
        entry="pivot",
        inputs=["`records`: list of dicts.", "`index`, `columns`, `values`: field names.", "`agg`: `'sum'`, `'count'` or `'mean'`."],
        output="A nested dict. Only combinations that occur appear (no zero filling).",
        constraints=[
            "Skip records where any of the three fields is missing or `None`.",
            "`count` counts records; `mean` is sum/count (unrounded).",
            "Raise `ValueError` for an unknown `agg`.",
            "Do not modify the input.",
        ],
        criteria=["Correct grouping on two keys.", "All three aggregations.", "Incomplete records skipped."],
        hints=[
            "Group values into lists per `(row, column)` first, aggregate afterwards.",
            "`groups.setdefault(row, {}).setdefault(col, []).append(value)` builds the nested structure.",
            "validate agg\nfor r in records: skip incomplete; append value to groups[row][col]\n"
            "return {row: {col: AGG(vals) for col, vals in cols.items()} for row, cols in groups.items()}",
        ],
        solution='''
AGGREGATIONS = {
    "sum": sum,
    "count": len,
    "mean": lambda values: sum(values) / len(values),
}


def pivot(records: list[dict], index: str, columns: str, values: str, agg: str = "sum") -> dict:
    """Two-level group-by: {index: {column: aggregate(values)}}."""
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {sorted(AGGREGATIONS)}, got {agg!r}")
    groups: dict = {}
    for record in records:
        row, col, value = record.get(index), record.get(columns), record.get(values)
        if row is None or col is None or value is None:
            continue
        groups.setdefault(row, {}).setdefault(col, []).append(value)
    func = AGGREGATIONS[agg]
    return {row: {col: func(vals) for col, vals in cols.items()} for row, cols in groups.items()}
''',
        explanation="A dispatch dict maps aggregation names to functions, avoiding an if/elif chain; nested `setdefault` builds the two-level groups in one pass.",
        edge_cases=["`value` of 0 is kept (only `None` is skipped).", "Unknown `agg` → `ValueError`.", "No valid records → `{}`."],
        mistakes=["Using `if not value` (drops zero revenue).", "Averaging partial averages.", "Mutating records while grouping."],
        tests=[
            T(([{"region": "EU", "month": "Jan", "revenue": 100.0}, {"region": "EU", "month": "Feb", "revenue": 80.0},
                {"region": "US", "month": "Jan", "revenue": 150.0}, {"region": "EU", "month": "Jan", "revenue": 20.5},
                {"region": "US", "month": "Feb", "revenue": None}, {"region": None, "month": "Feb", "revenue": 5.0}],
               "region", "month", "revenue"),
              {"EU": {"Jan": 120.5, "Feb": 80.0}, "US": {"Jan": 150.0}}, example=True, note="sum with incomplete rows"),
            T(([{"region": "EU", "month": "Jan", "revenue": 100.0}, {"region": "EU", "month": "Jan", "revenue": 20.5}],
               "region", "month", "revenue", "mean"), {"EU": {"Jan": 60.25}}, note="mean"),
            T(([{"r": "a", "c": "x", "v": 0}, {"r": "a", "c": "x", "v": 0}], "r", "c", "v", "count"), {"a": {"x": 2}},
              note="count; zero values kept"),
            T(([], "r", "c", "v"), {}, example=True, note="boundary: no records"),
            T(([{"r": "a", "c": "x", "v": 1}], "r", "c", "v", "median"), raises="ValueError", note="unknown agg"),
        ],
        complexity="O(n) time and space.",
        extension="Add a `fill_value` parameter that fills missing (row, column) combinations.",
    ),
    # ------------------------------------------------------------------------ Q051
    Question(
        id=51,
        title="Forward-Fill Sensor Gaps with a Limit",
        difficulty="Intermediate",
        category=DATA,
        roles=[DS_ROLE, MLE, DE],
        skills=["stateful iteration", "None handling", "limits on imputation"],
        minutes=20,
        scenario=(
            "IoT temperature readings arrive every minute, but the network drops some readings "
            "(`None`). For short gaps it is reasonable to carry the last reading forward; for "
            "long outages, imputing would hide a real problem."
        ),
        problem=(
            "Write `forward_fill(values, limit=None)` that replaces `None` with the most recent "
            "non-`None` value, filling at most `limit` consecutive gaps."
        ),
        signature="def forward_fill(values: list, limit: int | None = None) -> list:",
        entry="forward_fill",
        inputs=["`values`: list of numbers or `None`.", "`limit`: `None` (unlimited) or an `int` ≥ 0."],
        output="A new list of the same length.",
        constraints=[
            "Leading `None` values (before any real value) stay `None`.",
            "Within a run of `None`s, only the first `limit` are filled; the rest stay `None`.",
            "`0` is a real value.",
            "Raise `ValueError` for a negative `limit`. Do not modify the input.",
        ],
        criteria=["Fills correctly.", "Respects the limit per gap.", "Leading gaps untouched."],
        hints=[
            "Track two things while looping: the last real value and how many gaps you have filled in the current run.",
            "Reset the counter whenever a real value appears.",
            "last = None; run = 0\nfor v in values:\n    if v is not None: last, run = v, 0; out.append(v)\n"
            "    else:\n        run += 1\n        out.append(last if last is not None and (limit is None or run <= limit) else None)",
        ],
        solution='''
def forward_fill(values: list, limit: int | None = None) -> list:
    """Carry the last observed value forward over at most ``limit`` gaps."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0 or None")
    filled = []
    last = None
    gap = 0
    for value in values:
        if value is not None:
            last, gap = value, 0
            filled.append(value)
            continue
        gap += 1
        can_fill = last is not None and (limit is None or gap <= limit)
        filled.append(last if can_fill else None)
    return filled
''',
        explanation="The gap counter resets on every real value, so the limit applies per run of missing values, matching pandas' `ffill(limit=...)` semantics.",
        edge_cases=["Leading `None` stays.", "`limit=0` fills nothing.", "Zero readings propagate."],
        mistakes=["`if value:` treats 0 as missing.", "Applying the limit globally rather than per gap.", "Filling in place."],
        tests=[
            T(([1, None, None, 4, None],), [1, 1, 1, 4, 4], example=True, note="unlimited"),
            T(([5, None, None, None, 7, None], 1), [5, 5, None, None, 7, 7], example=True, note="boundary: limit 1 per gap"),
            T(([None, 2, None],), [None, 2, 2], note="leading None stays"),
            T(([],), [], note="empty"),
            T(([0, None],), [0, 0], note="zero is a value"),
            T(([1, None], 0), [1, None], note="limit 0"),
            T(([1], -1), raises="ValueError", note="negative limit"),
        ],
        complexity="O(n) time and space.",
        extension="Implement linear interpolation for interior gaps instead of carrying the value.",
    ),
    # ------------------------------------------------------------------------ Q052
    Question(
        id=52,
        title="Compute Percentiles with Linear Interpolation",
        difficulty="Intermediate",
        category=DATA,
        roles=[DS_ROLE, MLE, AIE],
        skills=["sorting", "index arithmetic", "interpolation", "non-mutation"],
        minutes=20,
        scenario=(
            "Latency SLOs are defined on percentiles (p50, p95, p99). Your monitoring script runs "
            "where NumPy is not installed, so you need a small, correct percentile function that "
            "matches NumPy's default ('linear') method."
        ),
        problem="Write `percentile(values, p)` returning the p-th percentile using linear interpolation between closest ranks.",
        signature="def percentile(values: list[float], p: float) -> float:",
        entry="percentile",
        inputs=["`values`: non-empty list of numbers (any order).", "`p`: number in [0, 100]."],
        output="A `float`.",
        constraints=[
            "Sort a copy of the values. `rank = p / 100 * (n - 1)`; `lo = floor(rank)`, `hi = ceil(rank)`; "
            "result = `v[lo] + (v[hi] - v[lo]) * (rank - lo)`.",
            "Raise `ValueError` for an empty list or `p` outside [0, 100].",
            "Do not modify the input (no in-place `.sort()` on it).",
        ],
        criteria=["Matches the formula for all p.", "Unsorted input handled.", "Validation.", "Input unchanged."],
        hints=[
            "Percentile p sits at a fractional position between two sorted elements.",
            "Compute `rank`, take `math.floor`/`math.ceil`, and interpolate by the fractional part.",
            "data = sorted(values)\nrank = p / 100 * (len(data) - 1)\nlo, hi = floor(rank), ceil(rank)\n"
            "return float(data[lo] + (data[hi] - data[lo]) * (rank - lo))",
        ],
        solution='''
import math


def percentile(values: list[float], p: float) -> float:
    """p-th percentile with linear interpolation (NumPy's default method)."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    data = sorted(values)
    rank = p / 100 * (len(data) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return float(data[lo] + (data[hi] - data[lo]) * (rank - lo))
''',
        explanation="`sorted()` returns a new list, leaving the caller's data untouched. When `rank` is an integer, `lo == hi` and the formula returns that element exactly.",
        edge_cases=["Single value → that value.", "p = 0 and p = 100 → min and max.", "Unsorted input."],
        mistakes=["`values.sort()` mutates the caller's list.", "Using `n` instead of `n - 1` in the rank.", "Rounding the rank (nearest-rank method gives different results)."],
        tests=[
            T(([15, 20, 35, 40, 50], 40), 29.0, example=True, note="interpolated"),
            T(([10], 95), 10.0, example=True, note="boundary: single value"),
            T(([50, 15, 40, 20, 35], 40), 29.0, note="unsorted input must not be modified"),
            T(([15, 20, 35, 40, 50], 0), 15.0, note="p = 0"),
            T(([15, 20, 35, 40, 50], 100), 50.0, note="p = 100"),
            T(([15, 20, 35, 40, 50], 50), 35.0, note="median"),
            T(([1, 2, 3, 4], 25), 1.75, note="even length"),
            T(([], 50), raises="ValueError", note="empty"),
            T(([1, 2], 101), raises="ValueError", note="p out of range"),
        ],
        complexity="O(n log n) time for the sort, O(n) space. For many percentiles, sort once and reuse.",
        extension="Accept a list of percentiles and return them all from a single sort.",
    ),
    # ------------------------------------------------------------------------ Q053
    Question(
        id=53,
        title="Bucket Timestamps into Fixed Time Windows",
        difficulty="Intermediate",
        category=DATA,
        roles=[DE, DS_ROLE, MLE],
        skills=["datetime parsing", "integer division for flooring", "formatting", "validation"],
        minutes=25,
        scenario=(
            "To chart request volume, you count requests per 15-minute window. The timestamps are "
            "UTC strings like `2024-05-01T10:14:59Z`."
        ),
        problem=(
            "Write `bucket_counts(timestamps, minutes)` that counts timestamps per window of "
            "`minutes` minutes, keyed by the window start formatted as `YYYY-MM-DDTHH:MM:00Z`."
        ),
        signature="def bucket_counts(timestamps: list[str], minutes: int) -> dict[str, int]:",
        entry="bucket_counts",
        inputs=["`timestamps`: list of strings in the exact format `%Y-%m-%dT%H:%M:%SZ`.", "`minutes`: window size."],
        output="`dict[str, int]` containing only non-empty windows.",
        constraints=[
            "Windows are aligned to midnight UTC: window start = floor(minute-of-day / minutes) × minutes.",
            "`minutes` must be positive and divide 1440 (minutes per day); otherwise `ValueError`.",
            "Raise `ValueError` for a malformed timestamp.",
        ],
        criteria=["Correct flooring at window edges.", "Correct key format.", "Validation."],
        hints=[
            "Convert each timestamp to minute-of-day, floor it, and convert back.",
            "`datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')` parses (and validates) the timestamp; "
            "`dt.replace(hour=..., minute=..., second=0)` builds the window start.",
            "validate minutes\nfor ts in timestamps:\n    dt = strptime(...)\n    start = (dt.hour * 60 + dt.minute) // minutes * minutes\n"
            "    key = dt.replace(hour=start // 60, minute=start % 60, second=0).strftime('%Y-%m-%dT%H:%M:00Z')\n    counts[key] += 1",
        ],
        solution='''
from datetime import datetime


def bucket_counts(timestamps: list[str], minutes: int) -> dict[str, int]:
    """Count timestamps per fixed window aligned to midnight UTC."""
    if minutes <= 0 or 1440 % minutes != 0:
        raise ValueError("minutes must be a positive divisor of 1440")
    counts: dict[str, int] = {}
    for ts in timestamps:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        start = (dt.hour * 60 + dt.minute) // minutes * minutes
        window = dt.replace(hour=start // 60, minute=start % 60, second=0)
        key = window.strftime("%Y-%m-%dT%H:%M:00Z")
        counts[key] = counts.get(key, 0) + 1
    return counts
''',
        explanation="Integer division floors the minute-of-day to the window boundary; requiring a divisor of 1440 guarantees windows never straddle midnight.",
        edge_cases=["`10:15:00` starts a new 15-minute window.", "`23:59` stays on the same date.", "Invalid window sizes rejected."],
        mistakes=["Rounding instead of flooring.", "String slicing that ignores seconds/validation.", "Allowing sizes like 7 that misalign at midnight."],
        tests=[
            T((["2024-05-01T10:03:12Z", "2024-05-01T10:14:59Z", "2024-05-01T10:15:00Z", "2024-05-01T11:59:59Z"], 15),
              {"2024-05-01T10:00:00Z": 2, "2024-05-01T10:15:00Z": 1, "2024-05-01T11:45:00Z": 1}, example=True,
              note="15-minute windows with an edge"),
            T((["2024-05-01T10:03:12Z", "2024-05-01T10:59:59Z", "2024-05-01T11:00:00Z"], 60),
              {"2024-05-01T10:00:00Z": 2, "2024-05-01T11:00:00Z": 1}, note="hourly"),
            T((["2024-05-01T23:59:00Z"], 30), {"2024-05-01T23:30:00Z": 1}, example=True, note="boundary: end of day"),
            T(([], 15), {}, note="no timestamps"),
            T((["2024-05-01T10:00:00Z"], 7), raises="ValueError", note="7 does not divide 1440"),
            T((["2024/05/01 10:00"], 15), raises="ValueError", note="malformed timestamp"),
        ],
        complexity="O(n) time, O(w) space for w non-empty windows.",
        extension="Return all windows in a range, including empty ones with a count of 0.",
    ),
    # ------------------------------------------------------------------------ Q054
    Question(
        id=54,
        title="Speed Up Unknown-ID Detection with Sets",
        difficulty="Intermediate",
        category=PERF,
        roles=[DE, PYDEV, MLE],
        skills=["set membership", "algorithmic complexity", "order-preserving de-duplication"],
        minutes=15,
        scenario=(
            "A data-quality check finds events whose user id is not in the user table:\n\n"
            "```python\ndef find_unknown_ids(event_ids, known_ids):\n    unknown = []\n"
            "    for uid in event_ids:\n        if uid not in known_ids and uid not in unknown:\n"
            "            unknown.append(uid)\n    return unknown\n```\n\n"
            "With 30,000 events and 30,000 users it takes many seconds, because `in` on a list "
            "scans the whole list."
        ),
        problem=(
            "Rewrite `find_unknown_ids(event_ids, known_ids)` so it returns the same result in "
            "roughly linear time."
        ),
        signature="def find_unknown_ids(event_ids: list[str], known_ids: list[str]) -> list[str]:",
        entry="find_unknown_ids",
        inputs=["`event_ids`: list of ids (with duplicates).", "`known_ids`: list of valid ids."],
        output="Unknown ids, each once, in order of first appearance in `event_ids`.",
        constraints=["Same output as the original function.", "Must handle 30,000 × 30,000 ids in well under a second.", "Do not modify the inputs."],
        criteria=["Identical semantics.", "Order of first appearance.", "Fast."],
        hints=[
            "Which data structure answers `x in collection` in O(1) on average?",
            "Convert `known_ids` to a set once, and track already-reported ids in another set.",
            "known = set(known_ids); seen = set(); result = []\nfor uid in event_ids:\n"
            "    if uid not in known and uid not in seen: seen.add(uid); result.append(uid)",
        ],
        solution='''
def find_unknown_ids(event_ids: list[str], known_ids: list[str]) -> list[str]:
    """Ids from events that are not known users, each once, in first-seen order."""
    known = set(known_ids)
    reported: set[str] = set()
    unknown = []
    for uid in event_ids:
        if uid not in known and uid not in reported:
            reported.add(uid)
            unknown.append(uid)
    return unknown
''',
        explanation=(
            "The original runs two linear scans per event: O(n·m). Building a set once costs O(m) "
            "and each lookup is O(1) on average, so the whole function is O(n + m). The list "
            "still provides the output order."
        ),
        edge_cases=["No events → `[]`.", "No known ids → every distinct event id.", "Duplicates reported once."],
        mistakes=["`list(set(event_ids) - set(known_ids))` loses order.", "Converting `known_ids` to a set inside the loop.", "Forgetting the de-duplication."],
        tests=[
            T((["a", "b", "c", "b", "d"], ["a", "c"]), ["b", "d"], example=True, note="duplicates, order"),
            T(([], ["x"]), [], example=True, note="boundary: no events"),
            T((["x", "x"], []), ["x"], note="no known ids"),
            T((["k1", "k2"], ["k2", "k1"]), [], note="all known"),
            CT(_q054_fast, note="30,000 x 30,000 ids in under a second"),
        ],
        complexity="O(n + m) average time; O(m + u) space.",
        extension="Measure the old and new versions with `timeit` at 1k, 10k and 30k ids and plot or tabulate the growth.",
    ),
    # ------------------------------------------------------------------------ Q055
    Question(
        id=55,
        title="Aggregate LLM Token Usage and Cost per Model",
        difficulty="Intermediate",
        category=INT,
        roles=[AIE, AAI, MLE],
        skills=["grouping", "dict.get defaults", "unit conversion", "None for unknown values"],
        minutes=25,
        scenario=(
            "Finance asks how much each LLM costs per month. The gateway logs one record per "
            "request with token counts; prices are quoted in USD per 1,000 tokens and differ for "
            "prompt and completion tokens. Failed requests are still billed."
        ),
        problem=(
            "Write `summarize_token_usage(records, prices)` that returns per-model request counts, "
            "error counts, token totals and cost."
        ),
        signature="def summarize_token_usage(records: list[dict], prices: dict) -> dict[str, dict]:",
        entry="summarize_token_usage",
        inputs=[
            "`records`: dicts with `model` (str), optional `prompt_tokens`/`completion_tokens` (ints; missing or `None` → 0) and `status` (`'ok'` or `'error'`).",
            "`prices`: `{model: {'prompt': usd_per_1k, 'completion': usd_per_1k}}`.",
        ],
        output=(
            "`{model: {'requests', 'errors', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_usd'}}`. "
            "`cost_usd` is rounded to 6 decimals, or `None` if the model has no price."
        ),
        constraints=[
            "Every record counts as a request; `status == 'error'` also counts as an error.",
            "Tokens of failed requests are included in totals and cost.",
            "`cost = prompt_tokens / 1000 * prompt_price + completion_tokens / 1000 * completion_price`.",
            "Do not modify the inputs.",
        ],
        criteria=["Correct counts and totals.", "Correct cost formula and rounding.", "Unknown models → `cost_usd` None."],
        hints=[
            "First aggregate token sums per model, then compute costs once per model.",
            "`record.get('prompt_tokens') or 0` turns `None`/missing into 0.",
            "for r in records: s = summary.setdefault(model, zero dict); s['requests'] += 1; ...\n"
            "for model, s in summary.items():\n    price = prices.get(model)\n    s['total_tokens'] = ...; s['cost_usd'] = round(...) if price else None",
        ],
        solution='''
def summarize_token_usage(records: list[dict], prices: dict) -> dict[str, dict]:
    """Per-model request, error, token and cost totals."""
    summary: dict[str, dict] = {}
    for record in records:
        stats = summary.setdefault(record["model"], {
            "requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0,
        })
        stats["requests"] += 1
        if record.get("status") == "error":
            stats["errors"] += 1
        stats["prompt_tokens"] += record.get("prompt_tokens") or 0
        stats["completion_tokens"] += record.get("completion_tokens") or 0

    for model, stats in summary.items():
        stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]
        price = prices.get(model)
        if price is None:
            stats["cost_usd"] = None
        else:
            cost = (stats["prompt_tokens"] / 1000 * price["prompt"]
                    + stats["completion_tokens"] / 1000 * price["completion"])
            stats["cost_usd"] = round(cost, 6)
    return summary
''',
        explanation="Separating accumulation from derived fields keeps the loop simple and computes each cost from exact integer totals, avoiding per-request rounding drift.",
        edge_cases=["Model without a price → `cost_usd: None`.", "Missing token fields count as 0.", "Errors still billed."],
        mistakes=["Skipping error records entirely.", "Dividing by 1000 with integer division (`//`).", "Rounding per request before summing."],
        tests=[
            T(([{"model": "gpt-small", "prompt_tokens": 1200, "completion_tokens": 300, "status": "ok"},
                {"model": "gpt-large", "prompt_tokens": 800, "completion_tokens": 200, "status": "ok"},
                {"model": "gpt-small", "prompt_tokens": 500, "completion_tokens": 0, "status": "error"},
                {"model": "gpt-small", "prompt_tokens": 300, "completion_tokens": 100, "status": "ok"},
                {"model": "local-llm", "prompt_tokens": 1000, "completion_tokens": 500, "status": "ok"}], PRICES),
              {"gpt-small": {"requests": 3, "errors": 1, "prompt_tokens": 2000, "completion_tokens": 400, "total_tokens": 2400, "cost_usd": 0.0016},
               "gpt-large": {"requests": 1, "errors": 0, "prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000, "cost_usd": 0.014},
               "local-llm": {"requests": 1, "errors": 0, "prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500, "cost_usd": None}},
              example=True, note="three models, one unpriced"),
            T(([], PRICES), {}, example=True, note="boundary: no records"),
            T(([{"model": "gpt-large", "status": "error", "prompt_tokens": None}], PRICES),
              {"gpt-large": {"requests": 1, "errors": 1, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}},
              note="missing tokens count as zero"),
            CT(_q055_sample, note="data/sample_data/token_usage.json"),
        ],
        complexity="O(n) time; O(m) space for m models.",
        extension="Break the summary down by day as well as model, using each record's timestamp.",
    ),
]
