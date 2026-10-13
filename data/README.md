# Sample datasets

All files in `sample_data/` are **synthetic** and small (a few KB each). They contain no personal information
and no real credentials. Some questions' tests check exact properties of these files (for example "34
parseable log lines"), so keep them unchanged if you want those checks to pass.

In the notebook, the path is available as `DATA_DIR` (exported by the practice system), e.g.
`read_config_lines(DATA_DIR / "pipeline_config.txt")`. Most questions use self-contained examples; the datasets
let you try your solutions on more realistic, messier input.

| File | Format | Used by | Contents and deliberate data-quality issues |
|---|---|---|---|
| `pipeline_config.txt` | text | Q018 | `key=value` settings with comments, blank lines and stray whitespace. |
| `application_logs.txt` | text | Q024 | 40 lines in `<timestamp> <LEVEL> [<service>] <message>` format: 34 valid (9 `ERROR`), 6 malformed (blank line, traceback lines, bad timestamp, unknown level, invalid service name). |
| `transactions.csv` | CSV | Q040, Q068 | 38 transactions (`transaction_id, customer_id, amount, currency, timestamp`). Invalid data rows 4, 9, 13, 18, 22, 27, 31, 35: non-numeric, negative, empty and `inf` amounts, missing customer id, unsupported currency, truncated row, empty timestamp. Mixed-case currencies and padded values. |
| `inference_requests.jsonl` | JSON Lines | Q041, Q084 | 51 lines: 48 inference request records (`request_id, timestamp, endpoint, model, latency_ms, status`) across three endpoints, plus malformed lines 7 (truncated JSON), 19 (array) and 33 (not JSON). |
| `model_config.json` | JSON | Q043 | A valid training configuration without a `seed` (so the default applies). |
| `token_usage.json` | JSON | Q055 | `prices` per 1K tokens for three models and 60 request records across four models (`local-llm` has no price); 6 errors; one `null` token count. |
| `rotated_logs/` | text + gzip | Q069 | `app.log`, `app.log.1`, `app.log.2.gz`, `app.log.3.gz` (oldest) plus an `app.log.bak` file that must be ignored. |
| `api_responses.json` | JSON | Q031 (exploration) | Three nested search-API responses, including an empty result and an error payload with `meta: null`. |
| `events.json` | JSON | Q076 | 80 shuffled clickstream events for 8 users (`user_id, event_type, timestamp`); 28 sessions with a 30-minute gap rule. |
| `model_metrics.json` | JSON | Q078 | 48 hourly snapshots of three metrics; one latency spike, one throughput drop, two missing GPU readings. |
| `prediction_logs.json` | JSON | Q083 | 120 ticket-routing predictions over four classes; 93 correct (accuracy 0.775). |
| `retry_logs.json` | JSON | Q085 | Attempt-level logs for 30 LLM requests: shuffled, 4 duplicated lines, and request `req-017` missing its second attempt. |
| `eval_results.json` | JSON | Q097 | Paired evaluation results of models `v1`/`v2` across four languages (improvement, neutral, regression and a small slice) plus 6 unpaired records. |
