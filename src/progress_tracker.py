"""Local, private progress tracking for the workbook.

Progress is stored as JSON on your own machine – no account, no network, and no
personal information (only question ids, counts, booleans and timestamps).

Location (first match wins):
  1. the ``PPW_PROGRESS_FILE`` environment variable, if set;
  2. ``<project root>/.progress/progress.json`` (git-ignored).

Reset with ``reset_progress()`` from the practice system, or simply delete the file.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable

from .schema import PROJECT_ROOT

FORMAT_VERSION = 1


def default_progress_path() -> Path:
    override = os.environ.get("PPW_PROGRESS_FILE")
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / ".progress" / "progress.json"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _empty_entry() -> dict:
    return {
        "attempts": 0,
        "passed": False,
        "best_tests_passed": 0,
        "total_tests": 0,
        "hints_used": [],
        "solution_viewed": False,
        "passed_independently": False,
        "first_passed_at": None,
        "last_attempt_at": None,
    }


class ProgressTracker:
    """Reads and writes the progress JSON file. Every update is saved immediately."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_progress_path()
        self.warning: str | None = None
        self.data = self._load()

    # ------------------------------------------------------------------ storage
    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": FORMAT_VERSION, "questions": {}, "tests": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("questions"), dict):
                raise ValueError("unexpected structure")
            data.setdefault("tests", [])
            data.setdefault("version", FORMAT_VERSION)
            return data
        except (ValueError, OSError) as exc:
            backup = self.path.with_suffix(self.path.suffix + ".corrupt")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            self.warning = (f"Progress file {self.path} could not be read ({exc}); it was moved to "
                            f"{backup.name} and progress starts fresh.")
            return {"version": FORMAT_VERSION, "questions": {}, "tests": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".progress.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------ updates
    def entry(self, qid: str) -> dict:
        return self.data["questions"].setdefault(qid, _empty_entry())

    def record_attempt(self, qid: str, n_passed: int, n_total: int) -> dict:
        entry = self.entry(qid)
        entry["attempts"] += 1
        entry["last_attempt_at"] = _now()
        entry["total_tests"] = n_total
        entry["best_tests_passed"] = max(entry["best_tests_passed"], n_passed)
        passed = n_total > 0 and n_passed == n_total
        if passed and not entry["passed"]:
            entry["passed"] = True
            entry["first_passed_at"] = entry["last_attempt_at"]
            entry["passed_independently"] = not entry["hints_used"] and not entry["solution_viewed"]
        self.save()
        return entry

    def record_hint(self, qid: str, level: int) -> None:
        entry = self.entry(qid)
        if level not in entry["hints_used"]:
            entry["hints_used"] = sorted([*entry["hints_used"], level])
        self.save()

    def record_solution_view(self, qid: str) -> None:
        self.entry(qid)["solution_viewed"] = True
        self.save()

    def record_test(self, summary: dict) -> None:
        self.data["tests"].append(summary)
        self.save()

    def reset(self, qids: Iterable[str] | None = None) -> None:
        if qids is None:
            self.data = {"version": FORMAT_VERSION, "questions": {}, "tests": []}
        else:
            for qid in qids:
                self.data["questions"].pop(qid, None)
        self.save()

    # ------------------------------------------------------------------ queries
    def status(self, qid: str) -> str:
        entry = self.data["questions"].get(qid)
        if not entry or entry["attempts"] == 0:
            return "not started"
        return "passed" if entry["passed"] else "attempted"

    def has_attempted(self, qid: str) -> bool:
        entry = self.data["questions"].get(qid)
        return bool(entry and entry["attempts"])
