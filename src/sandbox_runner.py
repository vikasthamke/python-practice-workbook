"""Run learner *source code* against a question's tests in a separate process.

``check(n, "def f(...): ...")`` uses this module. The source is written to a
temporary directory and executed by a fresh ``python -I`` interpreter with a
wall-clock timeout, so an infinite loop or a crash cannot hang or kill the
notebook kernel, and the learner's code never runs inside the checker process.

Security limitations (please read): this is *process isolation*, not a security
sandbox. The child process runs with your user account's permissions and can
read/write your files and use the network. Only check code you wrote or trust.
For untrusted submissions use a container or VM with no credentials mounted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKER = "__PPW_RESULT__"


def run_source(qid: str, source: str, timeout: float = 60.0) -> dict:
    """Execute ``source`` in a subprocess and return the serialized test results."""
    with tempfile.TemporaryDirectory(prefix="ppw_submission_") as workdir:
        submission = Path(workdir) / "submission.py"
        submission.write_text(source, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
        for name in ("SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "HOME"):
            if name in os.environ:
                env[name] = os.environ[name]
        cmd = [sys.executable, "-I", str(Path(__file__).resolve()), str(PROJECT_ROOT), qid, str(submission)]
        try:
            proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"error": f"Timeout: your code did not finish within {timeout:g}s (possible infinite loop?)"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    stderr = proc.stderr.strip().splitlines()
    return {"error": "Your code could not be run: " + (stderr[-1] if stderr else f"exit code {proc.returncode}")}


def _child_main(root: str, qid: str, submission: str) -> None:
    sys.path.insert(0, root)
    import types

    from src.question_bank import get_question
    from src.validators import run_tests

    question = get_question(qid)
    module = types.ModuleType("submission")
    sys.modules["submission"] = module
    source = Path(submission).read_text(encoding="utf-8")
    try:
        exec(compile(source, "submission.py", "exec"), module.__dict__)  # learner code, in this child only
    except Exception as exc:  # syntax errors, NameErrors at import time, ...
        print(MARKER + json.dumps({"error": f"Your code raised {type(exc).__name__} while loading: {exc}"}))
        return
    impl = module.__dict__.get(question.entry)
    if impl is None:
        print(MARKER + json.dumps({"error": f"Your code does not define {question.entry!r}"}))
        return
    results = run_tests(question, impl)
    print(MARKER + json.dumps({"cases": [r.to_dict() for r in results]}))


if __name__ == "__main__":
    _child_main(*sys.argv[1:4])
