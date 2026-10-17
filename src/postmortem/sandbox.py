"""A real sandbox: real files, real tools, a real verifier.

Postmortem's dataset has to be *observed* failures, not faults injected into a
replayed success. That means something has to actually run and actually fail.

So each task here is a small broken Python module plus a test that decides the
outcome by executing the code. An agent works on a real temporary directory
through real tools (list, read, grep, patch, run tests), and the verifier's
verdict is the label - nobody writes "this failed" by hand.

What is *not* a model: the agent policies (``policies.py``) are scripted. That
is stated everywhere this corpus is described, because "real trace, scripted
policy" and "model-generated trace" are different claims. The policies exist to
produce the failure *shapes* a real agent produces - wrong file, stale context,
loop, budget exhaustion, premature success - and the harness is the same one a
model arm will run through when the GPU frees.
"""

from __future__ import annotations

import io
import re
import shutil
import tempfile
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task:
    name: str
    files: dict[str, str]
    target_file: str
    test_source: str
    fix_hint: str  # the substring a correct patch must introduce
    broken_line: str  # the substring a correct patch must remove

    def materialise(self, root: Path) -> None:
        for rel, body in self.files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")


TASKS: list[Task] = [
    Task(
        name="off-by-one-slice",
        files={
            "pkg/__init__.py": "",
            "pkg/text.py": (
                "def last_words(text, n):\n"
                "    \"\"\"Return the last n words of text.\"\"\"\n"
                "    words = text.split()\n"
                "    return words[-n + 1:]\n"
            ),
            "pkg/numbers.py": "def mean(xs):\n    return sum(xs) / len(xs)\n",
            "README.md": "Text helpers.\n",
        },
        target_file="pkg/text.py",
        test_source=(
            "from pkg.text import last_words\n"
            "assert last_words('a b c d', 2) == ['c', 'd'], last_words('a b c d', 2)\n"
            "assert last_words('one two three', 1) == ['three']\n"
        ),
        fix_hint="words[-n:]",
        broken_line="words[-n + 1:]",
    ),
    Task(
        name="wrong-default",
        files={
            "pkg/__init__.py": "",
            "pkg/cfg.py": (
                "DEFAULTS = {'retries': 0, 'timeout': 30}\n\n\n"
                "def retries(config=None):\n"
                "    config = config or {}\n"
                "    return config.get('retries', DEFAULTS['retries'])\n"
            ),
            "pkg/util.py": "def clamp(v, lo, hi):\n    return max(lo, min(hi, v))\n",
        },
        target_file="pkg/cfg.py",
        test_source=(
            "from pkg.cfg import retries\n"
            "assert retries() == 3, retries()\n"
            "assert retries({'retries': 1}) == 1\n"
        ),
        fix_hint="'retries': 3",
        broken_line="'retries': 0",
    ),
    Task(
        name="missing-strip",
        files={
            "pkg/__init__.py": "",
            "pkg/parse.py": (
                "def parse_kv(line):\n"
                "    key, _, value = line.partition('=')\n"
                "    return key, value\n"
            ),
            "pkg/text.py": "def title(s):\n    return s.title()\n",
        },
        target_file="pkg/parse.py",
        test_source=(
            "from pkg.parse import parse_kv\n"
            "assert parse_kv(' a = b ') == ('a', 'b'), parse_kv(' a = b ')\n"
        ),
        # The first version of this pair replaced the whole return statement with a
        # bare expression, so the "fix" deleted the `return` and no agent could ever
        # pass this task. It showed up as 32 runs labelled unrecoverable_from_start -
        # exactly one task times eight policies times four seeds, which is what
        # pointed at the fixture rather than at the agents.
        fix_hint="return key.strip(), value.strip()",
        broken_line="return key, value",
    ),
    Task(
        name="inclusive-range",
        files={
            "pkg/__init__.py": "",
            "pkg/seq.py": "def upto(n):\n    return list(range(1, n))\n",
            "pkg/numbers.py": "def double(x):\n    return x * 2\n",
        },
        target_file="pkg/seq.py",
        test_source=(
            "from pkg.seq import upto\n"
            "assert upto(3) == [1, 2, 3], upto(3)\n"
        ),
        fix_hint="range(1, n + 1)",
        broken_line="range(1, n)",
    ),
    Task(
        name="case-sensitive-lookup",
        files={
            "pkg/__init__.py": "",
            "pkg/table.py": (
                "ROWS = {'alpha': 1, 'beta': 2}\n\n\n"
                "def lookup(name):\n"
                "    return ROWS.get(name)\n"
            ),
            "pkg/cfg.py": "DEFAULTS = {'debug': False}\n",
        },
        target_file="pkg/table.py",
        test_source=(
            "from pkg.table import lookup\n"
            "assert lookup('ALPHA') == 1, lookup('ALPHA')\n"
            "assert lookup('beta') == 2\n"
        ),
        fix_hint="ROWS.get(name.lower())",
        broken_line="ROWS.get(name)",
    ),
]


@dataclass
class ToolCall:
    tool: str
    args: dict
    ok: bool
    output: str
    error: str = ""

    def to_dict(self) -> dict:
        return {"tool": self.tool, "args": self.args, "ok": self.ok,
                "output": self.output[:600], "error": self.error}


@dataclass
class Sandbox:
    """A real directory with real tools. Every call is recorded as it happened."""

    task: Task
    root: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="pm-")))
    calls: list[ToolCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task.materialise(self.root)

    # --- tools ------------------------------------------------------------------
    def list_files(self) -> ToolCall:
        names = sorted(str(p.relative_to(self.root)).replace("\\", "/")
                       for p in self.root.rglob("*") if p.is_file())
        call = ToolCall("list_files", {}, True, "\n".join(names))
        self.calls.append(call)
        return call

    def read_file(self, path: str) -> ToolCall:
        target = self.root / path
        if target.is_dir():
            call = ToolCall("read_file", {"path": path}, False, "", "IsADirectoryError: " + path)
        elif not target.exists():
            call = ToolCall("read_file", {"path": path}, False, "", "FileNotFoundError: " + path)
        else:
            call = ToolCall("read_file", {"path": path}, True, target.read_text(encoding="utf-8"))
        self.calls.append(call)
        return call

    def grep(self, pattern: str) -> ToolCall:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            call = ToolCall("grep", {"pattern": pattern}, False, "", f"re.error: {exc}")
            self.calls.append(call)
            return call
        hits = []
        for p in sorted(self.root.rglob("*.py")):
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(self.root)}:{i}: {line.strip()}")
        call = ToolCall("grep", {"pattern": pattern}, True, "\n".join(hits) or "(no matches)")
        self.calls.append(call)
        return call

    def patch(self, path: str, old: str, new: str) -> ToolCall:
        target = self.root / path
        if not target.exists():
            call = ToolCall("patch", {"path": path, "old": old, "new": new}, False, "",
                            "FileNotFoundError: " + path)
        else:
            body = target.read_text(encoding="utf-8")
            if old not in body:
                call = ToolCall("patch", {"path": path, "old": old, "new": new}, False, "",
                                "PatchDidNotApply: old text not found")
            else:
                target.write_text(body.replace(old, new, 1), encoding="utf-8")
                call = ToolCall("patch", {"path": path, "old": old, "new": new}, True,
                                f"applied to {path}")
        self.calls.append(call)
        return call

    def run_tests(self) -> ToolCall:
        """Execute the task's test against the current files, in-process."""
        import subprocess
        import sys

        test_path = self.root / "_check.py"
        test_path.write_text(self.task.test_source, encoding="utf-8")
        try:
            proc = subprocess.run(
                # -s only (skip user site). NOT -I: that implies -P, which keeps the
                # script's own directory off sys.path, so every test failed with
                # ModuleNotFoundError and every trajectory looked like a failure -
                # including the competent policy, which is how the bug was caught.
                [sys.executable, "-s", str(test_path)],
                cwd=self.root, capture_output=True, text=True, timeout=60, check=False,
            )
            ok = proc.returncode == 0
            out = (proc.stdout or "") + (proc.stderr or "")
            call = ToolCall("run_tests", {}, ok, out[-800:], "" if ok else "tests failed")
        except subprocess.TimeoutExpired:
            call = ToolCall("run_tests", {}, False, "", "TimeoutExpired after 60s")
        except Exception as exc:  # noqa: BLE001
            call = ToolCall("run_tests", {}, False, "", f"{type(exc).__name__}: {exc}")
        self.calls.append(call)
        return call

    def verify(self) -> bool:
        """The verdict. This - not an opinion - is what labels a trajectory."""
        import subprocess
        import sys

        test_path = self.root / "_verify.py"
        test_path.write_text(self.task.test_source, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-s", str(test_path)], cwd=self.root,
                                  capture_output=True, text=True, timeout=60, check=False)
            return proc.returncode == 0
        except Exception:  # noqa: BLE001
            return False

    def snapshot(self) -> dict[str, str]:
        return {
            str(p.relative_to(self.root)).replace("\\", "/"): p.read_text(encoding="utf-8")
            for p in sorted(self.root.rglob("*.py"))
        }

    def restore(self, snap: dict[str, str]) -> None:
        for rel, body in snap.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def run_in_process(source: str, cwd: Path) -> tuple[bool, str]:
    """Helper kept for tests: run a snippet and report success plus output."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(source, "<snippet>", "exec"), {"__name__": "__main__"})  # noqa: S102
        return True, buf.getvalue()
    except Exception:  # noqa: BLE001
        return False, buf.getvalue() + traceback.format_exc()
