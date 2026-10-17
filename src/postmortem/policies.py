"""Agent behaviours: the failure shapes a real agent produces.

Each behaviour drives the sandbox through real tools. No model takes part. They
exist because the attribution engine has to be built and measured against
failures whose cause is known by construction, and because the same harness will
later run a model arm through the identical tools.

The named failure modes come from the published taxonomies of agent failure
(wrong target, stale context, tool misuse, loop/no-progress, budget exhaustion,
premature success claim) rather than from whatever happened to be easy to write.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from .sandbox import Sandbox, Task

MAX_STEPS = 12


@dataclass
class Step:
    index: int
    intent: str
    tool: str
    args: dict
    ok: bool
    output: str
    error: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "intent": self.intent,
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "output": self.output[:600],
            "error": self.error,
        }


Policy = Callable[[Sandbox, random.Random], list[Step]]


def _step(box: Sandbox, index: int, intent: str, call) -> Step:
    return Step(index=index, intent=intent, tool=call.tool, args=call.args,
                ok=call.ok, output=call.output, error=call.error)


def competent(box: Sandbox, rng: random.Random) -> list[Step]:
    """Locate, read, patch, test. The recovery policy replay uses."""
    task = box.task
    steps = [_step(box, 0, "survey the repository", box.list_files())]
    steps.append(_step(box, 1, "find the broken expression", box.grep(_safe_pattern(task.broken_line))))
    steps.append(_step(box, 2, "read the file that matched", box.read_file(task.target_file)))
    steps.append(_step(box, 3, "apply the fix",
                       box.patch(task.target_file, task.broken_line, task.fix_hint)))
    steps.append(_step(box, 4, "run the tests", box.run_tests()))
    return steps


def _safe_pattern(text: str) -> str:
    import re

    return re.escape(text.strip())


def wrong_file(box: Sandbox, rng: random.Random) -> list[Step]:
    """Patches a file that does not contain the bug: the classic wrong target."""
    task = box.task
    others = [f for f in task.files if f.endswith(".py") and f != task.target_file and "__init__" not in f]
    victim = rng.choice(others) if others else task.target_file
    steps = [_step(box, 0, "survey the repository", box.list_files())]
    steps.append(_step(box, 1, "read a plausible-looking file", box.read_file(victim)))
    steps.append(_step(box, 2, "patch the wrong file",
                       box.patch(victim, "def ", "def  ")))
    steps.append(_step(box, 3, "run the tests", box.run_tests()))
    steps.append(_step(box, 4, "retry the same patch",
                       box.patch(victim, "def  ", "def   ")))
    steps.append(_step(box, 5, "run the tests again", box.run_tests()))
    return steps


def stale_context(box: Sandbox, rng: random.Random) -> list[Step]:
    """Reads the file, mutates it, then patches against the version it remembers."""
    task = box.task
    steps = [_step(box, 0, "read the target file", box.read_file(task.target_file))]
    steps.append(_step(box, 1, "reformat the file first",
                       box.patch(task.target_file, task.broken_line, task.broken_line + "  # noted")))
    steps.append(_step(box, 2, "apply the fix against the remembered text",
                       box.patch(task.target_file, task.broken_line + "\n", task.fix_hint + "\n")))
    steps.append(_step(box, 3, "run the tests", box.run_tests()))
    return steps


def tool_misuse(box: Sandbox, rng: random.Random) -> list[Step]:
    """Calls tools with arguments of the wrong kind, then may recover."""
    task = box.task
    steps = [_step(box, 0, "read a directory as a file", box.read_file("pkg"))]
    steps.append(_step(box, 1, "search with a broken pattern", box.grep("(unclosed")))
    steps.append(_step(box, 2, "survey the repository", box.list_files()))
    if rng.random() < 0.5:
        steps.append(_step(box, 3, "read the target file", box.read_file(task.target_file)))
        steps.append(_step(box, 4, "apply the fix",
                           box.patch(task.target_file, task.broken_line, task.fix_hint)))
        steps.append(_step(box, 5, "run the tests", box.run_tests()))
    else:
        steps.append(_step(box, 3, "run the tests without changing anything", box.run_tests()))
    return steps


def looper(box: Sandbox, rng: random.Random) -> list[Step]:
    """Repeats one action until the budget runs out: no progress, no error either."""
    steps = []
    for i in range(6):
        steps.append(_step(box, i, "search again for the bug", box.grep("def ")))
    steps.append(_step(box, 6, "run the tests", box.run_tests()))
    return steps


def overconfident(box: Sandbox, rng: random.Random) -> list[Step]:
    """Skips reading and writes a patch that breaks the file."""
    task = box.task
    steps = [_step(box, 0, "patch without reading",
                   box.patch(task.target_file, "def ", "def \n"))]
    steps.append(_step(box, 1, "run the tests", box.run_tests()))
    steps.append(_step(box, 2, "read the file after the failure", box.read_file(task.target_file)))
    steps.append(_step(box, 3, "run the tests again", box.run_tests()))
    return steps


def premature_success(box: Sandbox, rng: random.Random) -> list[Step]:
    """Does part of the work, sees a failing test, and stops anyway."""
    task = box.task
    steps = [_step(box, 0, "survey the repository", box.list_files())]
    steps.append(_step(box, 1, "read the target file", box.read_file(task.target_file)))
    steps.append(_step(box, 2, "run the tests before fixing anything", box.run_tests()))
    return steps


def recovers_late(box: Sandbox, rng: random.Random) -> list[Step]:
    """Goes wrong first and fixes it afterwards: the trajectory succeeds.

    Successful-but-messy runs matter. A detector that only ever sees failures
    learns to call everything a failure.
    """
    task = box.task
    others = [f for f in task.files if f.endswith(".py") and f != task.target_file and "__init__" not in f]
    victim = rng.choice(others) if others else task.target_file
    steps = [_step(box, 0, "read the wrong file", box.read_file(victim))]
    steps.append(_step(box, 1, "run the tests", box.run_tests()))
    steps.append(_step(box, 2, "search for the real cause", box.grep(_safe_pattern(task.broken_line))))
    steps.append(_step(box, 3, "read the right file", box.read_file(task.target_file)))
    steps.append(_step(box, 4, "apply the fix",
                       box.patch(task.target_file, task.broken_line, task.fix_hint)))
    steps.append(_step(box, 5, "run the tests", box.run_tests()))
    return steps


POLICIES: dict[str, Policy] = {
    "competent": competent,
    "wrong_file": wrong_file,
    "stale_context": stale_context,
    "tool_misuse": tool_misuse,
    "looper": looper,
    "overconfident": overconfident,
    "premature_success": premature_success,
    "recovers_late": recovers_late,
}

# What each policy is *built* to do wrong. This is the construction label, kept
# separate from what the attribution engine predicts, so the two can be compared.
INTENDED_FAULT = {
    "competent": "",
    "wrong_file": "wrong_target",
    "stale_context": "stale_context",
    "tool_misuse": "tool_misuse",
    "looper": "no_progress",
    "overconfident": "unverified_edit",
    "premature_success": "premature_stop",
    "recovers_late": "",
}


def run_policy(name: str, task: Task, seed: int) -> tuple[list[Step], bool, Sandbox]:
    rng = random.Random(seed)
    box = Sandbox(task=task)
    steps = POLICIES[name](box, rng)
    return steps[:MAX_STEPS], box.verify(), box
