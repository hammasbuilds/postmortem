"""Collect trajectories and label them by replay.

    python -m postmortem.collect --seeds 3 --out "C:/d drive data/trace-100k/v1"

The label nobody writes by hand
-------------------------------
A failed run's *breaking step* is found by replaying it. For each prefix of the
trajectory, the harness rebuilds the sandbox, re-executes that prefix exactly,
then hands the result to the recovery policy and asks the verifier. A step is
the breaking step when recovery still succeeds from the state before it and
fails from the state after it.

That makes the label a property of the execution, not of a judgement: if a
different step turns out to be the point of no return, the replay says so.

Cost is honest too: labelling one trajectory runs the verifier once per prefix,
which is why this is a corpus to grow in the background rather than a thing to
recompute on every request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .policies import INTENDED_FAULT, POLICIES, Step, competent, run_policy
from .sandbox import TASKS, Sandbox, Task

DEFAULT_OUT = Path("C:/d drive data/trace-100k/v1")


def _replay(box: Sandbox, steps: list[Step]) -> None:
    """Re-execute a prefix exactly as it was recorded."""
    for s in steps:
        if s.tool == "list_files":
            box.list_files()
        elif s.tool == "read_file":
            box.read_file(s.args.get("path", ""))
        elif s.tool == "grep":
            box.grep(s.args.get("pattern", ""))
        elif s.tool == "patch":
            box.patch(s.args.get("path", ""), s.args.get("old", ""), s.args.get("new", ""))
        elif s.tool == "run_tests":
            box.run_tests()


def recoverability(task: Task, steps: list[Step], seed: int) -> list[bool]:
    """Can the recovery policy still succeed after each prefix? Index k = after k steps."""
    out: list[bool] = []
    import random

    for k in range(len(steps) + 1):
        box = Sandbox(task=task)
        try:
            _replay(box, steps[:k])
            competent(box, random.Random(seed))
            out.append(box.verify())
        finally:
            box.cleanup()
    return out


def breaking_step(recov: list[bool]) -> int | None:
    """The first step after which recovery stops working."""
    for k in range(1, len(recov)):
        if recov[k - 1] and not recov[k]:
            return k - 1  # the step at index k-1 is the one that did it
    return None


def label_kind(recov: list[bool], outcome: str) -> str:
    """What kind of failure the replay found. Three outcomes, not two.

    "No breaking step" was collapsing two different things: a run where every
    prefix is still recoverable (the agent simply stopped, or looped, without
    doing damage) and a run that was never recoverable at all. They need
    different advice, so they get different labels.
    """
    if outcome == "success":
        return "succeeded"
    if breaking_step(recov) is not None:
        return "destructive_step"
    if recov and not recov[0]:
        return "unrecoverable_from_start"
    return "no_fatal_step"


def collect(seeds: int, tasks: list[Task] | None = None, policies: list[str] | None = None) -> list[dict]:
    tasks = tasks or TASKS
    policies = policies or list(POLICIES)
    rows: list[dict] = []
    for task in tasks:
        for policy in policies:
            for seed in range(seeds):
                started = time.time()
                steps, success, box = run_policy(policy, task, seed=seed * 97 + 13)
                box.cleanup()
                recov = recoverability(task, steps, seed)
                k = breaking_step(recov)
                trace_id = hashlib.sha256(
                    f"{task.name}|{policy}|{seed}".encode()
                ).hexdigest()[:16]
                rows.append(
                    {
                        "trace_id": trace_id,
                        "task": task.name,
                        "policy": policy,
                        "seed": seed,
                        "steps": [s.to_dict() for s in steps],
                        "outcome": "success" if success else "failure",
                        "verifier": "task test executed in a subprocess",
                        "breaking_step": k,
                        "label_kind": label_kind(recov, "success" if success else "failure"),
                        "recoverable_after_step": recov,
                        "intended_fault": INTENDED_FAULT.get(policy, ""),
                        "seconds_to_label": round(time.time() - started, 2),
                        "policy_is_a_model": False,
                    }
                )
    return rows


def stats(rows: list[dict]) -> dict:
    import collections

    outcomes = collections.Counter(r["outcome"] for r in rows)
    by_policy = collections.Counter(r["policy"] for r in rows)
    failures = [r for r in rows if r["outcome"] == "failure"]
    labelled = [r for r in failures if r["breaking_step"] is not None]
    kinds = collections.Counter(r.get("label_kind", "") for r in rows)
    return {
        "trajectories": len(rows),
        "steps": sum(len(r["steps"]) for r in rows),
        "outcomes": dict(outcomes),
        "by_policy": dict(by_policy),
        "failures": len(failures),
        "failures_with_a_replay_label": len(labelled),
        "failures_without_a_label": len(failures) - len(labelled),
        "label_kinds": dict(kinds),
        "unlabelled_note": (
            "A failure with no breaking step is not unlabelled: it is either "
            "no_fatal_step (every prefix still recoverable - the agent stopped or looped "
            "without doing damage) or unrecoverable_from_start. Both are reported."
        ),
        "tasks": sorted({r["task"] for r in rows}),
        "seconds_total": round(sum(r["seconds_to_label"] for r in rows), 1),
        "policy_is_a_model": False,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Collect and replay-label agent trajectories")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--policies", nargs="*", default=None)
    args = ap.parse_args(argv)

    rows = collect(args.seeds, policies=args.policies)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "traces.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = stats(rows)
    (args.out / "stats.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(s, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
