"""Score attribution against replay, with baselines that could beat it.

    python -m postmortem.evaluate

Replay is the ground truth: it re-ran every prefix and asked the verifier. The
engine never sees it. The split is hashed on trace id, so the retrieval library
and the scored traces do not overlap.

Three baselines, because "our number" means nothing alone:

* **last step** - blame the final step. Surprisingly strong on short traces, and
  the baseline most tools effectively implement.
* **first failing call** - blame the first step whose tool errored. This is what
  a log viewer shows you.
* **output only** - the evidence an output-only system has: blame nothing, and
  predict the majority label kind. It is the floor the full-trace claim has to clear.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .attribute import Library, attribute, split_of

DEFAULT_DATA = Path("C:/d drive data/trace-100k/v1")


def _last_step(trace: dict) -> int | None:
    steps = trace.get("steps", [])
    return len(steps) - 1 if steps else None


def _first_failing(trace: dict) -> int | None:
    for i, s in enumerate(trace.get("steps", [])):
        if not s.get("ok"):
            return i
    return None


def evaluate(data_dir: Path = DEFAULT_DATA) -> dict:
    library = Library.load(data_dir, split="train")
    rows = [
        json.loads(line)
        for line in (data_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = [r for r in rows if split_of(r["trace_id"]) == "holdout"]
    failures = [r for r in scored if r["outcome"] == "failure"]

    majority_kind = collections.Counter(
        r.get("label_kind") for r in library.rows if r["outcome"] == "failure"
    ).most_common(1)
    majority_kind = majority_kind[0][0] if majority_kind else "no_fatal_step"

    engine = {"step_exact": 0, "kind": 0, "fault": 0}
    base_last = {"step_exact": 0}
    base_first = {"step_exact": 0}
    base_output = {"kind": 0}
    by_kind: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    by_layer: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    confusion: collections.Counter = collections.Counter()

    for r in failures:
        gold_step = r.get("breaking_step")
        gold_kind = r.get("label_kind")
        gold_fault = r.get("intended_fault") or ""

        a = attribute({k: v for k, v in r.items() if k not in ("breaking_step", "label_kind", "intended_fault", "recoverable_after_step")}, library)

        if a.breaking_step == gold_step:
            engine["step_exact"] += 1
        if a.label_kind == gold_kind:
            engine["kind"] += 1
            by_kind[gold_kind][0] += 1
        by_kind[gold_kind][1] += 1
        if gold_fault and a.fault_class == gold_fault:
            engine["fault"] += 1
        confusion[f"{gold_fault or '(none)'} -> {a.fault_class or '(none)'}"] += 1

        by_layer[a.layer][1] += 1
        if a.breaking_step == gold_step:
            by_layer[a.layer][0] += 1

        if _last_step(r) == gold_step:
            base_last["step_exact"] += 1
        if _first_failing(r) == gold_step:
            base_first["step_exact"] += 1
        if majority_kind == gold_kind:
            base_output["kind"] += 1

    n = len(failures) or 1
    with_fault = sum(1 for r in failures if r.get("intended_fault")) or 1
    return {
        "trajectories_total": len(rows),
        "holdout_trajectories": len(scored),
        "holdout_failures_scored": len(failures),
        "library_trajectories": len(library.rows),
        "engine": {
            "breaking_step_exact": round(engine["step_exact"] / n, 4),
            "label_kind": round(engine["kind"] / n, 4),
            "fault_class": round(engine["fault"] / with_fault, 4),
        },
        "baselines": {
            "blame_the_last_step": round(base_last["step_exact"] / n, 4),
            "blame_the_first_failing_call": round(base_first["step_exact"] / n, 4),
            "output_only_majority_kind": round(base_output["kind"] / n, 4),
        },
        "label_kind_recall": {
            k: {"correct": v[0], "total": v[1], "recall": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in sorted(by_kind.items())
        },
        "by_layer": {
            k: {"traces": v[1], "step_exact": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in sorted(by_layer.items())
        },
        "fault_confusion_top": dict(confusion.most_common(10)),
        "ground_truth": "replay: every prefix re-run and handed to the verifier",
        "model_used": None,
        "model_took_part": False,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score attribution against replay")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not (args.data / "traces.jsonl").exists():
        print(f"no traces at {args.data} - run: python -m postmortem.collect")
        return 2
    report = evaluate(args.data)
    out = args.out or (args.data / "benchmark.json")
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
