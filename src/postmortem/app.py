"""Postmortem — agent failure attribution. Web UI over the engine.

    uv run python -m postmortem.app --port 8830
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from . import report as report_mod
from .attribute import Library, attribute, split_of
from .collect import collect
from .policies import INTENDED_FAULT, POLICIES
from .sandbox import TASKS
from .web import App, Raw

UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"
DEFAULT_DATA = Path("C:/d drive data/trace-100k/v1")
SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def build_app(data_dir: Path = DEFAULT_DATA) -> App:
    app = App(UI_DIR, title="postmortem")
    library = Library.load(data_dir, split="train")
    all_rows = Library.load(data_dir, split="all").rows
    holdout = [r for r in all_rows if split_of(r["trace_id"]) == "holdout"]
    last: dict = {}

    benchmark_path = data_dir / "benchmark.json"
    stats_path = data_dir / "stats.json"

    @app.route("/api/stats")
    def _stats(_payload: dict) -> dict:
        corpus = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        return {
            "library": library.stats,
            "corpus": corpus,
            "holdout_traces": len(holdout),
            "tasks": [t.name for t in TASKS],
            "policies": {k: INTENDED_FAULT.get(k, "") for k in POLICIES},
        }

    @app.route("/api/benchmark")
    def _benchmark(_payload: dict) -> dict:
        if benchmark_path.exists():
            return json.loads(benchmark_path.read_text(encoding="utf-8"))
        return {"note": "No benchmark yet. Run: python -m postmortem.evaluate"}

    @app.route("/api/example")
    def _example(payload: dict) -> dict:
        """A held-out trace, with its replay label stripped out of the input."""
        pool = holdout or all_rows
        wanted = payload.get("policy")
        if wanted:
            pool = [r for r in pool if r["policy"] == wanted] or pool
        if not pool:
            return {"error": "no traces yet - run python -m postmortem.collect"}
        row = random.choice(pool)
        visible = {k: v for k, v in row.items()
                   if k not in ("breaking_step", "label_kind", "intended_fault", "recoverable_after_step")}
        return {
            "trace": visible,
            "replay_truth": {
                "breaking_step": row.get("breaking_step"),
                "label_kind": row.get("label_kind"),
                "fault_from_policy": row.get("intended_fault"),
            },
        }

    @app.route("/api/run")
    def _run(payload: dict) -> dict:
        """Run a live trajectory in the sandbox and attribute it immediately."""
        policy = payload.get("policy") or "wrong_file"
        task = payload.get("task") or TASKS[0].name
        if policy not in POLICIES:
            return {"error": f"unknown policy {policy}"}
        chosen = [t for t in TASKS if t.name == task] or [TASKS[0]]
        rows = collect(seeds=1, tasks=chosen, policies=[policy])
        row = rows[0]
        visible = {k: v for k, v in row.items()
                   if k not in ("breaking_step", "label_kind", "intended_fault", "recoverable_after_step")}
        a = attribute(visible, library)
        payload_out = dict(visible)
        payload_out["attribution"] = a.to_dict()
        last.clear()
        last.update(payload_out)
        return {
            "trace": visible,
            "attribution": a.to_dict(),
            "replay_truth": {
                "breaking_step": row.get("breaking_step"),
                "label_kind": row.get("label_kind"),
                "fault_from_policy": row.get("intended_fault"),
            },
            "seconds_to_label_by_replay": row.get("seconds_to_label"),
        }

    @app.route("/api/attribute")
    def _attribute(payload: dict) -> dict:
        trace = payload.get("trace")
        if isinstance(trace, str):
            try:
                trace = json.loads(trace)
            except json.JSONDecodeError as exc:
                return {"error": f"that is not valid JSON: {exc}"}
        if not isinstance(trace, dict) or "steps" not in trace:
            return {"error": "paste a trace object with a 'steps' list"}
        a = attribute(trace, library)
        out = dict(trace)
        out["attribution"] = a.to_dict()
        last.clear()
        last.update(out)
        return {"attribution": a.to_dict(), "steps": len(trace.get("steps", []))}

    def _report(payload: dict, kind: str) -> Raw:
        if not last:
            return Raw(b"Attribute a trace first.", "text/plain; charset=utf-8")
        name = payload.get("name") or f"{last.get('task','run')}-{last.get('policy','')}"
        stem = SAFE.sub("-", f"postmortem-{name}").strip("-")[:60]
        if kind == "pdf":
            return Raw(report_mod.pdf(last, name=name), "application/pdf", f"{stem}.pdf")
        if kind == "html":
            return Raw(report_mod.html(last, name=name).encode("utf-8"),
                       "text/html; charset=utf-8", f"{stem}.html")
        return Raw(report_mod.markdown(last, name=name).encode("utf-8"),
                   "text/markdown; charset=utf-8", f"{stem}.md")

    @app.route("/api/report.pdf")
    def _pdf(payload: dict) -> Raw:
        return _report(payload, "pdf")

    @app.route("/api/report.md")
    def _md(payload: dict) -> Raw:
        return _report(payload, "md")

    @app.route("/api/report.html")
    def _html(payload: dict) -> Raw:
        return _report(payload, "html")

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the Postmortem console")
    ap.add_argument("--port", type=int, default=8830)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args(argv)

    app = build_app(args.data)
    print(f"data: {args.data}", flush=True)
    app.serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
