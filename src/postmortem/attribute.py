"""Attribution: which step broke the run, and what kind of failure it was.

Methodology, and why no model
-----------------------------
The published work on agent failure attribution agrees on one thing that decides
this design: **full traces beat output-only evidence by a wide margin** (one 2026
study reports attribution accuracy up to 76% higher with the whole trace than
with final output alone). So the engine reads every step, not the verdict.

It works in three layers, cheapest first, and says which layer answered:

1. **Causal rules over the trace.** A failed tool call whose output is then used
   downstream, an edit never followed by a test, the same action repeated with
   the same result, a budget spent without progress. These are the failure shapes
   the taxonomies name, and they are decidable from the trace.
2. **Progress accounting.** Each step is scored for whether it changed the world
   (an applied patch), narrowed it (a search that returned fewer places to look)
   or neither. The breaking step is usually the last step after which progress
   stops - and a run where nothing ever changed the world has no destructive
   step at all, which is a different diagnosis.
3. **Retrieval over labelled past failures** (BM25, stdlib): the nearest
   replay-labelled trajectory suggests the fault class. This is the only layer
   that needs the corpus, and the UI says when it was used.

Replay is the ground truth and it is not available at review time - a user pastes
one trace and cannot re-run it - so the engine predicts what replay would have
said, and ``evaluate.py`` scores it against replay on the held-out corpus.
"""

from __future__ import annotations

import collections
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

FAULT_CLASSES = (
    "wrong_target",
    "stale_context",
    "tool_misuse",
    "no_progress",
    "unverified_edit",
    "premature_stop",
    "unrecoverable_task",
)

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]+")


def tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(self.len)) if docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in collections.Counter(doc).items():
                self.postings[term].append((i, tf))
        self.n = len(docs)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5)) if df else 0.0

    def search(self, query: list[str], limit: int = 5) -> list[tuple[int, float]]:
        scores: dict[int, float] = collections.defaultdict(float)
        for term in query:
            idf = self.idf(term)
            if idf == 0.0:
                continue
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]


@dataclass
class Library:
    """Replay-labelled trajectories, for the retrieval layer."""

    rows: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str, *, split: str = "train") -> Library:
        p = Path(path)
        rows: list[dict] = []
        if p.is_dir():
            p = p / "traces.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if split == "all" or split_of(row["trace_id"]) == split:
                        rows.append(row)
        return cls(rows=rows)

    def __post_init__(self) -> None:
        self.bm25 = BM25([tokens(_trace_text(r)) for r in self.rows])

    @property
    def stats(self) -> dict:
        return {
            "trajectories": len(self.rows),
            "label_kinds": dict(collections.Counter(r.get("label_kind", "") for r in self.rows)),
            "tasks": sorted({r["task"] for r in self.rows}),
            "policies": sorted({r["policy"] for r in self.rows}),
        }


def split_of(trace_id: str, *, holdout: float = 0.3) -> str:
    """Hashed split so the retrieval library and the scoring set never overlap."""
    h = int(trace_id[:8], 16) / 0xFFFFFFFF
    return "holdout" if h < holdout else "train"


def _trace_text(row: dict) -> str:
    bits = [row.get("task", ""), ""]
    for s in row.get("steps", []):
        bits.append(f"{s.get('tool','')} {s.get('intent','')} {s.get('error','')}")
    return " ".join(bits)


@dataclass
class Attribution:
    breaking_step: int | None
    first_wrong_step: int | None
    label_kind: str
    fault_class: str
    confidence: float
    layer: str
    evidence: list[dict] = field(default_factory=list)
    neighbours: list[dict] = field(default_factory=list)
    advice: str = ""
    step_notes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "breaking_step": self.breaking_step,
            "first_wrong_step": self.first_wrong_step,
            "label_kind": self.label_kind,
            "fault_class": self.fault_class,
            "confidence": round(self.confidence, 3),
            "layer": self.layer,
            "evidence": self.evidence,
            "neighbours": self.neighbours,
            "advice": self.advice,
            "step_notes": self.step_notes,
        }


ADVICE = {
    "wrong_target": "The run edited something the failing test never touches. Make the agent name the "
                    "file it believes is at fault and check that the failure mentions it.",
    "stale_context": "An edit was applied against text the file no longer contained. Re-read before "
                     "each edit, or make edits fail loudly when their context is gone.",
    "tool_misuse": "A tool was called with arguments of the wrong kind. Validate arguments at the "
                   "tool boundary so the run stops at the mistake instead of carrying it forward.",
    "no_progress": "The run repeated the same action without changing anything. Detect repeated "
                   "identical calls and force a different move.",
    "unverified_edit": "The run changed files and never ran the test afterwards. Require a check "
                       "after every edit.",
    "premature_stop": "Nothing fatal happened - the run simply stopped before the task was done. "
                      "Make the stop condition the verifier's verdict, not the agent's opinion.",
    "unrecoverable_task": "No prefix of this run was recoverable, so the task or the environment is "
                          "the suspect, not a step.",
}


def _progress(steps: list[dict]) -> list[dict]:
    """Per-step accounting: did this step change the world, narrow it, or neither?"""
    notes: list[dict] = []
    seen: collections.Counter = collections.Counter()
    for i, s in enumerate(steps):
        tool, ok = s.get("tool", ""), bool(s.get("ok"))
        key = f"{tool}:{json.dumps(s.get('args', {}), sort_keys=True)}"
        seen[key] += 1
        kind = "neither"
        if tool == "patch" and ok:
            kind = "changed the files"
        elif tool in ("grep", "read_file", "list_files") and ok:
            kind = "narrowed the search"
        elif tool == "run_tests":
            kind = "checked the result"
        notes.append(
            {
                "index": i,
                "tool": tool,
                "ok": ok,
                "effect": kind,
                "repeat": seen[key],
                "error": s.get("error", ""),
                # The timeline is unreadable without these: "patch" does not say
                # which file, and "grep" does not say what was searched for.
                "args": {k: str(v)[:120] for k, v in (s.get("args") or {}).items()},
            }
        )
    return notes


_BUILD_BREAK = ("SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError",
                "TabError", "NameError: name")


def _destroyed_at(notes: list[dict], steps: list[dict]) -> int | None:
    """The applied edit after which a check could no longer load the code.

    This is the one destructive signature a trace can show without re-running
    anything: the test stopped failing on an assertion and started failing on the
    code not loading. Returns the index of the edit that precedes it.
    """
    last_patch: int | None = None
    for n, raw in zip(notes, steps):
        if n["tool"] == "patch" and n["ok"]:
            last_patch = n["index"]
        elif n["tool"] == "run_tests" and last_patch is not None:
            blob = f"{raw.get('output', '')} {raw.get('error', '')}"
            if any(sig in blob for sig in _BUILD_BREAK):
                return last_patch
    return None


def attribute(trace: dict, library: Library | None = None) -> Attribution:
    """Predict what a replay would have concluded about this trace."""
    steps = trace.get("steps", [])
    notes = _progress(steps)
    evidence: list[dict] = []

    if not steps:
        return Attribution(None, None, "no_fatal_step", "premature_stop", 0.2, "rules",
                           advice=ADVICE["premature_stop"])

    outcome = trace.get("outcome")
    if outcome == "success":
        return Attribution(None, None, "succeeded", "", 0.95, "rules",
                           advice="The verifier passed. Nothing to attribute.",
                           step_notes=notes)

    patches = [n for n in notes if n["tool"] == "patch"]
    applied = [n for n in patches if n["ok"]]
    failed_calls = [n for n in notes if not n["ok"]]
    tests = [n for n in notes if n["tool"] == "run_tests"]
    repeats = [n for n in notes if n["repeat"] >= 3]

    # Layer 1: causal rules, in order of how specific the evidence is.
    # Two different facts, kept apart. "Breaking step" is the point of no return,
    # which only exists when something was damaged; "first wrong step" is where the
    # run first went off. Reporting one as the other produced a verdict that read
    # "Step 1 broke the run - no fatal step", which is a contradiction on its face.
    candidate: tuple[int | None, int | None, str, str, float] | None = None

    # Which step, if any, left the repository in a state recovery cannot fix. The
    # only trace-visible signature of that is an applied edit followed by a check
    # that fails to even load the code. Everything else is a wrong-but-reversible
    # move, and replay agrees: a failed patch changed nothing, so it cannot be
    # fatal - the first version called stale_context "destructive" and replay said
    # "no fatal step" every time.
    broke_the_build = _destroyed_at(notes, steps)

    stale = [n for n in patches if not n["ok"] and "PatchDidNotApply" in (n["error"] or "")]
    if stale:
        n = stale[0]
        candidate = (broke_the_build, n["index"],
                     "destructive_step" if broke_the_build is not None else "no_fatal_step",
                     "stale_context", 0.8)
        evidence.append({"rule": "an edit was applied against text the file no longer held",
                         "step": n["index"], "error": n["error"],
                         "note": "the edit did not apply, so nothing was damaged"})

    if candidate is None:
        misuse = [n for n in failed_calls if n["tool"] in ("read_file", "grep")
                  and any(k in (n["error"] or "") for k in ("IsADirectoryError", "re.error", "FileNotFoundError"))]
        if misuse:
            n = misuse[0]
            candidate = (broke_the_build, n["index"],
                         "destructive_step" if broke_the_build is not None else "no_fatal_step",
                         "tool_misuse", 0.72)
            evidence.append({"rule": "a tool was called with an argument of the wrong kind",
                             "step": n["index"], "error": n["error"]})

    if candidate is None and applied and not tests:
        n = applied[-1]
        candidate = (broke_the_build, n["index"],
                     "destructive_step" if broke_the_build is not None else "no_fatal_step",
                     "unverified_edit", 0.7)
        evidence.append({"rule": "files were changed and never checked afterwards", "step": n["index"]})

    if candidate is None and repeats:
        n = repeats[0]
        candidate = (None, n["index"], "no_fatal_step", "no_progress", 0.68)
        evidence.append({"rule": f"the same call was made {n['repeat']} times with the same result",
                         "step": n["index"]})

    if candidate is None and applied:
        # Something was changed, the tests still fail: the usual cause is that the
        # edit was not where the failure lives. Reversible unless it broke the build.
        n = applied[-1]
        candidate = (
            broke_the_build,
            n["index"],
            "destructive_step" if broke_the_build is not None else "no_fatal_step",
            "wrong_target",
            0.6,
        )
        evidence.append({"rule": "an edit applied cleanly and the check still failed",
                         "step": n["index"],
                         "note": "nothing was damaged, so recovery was still possible"
                         if broke_the_build is None else "a later check could not load the code"})

    if candidate is None:
        # Nothing was ever changed. The run simply stopped.
        last = notes[-1]
        candidate = (None, last["index"], "no_fatal_step", "premature_stop", 0.66)
        evidence.append({"rule": "no step changed any file, so no step can be the fatal one",
                         "last_step": last["index"]})

    step, first_wrong, kind, fault, conf = candidate
    layer = "rules"

    # Layer 3: retrieval, used to corroborate or to supply a class when the rules
    # were weak. It never silently overrides a confident rule.
    neighbours: list[dict] = []
    if library and library.rows:
        hits = library.bm25.search(tokens(_trace_text(trace)), limit=5)
        votes: collections.Counter = collections.Counter()
        shown: set[tuple] = set()
        for i, score in hits:
            row = library.rows[i]
            # One row per distinct kind of neighbour. The library holds many runs of
            # the same task and behaviour, and five identical lines are one piece of
            # evidence wearing five hats.
            signature = (row["task"], row.get("intended_fault", ""), row.get("label_kind", ""))
            if signature in shown:
                if row.get("intended_fault"):
                    votes[row["intended_fault"]] += score
                continue
            shown.add(signature)
            neighbours.append({
                "trace_id": row["trace_id"],
                "task": row["task"],
                "bm25": round(score, 2),
                "label_kind": row.get("label_kind", ""),
                "breaking_step": row.get("breaking_step"),
                "fault_from_replay": row.get("intended_fault", ""),
            })
            if row.get("intended_fault"):
                votes[row["intended_fault"]] += score
        if votes:
            top, top_score = votes.most_common(1)[0]
            if conf < 0.65:
                fault, conf, layer = top, min(0.75, conf + 0.1), "rules+retrieval"
                evidence.append({"rule": "nearest labelled failures agree on this class",
                                 "class": top, "bm25_total": round(top_score, 2)})
            elif top == fault:
                conf, layer = min(0.95, conf + 0.08), "rules+retrieval"
                evidence.append({"rule": "nearest labelled failures agree with the rule",
                                 "class": top})

    return Attribution(
        breaking_step=step,
        first_wrong_step=first_wrong,
        label_kind=kind,
        fault_class=fault,
        confidence=conf,
        layer=layer,
        evidence=evidence,
        neighbours=neighbours,
        advice=ADVICE.get(fault, ""),
        step_notes=notes,
    )
