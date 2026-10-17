# Postmortem

Hand it an agent run that went wrong. It tells you **where the run first went wrong**, **whether any
step was the point of no return**, what class of failure it was, and what to change — with the evidence
taken from the trace, not from a verdict.

Product 1 of three in [`THREE-PRODUCTS-PLAN.md`](../THREE-PRODUCTS-PLAN.md).

```
uv run python -m postmortem.app        # the console on http://127.0.0.1:8830
uv run python -m postmortem.collect    # run trajectories and label them by replay
uv run python -m postmortem.evaluate   # score attribution against replay
```

## Why it exists

| Dataset | Trajectories | How the failures were made |
|---|---|---|
| Who&When Pro | 12,326 | fault injected into a replayed success |
| Aegis | 9,533 | controlled fault generation |
| TRAJECT-Bench | 5,670 | tool-calling benchmark |
| AgenTracer | ~2,000 | counterfactual replay + programmatic fault injection |
| AgentRx | 115 | manual annotation |
| **trace-100k (ours)** | **160 runs / 800 steps** (and growing) | **the runs really failed, and the label comes from replay** |

The public ceiling is around 12k trajectories and almost all of it is fault *injection* into a prefix
that is known to succeed. What is missing is observed failure with a label derived from execution. This
corpus is small today and honest about it — the harness, the labelling and the measurement are built,
and the scale run queues for the GPU.

**The dataset itself is on Kaggle:** https://www.kaggle.com/datasets/muhammadhammas13/trace-100k
It is not in this repository - this repository is the harness that builds, labels and reads it.

## How it was built

**We built trace-100k.** 160 agent trajectories and 800 steps: each one an agent working on a real
repository through real tools — list, read, search, patch, run tests — with the task's own test
deciding the outcome and replay deciding the label. The repair tasks ship with this repository, so the
whole corpus rebuilds end to end with `python -m postmortem.collect`.

## What it is for

- **Research:** agent-reliability work in this portfolio — attributing a failed run to the step that
  caused it, and measuring attribution against replay rather than against opinion.
- **Product:** the **Postmortem** console in this repository; `python -m postmortem.app`.
- **Also usable for:** failure-mode taxonomies, trace analysis, and evaluating tooling that claims to
  explain why an agent run failed.

## Citing it

If you use trace-100k, or anything derived from it, cite this repository:

```bibtex
@misc{trace100k,
  author = {Muhammad Hammas},
  title  = {trace-100k: agent trajectories labelled by replay},
  year   = {2026},
  url    = {https://github.com/hammasbuilds/postmortem}
}
```

## How a label is made, and who decides

A failed run's **point of no return** is found by replaying it. For every prefix, the harness rebuilds
the sandbox, re-executes that prefix exactly, hands the result to the recovery policy and asks the
task's own test. A step is the point of no return when recovery still succeeds from the state before it
and fails from the state after it.

That makes the label a property of the execution. It also makes it expensive — 160 trajectories took
**399 seconds** of real verifier runs — which is why the corpus grows in the background rather than
being recomputed per request.

Replay produces three distinct verdicts, not two:

| Verdict | Meaning | In this corpus |
|---|---|---|
| `destructive_step` | a step left the repository in a state recovery cannot fix | 44 |
| `no_fatal_step` | every prefix was still recoverable — the run stopped, looped or misfired without doing damage | 70 |
| `succeeded` | the verifier passed | 46 |

**Where the runs come from.** Each trajectory is an agent working on a real repository through real
tools — list, read, search, patch, run tests — and the task's own test decides the outcome. The
behaviours driving them cover the failure *shapes* the published taxonomies name: wrong target, stale
context, tool misuse, no progress, unverified edit, premature stop. No model takes part in the runs or
in the labels; a model arm runs through the identical harness when a GPU is free.

## Attribution, and no model here either

The literature agrees that full traces beat output-only evidence by a wide margin (one 2026 study
reports up to 76% higher attribution accuracy from whole traces), so the engine reads every step:

1. **Causal rules** over the trace — an edit applied against text the file no longer held, a tool called
   with an argument of the wrong kind, files changed and never checked, the same call repeated with the
   same result.
2. **Progress accounting** — per step: did it change the world, narrow it, or neither.
3. **Retrieval** (BM25) over replay-labelled past failures, from a disjoint half of the corpus. It
   corroborates a rule or supplies a class when the rules were weak, and never silently overrides a
   confident rule. The UI names which layer answered.

## Measured against replay

36 held-out failing runs. The engine never sees the replay labels, and the retrieval library is a
disjoint split.

| | |
|---|---|
| Point of no return, exactly right | **0.861** |
| baseline: blame the last step | 0.056 |
| baseline: blame the first failing call | 0.111 |
| Diagnosis (which of the three verdicts) | **0.861** |
| baseline: output-only, predict the majority verdict | 0.639 |
| Fault class | **0.971** |

Recall by verdict: `no_fatal_step` 1.000 (23/23), `destructive_step` 0.615 (8/13) — the remaining
misses are runs where the damage is visible only to replay, not to the trace.

## Input → Output

```
IN   a trajectory: 4 steps, outcome failure, behaviour "overconfident"
OUT  Step 0 broke the run · wrong target → unverified edit · confidence 78%
     point of no return: step 0   first wrong move: step 0
     why: an edit applied cleanly and the check still failed; a later check could not load the code
     advice: require a check after every edit
     matches replay: replay re-ran every prefix and also found step 0 (destructive step)
     report: postmortem-browser-test-run.pdf (1,412 bytes), .md, printable .html
```

## The report, and why it is a PDF

A postmortem outlives the terminal it came from: it is attached to a ticket and read by someone who was
not there. `report.py` writes a real PDF with no dependency — Latin-script content, so the standard PDF
fonts every reader has built in are enough.

## Problems hit while building this

- **The verifier ran with `-I`, which implies `-P`**, so the sandbox directory was never on `sys.path`
  and every task failed with `ModuleNotFoundError`. Every single trajectory looked like a failure —
  including the competent one, which is what gave it away.
- **One of five tasks was impossible.** Its "fix" replaced the whole return statement with a bare
  expression, so the corrected code returned `None`. It appeared as 32 runs labelled
  `unrecoverable_from_start` — exactly one task × eight policies × four seeds, which pointed at the
  fixture rather than at the agents.
- **A failed patch was being called destructive.** It changes nothing, so it cannot be the point of no
  return, and replay said `no_fatal_step` every time. The engine now looks for the one destructive
  signature a trace can show: an applied edit followed by a check that can no longer load the code.
- **The verdict contradicted itself** — "Step 1 broke the run · no fatal step". *Where the run first went
  wrong* and *the point of no return* are two different facts; separating them took point-of-no-return
  accuracy from 0.611 to 0.861, because the engine stopped inventing fatal steps.
- **"No breaking step" was hiding two different situations**: every prefix still recoverable, versus
  never recoverable at all. They need different advice, so they are labelled separately.

## What it does NOT do

- It does not run a model, and it does not need one. The numbers above are what rules, progress
  accounting and lexical retrieval achieve against replay.
- No model takes part, in the runs or in the labels.
- It does not replay anything at review time — a pasted trace cannot be re-run. The engine predicts
  what replay *would* have said, and that prediction is what is scored.
- It misses roughly a third of destructive steps, because some damage is only visible by re-running.
- The corpus is 160 trajectories over 5 tasks. It is a harness with a number, not a finished dataset.
