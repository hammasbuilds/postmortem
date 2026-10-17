"""Tests for the sandbox, the replay labelling and the attribution engine.

Each test is here because of something that actually went wrong while building
this: a verifier that failed every task, a fixture no agent could pass, a
destructive label for a step that destroyed nothing, and a verdict that
contradicted itself.
"""

from __future__ import annotations

import pytest

from postmortem.attribute import Library, attribute
from postmortem.collect import breaking_step, label_kind, recoverability
from postmortem.policies import INTENDED_FAULT, POLICIES, run_policy
from postmortem.report import markdown, pdf
from postmortem.sandbox import TASKS, Sandbox


def step(tool, ok=True, *, args=None, error="", output=""):
    return {"tool": tool, "ok": ok, "args": args or {}, "error": error, "output": output}


# --- the sandbox and its verifier ----------------------------------------------


@pytest.mark.parametrize("task", TASKS, ids=[t.name for t in TASKS])
def test_every_task_starts_broken(task):
    box = Sandbox(task=task)
    try:
        assert box.verify() is False, f"{task.name} passes before anything is fixed"
    finally:
        box.cleanup()


@pytest.mark.parametrize("task", TASKS, ids=[t.name for t in TASKS])
def test_every_task_is_solvable(task):
    """A task no agent can pass is a broken fixture, not a hard problem.

    One of the five was exactly that: its fix replaced the whole return statement
    with a bare expression, so the "fixed" code returned None. It showed up as 32
    runs labelled unrecoverable_from_start - one task times eight policies times
    four seeds - which is what pointed at the fixture.
    """
    steps, _ok, box = run_policy("competent", task, seed=1)
    try:
        assert _ok, f"{task.name} cannot be solved by the recovery policy"
        assert any(s.tool == "patch" and s.ok for s in steps)
    finally:
        box.cleanup()


def test_verifier_can_import_the_package_under_test():
    """The verifier ran with -I, which implies -P: the sandbox directory was not on
    sys.path, so every task failed with ModuleNotFoundError and every trajectory
    looked like a failure - including the competent one."""
    task = TASKS[0]
    steps, _ok, box = run_policy("competent", task, seed=2)
    try:
        tests = [s for s in steps if s.tool == "run_tests"]
        assert tests and tests[-1].ok
        assert "ModuleNotFoundError" not in (tests[-1].output or "")
    finally:
        box.cleanup()


def test_patch_against_missing_text_does_not_change_the_file():
    task = TASKS[0]
    box = Sandbox(task=task)
    try:
        before = box.snapshot()
        call = box.patch(task.target_file, "text that is not there", "x")
        assert not call.ok and "PatchDidNotApply" in call.error
        assert box.snapshot() == before
    finally:
        box.cleanup()


# --- replay labelling -----------------------------------------------------------


def test_replay_finds_no_fatal_step_when_nothing_was_damaged():
    task = TASKS[0]
    steps, _ok, box = run_policy("premature_success", task, seed=3)
    box.cleanup()
    recov = recoverability(task, steps, seed=3)
    assert all(recov), recov
    assert breaking_step(recov) is None
    assert label_kind(recov, "failure") == "no_fatal_step"


def test_replay_finds_a_fatal_step_when_the_code_was_broken():
    task = TASKS[0]
    steps, _ok, box = run_policy("overconfident", task, seed=3)
    box.cleanup()
    recov = recoverability(task, steps, seed=3)
    k = breaking_step(recov)
    assert k is not None, recov
    assert label_kind(recov, "failure") == "destructive_step"


def test_label_kind_separates_the_three_situations():
    assert label_kind([True, True], "success") == "succeeded"
    assert label_kind([True, False], "failure") == "destructive_step"
    assert label_kind([True, True], "failure") == "no_fatal_step"
    assert label_kind([False, False], "failure") == "unrecoverable_from_start"


# --- the attribution engine -----------------------------------------------------


def test_success_is_not_attributed():
    a = attribute({"outcome": "success", "steps": [step("run_tests")]})
    assert a.label_kind == "succeeded"
    assert a.breaking_step is None and a.fault_class == ""


def test_a_failed_patch_is_never_the_point_of_no_return():
    """It changed nothing, so it cannot be fatal - replay says no_fatal_step."""
    a = attribute({"outcome": "failure", "steps": [
        step("read_file", args={"path": "pkg/text.py"}),
        step("patch", False, error="PatchDidNotApply: old text not found"),
        step("run_tests", False, output="AssertionError", error="tests failed"),
    ]})
    assert a.fault_class == "stale_context"
    assert a.label_kind == "no_fatal_step"
    assert a.breaking_step is None
    assert a.first_wrong_step == 1


def test_an_edit_that_breaks_the_build_is_the_point_of_no_return():
    a = attribute({"outcome": "failure", "steps": [
        step("patch", True, args={"path": "pkg/text.py"}),
        step("run_tests", False, output="SyntaxError: invalid syntax", error="tests failed"),
    ]})
    assert a.breaking_step == 0
    assert a.label_kind == "destructive_step"


def test_tool_misuse_is_reported_without_claiming_damage():
    """The verdict used to read 'Step 1 broke the run - no fatal step'."""
    a = attribute({"outcome": "failure", "steps": [
        step("list_files"),
        step("read_file", False, args={"path": "pkg"}, error="IsADirectoryError: pkg"),
        step("run_tests", False, output="AssertionError", error="tests failed"),
    ]})
    assert a.fault_class == "tool_misuse"
    assert a.breaking_step is None
    assert a.first_wrong_step == 1


def test_repeated_identical_calls_are_called_out():
    steps = [step("grep", True, args={"pattern": "def "}) for _ in range(5)]
    steps.append(step("run_tests", False, error="tests failed"))
    a = attribute({"outcome": "failure", "steps": steps})
    assert a.fault_class == "no_progress"
    assert a.breaking_step is None


def test_a_run_that_changed_nothing_is_a_premature_stop():
    a = attribute({"outcome": "failure", "steps": [
        step("list_files"), step("read_file", args={"path": "pkg/text.py"}),
        step("run_tests", False, output="AssertionError", error="tests failed"),
    ]})
    assert a.fault_class in ("premature_stop", "wrong_target")
    assert a.breaking_step is None


def test_an_unverified_edit_is_named():
    a = attribute({"outcome": "failure", "steps": [
        step("patch", True, args={"path": "pkg/text.py"}),
    ]})
    assert a.fault_class == "unverified_edit"


def test_every_fault_class_has_advice():
    from postmortem.attribute import ADVICE, FAULT_CLASSES

    for cls in FAULT_CLASSES:
        assert ADVICE.get(cls), cls


def test_advice_exists_for_every_policy_fault():
    from postmortem.attribute import ADVICE

    for policy, fault in INTENDED_FAULT.items():
        if fault:
            assert ADVICE.get(fault), f"{policy} -> {fault}"


def test_empty_library_does_not_break_attribution():
    a = attribute({"outcome": "failure", "steps": [step("run_tests", False, error="tests failed")]},
                  Library(rows=[]))
    assert a.fault_class
    assert a.neighbours == []


def test_all_policies_produce_steps():
    task = TASKS[1]
    for name in POLICIES:
        steps, _ok, box = run_policy(name, task, seed=5)
        box.cleanup()
        assert steps, name
        assert all(s.tool for s in steps), name


# --- reports --------------------------------------------------------------------


def _attributed_trace():
    trace = {"outcome": "failure", "task": "t", "policy": "p", "steps": [
        step("patch", True), step("run_tests", False, output="SyntaxError", error="tests failed")]}
    a = attribute(trace)
    out = dict(trace)
    out["attribution"] = a.to_dict()
    return out


def test_pdf_report_is_a_pdf():
    blob = pdf(_attributed_trace(), name="test")
    assert blob.startswith(b"%PDF-1.4")
    assert b"startxref" in blob and blob.rstrip().endswith(b"%%EOF")


def test_markdown_report_states_both_facts():
    md = markdown(_attributed_trace(), name="test")
    assert "Point of no return" in md
    assert "First wrong move" in md
    assert "What to change" in md
