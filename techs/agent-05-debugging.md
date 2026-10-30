# agent-05 · Debugging

For agents that find the cause of a bug, a failing test or a wrong result.
Extends [dev-08 · Debugging](dev-08-debugging.md). When you fix the bug, [agent-04 · Implement](agent-04-implement.md) also applies.
Put scratch files outside the repository ([agent-03 · Shared state](agent-03-parallel.md#shared-state)).

## When a test fails

This applies to each test run, also a run for a different task.

1. Find why the test fails. Do not assume the cause.
2. If the physics in the test is incorrect, tell the maintainer and ask for a decision.
3. Do not change the code to make the test pass until a correct physics test verifies the change.

A test that passes is not proof of correct physics.
A change that only makes the test pass can hide an incorrect result.

## Ask first

Ask the maintainer ([agent-01 · Ask first](agent-01-harness.md#ask-first)) if the cause is a decision in `techs/`,
or if a fix needs a change of the public API.
Do not add `xfail` or `skip` to get a green run.

## Report the cause

Report the cause, the evidence and the fix.
If you did not find the cause, report the reproduction and what you ruled out.
