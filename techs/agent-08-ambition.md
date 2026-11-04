# agent-08 · Ambition

For agents that make new code, a new feature or a plan for one.
Use it with [agent-04 · Implement](agent-04-implement.md). The other agent rules stay in force.
The ideas come from the be-ambitious skill and its source article ([overview-02 · Attribution](overview-02-attribution.md)).

Most agents work at the size of the task that the user typed.
Aim higher: make the result better, not bigger.

## The rules

1. **Better, not bigger.** Raise the outcome, the standard or the set of methods.
   Do not add deliverables, options or side tasks. This agrees with
   [agent-04 · Write lean code](agent-04-implement.md#write-lean-code).
   If a proposal makes the work larger and not better, remove it.
2. **Use the methods of the expert, not their words.** When you use a field, use its methods and tools.
   Words from a field without its methods change nothing.
3. **An outside judge decides.** Do not grade your own work.
   The judge is something other than you or the user. See [Outside judges in sisl](#outside-judges-in-sisl).
4. **Treat a hard target as possible.** Set a specific, hard target. Then work as if it is possible.
   Report the true result against the judge. A missed hard target with true numbers is a good result.
   A false result is never acceptable.
5. **Get the access. Do not invent it.** If data, a tool or access is missing, name it.
   Tell what it would make possible. Do not invent a source, a tool or a result.
6. **Ambition raises the standard, not the permissions.**
   [agent-01 · Ask first](agent-01-harness.md#ask-first) and [agent-02](agent-02-git.md) stay in force.

## Select the types that apply

Use one or two types for most tasks. Do not use a type only to use it.
Speed (the same work in less time) is not a type here.

| Type | Use it when | Do not use it when |
|---|---|---|
| Altitude | The task serves an outcome beyond itself. | The task is the outcome. |
| Depth | The result must be correct and lasting, for example physics or a hot path. | The work is scratch or a one-time script. |
| Range | Users must understand the result, or the usual method has no more ideas. | A narrow fix has one correct answer. |

### Altitude: aim at the outcome

Ask what the user wants to happen next. What is the goal beyond the goal?
Write three steps, each more ambitious than the last. An example for "fix the wrong band structure of this Hamiltonian":

1. Fix the case, with a test that fails before the fix.
2. Find each case of this kind in sisl, and fix the cause, not the symptom.
3. Make this class of error impossible to miss, for example with a check of a physics invariant in the tests.

The top step often needs preparation, for example a reference data set or a benchmark.
Write that preparation in the plan.
If a step has no outside judge, go down one step.

### Depth: work like the best expert of the field

Go one level below the task, to the data, the equations and the numerics.
Answer these four questions before you write the plan:

1. What would a top expert do here that the user did not ask for?
2. What is the most raw material to start from? For example, the equations, the file format specification
   or the output of the reference code, not a summary.
3. Which methods and tools would the expert use?
   For example, a convergence study, an analytic limit, a symmetry, or a profile instead of a guess.
4. What, other than you or the user, shows that the result is correct?

Also:

- **Set a hard target.** For example, machine precision instead of "close enough",
  or 10 times faster when the user asked for 2 times.
  Keep [overview-01 · Precision is a requirement](overview-01-concepts.md#precision-is-a-requirement):
  speed never costs precision.
- **Test what everyone believes.** Reproduce the belief from the raw data before you build on it.

### Range: use other fields

Replace the nouns of the task with verbs. Then the verbs tell which fields can help.
For example, "a plot of the PDOS" becomes "show a student which orbitals cause a peak, in one look".
Then select the fields and give each a task:

- Numerical analysis: error bounds, stability, conditioning.
- High-performance computing: memory layout, vectorization, scaling ([dev-03 · Performance](dev-03-performance.md)).
- Other codes in the field: the conventions and formats that users already know.
- Teaching: what a master or PhD student needs to use the feature correctly ([dev-06 · Public documentation](dev-06-public-docs.md)).

Use the method of each field, not only its words.
If you can start subagents, give each field one subagent with its method and its tools.

## Outside judges in sisl

Use the strongest judge that you can run:

- An analytic result or a known value from the literature, for example the band structure of graphene.
- A comparison with a reference code or a reference output file (IO tests need `SISL_FILES_TESTS`).
- A physics invariant: hermiticity, a sum rule, a symmetry, gauge invariance, charge conservation.
- A convergence study: the result converges with k-points, grid spacing or cutoff.
- A test that fails before the change and passes after it ([dev-02 · Testing](dev-02-testing.md)).
- A benchmark or a regression test for a hot path ([dev-03 · Performance](dev-03-performance.md)).

## Output

Before you implement a new feature, give the maintainer a short plan in this order.
Make the plan as short as the task permits.

1. The goal beyond the goal, in one line.
2. The types that apply, each with one line why. Name the types that you did not use.
3. The plan for each type: the three Altitude steps, the four Depth answers and the hard target,
   the Range verbs and the fields with their methods.
4. What the plan needs: data, tools and access. Mark each as available or missing.
5. The outside judge and what counts as a pass.
6. The next step: the highest step that you can complete and check with the access that you have.

Then continue as follows:

- If the step only raises the standard of the task, start it.
- If the step replaces the task, changes the public API, or is in [agent-01 · Ask first](agent-01-harness.md#ask-first),
  ask the maintainer and wait.
- If even the first step needs access that you do not have, ask for it and stop.

Sources:
URL[10]: https://github.com/robonuggets/be-ambitious/blob/main/SKILL.md
URL[8]: https://felixrieseberg.com/how-to-be-more-ambitious-a-tutorial-for-humans-with-models/
