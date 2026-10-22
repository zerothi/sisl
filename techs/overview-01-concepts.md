# overview-01 · Concepts

These principles apply to all code in sisl.
The other chapters assume them.

## Code carries little prose

Code has few comments.
A comment explains only a decision that is out of the ordinary. Keep it short.

- The WHY of a decision goes in the `techs/` chapter that owns the topic.
- Do not write comments that repeat WHAT the code does. The code shows that.
- When you touch code with long explanatory comments, remove them.
  Move the important text into `techs/` in the same commit.
- When a change makes a statement in `techs/` wrong, correct the chapter in the same commit.

Reason: lean code is faster to read. Decisions in `techs/` are easy to find and to challenge.

## Documentation is in the file

Doc-strings are not comments. Keep them.

- Each public function, class and method has a doc-string.
- Doc-strings use the numpy style. Sphinx reads them with `sphinx.ext.napoleon` (see `docs/conf.py`).
- The Sphinx build reads the doc-strings and makes the API documentation.

## Precision is a requirement

sisl is a physics code. Results must be exact to numerical precision.

- Check the correctness of each algorithm change.
- Keep the dtype of the input when possible. Do not change the precision silently.
- After a change, check correctness and consistency again.
- A fast result that is wrong has no value.

## Performance is a requirement

Users run sisl on large systems. Slow code blocks their work.
See [dev-03 · Performance](dev-03-performance.md).

## Few strong classes

- Prefer a small number of strong classes that are easy to extend.
- If a small generalization covers many cases, use it. Do not add many subclasses.
- Register new functionality on the existing classes.
  See [sisl-20 · Geometry](sisl-20-geometry.md) for the dispatch pattern.

## End-users guide the design

End-users are master students and new PhD students.

- Keep the API coherent. Similar methods have similar arguments.
- Make errors clear. Tell the user what to do.
- Make it possible to extend sisl without a change to sisl.
