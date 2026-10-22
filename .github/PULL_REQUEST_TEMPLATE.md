<!--
Keep the description lean, brief and to the point.
See techs/dev-07-git.md, section "Merge request description".
-->

<!-- Overview: the solution in one short paragraph. One line is sufficient. -->


<!-- Problem: if this solves a problem, tell it in one short paragraph. Remove it otherwise. -->
Closes #

<!--
Details: the important changes and decisions. Do not repeat the diff.
A bug fix names the test that shows the correct and the incorrect behavior.
-->


<!-- Feel free to remove check-list items aren't relevant to your change -->

 - [ ] Added tests for new/changed functions? A bug fix has a test that fails before the fix.
 - [ ] Documentation for functionality in `docs/`
 - [ ] Changes documented in `changes/<num>.<type>.rst`
 - [ ] AI-assisted code: commits carry an `Assisted-by` trailer, and a human reviewed each line.
 <!-- num can be either an issue or PR number.
 Note! You can do both in the same PR, if you want references to both!
 -->

<!--
AI-assisted code is welcome. You are responsible for each line you submit.

- Let the agent read AGENTS.md first. It points to the rules in techs/.
- Keep the code lean: only the code that the change needs.
  No unneeded options, abstractions, checks or copies of sisl functions.
- Mark each assisted commit with one trailer, and no Co-authored-by:
  Assisted-by: <agent>:<model>, for example Assisted-by: claude-code:claude-opus-5-5
- Review the full change yourself before you open the PR.
  Remove scratch files, tool configuration and generated files.
- Do not submit code that you cannot explain.
-->

<!--
Creating a PR will check whether the pre-commit hooks
have run, and if it fails, you should do this manually.

Please see here: https://zerothi.github.io/sisl/contribute.html
on how to enable the pre-commit hooks enabled in `sisl`.

The short message is:
- run `isort .` (version=6.0.0) at the top level
- run `black .` (version=25.1.0) at top-level
-->
