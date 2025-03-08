
.. _devindex:

Contributing to sisl
====================

The sisl code is open-source, and thus we encourage external users to contribute
back to the code base.

Even if you are a non-coder, your contributions are valuable to the community!
Please do remember that open-source projects benefits from interaction!

There are many aspects of useful contributions:

- Code maintenance and development
- Creating tutorials and extending documentation
- Finding bugs or typos
- Development of the website
- Missing a useful feature

We understand that people have different backgrounds, and thus different
experience in coding. We try to engage as much as possible with ticket creators.

In the following of this document you will find information related to the specifics
of making a development workflow for `sisl`.



Summary of development process
------------------------------


Here is a short summary of how to do developments with `sisl`.


1. If you are a first time contributor, you need to clone your repository
   and setup a few things.

   The procedure enables one to follow the upstream changes, while simultaneously
   have a fork where one has write access.

   * Go to `github.com/zerothi/sisl <sisl-git_>`_ and click the "fork" button to
     create your own copy of the code base.

   * Clone the fork to your local machine:

     .. code:: bash

        git clone https://github.com/<your-username>/sisl.git

   * Add the upstream repository:

     .. code:: bash

        git remote add upstream https://github.com/zerothi/sisl.git

   * Enable the `pre-commit <https://pre-commit.com>`_ hooks

     .. code:: bash

        python -m pip install pre-commit
        pre-commit install

     This will run specific checks before you commit things to the repo.
     It ensures a consistency in the project.

2. Installing the project in development mode.

   It is advised to install the project in *editable* mode for faster
   turn-around times.
   Please follow :ref:`these instructions <installation-testing>`.

3. Developing your contribution.

   First start by ensuring you have the latest changes on the ``main``
   branch.

   .. code:: bash

      git checkout main
      git pull upstream main

   If you are fixing an already opened issue (say :issue:`42`) it is advised
   to name your branch according to the issue number following a sensible name:

   .. code:: bash

      git checkout -b 42-enhancing-doc

   If no issue has been created, then just name it sensibly.

   Do all your commits locally as you progress.

   Be sure to document your changes, and write sensible documentation
   for the API.

4. To submit your contribution:

   * Push your changes back to your fork on GitHub:

     .. code:: bash

        git push origin 42-enhancing-doc

   * Go to `sisl's pull request site <pr_>`_.
     The new branch will show up with a green Pull Request
     button. Make sure the title and message are clear, concise, and self-
     explanatory. Then click the button to submit it.

   * Likely, your contribution will need a comment for the release notes.
     Please add one in ``/changes/`` by following the instructions found in
     the ``README.rst`` there.

5. Review process.

   The maintainers of `sisl` will do their best to respond as fast as possible.
   But first ensure that the CI runs successfully, if not, maintainers will likely
   wait untill it succeeds before taking any action.


Contribute external code
------------------------

External toolbox codes may be contributed `here <issue_>`_, then press
"Issue" and select *Contribute toolbox*.

There are two cases of external contributions:

1. If the code is directly integrable into sisl it will be merged into the sisl source.

2. If the code is showing how to use sisl to calculate some physical quantity but is not a general
   implementation, it will be placed in toolbox directory.

Either way, any contribution is very welcome.



Contribute additional tests
---------------------------

Additional test files should be added to `this repository <sisl-files_>`_.
Please follow the guidelines there, or open up an issue at that repository
for specific details.
