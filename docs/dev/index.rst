
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


.. _devSummary:
Summary of development process
------------------------------

You are free to choose your development environment, but we recommend using a conda virtual
environment because things are very easy to set up. To create one, install `miniforge`_
and then create a new python environment. E.g. to create an environment named ``sisl_dev``
with the latest python version do:

.. code:: bash

   conda create -n sisl_dev python

Then activate it doing:

.. code:: bash

   conda activate sisl_dev

Here is a short summary of how to do developments with `sisl`.

1. Install the development dependencies. They can be found :ref:`here<install>`. If you are in a
   conda environment, installing them is as simple as:

   .. code:: bash

      conda install -c conda-forge compilers cmake pandoc


2. If you are a first time contributor, you need to clone your forked repository
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

     This will run specific checks before you commit things to the repository.
     It ensures consistency in the project.

3. Installing the project in development mode.

   It is advised to install the project in *editable* mode for faster
   turn-around times.

   .. code:: bash

      python -m pip install -e .

   For further details, see
   :ref:`the editable|pip instructions <installation-pip>`.

4. Developing your contribution.

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

5. To submit your contribution:

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

6. Review process.

   The maintainers of `sisl` will do their best to respond as fast as possible.
   But first ensure that the CI runs successfully, if not, maintainers will likely
   wait until it succeeds before taking any action.


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


Contribute to the docs
----------------------

To contribute to the documentation one needs to install `pandoc` first (see
:ref:`Summary of development process<devSummary>`). Then follow these steps:

1. Sitting inside the `sisl` tree, install the `sisl` documentation via:

   .. code:: bash

      pip install -e .[docs]

2. Download tutorial files outside the `sisl` repository:

   .. code:: bash

      git submodule init
      git submodule update

3. **OPTIONAL** : If your are not contributing specifically to the notebooks,
   you may consider deactivating their compilation by commenting out (or eliminating)
   this line ``"nbsphinx",`` in ``sisl/docs/conf.py`` file. This will enormously speed up
   building times (see below).

4. Within the docs folder (``sisl/docs``) do:

   .. code:: bash

      make html

   This will build the documentation in the ``sisl/docs/build/html`` folder. Open any
   **.html** file sitting there in your browser to visualize the built docs. Note that
   `index.html` is the "home page" of the documentation.

5. The easiest thing that you can do now is to modify one of the **.rst** files
   (reStructuredText, or reST) sitting in ``sisl/docs``. Then build again (``make html``)
   and check your changes in the browser.

6. Once happy with your changes, *push* them to your fork and create a PR following the
   instructions in :ref:`Summary of development process<devSummary>`.
