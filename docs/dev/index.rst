
.. _dev.index:

Contributing to sisl
====================

The sisl code is open-source, and thus we encourage external users to contribute
back to the code base.

Even if you are a non-coder, your contributions are valuable to the community!
Please do remember that open-source projects benefits from interaction!

There are many aspects of useful contributions:

- Creating tutorials and extending documentation
- Code maintenance and development
- Finding bugs or typos
- Development of the website
- Missing a useful feature

We understand that people have different backgrounds, and thus different
experience in coding. We try to engage as much as possible with ticket creators, and
guide.

In the following of this document you will find information related to the specifics
of making a development workflow for `sisl`.


.. _dev.summary:

Summary of development process
------------------------------

You are free to choose your development environment, but for new Python developers,
we recommend using a ``conda`` environment because compilers are easier to set up.
To create one, install `miniforge`_ and then create a new environment:

.. code:: bash

   conda create -n sisl-dev -c conda-forge python compilers cmake pandoc

Then activate it doing:

.. code:: bash

   conda activate sisl-dev

Here is a short summary of how to do developments with `sisl`.

#. Install the development dependencies, see :ref:`here <install>`.

   Note, in particular if you want to build the documentation locally, then `pandoc`_
   is required.


#. If you are a first time contributor, you need to clone your forked repository
   and setup a few things.

   The procedure enables one to follow the upstream changes, while simultaneously
   have a fork where one has write access.

   * Go to `github.com/zerothi/sisl <sisl-git_>`_ and click the :guilabel:`Fork` button to
     create your own copy of the code base.

   * Clone the fork to your local machine:

     .. tab:: SSH

        .. code:: bash

           git clone git@github.com:<your-username>/sisl.git

     .. tab:: HTML

        .. code:: bash

           git clone https://github.com/<your-username>/sisl.git

     And move to the folder ``cd sisl``.

   * Add the upstream repository:

     .. code:: bash

        git remote add upstream https://github.com/zerothi/sisl.git

   * Enable the `pre-commit <https://pre-commit.com>`_ hooks

     .. code:: bash

        python -m pip install pre-commit
        pre-commit install

     This will run specific checks before you commit things to the repository.
     It ensures consistency in the project.

#. Installing the project in development mode.

   It is advised to install the project in *editable* mode for faster
   turn-around times.

   .. code:: bash

      python -m pip install -e .

   For further details, see
   :ref:`the editable|pip instructions <installation-pip>`.

#. Developing your contribution.

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


#. Submit your contribution:

   * Push your changes back to your fork on GitHub:

     .. code:: bash

        git push origin 42-enhancing-doc

   * Go to `sisl's pull request site <pr_>`_.
     The new branch will show up with a green Pull Request
     button. Make sure the title and message are clear, concise, and self-
     explanatory. Then click the button to submit it.

   * Likely, your contribution will need a comment for the release notes.
     Please add one in ``/changes/`` by following the instructions found in
     the ``/changes/README.rst``.

#. Review process.

   The maintainers of `sisl` will do their best to respond as fast as possible.
   But first ensure that the CI runs successfully, if not, maintainers will likely
   wait until it succeeds before taking any action.



Contribute external code
------------------------

External toolbox codes may be contributed `here <issue_>`_, then press
:guilabel:`Issue` and select :guilabel:`Contribute toolbox`.

There are two cases of external contributions:

#. If the code is integrable into sisl it will be merged into the sisl source.

#. If the code is showing how to use sisl to calculate some physical quantity but is not a general
   implementation, it will be placed in toolbox directory.

Either way, any contribution is very welcome!



Contribute additional tests
---------------------------

Additional test files should be added to `this repository <sisl-files_>`_.
Please follow the guidelines there, or open up an issue at that repository
for specific details.


Contribute to the docs
----------------------

To contribute to the documentation one needs to install `pandoc`_ first (see
:ref:`dev.summary`). Then follow these steps:

#. Sitting inside the `sisl` tree, install the `sisl` documentation via:

   .. code:: bash

      pip install -e .[docs]

#. Download tutorial files accompanying the `sisl` repository:

   .. code:: bash

      git submodule init
      git submodule update

#. **OPTIONAL**

   If you are not contributing specifically to the notebooks,
   you may consider deactivating their compilation by creating this environment
   variable to drastically speed up build time:

   .. code:: bash

      export _SISL_DOC_SKIP=notebook

#. Within the docs folder (``/docs``) do:

   .. code:: bash

      make

   This will build the documentation in the ``/docs/build/html`` folder. Open the
   ``docs/build/html/index.html`` to visualize the built documentation.

#. The easiest thing that you can do now is to modify one of the ``.rst`` files
   (reStructuredText, or reST) sitting in ``/docs``. Then build again (``make``)
   and check your changes in the browser.

#. Once happy with your changes, *push* them to your fork and create a PR following the
   instructions under *To submit your contribution* in :ref:`dev.summary`.
