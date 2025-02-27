:orphan:

Changelog
=========

This directory contains *news fragments* which are short files that contain a
small **ReST**-formatted text that will be added to the next what's new page.

Make sure to use full sentences with correct case and punctuation, and please
try to use Sphinx intersphinx using backticks. The fragment should have a
header line and an underline using ``------``.

Each file should be named like ``<PULL REQUEST>.<TYPE>.rst``, where
``<PULL REQUEST>`` is a pull request number, and ``<TYPE>`` is one of:

* ``feat``: New user facing features like ``kwargs``.
* ``fix``: A fix for the code-base, could be a bugfix, or behavioral.
* ``change``: Changes to API, and other operational changes.
* ``highlight``: Adds a highlight bullet point to use as a possibly highlight

It is possible to add more files with different categories (and text), but
same pull request if all are relevant. For example a new feature might change
the API of other related functions.

All categories should be formatted as simple git commits:
1) a one-line header, and possibly 2) a more detailed explanation (not required).

So for example: ``123.feat.rst`` would have the content::

    ``my_new_feature`` option for `my_favorite_function`

    The ``my_new_feature`` option is now available for `my_favorite_function`.
    To use it, write ``sisl.my_favorite_function(..., my_new_feature=True)``.

``highlight`` is usually formatted as bulled points making the fragment
``* This is a highlight``.

Note the use of single-backticks to get an internal link (assuming
``my_favorite_function`` is exported from the ``sisl`` namespace),
and double-backticks for code.

If you are unsure what pull request type to use, don't hesitate to ask in your
PR.

``towncrier`` is required to build the docs; it will be automatically run when
you build the docs locally. You can also run ``towncrier
build --draft --version 1.18`` if you want to get a preview of how your change
will look in the final release notes.

.. note::

    This README was adapted from the numpy changelog readme, which again was
    adapted from pytest:
    This README was adapted from the pytest changelog readme under the terms of
    the MIT licence.
