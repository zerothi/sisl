.. _release_notes:

*************
Release Notes
*************

Sometimes the API of `sisl` changes and users needs to adapt their
scripts to the API changes.

Generally this can be accommodated by using a code ``if`` block:

.. code-block::

    if sisl.__version_tuple__[:3] >= (0, 16, 0):
       pass
    else:
       pass

this will allow one to reliably test different versions.

We will generally advice users to follow the latest releases as bug-fixes
may supersede API changes and bring more performance overall.

In any case the following list of release notes may be used to check changes
between versions.


.. toctree::
   :maxdepth: 2

   0.16.2 <release/0.16.2-notes.rst>
   0.16.1 <release/0.16.1-notes.rst>

.. toctree::
   :maxdepth: 1

   0.16.0 <release/0.16.0-notes.rst>



Changelogs
----------

The old format changelogs (for older release <=0.15.2) is contained
here.

.. toctree::
   :maxdepth: 1

   changelog/index.rst
