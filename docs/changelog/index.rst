.. _release_notes:

Release notes
#############

Sometimes the API of `sisl` changes and users needs to adapt their
scripts to the API changes.

Generally this can be accommodated by using a code ``if`` block:

.. code::
   
    if sisl.__version_tuple__[:3] >= (0, 13, 0):
       pass
    else:
       pass

this will allow one to reliably test different versions.

We will generally advice users to follow the latest releases as bug-fixes
may superseede API changes and bring more performance overall.

In any case the following list of release notes may be used to check changes
between versions.


.. toctree::
   :maxdepth: 1

   v0.13.0
   v0.12.2
   v0.12.1
   v0.12.0
   v0.11.0
   v0.10.0
   v0.9.8
   v0.9.7
   v0.9.6
   v0.9.5
   v0.9.4
   v0.9.3
   v0.9.2
   v0.9.1
   v0.9.0
   v0.8.5
   v0.8.4
   v0.8.3
   v0.8.2
   v0.8.1
   v0.8.0

