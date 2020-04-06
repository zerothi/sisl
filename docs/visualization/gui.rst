The graphical interface
=========================

How to use it?
-------------

Well, it is pretty easy. Just make sure you are in your sisl virtual environment.

#### From a terminal you can open it like:

.. code:: bash
    python -m sisl.viz.GUI

#### Or inside a jupyter notebook:

.. code:: python

    from sisl.viz import GUI

    GUI.launch()

Then you will have access to the GUI session under `GUI.session` and you can interact with it and change it as you wish. Because, you know, some things just need to be coded :)

You can also change the current session using `GUI.set_session`.

This also applies to executing from the terminal. In that case an interactive console opens up.

Check out `this notebook tutorial <https://github.com/pfebrer96/Sisl-viz-tutorials/blob/master/GUI%20with%20Python%20Demo.ipynb>`_ for a sneak peek of how you can interact with the GUI session. You won't regret it!