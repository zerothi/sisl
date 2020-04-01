This is the graphical interface.

The interface is an extension of the [sisl visualization module](https://github.com/pfebrer96/sisl/tree/GUI/sisl/viz).

How to use it?
-------------

Well, it is pretty easy. Just make sure you are in your sisl virtual environment.

#### From a terminal you can open it like:

`python -m sisl.viz.GUI`

#### Or inside a jupyter notebook:

```python
from sisl.viz import GUI

GUI.launch()
```

Then you will have access to the GUI session under `GUI.session` and you can interact with it and change it as you wish. Because, you know, some things just need to be coded :)

You can also change the current session using `GUI.set_session`.

This also applies to executing from the terminal. In that case an interactive console opens up.