Getting started
----------------

Installation
^^^^^^^^^^^^

Blender offers the possibility to control things using python. However, it has its own python interpreter and it's not easy to
install new packages in it. If we want to use ``sisl`` within it, our best bet is to create a virtual environment and then let
blender use that python interpreter. The environment that we create, though, **MUST have exactly the same python version** as
the one shipped with blender.

Following, you have a step by step guide to get blender ready for plotting with sisl:

1. **Install blender**. You can install by downloading it directly from their official webpage, or in any other way.
Check `their installation documentation <https://docs.blender.org/manual/en/latest/getting_started/installing/index.html>`_

In ubuntu we can install it with:

.. code-block:: bash

    snap install blender

2. **Find out blender's python version**. You should check what is the version that blender is
shipped with. Being `blender` the name of the executable, you can run:

.. code-block:: bash

    blender -b --python-expr "import sys; print(f'PYTHON VERSION: {sys.version}')"

In blender 3.6 it gives an output that looks like this:

.. code-block:: bash

    Blender 3.6.3 (hash d3e6b08276ba built 2023-09-21 06:13:29)
    PYTHON VERSION: 3.10.12 (main, Aug 14 2023, 22:14:01) [GCC 11.2.1 20220127 (Red Hat 11.2.1-9)]

    Blender quit

Therefore, we know that **blender 3.6.3 uses python 3.10.12.**

3. **Create an environment with that python version** and install sisl (*skip if you have it already*).
In this case, we will use conda as the environment manager, since it lets us very easily select the python version.
You probably don't need the exact micro version. In our case asking for ``3.10`` is enough:

.. code-block:: bash

    conda create -n blender-python python=3.10

Then install all the packages you want to use in blender:

.. code-block:: bash

    conda activate blender-python
    python -m pip install sisl[viz]

4. **Find the path to the python libraries of your environment**. There are many ways to get this.
In conda, this path is in the ``CONDA_PREFIX`` environment variable. So you can just:

.. code-block:: bash

    $> echo $CONDA_PREFIX
    /home/miniconda3/envs/blender-python

5. **Tell blender to use the libraries in your environment**. This is done with the ``BLENDER_SYSTEM_PYTHON`` variable,
so you need to define it somehow for the blender process. You can specify it every time you use blender:

.. code-block:: bash

    BLENDER_SYSTEM_PYTHON=/home/miniconda3/envs/blender-python blender

or set it in your initialization files (recommended). E.g. in linux you just include this line in ``~/.bashrc``:

.. code-block:: bash

    export BLENDER_SYSTEM_PYTHON=/home/miniconda3/envs/blender-python

If everything went right, you should now be able to:

.. code-block:: bash

    blender -b --python-expr "import sisl"

and it shouldn't raise any error. Congratulations, you are ready to use sisl with blender!

First steps
^^^^^^^^^^^

Now that you have everything set up, let's open blender.

At first, blender might look intimidating because of all the options that it has, but we'll
keep it very simple. Our aim is just to show you how to use ``sisl``, the rest is in your hands.
What you see in the center of the screen is the default cube, you can just delete it. If its selected,
just press ``Supr``.

Currently, you are now in the ``Layout`` tab. The easiest way to start programming is to go to the
``Scripting`` tab. It is the last tab at the right of the tool bar.

You should see an interactive console and a text editor to write our scripts. Let's make our first
plot using the console!

We want to plot graphene, so the simplest way is

.. code-block:: python

    import sisl as si
    geom_plot = si.geom.graphene().plot(backend="blender", bonds_scale=0.01)
    geom_plot.show()

If we write these lines on the console, we should get the graphene structure in the viewport.

This is the 3D model. To get an image, we need to **render**. Rendering is a process that generates an image
from a camera (in our case the camera is the black wireframe that we see in the viewport) and the 3D model (objects, materials, lighting...).
We can trigger our first render by pressing ``F12`` or ``Render > Render image``.

There are infinite things that you can tweak in blender, but one important thing to know about is the rendering engine.
For this image, you have used ``Eeve`` which is the default engine. It is very fast, which makes it suitable for real-time
rendering applications. For single images that you want to publish, it is **usually worth it to use the** ``Cycles``
**engine**. This engine does more complex calculations (following light rays as they travel through the scene). You can change it
in the right hand side of the window, by clicking the tab with the microwave icon (*Render properties*). This should give you more realistic
looking results.

Now you **know how to use sisl inside blender** play with all the settings of the ``GeometryPlot``, move the camera,
change the lighting, the background, etc... to **get amazing images for your talks or publications**!

Notice that not only ``GeometryPlot`` has support for blender, ``GridPlot`` and ``GeometryPlot`` also support it.
Try to plot a grid and let's see how it looks!
