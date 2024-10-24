.. currentmodule:: sisl.viz

.. _toc-sisl-viz:

Visualization
----------------------------

This is a full visualization framework developed specifically for ``sisl`` to visualize
all the processing that you can do with the package. The aim is to provide a **high-level API**
to generate plots through sisl, so that you can avoid **boilerplate code** and the iteration loop
with your results can be as fast as possible.

The plots that you can generate with it are **not bound to a specific visualization framework**. Instead, the users
can choose the one that they want based on their taste or on what is available in their environment. Currently,
there is support for visualizing the plots with `plotly`_, `matplotlib`_, `blender <https://www.blender.org/>`_. The flexibility of the framework
allows for the user to **extend the visualizing options** quite simply without modifying ``sisl``'s internal code.

The framework started as a GUI, but then evolved to make it usable by ``sisl`` users directly. Therefore,
it can serve as a very robust (highly tested) and featureful **backend to integrate visualizations into graphical interfaces**.
An example of this is `the sisl-gui package <https://pypi.org/project/sisl-gui/>`_.

Basic Tutorials
^^^^^^^^^^^^^^^

Following, you will find some tutorials that will introduce you to the framework.

.. nbgallery::
    :name: viz-tutorials-gallery

    basic-tutorials/Demo.ipynb


Showcase of plot classes
^^^^^^^^^^^^^^^^^^^^^^^^

The following notebooks will help you develop a deeper understanding of what each plot class is capable of.

.. nbgallery::
    :name: viz-plotly-showcase-gallery

    showcase/GeometryPlot.ipynb
    showcase/SitesPlot.ipynb
    showcase/GridPlot.ipynb
    showcase/AtomicMatrixPlot.ipynb
    showcase/BandsPlot.ipynb
    showcase/FatbandsPlot.ipynb
    showcase/PdosPlot.ipynb
    showcase/WavefunctionPlot.ipynb

Using it with blender
^^^^^^^^^^^^^^^^^^^^^

The sisl visualization framework is independent of the plotting backend used, but `blender <https://www.blender.org/>`_ is a special one. Therefore
we dedicate this section to it with the hope of making the usage of it less confusing.

.. nbgallery::
    :name: viz-plotly-blender

    blender/Getting started.rst
    blender/First animation.rst

Combining plots
^^^^^^^^^^^^^^^

Have two plots that you would like to see displayed together, maybe as an animation or subplots? You've come
to the right place!

.. nbgallery::
    :name: viz-plotly-combining-plots-gallery

    combining-plots/Intro to combining plots.ipynb

Do it yourself
^^^^^^^^^^^^^^

Whether you feel like you would like to customize the framework a bit to fit your needs, you are a `sisl`
developer or you are building a framework (e.g. GUI) around `sisl.viz`, this is your section!

.. nbgallery::
    :name: viz-plotly-diy-gallery

    diy/Building a new plot.ipynb
    diy/Adding new backends.ipynb


.. note::
    Consider contributing to the package if you build a useful extension. The community will appreciate it! :)
