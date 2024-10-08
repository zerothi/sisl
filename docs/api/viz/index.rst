.. _viz:

=============
Visualization
=============

.. module:: sisl.viz

The visualization module contains tools to plot common visualizations, as well
as to create custom visualizations that support multiple plotting backends
automatically.

Plot classes
-----------------

Plot classes are workflow classes that implement some specific plotting.

.. autosummary::
    :toctree: generated/

    Plot
    BandsPlot
    FatbandsPlot
    GeometryPlot
    SitesPlot
    GridPlot
    WavefunctionPlot
    PdosPlot
    AtomicMatrixPlot

Utilities
---------

Utilities to build custom plots

.. autosummary::
    :toctree: generated/

    get_figure
    merge_plots
    subplots
    animation
    Figure
