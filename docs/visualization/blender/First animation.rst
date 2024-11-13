First animation
---------------

Below is a script that generates an animation of graphene breathing in blender:

.. code-block:: python

    import sisl as si
    from sisl.viz import merge_plots

    plots = []
    for color, opacity, scale in zip(["red", "orange", "green"], [1, 0.2, 1], [0.5, 1, 0.5]):
        geom_plot = si.geom.graphene().plot(backend="blender",
            atoms_style={"color": color, "opacity": opacity},
            bonds_scale=0.01,
            atoms_scale=scale
        )

        plots.append(geom_plot)

    merge_plots(*plots, backend="blender", composite_method="animation", interpolated_frames=50).show()

.. raw:: html

    <blockquote class="imgur-embed-pub" lang="en" data-id="AOfYrOD"><a href="https://imgur.com/AOfYrOD">View post on imgur.com</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
