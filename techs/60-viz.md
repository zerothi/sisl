# 60 · Visualization

Owns: `src/sisl/viz/*`, `src/sisl/_lazy_viz.py`, `src/sisl/_nodify.py`

Paths are relative to `src/sisl/`.

## Lazy loading

`import sisl` does not import `sisl.viz`.

- `_lazy_viz.py:set_viz_placeholders` sets `cls.plot` to a `PlotHandlerPlaceholder` on each class in `lazy_viz_classes`.
- The placeholder is a descriptor. The first access to `.plot` imports `sisl.viz`.
- `sisl.viz` removes the placeholders. Then `viz/_plotables_register.py` attaches the real plot handlers.

When you register a new plotable class, add it to `lazy_viz_classes` too.
Without it, `.plot` is not available before an explicit `import sisl.viz`.

## Dependencies

- `nodify` is required for viz. Without it, `import sisl.viz` raises `ModuleNotFoundError` with install hints.
- The backends are optional: `plotly`, `matplotlib`, `py3Dmol`, `bpy` (Blender).
  A missing backend becomes a `NotAvailableFigure` subclass. It raises only when a user selects it.
- The install extras are in `pyproject.toml`: `viz`, `viz-plotly`, `viz-matplotlib`, `viz-blender`, `viz-ase`.

## Architecture

A plot is a `nodify.Workflow` (`viz/plot.py`).

| Layer | Folder | Content |
|---|---|---|
| Data sources | `viz/data_sources/` | `DataSource` nodes: inputs that are evaluated lazily. |
| Data | `viz/data/` | Data classes (`BandsData`, `PDOSData`, ...). |
| Processors | `viz/processors/` | Functions that transform data. |
| Plotters | `viz/plotters/` | Functions that make plot actions. |
| Plots | `viz/plots/` | One function and one thin `Plot` class for each plot type. |
| Figures | `viz/figure/` | Backends. |

### Plot actions

Plotters do not draw. They return plot actions: `dict(method=..., args=..., kwargs=...)`.
`viz/plotters/plot_actions.py` makes one action function for each public `draw_*` method of `Figure`.
A `Figure` replays the actions.
Reason: plotters are independent of the backend, and the actions can be serialized.

### Plot behaviour

- `Plot.__getattr__` sends unknown attributes to the figure. Thus `plot.show()` works.
  It calls `.get()` first. This can start a computation.
- Use `update_inputs(...)` to change a plot. `update_settings` is deprecated (0.15, removal in 0.17).
- `Plot.plot_class_key()` is the class name without `Plot`, in lower case. It is the default registration name.

## Add a plot type

1. Write a function `xxx_plot(...)` in `viz/plots/`. It returns `get_figure(backend=..., plot_actions=...)`.
2. Add `class XxxPlot(Plot): function = staticmethod(xxx_plot)`. Export it in `viz/plots/__init__.py`.
3. Register it in `viz/_plotables_register.py` with `register_plotable` (`viz/_plotables.py`).
4. If the sisl class is new to viz, add it to `lazy_viz_classes`.

`register_plotable`:

- Adds a `ClassPlotHandler` (a `ClassDispatcher`) to the class on the first registration.
  Handlers of parent classes are chained.
- `obj.plot()` calls the default plot. Without a default it raises `TypeError`.
- `obj.plot.<name>()` calls one plot.
- `register_data_source` and `register_sile_method` are helpers for data classes and Siles.

## Add a backend

1. Subclass `Figure` (`viz/figure/figure.py`). Example: `PlotlyFigure` in `viz/figure/plotly.py`.
2. Implement `_init_figure` and the other `_init_figure_*` methods that the backend supports.
3. Override the `draw_*` methods. Some base `draw_*` methods fall back to simpler draws.
4. Add a guarded import and a `BACKENDS` entry in `viz/figure/__init__.py`.

`Figure.to(key)` converts a figure to another backend. It replays the plot actions.

## Testing

- Without `nodify`, `conftest.py` does not collect `sisl/viz`.
- Markers: `viz`, `processors`, `data`, `plots`, `plotters`.
- Backend tests use a parameter over `["plotly", "matplotlib"]` with `pytest.importorskip(backend)`.

## Known stale code

- `viz/_splot.py` says that it does not work with the current viz.
- `viz/processors/wavefunction.py` imports `sisl.viz.nodes.node`. That module does not exist.

Do not use these as examples.
