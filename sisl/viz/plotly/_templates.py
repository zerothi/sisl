"""
Plotly templates should be defined in this file
"""
import itertools

import plotly.io as pio
import plotly.graph_objs as go

__all__ = ["get_plotly_template", "add_plotly_template",
    "set_default_plotly_template", "available_plotly_templates"]


def get_plotly_template(name):
    """
    Gets a plotly template from plotly global space.

    Doing `get_plotly_template(name)` is equivalent to
    `plotly.io.templates[name]`.

    Parameters
    ----------
    name: str
        the name of the plotly template
    """
    return pio.templates[name]


def add_plotly_template(name, template, default=False):
    """
    Adds a plotly template to plotly's register.

    In this way the visualization module can profit from it.

    Parameters
    -----------
    name: str
        the name of the plotly template that you want to add
    template: dict or plotly.graph_objs.layout.Template
        the template that you want to add. 
        See https://plotly.com/python/templates/ to understand how they work.
    default: bool, optional
        whether this template should be set as the default during this runtime.

        If you want a permanent default, consider using the opportunity that 'user_customs'
        gives you to customize the sisl visualization package by acting every time the
        package is imported.
    """
    pio.templates[name] = template

    if default:
        set_default_plotly_template(name)

    return


def set_default_plotly_template(name):
    """
    Sets a template as the default during this runtime.

    If you want a permanent default, consider using the opportunity that 'user_customs'
    gives you to customize the sisl visualization package by acting every time the
    package is imported.

    Parameters
    -----------
    name: str
        the name of the template that you want to use as default
    """
    pio.templates.default = name


def available_plotly_templates():
    """
    Gets a list of the plotly templates that are currently available.

    Returns
    ---------
    list
        the list with all the template's names.
    """
    list(pio.templates.keys())

pio.templates["sisl"] = go.layout.Template(
    layout={
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        **{f"{ax}_{key}": val for ax, (key, val) in itertools.product(
            ("xaxis", "yaxis"),
            (("visible", True), ("showline", True), ("linewidth", 1), ("mirror", True),
             ("color", "black"), ("showgrid", False), ("gridcolor", "#ccc"), ("gridwidth", 1),
             ("zeroline", False), ("zerolinecolor", "#ccc"), ("zerolinewidth", 1),
             ("ticks", "outside"), ("ticklen", 5), ("ticksuffix", " "))
        )},
        "hovermode": "closest",
        "scene": {
            **{f"{ax}_{key}": val for ax, (key, val) in itertools.product(
                ("xaxis", "yaxis", "zaxis"),
                (("visible", True), ("showline", True), ("linewidth", 1), ("mirror", True),
                 ("color", "black"), ("showgrid",
                                      False), ("gridcolor", "#ccc"), ("gridwidth", 1),
                    ("zeroline", False), ("zerolinecolor",
                                          "#ccc"), ("zerolinewidth", 1),
                    ("ticks", "outside"), ("ticklen", 5), ("ticksuffix", " "))
            )},
        }
        #"editrevision": True
        #"title": {"xref": "paper", "x": 0.5, "text": "Whhhhhhhat up", "pad": {"b": 0}}
    },
)

pio.templates["sisl_dark"] = go.layout.Template(
    layout={
        "plot_bgcolor": "black",
        "paper_bgcolor": "black",
        **{f"{ax}_{key}": val for ax, (key, val) in itertools.product(
            ("xaxis", "yaxis"),
            (("visible", True), ("showline", True), ("linewidth", 1), ("mirror", True),
             ("color", "white"), ("showgrid",
                                  False), ("gridcolor", "#ccc"), ("gridwidth", 1),
             ("zeroline", False), ("zerolinecolor", "#ccc"), ("zerolinewidth", 1),
             ("ticks", "outside"), ("ticklen", 5), ("ticksuffix", " "))
        )},
        "font": {'color': 'white'},
        "hovermode": "closest",
        "scene": {
            **{f"{ax}_{key}": val for ax, (key, val) in itertools.product(
                ("xaxis", "yaxis", "zaxis"),
                (("visible", True), ("showline", True), ("linewidth", 1), ("mirror", True),
                 ("color", "white"), ("showgrid",
                                      False), ("gridcolor", "#ccc"), ("gridwidth", 1),
                    ("zeroline", False), ("zerolinecolor",
                                          "#ccc"), ("zerolinewidth", 1),
                    ("ticks", "outside"), ("ticklen", 5), ("ticksuffix", " "))
            )},
        }
        #"editrevision": True
        #"title": {"xref": "paper", "x": 0.5, "text": "Whhhhhhhat up", "pad": {"b": 0}}
    },
)

# This will be the default one for the sisl.viz.plotly module
pio.templates.default = "sisl"
