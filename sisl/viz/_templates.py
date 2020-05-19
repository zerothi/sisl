'''
Plotly templates should be defined in this file
'''
import itertools

import plotly.io as pio
import plotly.graph_objs as go

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

# This will be the default one for the sisl.viz module
pio.templates.default = "sisl"
