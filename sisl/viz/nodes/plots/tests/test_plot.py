import pytest
from sisl.viz.nodes import Plot
from sisl.viz.nodes.plotters import PlotterNode

def test_noinputs_plot_from_func():

    @Plot.from_func
    def draw_point():
        return PlotterNode.draw_scatter(x=[0], y=[2])

    plot = draw_point(backend="plotly").get()
    
    assert plot.data[0].x == (0,)
    assert plot.data[0].y == (2,)

def test_noinputs_plot_from_class():

    class DrawPoint(Plot):
        @staticmethod
        def _workflow():
            return PlotterNode.draw_scatter(x=[0], y=[2])

    plot = DrawPoint(backend="plotly").get()
    
    assert plot.data[0].x == (0,)
    assert plot.data[0].y == (2,)