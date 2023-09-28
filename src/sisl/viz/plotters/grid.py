import sisl.viz.plotters.plot_actions as plot_actions
from sisl.viz.processors.grid import get_isos


def draw_grid(data, isos=[], colorscale=None, crange=None, cmid=None, smooth=False):

    to_plot = []
    
    ndim = data.ndim

    if ndim == 1:
        to_plot.append(
            plot_actions.draw_line(x=data.x, y=data.values)
        )
    elif ndim == 2:
        transposed = data.transpose("y", "x")

        cmin, cmax = crange if crange is not None else (None, None)

        to_plot.append(
            plot_actions.init_coloraxis(name="grid_color", cmin=cmin, cmax=cmax, cmid=cmid, colorscale=colorscale)
        )
        

        to_plot.append(
            plot_actions.draw_heatmap(values=transposed.values, x=data.x, y=data.y, name="HEAT", zsmooth="best" if smooth else False, coloraxis="grid_color")
        )

        dx = data.x[1] - data.x[0]
        dy = data.y[1] - data.y[0]

        iso_lines = get_isos(transposed, isos)
        for iso_line in iso_lines:
            iso_line['line'] = {
                "color": iso_line.pop("color", None),
                "opacity": iso_line.pop("opacity", None),
                "width": iso_line.pop("width", None),
                **iso_line.get("line", {})
            }
            to_plot.append(
                plot_actions.draw_line(**iso_line)
            )
    elif ndim == 3:
        isosurfaces = get_isos(data, isos)
        
        for isosurface in isosurfaces:
            to_plot.append(
                plot_actions.draw_mesh_3D(**isosurface)
            )    
    
    if ndim > 1:
        to_plot.append(
            plot_actions.set_axes_equal()
        )

    return to_plot