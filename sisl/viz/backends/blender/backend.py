from ..templates.backend import Backend, MultiplePlotBackend
from ...plot import MultiplePlot

import bpy


class BlenderBackend(Backend):

    figure = None

    def clear(self):
        """ Clears the blender scene so that data can be reset
        Parameters
        --------
        """
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False, confirm=False)

        return self

    @staticmethod
    def _to_rgb_color(color):

        if isinstance(color, str):
            try:
                import matplotlib.colors

                color = matplotlib.colors.to_rgb(color)
            except ModuleNotFoundError:
                raise ValueError("Blender does not understand string colors."+
                    "Please provide the color in rgb (tuple of length 3, values from 0 to 1) or install matplotlib so that we can convert it."
                )

        return color


class BlenderMultiplePlotBackend(MultiplePlotBackend, BlenderBackend):

    def draw(self, backend_info, childs):
        # Start assigning each plot to a position of the layout
        for child in childs:
            self._draw_child_in_scene(child)

    def _draw_child_in_ax(self, child):
        child.get_figure(clear_fig=False)

MultiplePlot._backends.register("blender", BlenderMultiplePlotBackend)