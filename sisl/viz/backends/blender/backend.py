from ..templates.backend import Backend, MultiplePlotBackend
from ...plot import MultiplePlot

import bpy


class BlenderBackend(Backend):
    """Generic backend for the blender framework.

    This is the first experiment with it, so it is quite simple.

    Everything is drawn in the same scene. On initialization, a collections
    dictionary is started. The keys should be the local name of a collection
    in the backend environment and the values are the actual collections.
    Plots should try to organize the items they draw in collections. However,
    as said before, this is just a proof of concept.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._collections = {}

    def draw_on(self, figure):
        self._plot.get_figure(backend=self._backend_name, clear_fig=False)

    def clear(self):
        """ Clears the blender scene so that data can be reset
        Parameters
        --------
        """
        
        for key, collection in self._collections.items():
    
            for obj in collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
                
            bpy.data.collections.remove(collection)

            del self._collections[key]

    def get_collection(self, key):
        if key not in self._collections:
            self._collections[key] = bpy.data.collections.new(key)
            bpy.context.scene.collection.children.link(self._collections[key])
        
        return self._collections[key]

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

    def draw(self, backend_info):
        children = backend_info["children"]
        # Start assigning each plot to a position of the layout
        for child in children:
            self._draw_child_in_scene(child)

    def _draw_child_in_ax(self, child):
        child.get_figure(clear_fig=False)

MultiplePlot.backends.register("blender", BlenderMultiplePlotBackend)