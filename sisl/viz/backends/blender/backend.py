from ..templates.backend import Backend, MultiplePlotBackend
from ...plot import MultiplePlot

import numpy as np

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
        """ Clears the blender scene so that data can be reset"""

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

    def draw_line3D(self, x, y, z, line={}, name="", **kwargs):
        """Draws a line using a bezier curve."""
        # First, generate the curve object
        bpy.ops.curve.primitive_bezier_curve_add()
        # Then get it from the context
        curve_obj = bpy.context.object
        # And give it a name
        if name is None:
            name = ""
        curve_obj.name = name 

        # Retrieve the curve from the object
        curve = curve_obj.data
        # And modify some attributes to make it look cylindric
        curve.dimensions = '3D'
        curve.fill_mode = 'FULL'
        width = line.get("width")
        curve.bevel_depth = width if width is not None else 0.1
        curve.bevel_resolution = 10
        # Clear all existing splines from the curve, as we are going to add them
        curve.splines.clear()
        
        xyz = np.array([x,y,z], dtype=float).T

        # To be compatible with other frameworks such as plotly and matplotlib,
        # we allow x, y and z to contain None values that indicate discontinuities
        # E.g.: x=[0, 1, None, 2, 3] means we should draw a line from 0 to 1 and another
        # from 2 to 3.
        # Here, we get the breakpoints (i.e. indices where there is a None). We add
        # -1 and None at the sides o facilitate iterating.
        breakpoint_indices = [-1, *np.where(np.isnan(xyz).any(axis=1))[0], None]
        
        # Now loop through all segments using the known breakpoints
        for start_i, end_i in zip(breakpoint_indices, breakpoint_indices[1:]):
            # Get the coordinates of the segment
            segment_xyz = xyz[start_i+1: end_i]
            
            # If there is nothing to draw, go to next segment
            if len(segment_xyz) == 0:
                continue
            
            # Create a new spline (within the curve, we are not creating a new object!)
            segment = curve.splines.new("BEZIER")
            # Splines by default have only 1 point, add as many as we need
            segment.bezier_points.add(len(segment_xyz) - 1)
            # Assign the coordinates to each point
            segment.bezier_points.foreach_set('co', np.ravel(segment_xyz))

            # We want linear interpolation between points. If we wanted cubic interpolation,
            # we would set this parameter to 3, for example.
            segment.resolution_u = 1
        
        # Give a color to our new curve object if it needs to be colored.
        self._color_obj(curve_obj, line.get("color", None), line.get("opacity", 1))

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

    @classmethod
    def _color_obj(cls, obj, color, opacity=1):
        """Utiity method to quickly color a given object.

        Parameters
        -----------
        obj: blender Object
            object to be colored
        color: str or array-like of shape (3,)
            color, it is converted to rgb using `matplotlib.colors.to_rgb`
        opacity:
            the opacity that should be given to the object. It doesn't
            work currently.
        """
        color = cls._to_rgb_color(color)

        if color is not None:
            mat = bpy.data.materials.new("material")
            mat.use_nodes = True

            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (*color, opacity)

            obj.active_material = mat


class BlenderMultiplePlotBackend(MultiplePlotBackend, BlenderBackend):

    def draw(self, backend_info):
        children = backend_info["children"]
        # Start assigning each plot to a position of the layout
        for child in children:
            self._draw_child_in_scene(child)

    def _draw_child_in_ax(self, child):
        child.get_figure(clear_fig=False)

MultiplePlot.backends.register("blender", BlenderMultiplePlotBackend)
