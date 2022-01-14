# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ..templates.backend import Backend, MultiplePlotBackend, AnimationBackend
from ...plot import MultiplePlot, Animation

import numpy as np

import bpy


def add_line_frame(ani_objects, child_objects, frame):
    """Creates the frames for a child plot lines.

    Given the objects of the lines collection in the animation, it uses
    the corresponding lines in the child to set keyframes.

    Parameters
    -----------
    ani_objects: CollectionObjects
        the objects of the Atoms collection in the animation.
    child_objects: CollectionObjects
        the objects of the Atoms collection in the child plot.
    frame: int
        the frame number to which the keyframe values should be set. 
    """
    # Loop through all objects in the collections
    for ani_obj, child_obj in zip(ani_objects, child_objects):
        # Each curve object has multiple splines
        for ani_spline, child_spline in zip(ani_obj.data.splines, child_obj.data.splines):
            # And each spline has multiple points
            for ani_point, child_point in zip(ani_spline.bezier_points, child_spline.bezier_points):
                # Set the position of that point
                ani_point.co = child_point.co
                ani_point.keyframe_insert(data_path="co", frame=frame)

        # Loop through all the materials that the object might have associated
        for ani_material, child_material in zip(ani_obj.data.materials, child_obj.data.materials):
            ani_mat_inputs = ani_material.node_tree.nodes["Principled BSDF"].inputs
            child_mat_inputs = child_material.node_tree.nodes["Principled BSDF"].inputs

            for input_key in ("Base Color", "Alpha"):
                ani_mat_inputs[input_key].default_value = child_mat_inputs[input_key].default_value
                ani_mat_inputs[input_key].keyframe_insert(data_path="default_value", frame=frame)


class BlenderBackend(Backend):
    """Generic backend for the blender framework.

    This is the first experiment with it, so it is quite simple.

    Everything is drawn in the same scene. On initialization, a collections
    dictionary is started. The keys should be the local name of a collection
    in the backend environment and the values are the actual collections.
    Plots should try to organize the items they draw in collections. However,
    as said before, this is just a proof of concept.
    """

    _animatable_collections = {
        "Lines": {"add_frame": add_line_frame},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This is the collection that will store everything related to the plot.
        self._collection = bpy.data.collections.new(f"sislplot_{self._plot.id}")
        self._collections = {}

    def draw_on(self, figure):
        self._plot.get_figure(backend=self._backend_name, clear_fig=False)

    def clear(self):
        """ Clears the blender scene so that data can be reset"""

        for key, collection in self._collections.items():

            for obj in collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)

            bpy.data.collections.remove(collection)

        self._collections = {}

    def get_collection(self, key):
        if key not in self._collections:
            self._collections[key] = bpy.data.collections.new(key)
            self._collection.children.link(self._collections[key])

        return self._collections[key]

    def draw_line3D(self, x, y, z, line={}, name="", collection=None, **kwargs):
        """Draws a line using a bezier curve."""
        if collection is None:
            collection = self.get_collection("Lines")
        # First, generate the curve object
        bpy.ops.curve.primitive_bezier_curve_add()
        # Then get it from the context
        curve_obj = bpy.context.object
        # And give it a name
        if name is None:
            name = ""
        curve_obj.name = name

        # Link the curve to our collection (remove it from the context one)
        context_col = bpy.context.collection
        if context_col is not collection:
            context_col.objects.unlink(curve_obj)
            collection.objects.link(curve_obj)

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

        xyz = np.array([x, y, z], dtype=float).T

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

            BSDF_inputs = mat.node_tree.nodes["Principled BSDF"].inputs

            BSDF_inputs["Base Color"].default_value = (*color, 1)
            BSDF_inputs["Alpha"].default_value = opacity

            obj.active_material = mat

    def show(self, *args, **kwargs):
        bpy.context.scene.collection.children.link(self._collection)


class BlenderMultiplePlotBackend(MultiplePlotBackend, BlenderBackend):

    def draw(self, backend_info):
        children = backend_info["children"]
        # Start assigning each plot to a position of the layout
        for child in children:
            self._draw_child_in_scene(child)

    def _draw_child_in_ax(self, child):
        child.get_figure(clear_fig=False)


class BlenderAnimationBackend(BlenderBackend, AnimationBackend):

    def draw(self, backend_info):

        # Get the collections that make sense to implement. This property is defined
        # in each backend. See for example BlenderGeometryBackend
        animatable_collections = backend_info["children"][0]._animatable_collections
        # Get the number of frames that should be interpolated between two animation frames.
        interpolated_frames = backend_info["interpolated_frames"]

        # Iterate over all collections
        for key, animate_config in animatable_collections.items():

            # Get the collection in the animation's instance
            collection = self.get_collection(key)
            # Copy all the objects from first child's collection
            for obj in backend_info["children"][0].get_collection(key).objects:
                new_obj = obj.copy()
                new_obj.data = obj.data.copy()
                # Some objects don't have materials associated.
                try:
                    new_obj.data.materials[0] = obj.data.materials[0].copy()
                except:
                    pass
                collection.objects.link(new_obj)

            # Loop over all child plots
            for i_plot, plot in enumerate(backend_info["children"]):
                # Calculate the frame number
                frame = i_plot * interpolated_frames
                # Ask the provided function to build the keyframes.
                animate_config["add_frame"](collection.objects, plot.get_collection(key).objects, frame=frame)


Animation.backends.register("blender", BlenderAnimationBackend)
MultiplePlot.backends.register("blender", BlenderMultiplePlotBackend)
