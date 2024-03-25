# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import collections
import itertools

import bpy
import numpy as np

from .figure import Figure


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
        for ani_spline, child_spline in zip(
            ani_obj.data.splines, child_obj.data.splines
        ):
            # And each spline has multiple points
            for ani_point, child_point in zip(
                ani_spline.bezier_points, child_spline.bezier_points
            ):
                # Set the position of that point
                ani_point.co = child_point.co
                ani_point.keyframe_insert(data_path="co", frame=frame)

        # Loop through all the materials that the object might have associated
        for ani_material, child_material in zip(
            ani_obj.data.materials, child_obj.data.materials
        ):
            ani_mat_inputs = ani_material.node_tree.nodes["Principled BSDF"].inputs
            child_mat_inputs = child_material.node_tree.nodes["Principled BSDF"].inputs

            for input_key in ("Base Color", "Alpha"):
                ani_mat_inputs[input_key].default_value = child_mat_inputs[
                    input_key
                ].default_value
                ani_mat_inputs[input_key].keyframe_insert(
                    data_path="default_value", frame=frame
                )


def add_atoms_frame(ani_objects, child_objects, frame):
    """Creates the frames for a child plot atoms.

    Given the objects of the Atoms collection in the animation, it uses
    the corresponding atoms in the child to set keyframes.

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
        # Set the atom position
        ani_obj.location = child_obj.location
        ani_obj.keyframe_insert(data_path="location", frame=frame)

        # Set the atom size
        ani_obj.scale = child_obj.scale
        ani_obj.keyframe_insert(data_path="scale", frame=frame)

        # Set the atom color and opacity
        ani_mat_inputs = (
            ani_obj.data.materials[0].node_tree.nodes["Principled BSDF"].inputs
        )
        child_mat_inputs = (
            child_obj.data.materials[0].node_tree.nodes["Principled BSDF"].inputs
        )

        for input_key in ("Base Color", "Alpha"):
            ani_mat_inputs[input_key].default_value = child_mat_inputs[
                input_key
            ].default_value
            ani_mat_inputs[input_key].keyframe_insert(
                data_path="default_value", frame=frame
            )


class BlenderFigure(Figure):
    """Generic canvas for the blender framework.

    This is the first experiment with it, so it is quite simple.

    Everything is drawn in the same scene. On initialization, a collections
    dictionary is started. The keys should be the local name of a collection
    in the canvas environment and the values are the actual collections.
    """

    # Experimental feature to adjust 2D plottings
    # _2D_scale =  (1, 1)

    _animatable_collections = {
        "Lines": {"add_frame": add_line_frame},
    }

    def _init_figure(self, *args, **kwargs):
        # This is the collection that will store everything related to the plot.
        self._collection = bpy.data.collections.new(f"sislplot_{id(self)}")
        self._collections = {}

    def _init_figure_animated(self, interpolated_frames: int = 5, **kwargs):
        self._animation_settings = {"interpolated_frames": interpolated_frames}
        return self._init_figure(**kwargs)

    def _iter_animation(self, plot_actions, interpolated_frames=5):
        interpolated_frames = self._animation_settings["interpolated_frames"]

        for i, section_actions in enumerate(plot_actions):
            frame = i * interpolated_frames

            sanitized_section_actions = []
            for action in section_actions:
                action_name = action["method"]
                if action_name.startswith("draw_"):
                    action = {
                        **action,
                        "kwargs": {**action.get("kwargs", {}), "frame": frame},
                    }

                sanitized_section_actions.append(action)

            yield sanitized_section_actions

    def draw_on(self, figure):
        self._plot.get_figure(backend=self._backend_name, clear_fig=False)

    def clear(self):
        """Clears the blender scene so that data can be reset"""

        for key, collection in self._collections.items():
            self.clear_collection(collection)

            bpy.data.collections.remove(collection)

        self._collections = {}

    def get_collection(self, key):
        if key not in self._collections:
            self._collections[key] = bpy.data.collections.new(key)
            self._collection.children.link(self._collections[key])

        return self._collections[key]

    def remove_collection(self, key):
        if key in self._collections:
            collection = self._collections[key]

            self.clear_collection(collection)

            bpy.data.collections.remove(collection)

            del self._collections[key]

    def clear_collection(self, collection):
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    def draw_line(
        self, x, y, name="", line={}, marker={}, text=None, row=None, col=None, **kwargs
    ):
        z = np.full_like(x, 0)
        # x = self._2D_scale[0] * x
        # y = self._2D_scale[1] * y
        return self.draw_line_3D(
            x,
            y,
            z,
            name=name,
            line=line,
            marker=marker,
            text=text,
            row=row,
            col=col,
            **kwargs,
        )

    def draw_scatter(
        self, x, y, name=None, marker={}, text=None, row=None, col=None, **kwargs
    ):
        z = np.full_like(x, 0)
        # x = self._2D_scale[0] * x
        # y = self._2D_scale[1] * y
        return self.draw_scatter_3D(
            x, y, z, name=name, marker=marker, text=text, row=row, col=col, **kwargs
        )

    def draw_line_3D(
        self, x, y, z, line={}, name="", collection=None, frame=None, **kwargs
    ):
        """Draws a line using a bezier curve."""
        if frame is not None:
            return self._animate_line_3D(
                x,
                y,
                z,
                line=line,
                name=name,
                collection=collection,
                frame=frame,
                **kwargs,
            )

        if collection is None:
            collection = self.get_collection(name)
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
        curve.dimensions = "3D"
        curve.fill_mode = "FULL"
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
        # -1 and None at the sides to facilitate iterating.
        breakpoint_indices = [-1, *np.where(np.isnan(xyz).any(axis=1))[0], None]

        # Now loop through all segments using the known breakpoints
        for start_i, end_i in zip(breakpoint_indices, breakpoint_indices[1:]):
            # Get the coordinates of the segment
            segment_xyz = xyz[start_i + 1 : end_i]

            # If there is nothing to draw, go to next segment
            if len(segment_xyz) == 0:
                continue

            # Create a new spline (within the curve, we are not creating a new object!)
            segment = curve.splines.new("BEZIER")
            # Splines by default have only 1 point, add as many as we need
            segment.bezier_points.add(len(segment_xyz) - 1)
            # Assign the coordinates to each point
            segment.bezier_points.foreach_set("co", np.ravel(segment_xyz))

            # We want linear interpolation between points. If we wanted cubic interpolation,
            # we would set this parameter to 3, for example.
            segment.resolution_u = 1

        # Give a color to our new curve object if it needs to be colored.
        self._color_obj(curve_obj, line.get("color", None), line.get("opacity", 1))

        return self

    def _animate_line_3D(
        self, x, y, z, line={}, name="", collection=None, frame=0, **kwargs
    ):
        if collection is None:
            collection = self.get_collection(name)

        # If this is the first frame, draw the object as usual
        if frame == 0:
            self.draw_line_3D(
                x,
                y,
                z,
                line=line,
                name=name,
                collection=collection,
                frame=None,
                **kwargs,
            )

        # Create a collection that we are just going to use to create new objects from which
        # to copy the properties.
        temp_collection_name = f"__animated_{name}"
        temp_collection = self.get_collection(temp_collection_name)
        self.clear_collection(temp_collection)

        self.draw_line_3D(
            x,
            y,
            z,
            line=line,
            name=name,
            collection=temp_collection,
            frame=None,
            **kwargs,
        )

        # Loop through all objects in the collections
        for ani_obj, child_obj in zip(collection.objects, temp_collection.objects):
            # Each curve object has multiple splines
            for ani_spline, child_spline in zip(
                ani_obj.data.splines, child_obj.data.splines
            ):
                # And each spline has multiple points
                for ani_point, child_point in zip(
                    ani_spline.bezier_points, child_spline.bezier_points
                ):
                    # Set the position of that point
                    ani_point.co = child_point.co
                    ani_point.keyframe_insert(data_path="co", frame=frame)

            # Loop through all the materials that the object might have associated
            for ani_material, child_material in zip(
                ani_obj.data.materials, child_obj.data.materials
            ):
                ani_mat_inputs = ani_material.node_tree.nodes["Principled BSDF"].inputs
                child_mat_inputs = child_material.node_tree.nodes[
                    "Principled BSDF"
                ].inputs

                for input_key in ("Base Color", "Alpha"):
                    ani_mat_inputs[input_key].default_value = child_mat_inputs[
                        input_key
                    ].default_value
                    ani_mat_inputs[input_key].keyframe_insert(
                        data_path="default_value", frame=frame
                    )

        # Remove the temporal collection
        self.remove_collection(temp_collection_name)

    def draw_balls_3D(
        self,
        x,
        y,
        z,
        name=None,
        marker={},
        row=None,
        col=None,
        collection=None,
        frame=None,
        **kwargs,
    ):
        if frame is not None:
            return self._animate_balls_3D(
                x,
                y,
                z,
                name=name,
                marker=marker,
                row=row,
                col=col,
                collection=collection,
                frame=frame,
                **kwargs,
            )

        if collection is None:
            collection = self.get_collection(name)

        bpy.ops.surface.primitive_nurbs_surface_sphere_add(
            radius=1, enter_editmode=False, align="WORLD"
        )
        template_ball = bpy.context.object
        bpy.context.collection.objects.unlink(template_ball)

        style = {
            "color": marker.get("color", "gray"),
            "opacity": marker.get("opacity", 1),
            "size": marker.get("size", 1),
        }

        for k, v in style.items():
            if (
                not isinstance(v, (collections.abc.Sequence, np.ndarray))
            ) or isinstance(v, str):
                style[k] = itertools.repeat(v)

        ball = template_ball
        for i, (x_i, y_i, z_i, color, opacity, size) in enumerate(
            zip(x, y, z, style["color"], style["opacity"], style["size"])
        ):
            if i > 0:
                ball = template_ball.copy()
                ball.data = template_ball.data.copy()

            ball.location = [x_i, y_i, z_i]
            ball.scale = (size, size, size)

            # Link the atom to the atoms collection
            collection.objects.link(ball)

            ball.name = f"{name}_{i}"
            ball.data.name = f"{name}_{i}"

            self._color_obj(ball, color, opacity=opacity)

    def _animate_balls_3D(
        self,
        x,
        y,
        z,
        name=None,
        marker={},
        row=None,
        col=None,
        collection=None,
        frame=0,
        **kwargs,
    ):
        if collection is None:
            collection = self.get_collection(name)

        # If this is the first frame, draw the object as usual
        if frame == 0:
            self.draw_balls_3D(
                x,
                y,
                z,
                marker=marker,
                name=name,
                row=row,
                col=col,
                collection=collection,
                frame=None,
                **kwargs,
            )

        # Create a collection that we are just going to use to create new objects from which
        # to copy the properties.
        temp_collection_name = f"__animated_{name}"
        temp_collection = self.get_collection(temp_collection_name)
        self.clear_collection(temp_collection)

        self.draw_balls_3D(
            x,
            y,
            z,
            marker=marker,
            name=name,
            row=row,
            col=col,
            collection=temp_collection,
            frame=None,
            **kwargs,
        )

        # Loop through all objects in the collections
        for ani_obj, child_obj in zip(collection.objects, temp_collection.objects):
            # Set the atom position
            ani_obj.location = child_obj.location
            ani_obj.keyframe_insert(data_path="location", frame=frame)

            # Set the atom size
            ani_obj.scale = child_obj.scale
            ani_obj.keyframe_insert(data_path="scale", frame=frame)

            # Set the atom color and opacity
            ani_mat_inputs = (
                ani_obj.data.materials[0].node_tree.nodes["Principled BSDF"].inputs
            )
            child_mat_inputs = (
                child_obj.data.materials[0].node_tree.nodes["Principled BSDF"].inputs
            )

            for input_key in ("Base Color", "Alpha"):
                ani_mat_inputs[input_key].default_value = child_mat_inputs[
                    input_key
                ].default_value
                ani_mat_inputs[input_key].keyframe_insert(
                    data_path="default_value", frame=frame
                )

        self.remove_collection(temp_collection_name)

    draw_scatter_3D = draw_balls_3D

    def draw_mesh_3D(
        self,
        vertices,
        faces,
        color=None,
        opacity=None,
        name="Mesh",
        row=None,
        col=None,
        **kwargs,
    ):
        col = self.get_collection(name)

        mesh = bpy.data.meshes.new(name)

        obj = bpy.data.objects.new(mesh.name, mesh)

        col.objects.link(obj)

        edges = []
        mesh.from_pydata(vertices, edges, faces.tolist())

        self._color_obj(obj, color, opacity)

    @staticmethod
    def _to_rgb_color(color):
        if isinstance(color, str):
            try:
                import matplotlib.colors

                color = matplotlib.colors.to_rgb(color)
            except ModuleNotFoundError:
                raise ValueError(
                    "Blender does not understand string colors."
                    + "Please provide the color in rgb (tuple of length 3, values from 0 to 1) or install matplotlib so that we can convert it."
                )

        return color

    @classmethod
    def _color_obj(cls, obj, color, opacity=1.0):
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
        if opacity is None:
            opacity = 1.0

        color = cls._to_rgb_color(color)

        if color is not None:
            mat = bpy.data.materials.new("material")
            mat.use_nodes = True

            BSDF_inputs = mat.node_tree.nodes["Principled BSDF"].inputs

            BSDF_inputs["Base Color"].default_value = (*color, 1)
            BSDF_inputs["Alpha"].default_value = opacity

            obj.active_material = mat

    def set_axis(self, *args, **kwargs):
        """There are no axes titles and these kind of things in blender.
        At least for now, we might implement it later."""

    def set_axes_equal(self, *args, **kwargs):
        """Axes are always "equal" in blender, so we do nothing here"""

    def show(self, *args, **kwargs):
        bpy.context.scene.collection.children.link(self._collection)


# class BlenderMultiplePlotBackend(MultiplePlotBackend, BlenderBackend):

#     def draw(self, backend_info):
#         children = backend_info["children"]
#         # Start assigning each plot to a position of the layout
#         for child in children:
#             self._draw_child_in_scene(child)

#     def _draw_child_in_ax(self, child):
#         child.get_figure(clear_fig=False)


# class BlenderAnimationBackend(BlenderBackend, AnimationBackend):

#     def draw(self, backend_info):

#         # Get the collections that make sense to implement. This property is defined
#         # in each backend. See for example BlenderGeometryBackend
#         animatable_collections = backend_info["children"][0]._animatable_collections
#         # Get the number of frames that should be interpolated between two animation frames.
#         interpolated_frames = backend_info["interpolated_frames"]

#         # Iterate over all collections
#         for key, animate_config in animatable_collections.items():

#             # Get the collection in the animation's instance
#             collection = self.get_collection(key)
#             # Copy all the objects from first child's collection
#             for obj in backend_info["children"][0].get_collection(key).objects:
#                 new_obj = obj.copy()
#                 new_obj.data = obj.data.copy()
#                 # Some objects don't have materials associated.
#                 try:
#                     new_obj.data.materials[0] = obj.data.materials[0].copy()
#                 except Exception:
#                     pass
#                 collection.objects.link(new_obj)

#             # Loop over all child plots
#             for i_plot, plot in enumerate(backend_info["children"]):
#                 # Calculate the frame number
#                 frame = i_plot * interpolated_frames
#                 # Ask the provided function to build the keyframes.
#                 animate_config["add_frame"](collection.objects, plot.get_collection(key).objects, frame=frame)


# class BlenderGeometryBackend(BlenderBackend, GeometryBackend):

#     _animatable_collections = {
#         **BlenderBackend._animatable_collections,
#         "Atoms": {"add_frame": add_atoms_frame},
#         "Unit cell": BlenderBackend._animatable_collections["Lines"]
#     }

#     def draw_1D(self, backend_info, **kwargs):
#         raise NotImplementedError("A way of drawing 1D geometry representations is not implemented for blender")

#     def draw_2D(self, backend_info, **kwargs):
#         raise NotImplementedError("A way of drawing 2D geometry representations is not implemented for blender")

#     def _draw_single_atom_3D(self, xyz, size, color="gray", name=None, opacity=1, vertices=15, **kwargs):

#         try:
#             atom = self._template_atom.copy()
#             atom.data = self._template_atom.data.copy()
#         except Exception:
#             bpy.ops.surface.primitive_nurbs_surface_sphere_add(radius=1, enter_editmode=False, align='WORLD')
#             self._template_atom = bpy.context.object
#             atom = self._template_atom
#             bpy.context.collection.objects.unlink(atom)

#         atom.location = xyz
#         atom.scale = (size, size, size)

#         # Link the atom to the atoms collection
#         atoms_col = self.get_collection("Atoms")
#         atoms_col.objects.link(atom)

#         atom.name = name
#         atom.data.name = name

#         self._color_obj(atom, color, opacity=opacity)

#     def _draw_bonds_3D(self, *args, line=None, **kwargs):
#         # Multiply the width of the bonds to 0.2, otherwise they look gigantic.
#         line = line or {}
#         line["width"] = 0.2 * line.get("width", 1)
#         # And call the method to draw bonds (which will use self.draw_line3D)
#         collection = self.get_collection("Bonds")
#         super()._draw_bonds_3D(*args, line=line, collection=collection, **kwargs)

#     def _draw_cell_3D_box(self, *args, width=None, **kwargs):
#         width = width or 0.1
#         # This method is only defined to provide a better default for the width in blender
#         # otherwise it looks gigantic, as the bonds
#         collection = self.get_collection("Unit cell")
#         super()._draw_cell_3D_box(*args, width=width, collection=collection, **kwargs)

# GeometryPlot.backends.register("blender", BlenderGeometryBackend)
