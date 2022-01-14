# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ....plots import GeometryPlot
from ..backend import BlenderBackend
from ...templates import GeometryBackend

import bpy

__all__ = ["BlenderGeometryBackend"]


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
        ani_mat_inputs = ani_obj.data.materials[0].node_tree.nodes["Principled BSDF"].inputs
        child_mat_inputs = child_obj.data.materials[0].node_tree.nodes["Principled BSDF"].inputs

        for input_key in ("Base Color", "Alpha"):
            ani_mat_inputs[input_key].default_value = child_mat_inputs[input_key].default_value
            ani_mat_inputs[input_key].keyframe_insert(data_path="default_value", frame=frame)


class BlenderGeometryBackend(BlenderBackend, GeometryBackend):

    _animatable_collections = {
        **BlenderBackend._animatable_collections,
        "Atoms": {"add_frame": add_atoms_frame},
        "Unit cell": BlenderBackend._animatable_collections["Lines"]
    }

    def draw_1D(self, backend_info, **kwargs):
        raise NotImplementedError("A way of drawing 1D geometry representations is not implemented for blender")

    def draw_2D(self, backend_info, **kwargs):
        raise NotImplementedError("A way of drawing 2D geometry representations is not implemented for blender")

    def _draw_single_atom_3D(self, xyz, size, color="gray", name=None, opacity=1, vertices=15, **kwargs):

        try:
            atom = self._template_atom.copy()
            atom.data = self._template_atom.data.copy()
        except:
            bpy.ops.surface.primitive_nurbs_surface_sphere_add(radius=1, enter_editmode=False, align='WORLD')
            self._template_atom = bpy.context.object
            atom = self._template_atom
            bpy.context.collection.objects.unlink(atom)

        atom.location = xyz
        atom.scale = (size, size, size)

        # Link the atom to the atoms collection
        atoms_col = self.get_collection("Atoms")
        atoms_col.objects.link(atom)

        atom.name = name
        atom.data.name = name

        self._color_obj(atom, color, opacity=opacity)

    def _draw_bonds_3D(self, *args, line=None, **kwargs):
        # Multiply the width of the bonds to 0.2, otherwise they look gigantic.
        line = line or {}
        line["width"] = 0.2 * line.get("width", 1)
        # And call the method to draw bonds (which will use self.draw_line3D)
        collection = self.get_collection("Bonds")
        super()._draw_bonds_3D(*args, line=line, collection=collection, **kwargs)

    def _draw_cell_3D_box(self, *args, width=None, **kwargs):
        width = width or 0.1
        # This method is only defined to provide a better default for the width in blender
        # otherwise it looks gigantic, as the bonds
        collection = self.get_collection("Unit cell")
        super()._draw_cell_3D_box(*args, width=width, collection=collection, **kwargs)

GeometryPlot.backends.register("blender", BlenderGeometryBackend)
