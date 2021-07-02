from ....plots import GeometryPlot
from ..backend import BlenderBackend
from ...templates import GeometryBackend

import bpy

class BlenderGeometryBackend(BlenderBackend, GeometryBackend):

    def draw_1D(self, backend_info, **kwargs):
        raise NotImplementedError("A way of drawing 1D geometry representations is not implemented for blender")

    def draw_2D(self, backend_info, **kwargs):
        raise NotImplementedError("A way of drawing 2D geometry representations is not implemented for blender")

    def draw_3D(self, backend_info, **kwargs):

        # For now, draw only the atoms
        for atom_props in backend_info["atoms_props"]:
            self._draw_single_atom_3D(**atom_props)

    def _draw_single_atom_3D(self, xyz, size, color="gray", name=None, vertices=15, **kwargs):

        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=vertices, ring_count=vertices,
            align='WORLD', enter_editmode=False,
            location=xyz, radius=size
        )

        atom = bpy.context.object

        # Unlink the atom from the default collection and link it to the Atoms collection
        context_col = bpy.context.collection
        atoms_col = self.get_collection("Atoms")
        if context_col is not atoms_col:
            context_col.objects.unlink(atom)
            atoms_col.objects.link(atom)

        atom.name = name
        atom.data.name = name

        color = self._to_rgb_color(color)

        if color is not None:
            mat = bpy.data.materials.new("material")
            mat.use_nodes = True

            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (*color, 1)

            atom.active_material = mat

        bpy.ops.object.shade_smooth()

GeometryPlot._backends.register("blender", BlenderGeometryBackend)