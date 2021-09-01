from ....plots import GeometryPlot
from ..backend import BlenderBackend
from ...templates import GeometryBackend

import bpy


class BlenderGeometryBackend(BlenderBackend, GeometryBackend):

    def draw_1D(self, backend_info, **kwargs):
        raise NotImplementedError("A way of drawing 1D geometry representations is not implemented for blender")

    def draw_2D(self, backend_info, **kwargs):
        raise NotImplementedError("A way of drawing 2D geometry representations is not implemented for blender")

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

        self._color_obj(atom, color, opacity=1)
        
        bpy.ops.object.shade_smooth()

    def _draw_bonds_3D(self, *args, line=None, **kwargs):
        # Set the width of the bonds to 0.2, otherwise they look gigantic.
        line = line or {}
        line["width"] = 0.2
        # And call the method to draw bonds (which will use self.draw_line3D)
        super()._draw_bonds_3D(*args, line=line, **kwargs)
    
    def _draw_cell_3D_box(self, *args, width=0.1, **kwargs):
        # This method is only defined to provide a better default for the width in blender
        # otherwise it looks gigantic, as the bonds
        super()._draw_cell_3D_box(*args, width=width, **kwargs)

GeometryPlot.backends.register("blender", BlenderGeometryBackend)
