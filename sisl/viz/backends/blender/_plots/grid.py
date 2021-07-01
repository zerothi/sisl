from ....plots import GridPlot
from ..backend import BlenderBackend
from ...templates import GridBackend

import bpy

class BlenderGridBackend(BlenderBackend, GridBackend):

    def draw_3D(self, backend_info, **kwargs):

        for isosurf in backend_info["isosurfaces"]:

            x, y, z = isosurf["vertices"].T
            I, J, K = isosurf["faces"].T

            mesh = bpy.data.meshes.new(isosurf["name"])

            obj = bpy.data.objects.new(mesh.name, mesh)

            col = bpy.data.collections.get("Grids")

            if col is None:
                col = bpy.data.collections.new("Grids")
                bpy.context.scene.collection.children.link(col)

            col.objects.link(obj)
            bpy.context.view_layer.objects.active = obj

            edges = []
            mesh.from_pydata(isosurf["vertices"], edges, isosurf["faces"].tolist())

            mat = bpy.data.materials.new("material")
            mat.use_nodes = True

            color = self._to_rgb_color(isosurf["color"])

            if color is not None:
                mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (*color, 1)

            mat.node_tree.nodes["Principled BSDF"].inputs[19].default_value = isosurf["opacity"]

            mesh.materials.append(mat)

GridPlot._backends.register("blender", BlenderGridBackend)