# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ....plots.grid import GridPlot
from ..backend import BlenderBackend
from ...templates import GridBackend

import bpy


class BlenderGridBackend(BlenderBackend, GridBackend):

    def draw_3D(self, backend_info, **kwargs):

        col = self.get_collection("Grid")

        for isosurf in backend_info["isosurfaces"]:

            mesh = bpy.data.meshes.new(isosurf["name"])

            obj = bpy.data.objects.new(mesh.name, mesh)

            col.objects.link(obj)
            #bpy.context.view_layer.objects.active = obj

            edges = []
            mesh.from_pydata(isosurf["vertices"], edges, isosurf["faces"].tolist())

            self._color_obj(obj, isosurf["color"], isosurf['opacity'])

            # mat = bpy.data.materials.new("material")
            # mat.use_nodes = True

            # color = self._to_rgb_color(isosurf["color"])

            # if color is not None:
            #     mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (*color, 1)

            # mat.node_tree.nodes["Principled BSDF"].inputs[19].default_value = isosurf["opacity"]

            # mesh.materials.append(mat)

GridPlot.backends.register("blender", BlenderGridBackend)
