# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import collections.abc
import itertools

import numpy as np
import py3Dmol

from .figure import Figure


class Py3DmolFigure(Figure):
    """Generic canvas for the py3Dmol framework"""

    def _init_figure(self, *args, **kwargs):
        self.figure = py3Dmol.view()

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
        """Draws a line."""

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

            points = [{"x": x, "y": y, "z": z} for x, y, z in segment_xyz]

            # If there's only two points, py3dmol doesn't display the curve,
            # probably because it can not smooth it.
            if len(points) == 2:
                points.append(points[-1])

            self.figure.addCurve(
                dict(
                    points=points,
                    radius=line.get("width", 0.1),
                    color=line.get("color"),
                    opacity=line.get("opacity", 1.0) or 1.0,
                    smooth=1,
                )
            )

        return self

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
        style = {
            "color": marker.get("color", "gray"),
            "opacity": marker.get("opacity", 1.0),
            "size": marker.get("size", 1.0),
        }

        for k, v in style.items():
            if (
                not isinstance(v, (collections.abc.Sequence, np.ndarray))
            ) or isinstance(v, str):
                style[k] = itertools.repeat(v)

        for i, (x_i, y_i, z_i, color, opacity, size) in enumerate(
            zip(x, y, z, style["color"], style["opacity"], style["size"])
        ):
            self.figure.addSphere(
                dict(
                    center={"x": float(x_i), "y": float(y_i), "z": float(z_i)},
                    radius=size,
                    color=color,
                    opacity=opacity,
                    quality=5.0,  # This does not work, but sphere quality is really bad by default
                )
            )

    draw_scatter_3D = draw_balls_3D

    def draw_arrows_3D(
        self,
        x,
        y,
        z,
        dxyz,
        arrowhead_scale=0.3,
        arrowhead_angle=15,
        scale: float = 1,
        row=None,
        col=None,
        line={},
        **kwargs,
    ):
        """Draws multiple arrows using the generic draw_line method.

        Parameters
        -----------
        xy: np.ndarray of shape (n_arrows, 2)
            the positions where the atoms start.
        dxy: np.ndarray of shape (n_arrows, 2)
            the arrow vector.
        arrow_head_scale: float, optional
            how big is the arrow head in comparison to the arrow vector.
        arrowhead_angle: angle
            the angle that the arrow head forms with the direction of the arrow (in degrees).
        scale: float, optional
            multiplying factor to display the arrows. It does not affect the underlying data,
            therefore if the data is somehow displayed it should be without the scale factor.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        """
        # Make sure we are dealing with numpy arrays
        xyz = np.array([x, y, z]).T
        dxyz = np.array(dxyz) * scale

        for (x, y, z), (dx, dy, dz) in zip(xyz, dxyz):
            self.figure.addArrow(
                dict(
                    start={"x": x, "y": y, "z": z},
                    end={"x": x + dx, "y": y + dy, "z": z + dz},
                    radius=line.get("width", 0.1),
                    color=line.get("color"),
                    opacity=line.get("opacity", 1.0),
                    radiusRatio=2,
                    mid=(1 - arrowhead_scale),
                )
            )

    def draw_mesh_3D(
        self,
        vertices,
        faces,
        color=None,
        opacity=None,
        name="Mesh",
        wireframe=False,
        row=None,
        col=None,
        **kwargs,
    ):
        def vec_to_dict(a, labels="xyz"):
            return dict(zip(labels, a))

        self.figure.addCustom(
            dict(
                vertexArr=[vec_to_dict(v) for v in vertices.astype(float)],
                faceArr=[int(x) for f in faces for x in f],
                color=color,
                opacity=float(opacity or 1.0),
                wireframe=wireframe,
            )
        )

    def set_axis(self, *args, **kwargs):
        """There are no axes titles and these kind of things in py3dmol.
        At least for now, we might implement it later."""

    def set_axes_equal(self, *args, **kwargs):
        """Axes are always "equal" in py3dmol, so we do nothing here"""

    def show(self, *args, **kwargs):
        self.figure.zoomTo()
        return self.figure.show()
