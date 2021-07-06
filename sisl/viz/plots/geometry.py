# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from functools import wraps
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from sisl import Geometry, PeriodicTable, Atom, AtomGhost
from sisl.utils.mathematics import fnorm
from ..plot import Plot, entry_point
from ..input_fields import (
    ProgramaticInput, FunctionInput, FloatInput,
    SwitchInput, DropdownInput, AtomSelect, GeomAxisSelect,
    FilePathInput, PlotableInput, IntegerInput, TextInput, Array1DInput
)
from ..plotutils import values_to_colors
from sisl._dispatcher import AbstractDispatch, ClassDispatcher


class BoundGeometry(AbstractDispatch):
    """
    Updates the plot after a method is run on the plot's geometry.
    """

    def __init__(self, geom, parent_plot):

        self.parent_plot = parent_plot
        super().__init__(geom)

    def dispatch(self, method):

        @wraps(method)
        def with_plot_update(*args, **kwargs):

            ret = method(*args, **kwargs)

            # Maybe the returned value is not a geometry
            if isinstance(ret, Geometry):
                self.parent_plot.update_settings(geometry=ret)
                return self.parent_plot.on_geom

            return ret

        return with_plot_update


class GeometryPlot(Plot):
    """
    Versatile representation of geometries. 

    This class contains all functions necessary to plot geometries in very diverse ways.

    Parameters
    -------------
    geometry: Geometry, optional

    geom_file: str, optional

    show_bonds: bool, optional

    axes:  optional
        The axis along which you want to see the geometry. You
        can provide as many axes as dimensions you want for your plot.
        Note that the order is important and will result in setting the plot
        axes diferently. For 2D and 1D representations, you can
        pass an arbitrary direction as an axis (array of shape (3,))
    dataaxis_1d: array-like or function, optional
        If you want a 1d representation, you can provide a data axis.
        It determines the second coordinate of the atoms.
        If it's a function, it will recieve the projected 1D coordinates and
        needs to returns the coordinates for the other axis as
        an array. If not provided, the other axis will just be 0 for all points.
    show_cell:  optional
        Specifies how the cell should be rendered. (False: not
        rendered, 'axes': render axes only, 'box': render a bounding box)
    atoms:  optional
        The atoms that are going to be displayed in the plot.
        This also has an impact on bonds (see the `bind_bonds_to_ats` and
        `show_atoms` parameters). If set to None, all atoms are
        displayed
    atoms_color: array-like, optional
        A list containing the color for each atom.
    atoms_size: array-like, optional
        A list containing the size for each atom.
    atoms_colorscale: str, optional
        The colorscale to use to map values to colors for the atoms.
        Only used if atoms_color is provided and is an array of values.
    nsc: array-like, optional
        number of times the geometry should be repeated
    atoms_vertices: int, optional
        In a 3D representation, the number of vertices that each atom sphere
        is composed of.
    bind_bonds_to_ats: bool, optional
        whether only the bonds that belong to an atom that is present should
        be displayed. If False, all bonds are displayed
        regardless of the `atoms` parameter
    show_atoms: bool, optional
        If set to False, it will not display atoms. Basically
        this is a shortcut for ``atoms = [], bind_bonds_to_ats=False``.
        Therefore, it will override these two parameters.
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    _plot_type = "Geometry"

    _parameters = (

        PlotableInput(key='geometry', name="Geometry",
            dtype=Geometry,
            default=None,
            help="A geometry object",
        ),

        FilePathInput(key="geom_file", name="Geometry file",
            group="dataread",
            default=None,
            help="A file name that can read a geometry",
        ),

        SwitchInput(key='show_bonds', name='Show bonds',
                    default=True,
                    help="Also show bonds between atoms."
        ),

        GeomAxisSelect(
            key="axes", name="Axes to display",
            default=["x", "y", "z"],
            help="""The axis along which you want to see the geometry. 
            You can provide as many axes as dimensions you want for your plot.
            Note that the order is important and will result in setting the plot axes diferently.
            For 2D and 1D representations, you can pass an arbitrary direction as an axis (array of shape (3,))"""
        ),

        ProgramaticInput(
            key="dataaxis_1d", name="1d data axis",
            default=None,
            dtype="array-like or function",
            help="""If you want a 1d representation, you can provide a data axis.
            It determines the second coordinate of the atoms.
            
            If it's a function, it will recieve the projected 1D coordinates and needs to returns 
            the coordinates for the other axis as an array.
            
            If not provided, the other axis will just be 0 for all points.
            """
        ),

        DropdownInput(key="show_cell", name="Cell display",
            default="box",
            width="s100% m50% l90%",
            params={
                'options': [
                    {'label': 'False', 'value': False},
                    {'label': 'axes', 'value': 'axes'},
                    {'label': 'box', 'value': 'box'}
                ],
                'isMulti': False,
                'isSearchable': True,
                'isClearable': False
            },
            help="""Specifies how the cell should be rendered. 
            (False: not rendered, 'axes': render axes only, 'box': render a bounding box)"""
        ),

        Array1DInput(
            key="nsc", name="Supercell",
            default=[1, 1, 1],
            params={
                'inputType': 'number',
                'shape': (3,),
                'extendable': False,
            },
            help="""Make the geometry larger by tiling it along each lattice vector"""
        ),

        AtomSelect(key="atoms", name="Atoms to display",
            default=None,
            params={
                "options": [],
                "isSearchable": True,
                "isMulti": True,
                "isClearable": True
            },
            help="""The atoms that are going to be displayed in the plot. 
            This also has an impact on bonds (see the `bind_bonds_to_ats` and `show_atoms` parameters).
            If set to None, all atoms are displayed"""
        ),

        ProgramaticInput(key="atoms_color", name="Atoms color",
            default=None,
            dtype="array-like",
            help="""A list containing the color for each atom."""
        ),

        ProgramaticInput(key="atoms_size", name="Atoms size",
            default=None,
            dtype="array-like",
            help="""A list containing the size for each atom."""
        ),

        TextInput(key="atoms_colorscale", name="Atoms vertices",
            default="viridis",
            help="""The colorscale to use to map values to colors for the atoms.
            Only used if atoms_color is provided and is an array of values."""
        ),

        IntegerInput(key="atoms_vertices", name="Atoms vertices",
            default=15,
            help="""In a 3D representation, the number of vertices that each atom sphere is composed of."""
        ),

        SwitchInput(key="bind_bonds_to_ats", name="Bind bonds to atoms",
            default=True,
            help="""whether only the bonds that belong to an atom that is present should be displayed.
            If False, all bonds are displayed regardless of the `atoms` parameter"""
        ),

        SwitchInput(key="show_atoms", name="Show atoms",
            default=True,
            help="""If set to False, it will not display atoms. 
            Basically this is a shortcut for ``atoms = [], bind_bonds_to_ats=False``.
            Therefore, it will override these two parameters."""
        ),

        IntegerInput(
            key="points_per_bond", name="Points per bond",
            default=10,
            help="Number of points that fill a bond in 2D in case each bond has a different color or different size. <br>More points will make it look more like a line but will slow plot rendering down."
        ),

    )

    # Colors of the atoms following CPK rules
    _atoms_colors = {
        "H": "#cccccc", # Should be white but the default background is white
        "O": "red",
        "Cl": "green",
        "N": "blue",
        "C": "grey",
        "S": "yellow",
        "P": "orange",
        "Au": "gold",
        "else": "pink"
    }

    _pt = PeriodicTable()

    _layout_defaults = {
        'xaxis_showgrid': False,
        'xaxis_zeroline': False,
        'yaxis_showgrid': False,
        'yaxis_zeroline': False,
    }

    _update_methods = {
        "read_data": [],
        "set_data": ["_prepare1D", "_prepare2D", "_prepare3D"],
        "get_figure": []
    }

    def _after_init(self):

        self._display_props = {
            "atoms": {
                "color": None,
                "size": None,
                "colorscale": "viridis"
            },
        }

    @entry_point('geometry')
    def _read_nosource(self, geometry):
        """
        Reads directly from a sisl geometry.
        """
        self.geometry = geometry or getattr(self, "geometry", None)

        if self.geometry is None:
            raise ValueError("No geometry has been provided.")

    @entry_point('geometry file')
    def _read_siesta_output(self, geom_file, root_fdf):
        """
        Reads from a sile that contains a geometry using the `read_geometry` method.
        """
        geom_file = geom_file or root_fdf

        self.geometry = self.get_sile(geom_file).read_geometry()

    def _after_read(self, show_bonds, nsc):
        # Tile the geometry. It shouldn't be done here, since we will need to calculate the bonds for
        # the whole supercell. FIND A SMARTER WAY!!
        for ax, reps in enumerate(nsc):
            self.geometry = self.geometry.tile(reps, ax)

        if show_bonds:
            self.bonds = self.find_all_bonds(self.geometry)

        self.get_param("atoms").update_options(self.geometry)

    def _atoms_props_nsc(self, *props):
        """
        Makes sure that atoms properties such as atoms_size or atoms_color are coherent with nsc.
        """
        def ensure_nsc(prop):
            list_like = isinstance(prop, (np.ndarray, list, tuple))
            if list_like and not self.geometry.na % len(prop):
                prop = np.tile(prop, self.geometry.na // len(prop))
            return prop

        return tuple(ensure_nsc(prop) for prop in props)

    def _set_data(self, axes, atoms, atoms_color, atoms_size, show_atoms, bind_bonds_to_ats, dataaxis_1d, show_cell, kwargs3d={}, kwargs2d={}, kwargs1d={}):
        self._ndim = len(axes)

        if show_atoms == False:
            atoms = []
            bind_bonds_to_ats = False

        # Account for supercell extensions
        atoms_color, atoms_size = self._atoms_props_nsc(atoms_color, atoms_size)

        atoms_kwargs = {"atoms": atoms, "atoms_color": atoms_color, "atoms_size": atoms_size}

        if self._ndim == 3:
            backend_info = self._prepare3D(**atoms_kwargs, bind_bonds_to_ats=bind_bonds_to_ats, **kwargs3d)
        elif self._ndim == 2:
            xaxis, yaxis = axes
            backend_info = self._prepare2D(xaxis=xaxis, yaxis=yaxis, **atoms_kwargs, bind_bonds_to_ats=bind_bonds_to_ats, **kwargs2d)
        elif self._ndim == 1:
            backend_info = self._prepare1D(atoms=atoms, coords_axis=axes[0], data_axis=dataaxis_1d, **kwargs1d)

        backend_info["ndim"] = self._ndim
        backend_info["show_cell"] = show_cell

        return backend_info

    # From here, we start to define all the helper methods:
    @property
    def on_geom(self):
        return BoundGeometry(self.geometry, self)

    @staticmethod
    def _sphere(center=[0, 0, 0], r=1, vertices=10):
        phi, theta = np.mgrid[0.0:np.pi: 1j*vertices, 0.0:2.0*np.pi: 1j*vertices]

        x = center[0] + r*np.sin(phi)*np.cos(theta)
        y = center[1] + r*np.sin(phi)*np.sin(theta)
        z = center[2] + r*np.cos(phi)

        return {'x': x, 'y': y, 'z': z}

    @classmethod
    def atom_color(cls, atom):

        atom = Atom(atom)

        ghost = isinstance(atom, AtomGhost)

        color = cls._atoms_colors.get(atom.symbol, cls._atoms_colors["else"])

        if ghost:
            color = (np.array(matplotlib.colors.to_rgb(color))*255).astype(int)
            color = f'rgba({",".join(color.astype(str))}, 0.4)'

        return color

    @staticmethod
    def find_all_bonds(geometry, tol=0.2):
        """
        Finds all bonds present in a geometry.

        Parameters
        -----------
        geometry: sisl.Geometry
            the structure where the bonds should be found.
        tol: float
            the fraction that the distance between atoms is allowed to differ from
            the "standard" in order to be considered a bond.

        Return
        ---------
        np.ndarray of shape (nbonds, 2)
            each item of the array contains the 2 indices of the atoms that participate in the
            bond.
        """
        pt = PeriodicTable()

        bonds = []
        for at in geometry:
            neighs = geometry.close(at, R=[0.1, 3])[-1]

            for neigh in neighs:
                summed_radius = pt.radius([abs(geometry.atoms[at].Z), abs(geometry.atoms[neigh % geometry.na].Z)]).sum()
                bond_thresh = (1+tol) * summed_radius
                if  bond_thresh > np.linalg.norm(geometry[neigh] - geometry[at]):
                    bonds.append(np.sort([at, neigh]))

        if bonds:
            return np.unique(bonds, axis=0)
        else:
            return bonds

    @staticmethod
    def _sanitize_axis(geometry, axis):

        if isinstance(axis, str):
            try:
                i = ["x", "y", "z"].index(axis)
                axis = np.zeros(3)
                axis[i] = 1
            except:
                i = ["a", "b", "c"].index(axis)
                axis = geometry.cell[i]
        elif isinstance(axis, int):
            i = axis
            axis = np.zeros(3)
            axis[i] = 1

        return np.array(axis)

    @staticmethod
    def _get_cell_corners(cell, unique=False):
        """Gets the coordinates of a cell's corners.

        Parameters
        ----------
        cell: np.ndarray of shape (3, 3)
            the cell for which you want the corner's coordinates.
        unique: bool, optional
            if `False`, a full path to draw a cell is returned.
            if `True`, only unique points are returned, in no particular order.
        
        Returns
        ---------
        np.ndarray of shape (x, 3)
            where x is 16 if unique=False and 8 if unique=True.
        """

        def xyz(coeffs):
            return np.dot(coeffs, cell)

        # Define the vertices of the cube. They follow an order so that we can
        # draw a line that represents the cell's box
        points = [
            (0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0),
            (0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 1, 1), (1, 1, 1),
            (1, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)
        ]

        if unique:
            points = np.unique(points, axis=0)

        return np.array([xyz(coeffs) for coeffs in points])

    @classmethod
    def _projected_1Dcoords(cls, geometry, xyz=None, axis="x"):
        """
        Moves the 3D positions of the atoms to a 2D supspace.

        In this way, we can plot the structure from the "point of view" that we want.

        Parameters
        ------------
        geometry: sisl.Geometry
            the geometry for which you want the projected coords
        xyz: array-like of shape (natoms, 3), optional
            the 3D coordinates that we want to project.
            otherwise they are taken from the geometry. 
        axis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.

        Returns
        ----------
        np.ndarray of shape (natoms, )
            the 2D coordinates of the geometry, with all positions projected into the plane
            defined by xaxis and yaxis.
        """
        if xyz is None:
            xyz = geometry.xyz

        # Get the directions that these axes represent if the provided input
        # is an axis index
        axis = cls._sanitize_axis(geometry, axis)

        return xyz.dot(axis)/np.linalg.norm(axis)
    
    @classmethod
    def _projected_2Dcoords(cls, geometry, xyz=None, xaxis="x", yaxis="y"):
        """
        Moves the 3D positions of the atoms to a 2D supspace.

        In this way, we can plot the structure from the "point of view" that we want.

        Parameters
        ------------
        geometry: sisl.Geometry
            the geometry for which you want the projected coords
        xyz: array-like of shape (natoms, 3), optional
            the 3D coordinates that we want to project.
            otherwise they are taken from the geometry. 
        xaxis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.
        yaxis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.

        Returns
        ----------
        np.ndarray of shape (2, natoms)
            the 2D coordinates of the geometry, with all positions projected into the plane
            defined by xaxis and yaxis.
        """
        if xyz is None:
            xyz = geometry.xyz

        # Get the directions that these axes represent if the provided input
        # is an axis index
        xaxis = cls._sanitize_axis(geometry, xaxis)
        yaxis = cls._sanitize_axis(geometry, yaxis)

        return np.array([xyz.dot(ax)/fnorm(ax) for ax in (xaxis, yaxis)])

    def _get_atoms_bonds(self, bonds, atom, geom=None, sanitize_atom=True):
        """
        Gets the bonds where the given atoms are involved
        """
        if atom is None:
            return bonds

        if sanitize_atom:
            geom = geom or self.geometry
            atom = geom._sanitize_atoms(atom)

        return [bond for bond in bonds if np.any([at in atom for at in bond])]

    #---------------------------------------------------
    #                  1D plotting
    #---------------------------------------------------

    def _prepare1D(self, atoms=None, coords_axis="x", data_axis=None, wrap_atoms=None, atoms_color=None, atoms_size=None, atoms_colorscale="viridis", **kwargs):
        """
        Returns a 1D representation of the plot's geometry.

        Parameters
        -----------
        atoms: array-like of int, optional
            the indices of the atoms that you want to plot
        coords_axis:  {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the axis onto which all the atoms are projected.
        data_axis: function or array-like, optional
            determines the second coordinate of the atoms

            If it's a function, it will recieve the projected 1D coordinates and needs to returns 
            the coordinates for the other axis as an array.

            If not provided, the other axis will just be 0 for all points.
        atoms_color: array-like, optional
            an array of colors or values that will be mapped into colors
        atoms_size: array-like, optional
            the size that each atom must have.
        atoms_colorscale: str or list, optional
            the name of a plotly colorscale or a list of colors.

            Only used if atoms_color is an array of values.
        wrap_atoms: function, optional
            function that takes the 2D positions of the atoms in the plot and returns a tuple of (args, kwargs),
            that are passed to self._atoms_scatter_trace2D.
            If not provided self._default_wrap_atoms is used.
        **kwargs: 
            passed directly to the atoms scatter trace
        """
        wrap_atoms = wrap_atoms or self._default_wrap_atoms1D
        traces = []

        atoms = self.geometry._sanitize_atoms(atoms)

        self._display_props["atoms"]["color"] = atoms_color
        self._display_props["atoms"]["size"] = atoms_size
        self._display_props["atoms"]["colorscale"] = atoms_colorscale

        x = self._projected_1Dcoords(self.geometry, self.geometry[atoms], axis=coords_axis)
        if data_axis is None:
            def data_axis(x):
                return np.zeros(x.shape[0])

        data_axis_name = data_axis.__name__ if callable(data_axis) else 'Data axis'
        if callable(data_axis):
            data_axis = np.array(data_axis(x))

        xy = np.array([x, data_axis])

        atoms_props = wrap_atoms(atoms, xy)

        return {
            "geometry": self.geometry, "xaxis": coords_axis, "yaxis": data_axis_name, "atoms_props": atoms_props,
        }

    def _default_wrap_atoms1D(self, ats, xy):

        extra_kwargs = {}

        predefined_colors = self._display_props["atoms"]["color"]

        if predefined_colors is None:
            color = [self.atom_color(atom.Z) for atom in self.geometry.atoms[ats]]
        else:
            color = predefined_colors
            extra_kwargs["marker_colorscale"] = self._display_props["atoms"]["colorscale"]

            if isinstance(color, (list, tuple, np.ndarray)):
                extra_kwargs["text"] = [f"Color: {c}" for c in color]

        predefined_sizes = self._display_props["atoms"]["size"]

        if predefined_sizes is None:
            size = [self._pt.radius(abs(atom.Z))*16 for atom in self.geometry.atoms[ats]]
        else:
            size = predefined_sizes

        return {
            "xy": xy,
            "text": [f'{self.geometry[at]}<br>{at} ({self.geometry.atoms[at].tag})' for at in ats],
            "name": "Atoms",
            "color": color,
            "size": size,
            **extra_kwargs
        }

    #---------------------------------------------------
    #                  2D plotting
    #---------------------------------------------------

    def _prepare2D(self, xaxis="x", yaxis="y", atoms=None, atoms_color=None, atoms_size=None, atoms_colorscale="viridis",
        show_bonds=True, bind_bonds_to_ats=True, bonds_together=True, points_per_bond=5,
        show_cell='box', wrap_atoms=None, wrap_bond=None):
        """Returns a 2D representation of the plot's geometry.

        Parameters
        -----------
        xaxis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.
        yaxis: {0,1,2, "x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
            If it's an int, it will interpreted as the index of the cell axis.
        atoms: array-like of int, optional
            the indices of the atoms that you want to plot
        atoms_color: array-like, optional
            an array of colors or values that will be mapped into colors
        atoms_size: array-like, optional
            the size that each atom must have.
        atoms_colorscale: str or list, optional
            the name of a plotly colorscale or a list of colors.
            Only used if atoms_color is an array of values.
        show_bonds: boolean, optional
            whether bonds should be plotted.
        bind_bonds_to_ats: boolean, optional
            whether only the bonds that belong to an atom that is present should be displayed.
            If False, all bonds are displayed regardless of the `atom` parameter.
        bonds_together: boolean, optional
            If set to True, it draws all bonds in one trace, which may be faster for rendering.
            The only limitation that it has is that you can't set individual widths.
            If you provide variable color and/or size for the bonds, bonds will be drawn as dots
            (if you use enough points per bond it almost looks like a line). If you don't like this, use individual
            bonds instead, but then note that you can not share a colorscale between bonds. This indirectly means that you 
            can not provide the color as a number, so you will need to calculate the colors yourself if you want
            a colorscale-like behavior.
        points_per_bond: int, optional
            If `bonds_together` is True and you provide a variable color or size (using `wrap_bonds`), this is
            the number of points that are used for each bond. See `bonds_together` for more info.
        show_cell: {False, "box", "axes"}, optional
            determines how the unit cell is represented.
        wrap_atoms: function, optional
            function that recieves the 2D coordinates and returns
            the args (array-like) and kwargs (dict) that go into self._atoms_scatter_trace2D()
            If not provided, self._default_wrap_atoms2D will be used.
            wrap_atom: function, optional
            function that recieves the index of an atom and returns
            the args (array-like) and kwargs (dict) that go into self._atom_trace3D()
            If not provided, self._default_wrap_atom3D will be used.
        wrap_bond: function, optional
            function that recieves "a bond" (list of 2 atom indices) and its coordinates ((x1,y1), (x2, y2)).
            It should return the args (array-like) and kwargs (dict) that go into `self._bond_trace2D()`
            If not provided, self._default_wrap_bond2D will be used.
        """
        wrap_atoms = wrap_atoms or self._default_wrap_atoms2D
        wrap_bond = wrap_bond or self._default_wrap_bond2D

        atoms = self.geometry._sanitize_atoms(atoms)

        self._display_props["atoms"]["color"] = atoms_color
        self._display_props["atoms"]["size"] = atoms_size
        self._display_props["atoms"]["colorscale"] = atoms_colorscale

        xy = self._projected_2Dcoords(self.geometry, self.geometry[atoms], xaxis=xaxis, yaxis=yaxis)

        # Add atoms
        atoms_props = wrap_atoms(atoms, xy)

        # Add bonds
        if show_bonds:
            # Define the actual bonds that we are going to draw depending on which
            # atoms are requested
            bonds = self.bonds
            if bind_bonds_to_ats:
                bonds = self._get_atoms_bonds(bonds, atoms, sanitize_atom=False)

            bonds_xyz = np.array([self.geometry[bond] for bond in bonds])
            if len(bonds_xyz) != 0:
                xys = self._projected_2Dcoords(self.geometry, bonds_xyz, xaxis=xaxis, yaxis=yaxis)

                # By reshaping we get the following: First axis -> bond (length: number of bonds),
                # Second axis -> atoms in the bond (length 2), Third axis -> coordinate (x, y)
                xys = xys.transpose((1, 2, 0))

                # Try to get the bonds colors (It might be that the user is not setting them)
                bonds_props = [wrap_bond(bond, xy) for bond, xy in zip(bonds, xys)]
            else:
                bonds_props = []
        else:
            bonds_props = []

        return {
            "geometry": self.geometry, "xaxis": xaxis, "yaxis": yaxis, "atoms_props": atoms_props,
            "bonds_props": bonds_props, "points_per_bond": points_per_bond,
        }

    def _default_wrap_atoms2D(self, ats, xy):
        return self._default_wrap_atoms1D(ats, xy)

    def _default_wrap_bond2D(self, bond, xys):
        return {
            "xys": xys,
        }

    #---------------------------------------------------
    #                  3D plotting
    #---------------------------------------------------

    def _prepare3D(self, wrap_atom=None, wrap_bond=None, show_cell='box',
        atoms=None, bind_bonds_to_ats=True, atoms_vertices=15, atoms_color=None, atoms_size=None, atoms_colorscale="viridis",
        show_bonds=True, cheap_bonds=True, cheap_atoms=False, cheap_bonds_kwargs={}):
        """Returns a 3D representation of the plot's geometry.

        Parameters
        -----------
        wrap_atom: function, optional
            function that recieves the index of an atom and returns
            the args (array-like) and kwargs (dict) that go into self._atom_trace3D()
            If not provided, self._default_wrap_atom3D will be used.
        wrap_bond: function, optional
            function that recieves "a bond" (list of 2 atom indices) and returns
            the args (array-like) and kwargs (dict) that go into self._bond_trace3D()
            If not provided, self._default_wrap_bond3D will be used.
        show_cell: {'axes', 'box', False}, optional
            defines how the unit cell is drawn
        atoms: array-like of int, optional
            the indices of the atoms that you want to plot
        bind_bonds_to_ats: boolean, optional
            whether only the bonds that belong to an atom that is present should be displayed.
            If False, all bonds are displayed regardless of the `atom` parameter
        atoms_vertices: int
            the "definition" of the atom sphere, if not in cheap mode. The more vertices, the more defined the sphere
            will be. However, it will also be more expensive to render.
        atoms_color: array-like, optional
            an array of colors or values that will be mapped into colors
        atoms_size: array-like, optional
            the size that each atom must have.
        atoms_colorscale: str or list, optional
            the name of a plotly colorscale or a list of colors.
            Only used if atoms_color is an array of values.
        cheap_bonds: boolean, optional
            If set to True, it draws all in one trace, which results in a dramatically faster rendering.
            The only limitation that it has is that you can't set individual widths.
        cheap_atoms: boolean, optional
            Whether atoms are drawn in a cheap way (all in one scatter trace). 
            If `False`, each atom is drawn individually as a sphere. It's more expensive, but by doing
            this you avoid variable size problems (and looks better).
        cheap_bonds_kwargs: dict, optional
            dict that is passed directly as keyword arguments to `self._bonds_trace3D`.
        """
        wrap_atom = wrap_atom or self._default_wrap_atom3D
        wrap_bond = wrap_bond or self._default_wrap_bond3D

        atoms = self.geometry._sanitize_atoms(atoms)

        self._display_props["atoms"]["colorscale"] = atoms_colorscale
        if atoms_color is not None:
            try:
                self._display_props["atoms"]["color"] = values_to_colors(atoms_color, self._display_props["atoms"]["colorscale"])
            except:
                self._display_props["atoms"]["color"] = atoms_color
        self._display_props["atoms"]["size"] = atoms_size

        atoms_props = [wrap_atom(at) for at in atoms]

        if show_bonds:
            # Try to get the bonds colors (It might be that the user is not setting them)
            bonds = self.bonds
            if bind_bonds_to_ats:
                bonds = self._get_atoms_bonds(bonds, atoms, sanitize_atom=False)
            bonds_props = [wrap_bond(bond) for bond in bonds]
        else:
            bonds = []
            bonds_props = []

        return {"geometry": self.geometry, "atoms": atoms, "bonds": bonds, "atoms_props": atoms_props, "bonds_props": bonds_props}

    def _default_wrap_atom3D(self, at):

        atom = self.geometry.atoms[at]

        predefined_colors = self._display_props["atoms"]["color"]
        if predefined_colors is None:
            color = self.atom_color(atom.Z)
        else:
            color = predefined_colors[at]

        predefined_sizes = self._display_props["atoms"]["size"]
        if predefined_sizes is None:
            size = self._pt.radius(abs(atom.Z))*0.6
        else:
            size = predefined_sizes[at]

        return {
            "xyz": self.geometry[at],
            "name": f'{at} ({atom.tag})',
            "color": color,
            "size": size,
            "opacity": 0.4 if isinstance(atom, AtomGhost) else 1
        }

    def _default_wrap_bond3D(self, bond):

        return {
            "xyz1": self.geometry[bond[0]],
            "xyz2": self.geometry[bond[1]],
            "r": 15
        }