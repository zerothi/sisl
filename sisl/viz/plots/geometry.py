# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from functools import wraps
import itertools
import re

from sisl.messages import warn

import numpy as np

from sisl import Geometry, PeriodicTable, Atom, AtomGhost
from sisl.utils import direction
from sisl.utils.mathematics import fnorm
from ..plot import Plot, entry_point
from ..input_fields import (
    ProgramaticInput, ColorInput, DictInput,
    BoolInput, OptionsInput, AtomSelect, GeomAxisSelect, QueriesInput,
    FilePathInput, PlotableInput, IntegerInput, FloatInput, TextInput, Array1DInput,
)
from ..plotutils import values_to_colors
from sisl._dispatcher import AbstractDispatch
from sisl._supercell import cell_invert


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
        A geometry object
    geom_file: str, optional
        A file name that can read a geometry
    show_bonds: bool, optional
        Show bonds between atoms.
    bonds_style: dict, optional
        Customize the style of the bonds by passing style specifications.
        Currently, you can only pass one style specification. Styling bonds
        individually is not supported yet, but it will be in the future.
        Structure of the dict: {          }
    axes:  optional
        The axis along which you want to see the geometry.              You
        can provide as many axes as dimensions you want for your plot.
        Note that the order is important and will result in setting the plot
        axes diferently.             For 2D and 1D representations, you can
        pass an arbitrary direction as an axis (array of shape (3,))
    dataaxis_1d: array-like or function, optional
        If you want a 1d representation, you can provide a data axis.
        It determines the second coordinate of the atoms.
        If it's a function, it will recieve the projected 1D coordinates and
        needs to returns              the coordinates for the other axis as
        an array.                          If not provided, the other axis
        will just be 0 for all points.
    show_cell:  optional
        Specifies how the cell should be rendered.              (False: not
        rendered, 'axes': render axes only, 'box': render a bounding box)
    nsc: array-like, optional
        Make the geometry larger by tiling it along each lattice vector
    atoms: dict, optional
        The atoms that are going to be displayed in the plot.
        This also has an impact on bonds (see the `bind_bonds_to_ats` and
        `show_atoms` parameters).             If set to None, all atoms are
        displayed   Structure of the dict: {         'index':    Structure of
        the dict: {         'in':  }         'fx':          'fy':
        'fz':          'x':          'y':          'z':          'Z':
        'neighbours':    Structure of the dict: {         'range':
        'R':          'neigh_tag':  }         'tag':          'seq':  }
    atoms_style: array-like of dict, optional
        Customize the style of the atoms by passing style specifications.
        Each style specification can have an "atoms" key to select the atoms
        for which             that style should be used. If an atom fits into
        more than one selector, the last             specification is used.
        Each item is a dict.    Structure of the dict: {         'atoms':
        Structure of the dict: {         'index':    Structure of the dict: {
        'in':  }         'fx':          'fy':          'fz':          'x':
        'y':          'z':          'Z':          'neighbours':    Structure
        of the dict: {         'range':          'R':          'neigh_tag':
        }         'tag':          'seq':  }         'color':          'size':
        'opacity':          'vertices': In a 3D representation, the number of
        vertices that each atom sphere is composed of. }
    arrows: array-like of dict, optional
        Add arrows centered at the atoms to display some vector property.
        You can add as many arrows as you want, each with different styles.
        Each item is a dict.    Structure of the dict: {         'atoms':
        Structure of the dict: {         'index':    Structure of the dict: {
        'in':  }         'fx':          'fy':          'fz':          'x':
        'y':          'z':          'Z':          'neighbours':    Structure
        of the dict: {         'range':          'R':          'neigh_tag':
        }         'tag':          'seq':  }         'data':          'scale':
        'color':          'width':          'name':
        'arrowhead_scale':          'arrowhead_angle':  }
    atoms_scale: float, optional
        A scaling factor for atom sizes. This is a very quick way to rescale.
    atoms_colorscale: str, optional
        The colorscale to use to map values to colors for the atoms.
        Only used if atoms_color is provided and is an array of values.
    bind_bonds_to_ats: bool, optional
        whether only the bonds that belong to an atom that is present should
        be displayed.             If False, all bonds are displayed
        regardless of the `atoms` parameter
    show_atoms: bool, optional
        If set to False, it will not display atoms.              Basically
        this is a shortcut for ``atoms = [], bind_bonds_to_ats=False``.
        Therefore, it will override these two parameters.
    points_per_bond: int, optional
        Number of points that fill a bond in 2D in case each bond has a
        different color or different size. More points will make it look
        more like a line but will slow plot rendering down.
    cell_style: dict, optional
        The style of the unit cell lines   Structure of the dict: {
        'color':          'width':          'opacity':  }
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    entry_points_order: array-like, optional
        Order with which entry points will be attempted.
    backend:  optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    _plot_type = "Geometry"

    _param_groups = (
        {
            "key": "cell",
            "name": "Cell display",
            "icon": "check_box_outline_blank",
            "description": "These are all inputs related to the geometry's cell."
        },

        {
            "key": "atoms",
            "name": "Atoms display",
            "icon": "album",
            "description": "Inputs related to which and how atoms are displayed."
        },

        {
            "key": "bonds",
            "name": "Bonds display",
            "icon": "power_input",
            "description": "Inputs related to which and how bonds are displayed."
        },

    )

    _parameters = (

        PlotableInput(key='geometry', name="Geometry",
            dtype=Geometry,
            default=None,
            group="dataread",
            help="A geometry object",
        ),

        FilePathInput(key="geom_file", name="Geometry file",
            group="dataread",
            default=None,
            help="A file name that can read a geometry",
        ),

        BoolInput(key='show_bonds', name='Show bonds',
            default=True,
            group="bonds",
            help="Show bonds between atoms."
        ),

        DictInput(key="bonds_style", name="Bonds style",
            default={},
            group="bonds",
            help = """Customize the style of the bonds by passing style specifications.
            Currently, you can only pass one style specification. Styling bonds 
            individually is not supported yet, but it will be in the future. 
            """,
            queryForm = [

                ColorInput(key="color", name="Color", default="#cccccc"),

                FloatInput(key="width", name="Width", default=None),

                FloatInput(key="opacity", name="Opacity",
                    default=1,
                    params={"min": 0, "max": 1},
                ),

            ]
        ),

        GeomAxisSelect(
            key="axes", name="Axes to display",
            default=["x", "y", "z"],
            group="cell",
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

        OptionsInput(key="show_cell", name="Cell display",
            default="box",
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
            group="cell",
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
            group="cell",
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
            group="atoms",
            help="""The atoms that are going to be displayed in the plot. 
            This also has an impact on bonds (see the `bind_bonds_to_ats` and `show_atoms` parameters).
            If set to None, all atoms are displayed"""
        ),

        QueriesInput(key="atoms_style", name="Atoms style",
            default=[],
            group="atoms",
            help = """Customize the style of the atoms by passing style specifications. 
            Each style specification can have an "atoms" key to select the atoms for which
            that style should be used. If an atom fits into more than one selector, the last
            specification is used.
            """,
            queryForm = [

                AtomSelect(key="atoms", name="Atoms", default=None),

                ColorInput(key="color", name="Color", default=None),

                FloatInput(key="size", name="Size", default=None),

                FloatInput(key="opacity", name="Opacity",
                    default=1,
                    params={"min": 0, "max": 1},
                ),

                IntegerInput(key="vertices", name="Vertices", default=15,
                    help="""In a 3D representation, the number of vertices that each atom sphere is composed of."""),

            ]
        ),

        QueriesInput(key="arrows", name="Arrows",
            default=[],
            group="atoms",
            help = """Add arrows centered at the atoms to display some vector property.
            You can add as many arrows as you want, each with different styles.""",
            queryForm = [

                AtomSelect(key="atoms", name="Atoms", default=None),

                Array1DInput(key="data", name="Data", default=None, params={"shape": (3,)}),

                FloatInput(key="scale", name="Scale", default=1),

                ColorInput(key="color", name="Color", default=None),

                FloatInput(key="width", name="Width", default=None),

                TextInput(key="name", name="Name", default=None),

                FloatInput(key="arrowhead_scale", name="Arrowhead scale", default=0.2),

                FloatInput(key="arrowhead_angle", name="Arrowhead angle", default=20),
            ]
        ),

        FloatInput(key="atoms_scale", name="Atoms scale",
            default=1.,
            group="atoms",
            help="A scaling factor for atom sizes. This is a very quick way to rescale."
        ),

        TextInput(key="atoms_colorscale", name="Atoms colorscale",
            group="atoms",
            default="viridis",
            help="""The colorscale to use to map values to colors for the atoms.
            Only used if atoms_color is provided and is an array of values."""
        ),

        BoolInput(key="bind_bonds_to_ats", name="Bind bonds to atoms",
            default=True,
            group="bonds",
            help="""whether only the bonds that belong to an atom that is present should be displayed.
            If False, all bonds are displayed regardless of the `atoms` parameter"""
        ),

        BoolInput(key="show_atoms", name="Show atoms",
            default=True,
            group="atoms",
            help="""If set to False, it will not display atoms. 
            Basically this is a shortcut for ``atoms = [], bind_bonds_to_ats=False``.
            Therefore, it will override these two parameters."""
        ),

        IntegerInput(
            key="points_per_bond", name="Points per bond",
            group="bonds",
            default=10,
            help="Number of points that fill a bond in 2D in case each bond has a different color or different size. <br>More points will make it look more like a line but will slow plot rendering down."
        ),

        DictInput(key="cell_style", name="Cell style",
            default={"color": "green"},
            group="cell",
            help="""The style of the unit cell lines""",
            fields=[
                ColorInput(key="color", name="Color", default="green"),

                FloatInput(key="width", name="Width", default=None),

                FloatInput(key="opacity", name="Opacity", default=1),
            ]
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

    _update_methods = {
        "read_data": [],
        "set_data": ["_prepare1D", "_prepare2D", "_prepare3D"],
        "get_figure": []
    }

    @entry_point('geometry', 0)
    def _read_nosource(self, geometry):
        """
        Reads directly from a sisl geometry.
        """
        self.geometry = geometry or getattr(self, "geometry", None)

        if self.geometry is None:
            raise ValueError("No geometry has been provided.")

    @entry_point('geometry file', 1)
    def _read_siesta_output(self, geom_file, root_fdf):
        """
        Reads from a sile that contains a geometry using the `read_geometry` method.
        """
        geom_file = geom_file or root_fdf

        self.geometry = self.get_sile(geom_file).read_geometry()

    def _after_read(self, show_bonds, nsc):
        # Tile the geometry. It shouldn't be done here, since we will need to calculate the bonds for
        # the whole supercell. FIND A SMARTER WAY!!
        self._tiled_geometry = self.geometry
        for ax, reps in enumerate(nsc):
            self._tiled_geometry = self._tiled_geometry.tile(reps, ax)

        if show_bonds:
            self.bonds = self.find_all_bonds(self._tiled_geometry)

        self.get_param("atoms").update_options(self.geometry)
        self.get_param("atoms_style").get_param("atoms").update_options(self.geometry)
        self.get_param("arrows").get_param("atoms").update_options(self.geometry)

    def _parse_atoms_style(self, atoms_style, ndim):
        """Parses the `atoms_style` setting to a dictionary of style specifications.

        Parameters
        -----------
        atoms_style:
            the value of the atoms_style setting.
        ndim: int
            the number of dimensions of the plot, only used for the default atom sizes. 
        """

        # Set the radius scale for the different representations (1D and 2D measure size in pixels,
        # while in 3D this is a real distance)
        radius_scale = [16, 16, 1][ndim-1]

        # Add the default styles first
        atoms_style = [
            {
                "color": [self.atom_color(atom.Z) for atom in self.geometry.atoms],
                "size": [self._pt.radius(abs(atom.Z))*radius_scale for atom in self.geometry.atoms],
                "opacity": [0.4 if isinstance(atom, AtomGhost) else 1 for atom in self.geometry.atoms],
                "vertices": 15,
            },
            *atoms_style
        ]

        def _tile_if_needed(atoms, spec):
            """Function that tiles an array style specification.

            It does so if the specification needs to be applied to more atoms
            than items are in the array."""
            if isinstance(spec, (tuple, list, np.ndarray)):
                n_ats = len(atoms)
                n_spec = len(spec)
                if n_ats != n_spec and n_ats % n_spec == 0:
                    spec = np.tile(spec, n_ats // n_spec)
            return spec

        # Initialize the styles.
        parsed_atoms_style = {
            "color": np.empty((self.geometry.na, ), dtype=object),
            "size": np.empty((self.geometry.na, ), dtype=float),
            "vertices": np.empty((self.geometry.na, ), dtype=int),
            "opacity": np.empty((self.geometry.na), dtype=float),
        }

        # Go specification by specification and apply the styles
        # to the corresponding atoms.
        for style_spec in atoms_style:
            atoms = self.geometry._sanitize_atoms(style_spec.get("atoms"))
            for key in parsed_atoms_style:
                if style_spec.get(key) is not None:
                    parsed_atoms_style[key][atoms] = _tile_if_needed(atoms, style_spec[key])

        return parsed_atoms_style

    def _parse_arrows(self, arrows, atoms, ndim, axes, nsc):
        arrows_param = self.get_param("arrows")

        def _sanitize_spec(arrow_spec):
            arrow_spec = arrows_param.complete_query(arrow_spec)

            arrow_spec["atoms"] = np.atleast_1d(self.geometry._sanitize_atoms(arrow_spec["atoms"]))
            arrow_atoms = arrow_spec["atoms"]

            not_displayed = set(arrow_atoms) - set(atoms)
            if not_displayed:
                warn(f"Arrow data for atoms {not_displayed} will not be displayed because these atoms are not displayed.")
            if set(atoms) == set(atoms) - set(arrow_atoms):
                # Then it makes no sense to store arrows, as nothing will be drawn
                return None

            arrow_data = np.full((self.geometry.na, ndim), np.nan, dtype=np.float64)
            provided_data = np.array(arrow_spec["data"])

            # Get the projected directions if we are not in 3D.
            if ndim == 1:
                provided_data = self._projected_1Dcoords(self.geometry, provided_data, axis=axes[0])
                provided_data = np.expand_dims(provided_data, axis=-1)
            elif ndim == 2:
                provided_data = self._projected_2Dcoords(self.geometry, provided_data, xaxis=axes[0], yaxis=axes[1])

            arrow_data[arrow_atoms] = provided_data
            arrow_spec["data"] = arrow_data[atoms]

            arrow_spec["data"] = self._tile_atomic_data(arrow_spec["data"])

            return arrow_spec

        arrows = [_sanitize_spec(arrow_spec) for arrow_spec in arrows]

        return [arrow_spec for arrow_spec in arrows if arrow_spec is not None]

    def _tile_atomic_data(self, data):
        tiles = np.ones(np.array(data).ndim, dtype=int)
        tiles[0] = self._tiled_geometry.na // self.geometry.na
        return np.tile(data, tiles)

    def _tiled_atoms(self, atoms):
        if len(atoms) == 0:
            return atoms

        n_tiles = self._tiled_geometry.na // self.geometry.na

        tiled_atoms = np.tile(atoms, n_tiles).reshape(-1, atoms.shape[0])

        tiled_atoms += np.linspace(0, self.geometry.na*(n_tiles - 1), n_tiles, dtype=int).reshape(-1, 1)
        return tiled_atoms.ravel()

    def _tiled_coords(self, atoms):
        return self._tiled_geometry[self._tiled_atoms(atoms)]

    def _set_data(self, axes,
        atoms, atoms_style, atoms_scale, atoms_colorscale, show_atoms, bind_bonds_to_ats, bonds_style,
        arrows, dataaxis_1d, show_cell, cell_style, nsc, kwargs3d={}, kwargs2d={}, kwargs1d={}):
        self._ndim = len(axes)

        if show_atoms == False:
            atoms = []
            bind_bonds_to_ats = False

        atoms = np.atleast_1d(self.geometry._sanitize_atoms(atoms))

        arrows = self._parse_arrows(arrows, atoms, self._ndim, axes, nsc)

        atoms_styles = self._parse_atoms_style(atoms_style, self._ndim)
        atoms_styles["colorscale"] = atoms_colorscale

        atoms_kwargs = {"atoms": atoms, "atoms_styles": atoms_styles, "atoms_scale": atoms_scale}

        if self._ndim == 3:
            xaxis, yaxis, zaxis = axes
            backend_info = self._prepare3D(
                **atoms_kwargs, bonds_styles=bonds_style,
                bind_bonds_to_ats=bind_bonds_to_ats, **kwargs3d
            )
        elif self._ndim == 2:
            xaxis, yaxis = axes
            backend_info = self._prepare2D(
                xaxis=xaxis, yaxis=yaxis, bonds_styles=bonds_style, **atoms_kwargs,
                bind_bonds_to_ats=bind_bonds_to_ats, nsc=nsc, **kwargs2d
            )
        elif self._ndim == 1:
            xaxis = axes[0]
            yaxis = dataaxis_1d
            backend_info = self._prepare1D(**atoms_kwargs, coords_axis=xaxis, data_axis=yaxis, nsc=nsc, **kwargs1d)

        # Define the axes titles
        backend_info["axes_titles"] = {
            "xaxis": self._get_ax_title(xaxis),
            "yaxis": self._get_ax_title(yaxis),
        }
        if self._ndim == 3:
            backend_info["axes_titles"]["zaxis"] = self._get_ax_title(zaxis)

        backend_info["ndim"] = self._ndim
        backend_info["show_cell"] = show_cell
        backend_info["arrows"] = arrows

        cell_style = self.get_param("cell_style").complete_dict(cell_style)
        backend_info["cell_style"] = cell_style

        return backend_info

    @staticmethod
    def _get_ax_title(ax):
        """Generates the title for a given axis"""
        if hasattr(ax, "__name__"):
            title = ax.__name__
        elif isinstance(ax, np.ndarray) and ax.shape == (3,):
            title = str(ax)
        elif not isinstance(ax, str):
            title = ""
        elif re.match("[+-]?[xXyYzZ]", ax):
            title = f'{ax.upper()} axis [Ang]'
        elif re.match("[+-]?[aAbBcC]", ax):
            title = f'{ax.upper()} lattice vector'
        else:
            title = ax

        return title

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
            import matplotlib.colors

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

            for neigh in neighs[neighs > at]:
                summed_radius = pt.radius([abs(geometry.atoms[at].Z), abs(geometry.atoms[neigh % geometry.na].Z)]).sum()
                bond_thresh = (1+tol) * summed_radius
                if  bond_thresh > fnorm(geometry[neigh] - geometry[at]):
                    bonds.append([at, neigh])

        return np.array(bonds, dtype=int)

    @staticmethod
    def _direction(ax, cell=None):
        if isinstance(ax, (int, str)):
            sign = 1
            # If the axis contains a -, we need to mirror the direction.
            if isinstance(ax, str) and ax[0] == "-":
                sign = -1
                ax = ax[1]
            ax = sign * direction(ax, abc=cell, xyz=np.diag([1., 1., 1.]))

        return ax

    @classmethod
    def _cross_product(cls, v1, v2, cell=None):
        """An enhanced version of the cross product.

        It is an enhanced version because both bectors accept strings that represent
        the cartesian axes or the lattice vectors (see `v1`, `v2` below). It has been built
        so that cross product between lattice vectors (-){"a", "b", "c"} follows the same rules
        as (-){"x", "y", "z"}
        Parameters
        ----------
        v1, v2: array-like of shape (3,) or (-){"x", "y", "z", "a", "b", "c"}
            The vectors to take the cross product of.
        cell: array-like of shape (3, 3)
            The cell of the structure, only needed if lattice vectors {"a", "b", "c"}
            are passed for `v1` and `v2`.
        """
        # Make abc follow the same rules as xyz to find the orthogonal direction
        # That is, a X b = c; -a X b = -c and so on.
        if isinstance(v1, str) and isinstance(v2, str):
            if re.match("([+-]?[abc]){2}", v1 + v2):
                v1 = v1.replace("a", "x").replace("b", "y").replace("c", "z")
                v2 = v2.replace("a", "x").replace("b", "y").replace("c", "z")
                ort = cls._cross_product(v1, v2)
                ort_ax = "abc"[np.where(ort != 0)[0][0]]
                if ort.sum() == -1:
                    ort_ax = "-" + ort_ax
                return cls._direction(ort_ax, cell)

        # If the vectors are not abc, we just need to take the cross product.
        return np.cross(cls._direction(v1, cell), cls._direction(v2, cell))

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
        if unique:
            verts = list(itertools.product([0, 1], [0, 1], [0, 1]))
        else:
            # Define the vertices of the cube. They follow an order so that we can
            # draw a line that represents the cell's box
            verts = [
                (0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1), (0, 1, 0),
                (np.nan, np.nan, np.nan),
                (0, 1, 1), (0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),
                (np.nan, np.nan, np.nan),
                (1, 1, 0), (1, 0, 0),
                (np.nan, np.nan, np.nan),
                (1, 1, 1), (1, 0, 1)
            ]

        verts = np.array(verts, dtype=np.float64)

        return verts.dot(cell)

    @classmethod
    def _projected_1Dcoords(cls, geometry, xyz=None, axis="x"):
        """
        Moves the 3D positions of the atoms to a 2D supspace.

        In this way, we can plot the structure from the "point of view" that we want.

        NOTE: If axis is one of {"a", "b", "c", "1", "2", "3"} the function doesn't
        project the coordinates in the direction of the lattice vector. The fractional
        coordinates, taking in consideration the three lattice vectors, are returned
        instead.

        Parameters
        ------------
        geometry: sisl.Geometry
            the geometry for which you want the projected coords
        xyz: array-like of shape (natoms, 3), optional
            the 3D coordinates that we want to project.
            otherwise they are taken from the geometry. 
        axis: {"x", "y", "z", "a", "b", "c", "1", "2", "3"} or array-like of shape 3, optional
            the direction to be displayed along the X axis.
        nsc: array-like of shape (3, ), optional
            only used if `axis` is a lattice vector. It is used to rescale everything to the unit
            cell lattice vectors, otherwise `GeometryPlot` doesn't play well with `GridPlot`.

        Returns
        ----------
        np.ndarray of shape (natoms, )
            the 1D coordinates of the geometry, with all positions projected into the line
            defined by axis.
        """
        if xyz is None:
            xyz = geometry.xyz

        if isinstance(axis, str) and axis in ("a", "b", "c", "0", "1", "2"):
            return cls._projected_2Dcoords(geometry, xyz, xaxis=axis, yaxis="a" if axis == "c" else "c")[..., 0]

        # Get the direction that the axis represents
        axis = cls._direction(axis, geometry.cell)

        return xyz.dot(axis/fnorm(axis)) / fnorm(axis)

    @classmethod
    def _projected_2Dcoords(cls, geometry, xyz=None, xaxis="x", yaxis="y"):
        """
        Moves the 3D positions of the atoms to a 2D supspace.

        In this way, we can plot the structure from the "point of view" that we want.

        NOTE: If xaxis/yaxis is one of {"a", "b", "c", "1", "2", "3"} the function doesn't
        project the coordinates in the direction of the lattice vector. The fractional
        coordinates, taking in consideration the three lattice vectors, are returned
        instead.

        Parameters
        ------------
        geometry: sisl.Geometry
            the geometry for which you want the projected coords
        xyz: array-like of shape (natoms, 3), optional
            the 3D coordinates that we want to project.
            otherwise they are taken from the geometry. 
        xaxis: {"x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
        yaxis: {"x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis.

        Returns
        ----------
        np.ndarray of shape (2, natoms)
            the 2D coordinates of the geometry, with all positions projected into the plane
            defined by xaxis and yaxis.
        """
        if xyz is None:
            xyz = geometry.xyz

        try:
            all_lattice_vecs = len(set([xaxis, yaxis]).intersection(["a", "b", "c"])) == 2
        except:
            # If set fails it is because xaxis/yaxis is unhashable, which means it
            # is a numpy array
            all_lattice_vecs = False

        if all_lattice_vecs:
            coord_indices = ["abc".index(ax) for ax in (xaxis, yaxis)]

            icell = cell_invert(geometry.cell)
        else:
            # Get the directions that these axes represent
            xaxis = cls._direction(xaxis, geometry.cell)
            yaxis = cls._direction(yaxis, geometry.cell)

            fake_cell = np.array([xaxis, yaxis, np.cross(xaxis, yaxis)], dtype=np.float64)
            icell = cell_invert(fake_cell)
            coord_indices = [0, 1]

        return np.dot(xyz, icell.T)[..., coord_indices]

    def _get_atoms_bonds(self, bonds, atoms):
        """
        Gets the bonds where the given atoms are involved
        """
        return [bond for bond in bonds if np.any([at in atoms for at in bond])]

    #---------------------------------------------------
    #                  1D plotting
    #---------------------------------------------------

    def _prepare1D(self, atoms=None, atoms_styles=None, coords_axis="x", data_axis=None, wrap_atoms=None, atoms_scale=1.,
        nsc=(1, 1, 1), **kwargs):
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
        atoms_styles: dict, optional
            dictionary containing all the style properties of the atoms, it should be build by `self._parse_atoms_style`.
        atoms_colorscale: str or list, optional
            the name of a plotly colorscale or a list of colors.

            Only used if atoms_color is an array of values.
        wrap_atoms: function, optional
            function that takes the 2D positions of the atoms in the plot and returns a tuple of (args, kwargs),
            that are passed to self._atoms_scatter_trace2D.
            If not provided self._default_wrap_atoms is used.
        nsc: array-like of shape (3,), optional
            the number of times the geometry has been tiled in each direction. This is only used to rescale
            fractional coordinates.
        **kwargs: 
            passed directly to the atoms scatter trace
        """
        wrap_atoms = wrap_atoms or self._default_wrap_atoms1D

        x = self._projected_1Dcoords(self.geometry, self._tiled_coords(atoms), axis=coords_axis)
        if data_axis is None:
            def data_axis(x):
                return np.zeros(x.shape[0])

        data_axis_name = data_axis.__name__ if callable(data_axis) else 'Data axis'
        if callable(data_axis):
            data_axis = np.array(data_axis(x))

        xy = np.array([x, data_axis]).T

        atoms_props = wrap_atoms(atoms, xy, atoms_styles)
        atoms_props["size"] *= atoms_scale

        return {
            "geometry": self.geometry, "xaxis": coords_axis, "yaxis": data_axis_name, "atoms_props": atoms_props, "bonds_props": []
        }

    def _default_wrap_atoms1D(self, ats, xy, atoms_styles):

        extra_kwargs = {}

        color = atoms_styles["color"][ats]

        try:
            color.astype(float)
            extra_kwargs["marker_colorscale"] = atoms_styles["colorscale"]
            extra_kwargs["text"] = self._tile_atomic_data([f"Color: {c}" for c in color])
        except ValueError:
            pass

        return {
            "xy": xy,
            "text": self._tile_atomic_data([f'{self.geometry[at]}<br>{at} ({self.geometry.atoms[at].tag})' for at in ats]),
            "name": "Atoms",
            **{k: self._tile_atomic_data(atoms_styles[k][ats]) for k in ("color", "size", "opacity")},
            **extra_kwargs
        }

    #---------------------------------------------------
    #                  2D plotting
    #---------------------------------------------------

    def _prepare2D(self, xaxis="x", yaxis="y",
        atoms=None, atoms_styles=None, atoms_scale=1.,
        show_bonds=True, bonds_styles=None, bind_bonds_to_ats=True,
        points_per_bond=5, wrap_atoms=None, wrap_bond=None, nsc=(1, 1, 1)):
        """Returns a 2D representation of the plot's geometry.

        Parameters
        -----------
        xaxis: {"x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
        yaxis: {"x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
            the direction to be displayed along the X axis. 
        atoms: array-like of int, optional
            the indices of the atoms that you want to plot
        atoms_styles: dict, optional
            dictionary containing all the style properties of the atoms, it should be build by `self._parse_atoms_style`.
        atoms_scale: float, optional
            a factor to multiply atom sizes by.
        atoms_colorscale: str or list, optional
            the name of a plotly colorscale or a list of colors.
            Only used if atoms_color is an array of values.
        show_bonds: boolean, optional
            whether bonds should be plotted.
        bind_bonds_to_ats: boolean, optional
            whether only the bonds that belong to an atom that is present should be displayed.
            If False, all bonds are displayed regardless of the `atom` parameter.
        bonds_styles: dict, optional
            dictionary containing all the style properties of the bonds.
        points_per_bond: int, optional
            If `bonds_together` is True and you provide a variable color or size (using `wrap_bonds`), this is
            the number of points that are used for each bond. See `bonds_together` for more info.
        wrap_atoms: function, optional
            function that recieves the 2D coordinates and returns
            the args (array-like) and kwargs (dict) that go into self._atoms_scatter_trace2D()
            If not provided, self._default_wrap_atoms2D will be used.
            wrap_atom: function, optional
            function that recieves the index of an atom and returns
            the args (array-like) and kwargs (dict) that go into self._atom_trace3D()
            If not provided, self._default_wrap_atoms3D will be used.
        wrap_bond: function, optional
            function that recieves "a bond" (list of 2 atom indices) and its coordinates ((x1,y1), (x2, y2)).
            It should return the args (array-like) and kwargs (dict) that go into `self._bond_trace2D()`
            If not provided, self._default_wrap_bond2D will be used.
        """
        wrap_atoms = wrap_atoms or self._default_wrap_atoms2D
        wrap_bond = wrap_bond or self._default_wrap_bond2D

        # We need to sort the geometry according to depth, because when atoms are drawn they can be one
        # on top of the other. The last atoms should be the ones on top.
        if len(atoms) > 0:
            depth_vector = self._cross_product(xaxis, yaxis, self.geometry.cell)
            sorted_atoms = np.concatenate(self.geometry.sort(atoms=atoms, vector=depth_vector, ret_atoms=True)[1])
        else:
            sorted_atoms = atoms
        xy = self._projected_2Dcoords(self.geometry, self._tiled_coords(sorted_atoms), xaxis=xaxis, yaxis=yaxis)

        # Add atoms
        atoms_props = wrap_atoms(sorted_atoms, xy, atoms_styles)
        atoms_props["size"] *= atoms_scale

        # Add bonds
        if show_bonds:
            # Define the actual bonds that we are going to draw depending on which
            # atoms are requested
            bonds = self.bonds
            if bind_bonds_to_ats:
                bonds = self._get_atoms_bonds(bonds, self._tiled_atoms(atoms))

            bonds_xyz = np.array([self._tiled_geometry[bond] for bond in bonds])
            if len(bonds_xyz) != 0:
                xys = self._projected_2Dcoords(self.geometry, bonds_xyz, xaxis=xaxis, yaxis=yaxis)

                # Try to get the bonds colors (It might be that the user is not setting them)
                bonds_props = [wrap_bond(bond, xy, bonds_styles) for bond, xy in zip(bonds, xys)]
            else:
                bonds_props = []
        else:
            bonds_props = []

        return {
            "geometry": self.geometry, "xaxis": xaxis, "yaxis": yaxis, "atoms_props": atoms_props,
            "bonds_props": bonds_props, "points_per_bond": points_per_bond,
        }

    def _default_wrap_atoms2D(self, ats, xy, atoms_styles):
        return self._default_wrap_atoms1D(ats, xy, atoms_styles)

    def _default_wrap_bond2D(self, bond, xys, bonds_styles):
        return {
            "xys": xys,
            **bonds_styles,
        }

    #---------------------------------------------------
    #                  3D plotting
    #---------------------------------------------------

    def _prepare3D(self, wrap_atoms=None, wrap_bond=None,
        atoms=None, atoms_styles=None, bind_bonds_to_ats=True, atoms_scale=1.,
        show_bonds=True, bonds_styles=None):
        """Returns a 3D representation of the plot's geometry.

        Parameters
        -----------
        wrap_atoms: function, optional
            function that recieves the index of the atoms and returns
            a dictionary with properties of the atoms.
            If not provided, self._default_wrap_atoms3D will be used.
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
        atoms_styles: dict, optional
            dictionary containing all the style properties of the atoms, it should be build by `self._parse_atoms_style`.
        """
        wrap_atoms = wrap_atoms or self._default_wrap_atoms3D
        wrap_bond = wrap_bond or self._default_wrap_bond3D

        try:
            atoms_styles["color"] = np.array(values_to_colors(atoms_styles["color"], atoms_styles["colorscale"]))
        except:
            pass

        atoms_props = wrap_atoms(atoms, atoms_styles)
        atoms_props["size"] *= atoms_scale

        if show_bonds:
            # Try to get the bonds colors (It might be that the user is not setting them)
            bonds = self.bonds
            if bind_bonds_to_ats:
                bonds = self._get_atoms_bonds(bonds, self._tiled_atoms(atoms))
            bonds_props = [wrap_bond(bond, bonds_styles) for bond in bonds]
        else:
            bonds = []
            bonds_props = []

        return {"geometry": self.geometry, "atoms_props": atoms_props, "bonds_props": bonds_props}

    def _default_wrap_atoms3D(self, ats, atoms_styles):

        return {
            "xyz": self._tiled_coords(ats),
            "name": self._tile_atomic_data([f'{at} ({self.geometry.atoms[at].tag})' for at in ats]),
            **{k: self._tile_atomic_data(atoms_styles[k][ats]) for k in ("color", "size", "vertices", "opacity")}
        }

    def _default_wrap_bond3D(self, bond, bonds_styles):

        return {
            "xyz1": self._tiled_geometry[bond[0]],
            "xyz2": self._tiled_geometry[bond[1]],
            #"r": 15,
            **bonds_styles,
        }
