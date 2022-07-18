# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from functools import partial

import sisl
from sisl.utils.mathematics import fnorm
from .geometry import GeometryPlot, BoundGeometry
from ..plotutils import find_files
from ..input_fields import TextInput, BoolInput,\
     IntegerInput, FloatInput, SileInput


class BondLengthMap(GeometryPlot):
    """
    Colorful representation of bond lengths.

    Parameters
    -------------
    geom_from_output: bool, optional
        In case the geometry is read from the fdf file, this will determine
        whether the input or the output geometry is taken.This setting
        will be ignored if geom_file is passed
    strain_ref: str or Geometry, optional
        The path to a geometry or a Geometry object used to calculate strain
        from.             This geometry will probably be the relaxed
        one             If provided, colors can indicate strain values.
        Otherwise they are just bond length
    strain: bool, optional
        Determines whether strain values should be displayed instead of
        lengths
    bond_thresh: float, optional
        Maximum distance between two atoms to draw a bond
    colorscale: str, optional
        This determines the colormap to be used for the bond lengths
        display.             You can see all valid colormaps here:
        https://plot.ly/python/builtin-colorscales/
        Note that you can reverse a color map by adding _r
    cmin: float, optional
        Minimum color scale
    cmax: float, optional
        Maximum color scale
    cmid: float, optional
        Sets the middle point of the color scale. Only meaningful in
        diverging colormaps             If this is set 'cmin' and 'cmax'
        are ignored. In strain representations this might be set to 0.
    colorbar: bool, optional
        Whether the color bar should be displayed or not.
    points_per_bond: int, optional
        Number of points that fill a bond. More points will make it look
        more like a line but will slow plot rendering down.
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

    _plot_type = "Bond length"

    _parameters = (

        BoolInput(
            key = "geom_from_output", name = "Geometry from output",
            default = True,
            group = "dataread",
            params = {
                "offLabel": "No",
                "onLabel": "Yes",
            },
            help = "In case the geometry is read from the fdf file, this will determine whether the input or the output geometry is taken.<br>This setting will be ignored if geom_file is passed"
        ),

        SileInput(
            key = "strain_ref", name = "Strain reference geometry",
            hasattr=['read_geometry'],
            dtype=(str, sisl.Geometry),
            group = "dataread",
            params = {
                "placeholder": "Write the path to your strain reference file here..."
            },
            help = """The path to a geometry or a Geometry object used to calculate strain from.<br>
            This geometry will probably be the relaxed one<br>
            If provided, colors can indicate strain values. Otherwise they are just bond length"""
        ),

        BoolInput(
            key = "strain", name = "Display strain",
            default = True,
            params = {
                "offLabel": False,
                "onLabel": True
            },
            help = """Determines whether strain values should be displayed instead of lengths"""
        ),

        FloatInput(
            key = "bond_thresh", name = "Bond length threshold",
            default = 1.7,
            params = {
                "step": 0.01
            },
            help = "Maximum distance between two atoms to draw a bond"
        ),

        TextInput(
            key="colorscale", name="Plotly colormap",
            default="viridis",
            params={
                "placeholder": "Write a valid plotly colormap here..."
            },
            help="""This determines the colormap to be used for the bond lengths display.<br>
            You can see all valid colormaps here: <a>https://plot.ly/python/builtin-colorscales/<a/><br>
            Note that you can reverse a color map by adding _r"""
        ),

        FloatInput(
            key = "cmin", name = "Color scale low limit",
            default = 0,
            params = {
                "step": 0.01
            },
            help="Minimum color scale"
        ),

        FloatInput(
            key = "cmax", name = "Color scale high limit",
            default = 0,
            params = {
                "step": 0.01
            },
            help="Maximum color scale"
        ),

        FloatInput(
            key = "cmid", name = "Color scale mid point",
            default = None,
            params = {
                "step": 0.01
            },
            help = """Sets the middle point of the color scale. Only meaningful in diverging colormaps<br>
            If this is set 'cmin' and 'cmax' are ignored. In strain representations this might be set to 0.
            """
        ),

        BoolInput(
            key='colorbar', name='Show colorbar',
            default=True,
            help="""Whether the color bar should be displayed or not."""
        ),

        IntegerInput(
            key="points_per_bond", name="Points per bond",
            default=10,
            help="Number of points that fill a bond. <br>More points will make it look more like a line but will slow plot rendering down."
        ),

    )

    _layout_defaults = {
        'xaxis_title': 'X [Ang]',
        'yaxis_title': "Y [Ang]",
        'yaxis_zeroline': False
    }

    @classmethod
    def _default_animation(self, wdir=None, frame_names=None, **kwargs):
        """By default, we will animate all the *XV files that we find"""
        geom_files = find_files(wdir, "*.XV", sort = True)

        return BondLengthMap.animated("geom_file", geom_files, wdir = wdir, **kwargs)

    @property
    def on_relaxed_geom(self):
        """
        Returns a bound geometry, which you can apply methods to so that the plot
        updates automatically.
        """
        return BoundGeometry(self.relaxed_geom, self)

    _read_geom = GeometryPlot.entry_points[0]
    _read_file = GeometryPlot.entry_points[1]

    def _read_strain_ref(self, ref):
        """Reads the strain reference, if there is any."""
        strain_ref = ref

        if isinstance(strain_ref, str):
            self.relaxed_geom = self.get_sile(strain_ref).read_geometry()
        elif isinstance(strain_ref, sisl.Geometry):
            self.relaxed_geom = strain_ref
        else:
            self.relaxed_geom = None

    def _after_read(self, strain_ref, nsc):
        self._read_strain_ref(strain_ref)

        is_strain_ref = self.relaxed_geom is not None

        self._tiled_geometry = self.geometry
        for ax, reps in enumerate(nsc):
            self._tiled_geometry = self._tiled_geometry.tile(reps, ax)
            if is_strain_ref:
                self.relaxed_geom = self.relaxed_geom.tile(reps, ax)

        self.geom_bonds = self.find_all_bonds(self._tiled_geometry)

        if is_strain_ref:
            self.relaxed_bonds = self.find_all_bonds(self.relaxed_geom)

        self.get_param("atoms").update_options(self.geometry)

    def _wrap_bond3D(self, bond, bonds_styles, show_strain=False):
        """
        Receives a bond and sets its color to the bond length for the 3D case
        """
        if show_strain:
            color = self._bond_strain(self.relaxed_geom, self._tiled_geometry, bond)
            name = f'Strain: {color:.3f}'
        else:
            color = self._bond_length(self._tiled_geometry, bond)
            name = f'{color:.3f} Ang'

        self.colors.append(color)

        return {
            **self._default_wrap_bond3D(bond, bonds_styles=bonds_styles),
            "color": color,
            "name": name
        }

    def _wrap_bond2D(self, bond, xys, bonds_styles, show_strain=False):
        """
        Receives a bond and sets its color to the bond length for the 2D case
        """
        if show_strain:
            color = self._bond_strain(self.relaxed_geom, self._tiled_geometry, bond)
            name = f'Strain: {color:.3f}'
        else:
            color = self._bond_length(self._tiled_geometry, bond)
            name = f'{color:.3f} Ang'

        self.colors.append(color)

        return {
            **self._default_wrap_bond2D(bond, xys, bonds_styles=bonds_styles),
            "color": color, "name": name
        }

    @staticmethod
    def _bond_length(geom, bond):
        """
        Returns the length of a bond between two atoms.

        Parameters
        ------------
        geom: Geometry
            the structure where the atoms are
        bond: array-like of two int
            the indices of the atoms that form the bond
        """
        return fnorm(geom[bond[1]] - geom[bond[0]])

    @staticmethod
    def _bond_strain(relaxed_geom, geom, bond):
        """
        Calculates the strain of a bond using a reference geometry.

        Parameters
        ------------
        relaxed_geom: Geometry
            the structure to take as a reference
        geom: Geometry
            the structure to take as the "current" one
        bond: array-like of two int
            the indices of the atoms that form the bond
        """
        relaxed_bl = BondLengthMap._bond_length(relaxed_geom, bond)
        bond_length = BondLengthMap._bond_length(geom, bond)

        return (bond_length - relaxed_bl) / relaxed_bl

    def _set_data(self, strain, axes, atoms, show_atoms, bind_bonds_to_ats, points_per_bond, cmin, cmax, colorscale, colorbar,
        kwargs3d={}, kwargs2d={}, kwargs1d={}):

        # Set the bonds to the relaxed ones if there is a strain reference
        show_strain = strain and hasattr(self, "relaxed_bonds")
        if show_strain:
            self.bonds = self.relaxed_bonds

            self.geometry.set_nsc(self.relaxed_geom.sc.nsc)
        else:
            self.bonds = self.geom_bonds

        # We will initialize the colors list so that it is filled by
        # the methods that generate them and we can at the end set the limits
        # of the color scale
        self.colors = []

        # Let GeometryPlot set the data
        for_backend = super()._set_data(
            kwargs3d={
                "wrap_bond": partial(self._wrap_bond3D, show_strain=show_strain),
                **kwargs3d
            },
            kwargs2d={
                "wrap_bond": partial(self._wrap_bond2D, show_strain=show_strain),
                "points_per_bond": points_per_bond,
                **kwargs2d
            },
            kwargs1d=kwargs1d
        )

        if self.colors:
            for_backend["bonds_coloraxis"] = {
                "cmin": cmin or min(self.colors),
                "cmax": cmax or max(self.colors),
                "colorscale": colorscale,
                'showscale': colorbar,
                'colorbar_title': 'Strain' if show_strain else 'Bond length [Ang]'
            }

        return for_backend
