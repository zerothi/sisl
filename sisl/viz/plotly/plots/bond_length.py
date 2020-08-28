import numpy as np
import pandas as pd
import itertools
from functools import partial

import sisl
from sisl.utils.mathematics import fnorm
from ..plot import Plot, entry_point
from .geometry import GeometryPlot, BoundGeometry
from ..plotutils import find_files
from ..input_fields import TextInput, FilePathInput, SwitchInput, ColorPicker, DropdownInput, \
     IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput, SileInput


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
    cmap: str, optional
        This determines the colormap to be used for the bond lengths
        display.             You can see all valid colormaps here:
        https://plot.ly/python/builtin-colorscales/
        Note that you can reverse a color map by adding _r
    cmin: float, optional

    cmax: float, optional

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

    geom_file: str, optional

    bonds: bool, optional

    axes:  optional
        The axis along which you want to see the geometry.              You
        can provide as many axes as dimensions you want for your plot.
        Note that the order is important and will result in setting the plot
        axes diferently.             For 2D and 1D representations, you can
        pass an arbitrary direction as an axis (array of shape (3,))
    dataaxis_1d:  optional
        If you want a 1d representation, you can provide a data axis.
        It should be a function that receives the 1d coordinate of each atom
        and             returns it's "data-coordinate", which will be in the
        y axis of the plot.             If not provided, the y axis will be
        all 0.
    cell:  optional
        Specifies how the cell should be rendered.              (False: not
        rendered, 'axes': render axes only, 'box': render a bounding box)
    atom:  optional
        The atoms that are going to be displayed in the plot.
        This also has an impact on bonds (see the `bind_bonds_to_ats` and
        `show_atoms` parameters).             If set to None, all atoms are
        displayed
    bind_bonds_to_ats: bool, optional
        whether only the bonds that belong to an atom that is present should
        be displayed.             If False, all bonds are displayed
        regardless of the `atom` parameter
    show_atoms: bool, optional
        If set to False, it will not display atoms.              Basically
        this is a shortcut for `atoms = [], bind_bonds_to_ats=False`.
        Therefore, it will override these two parameters.
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    _plot_type = "Bond length"

    _parameters = (

        SwitchInput(
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
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write the path to your strain reference file here..."
            },
            help = """The path to a geometry or a Geometry object used to calculate strain from.<br>
            This geometry will probably be the relaxed one<br>
            If provided, colors can indicate strain values. Otherwise they are just bond length"""
        ),

        SwitchInput(
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
            key = "cmap", name = "Plotly colormap",
            default = "viridis",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write a valid plotly colormap here..."
            },
            help = """This determines the colormap to be used for the bond lengths display.<br>
            You can see all valid colormaps here: <a>https://plot.ly/python/builtin-colorscales/<a/><br>
            Note that you can reverse a color map by adding _r"""
        ),

        # IntegerInput(
        #     key = "tileX", name = "Tile first axis",
        #     default = 1,
        #     params = {
        #         "min": 1
        #     },
        #     help = "Number of unit cells to display along the first axis"
        # ),

        # IntegerInput(
        #     key = "tileY", name = "Tile second axis",
        #     default = 1,
        #     params = {
        #         "min": 1
        #     },
        #     help = "Number of unit cells to display along the second axis"
        # ),

        # IntegerInput(
        #     key = "tileZ", name = "Tile third axis",
        #     default = 1,
        #     params = {
        #         "min": 1
        #     },
        #     help = "Number of unit cells to display along the third axis"
        # ),

        FloatInput(
            key = "cmin", name = "Color scale low limit",
            default = 0,
            params = {
                "step": 0.01
            }
        ),

        FloatInput(
            key = "cmax", name = "Color scale high limit",
            default = 0,
            params = {
                "step": 0.01
            }
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

        SwitchInput(
            key='colorbar', name='Show colorbar',
            default=True,
            help="""Whether the color bar should be displayed or not."""
        ),

        IntegerInput(
            key = "points_per_bond", name = "Points per bond",
            default = 10,
            help = "Number of points that fill a bond. <br>More points will make it look more like a line but will slow plot rendering down."
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

    @entry_point('geometry')
    def _read_nosource(self):

        GeometryPlot._read_nosource(self)

        self._read_strain_ref()

    @entry_point('geom_file')
    def _read_siesta_output(self):

        GeometryPlot._read_siesta_output(self)

        self._read_strain_ref()

    def _read_strain_ref(self):
        """Reads the strain reference, if there is any."""
        strain_ref = self.setting("strain_ref")

        if isinstance(strain_ref, str):
            self.relaxed_geom = self.get_sile(strain_ref).read_geometry()
        elif isinstance(strain_ref, sisl.Geometry):
            self.relaxed_geom = strain_ref

    def _after_read(self):
        self.geom_bonds = self.find_all_bonds(self.geometry)

        if getattr(self, "relaxed_geom", None):
            self.relaxed_bonds = self.find_all_bonds(self.relaxed_geom)

        self.get_param("atoms").update_options(self.geometry)

    def _wrap_bond3D(self, bond, strain=False):
        """
        Receives a bond and sets its color to the bond length for the 3D case
        """
        if strain:
            color = self._bond_strain(self.relaxed_geom, self.geometry, bond)
            name = f'Strain: {color:.3f}'
        else:
            color = self._bond_length(self.geometry, bond)
            name = f'{color:.3f} Ang'

        self.colors.append(color)

        return (*self.geometry[bond], 15), {"color": color, "name": name}

    def _wrap_bond2D(self, bond, xys, strain=False):
        """
        Receives a bond and sets its color to the bond length for the 2D case
        """
        if strain:
            color = self._bond_strain(self.relaxed_geom, self.geometry, bond)
            name = f'Strain: {color:.3f}'
        else:
            color = self._bond_length(self.geometry, bond)
            name = f'{color:.3f} Ang'

        self.colors.append(color)

        return (*xys, ), {"color": color, "name": name}

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

    def _set_data(self):
        axes = self.setting("axes")
        bonds = self.setting('bonds')
        ndims = len(axes)
        cell_rendering = self.setting("cell")
        if self.setting("show_atoms") == False:
            atoms = []
            bind_bonds_to_ats = False
        else:
            atoms = self.setting("atoms")
            bind_bonds_to_ats = self.setting("bind_bonds_to_ats")
        atoms_color = self.setting("atoms_color")
        atoms_colorscale = self.setting("atoms_colorscale")
        atoms_size = self.setting("atoms_size")
        atoms_vertices = self.setting("atoms_vertices")

        # Set the bonds to the relaxed ones if there is a strain reference
        show_strain = self.setting("strain")
        show_strain = show_strain and hasattr(self, "relaxed_bonds")
        if show_strain:
            self.bonds = self.relaxed_bonds

            self.geometry.set_nsc(self.relaxed_geom.sc.nsc)
        else:
            self.bonds = self.geom_bonds

        # We will initialize the colors list so that it is filled by
        # the methods that generate them and we can at the end set the limits
        # of the color scale
        self.colors = []

        common_kwargs = {'cell': cell_rendering, 'show_bonds': bonds,
            'atoms': atoms, "atoms_color": atoms_color, "atoms_size": atoms_size, "atoms_colorscale": atoms_colorscale,
            'bind_bonds_to_ats': bind_bonds_to_ats
        }

        if ndims == 3:
            self._plot_geom3D(cheap_bonds=True,
                wrap_bond=partial(self._wrap_bond3D, strain=show_strain),
                atoms_vertices=atoms_vertices,
                **common_kwargs
            )
        elif ndims == 2:
            xaxis, yaxis = axes
            points_per_bond = self.setting("points_per_bond")

            self._plot_geom2D(
                xaxis=xaxis, yaxis=yaxis,
                bonds_together=True, points_per_bond=points_per_bond,
                wrap_bond=partial(self._wrap_bond2D, strain=show_strain),
                **common_kwargs
            )

            self.update_layout(xaxis_title=f'Axis {xaxis} [Ang]', yaxis_title=f'Axis {yaxis} [Ang]')
        elif ndims == 1:
            raise NotImplementedError("Does it make sense to implement 1 dimensional bond length maps? If so, post an issue on sisl's github page. Thanks!")

        showscale = self.setting('colorbar')

        if self.colors:
            self.update_layout(coloraxis={"cmin": self.setting("cmin") or min(self.colors),
                                        "cmax": self.setting("cmax") or max(self.colors),
                                        "colorscale": self.setting("cmap"),
                                        'showscale': showscale,
                                        'colorbar_title': 'Strain' if show_strain else 'Bond length [Ang]'})

        self.update_layout(legend_orientation='h')
