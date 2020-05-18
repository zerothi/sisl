import numpy as np
import pandas as pd
import itertools
from functools import partial

import sisl
from ..plot import Plot
from .geometry import GeometryPlot, BoundGeometry
from ..plotutils import find_files
from ..input_fields import TextInput, FilePathInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput

class BondLengthMap(GeometryPlot):
    
    '''
    Colorful representation of bond lengths.

    Parameters
    -------------
    %%configurable_settings%%
    '''

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

        FilePathInput(
            key = "strain_ref", name = "Strain reference geometry",
            default = None,
            dtype=(str, sisl.Geometry),
            group = "dataread",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write the path to your strain reference file here..."
            },
            help = '''The path to a geometry or a Geometry object used to calculate strain from.<br>
            This geometry will probably be the relaxed one<br>
            If provided, colors can indicate strain values. Otherwise they are just bond length'''
        ),

        SwitchInput(
            key = "strain", name = "Display strain",
            default = True,
            params = {
                "offLabel": False,
                "onLabel": True
            },
            help = '''Determines whether strain values should be displayed instead of lengths'''
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
            help = '''This determines the colormap to be used for the bond lengths display.<br>
            You can see all valid colormaps here: <a>https://plot.ly/python/builtin-colorscales/<a/><br>
            Note that you can reverse a color map by adding _r'''
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
            help = '''Sets the middle point of the color scale. Only meaningful in diverging colormaps<br>
            If this is set 'cmin' and 'cmax' are ignored. In strain representations this might be set to 0.
            '''
        ),

        SwitchInput(
            key='colorbar', name='Show colorbar',
            default=True,
            help='''Whether the color bar should be displayed or not.'''
        ),
        
        IntegerInput(
            key = "points_per_bond", name = "Points per bond",
            default = 10,
            help = "Number of points that fill a bond. <br>More points will make it look more like a line but will slow plot rendering down."
        ),
    
    )

    _layout_defaults = {
        'xaxis_title': 'X (Ang)', 
        'yaxis_title': "Y (Ang)",
        'yaxis_zeroline': False
    }
    
    @classmethod
    def _default_animation(self, wdir = None, frameNames = None, **kwargs):
        
        geomsFiles = find_files(wdir, "*.XV", sort = True)

        return BondLengthMap.animated("geom_file", geomsFiles, wdir = wdir, **kwargs)

    @property
    def on_relaxed_geom(self):
        return BoundGeometry(self.relaxed_geom, self)

    def _read_nosource(self):

        GeometryPlot._read_nosource(self)

        self._read_strain_ref()

    def _read_siesta_output(self):
        
        GeometryPlot._read_siesta_output(self)

        self._read_strain_ref()

    def _read_strain_ref(self):

        strain_ref = self.setting("strain_ref")

        if isinstance(strain_ref, str):
            self.relaxed_geom = self.get_sile(strain_ref).read_geometry()
        elif isinstance(strain_ref, sisl.Geometry):
            self.relaxed_geom = strain_ref

    def _after_read(self):

        self.geom_bonds = self.find_all_bonds(self.geom)

        if getattr(self, "relaxed_geom", None):
            self.relaxed_bonds = self.find_all_bonds(self.relaxed_geom)
        
        self.get_param("atom").update_options(self.geom)
    
    def _wrap_bond3D(self, bond, strain=False):
        '''
        Receives a bond and sets its color to the bond length
        '''

        if strain:
            color = self._bond_strain(self.relaxed_geom, self.geom, bond)
            name = f'Strain: {color:.3f}'
        else:
            color = self._bond_length(self.geom, bond)
            name = f'{color:.3f} Ang'
        
        self.colors.append(color)

        return (*self.geom[bond], 15), {"color": color, "name": name }
    
    def _wrap_bond2D(self, bond, xys, strain=False):

        if strain:
            color = self._bond_strain(self.relaxed_geom, self.geom, bond)
            name = f'Strain: {color:.3f}'
        else:
            color = self._bond_length(self.geom, bond)
            name = f'{color:.3f} Ang'

        self.colors.append(color)

        return (*xys, ), {"color": color, "name": name}
    
    @staticmethod
    def _bond_length(geom, bond):
        return np.linalg.norm(geom[bond[1]] - geom[bond[0]])
    
    @staticmethod
    def _bond_strain(relaxed_geom, geom, bond):

        relaxed_bl = BondLengthMap._bond_length(relaxed_geom, bond)
        bond_length = BondLengthMap._bond_length(geom, bond)

        return (bond_length - relaxed_bl) / relaxed_bl

    def _set_data(self):

        axes = self.setting("axes")
        ndims = len(axes)
        cell_rendering = self.setting("cell")
        if self.setting("show_atoms") == False:
            atom = []
            bind_bonds_to_ats = False
        else:
            atom = self.setting("atom")
            bind_bonds_to_ats = self.setting("bind_bonds_to_ats")
        
        # Set the bonds to the relaxed ones if there is a strain reference
        show_strain = self.setting("strain")
        show_strain = show_strain and hasattr(self, "relaxed_bonds")
        if show_strain:
            self.bonds = self.relaxed_bonds

            self.geom.set_nsc(self.relaxed_geom.sc.nsc)
        else:
            self.bonds = self.geom_bonds

        # We will initialize the colors list so that it is filled by
        # the methods that generate them and we can at the end set the limits
        # of the color scale
        self.colors = []

        if ndims == 3:
            self._plot_geom3D(cell=cell_rendering, cheap_bonds=True,
                wrap_bond=partial(self._wrap_bond3D, strain=show_strain), 
                atom=atom, bind_bonds_to_ats=bind_bonds_to_ats
            )
        elif ndims == 2:
            xaxis, yaxis = axes
            points_per_bond = self.setting("points_per_bond")

            self._plot_geom2D(
                xaxis=xaxis, yaxis=yaxis, cell=cell_rendering,
                bonds_together=True, points_per_bond=points_per_bond,
                wrap_bond=partial(self._wrap_bond2D, strain=show_strain),
                atom=atom, bind_bonds_to_ats=bind_bonds_to_ats
            )

            self.update_layout(xaxis_title=f'Axis {xaxis} (Ang)', yaxis_title=f'Axis {yaxis} (Ang)')
        elif ndims == 1:
            raise NotImplementedError("Does it make sense to implement 1 dimensional bond length maps? If so, post an issue on sisl's github page. Thanks!")

        showscale = self.setting('colorbar')
        
        self.update_layout(coloraxis={"cmin": self.setting("cmin") or min(self.colors) ,
                                      "cmax": self.setting("cmax") or max(self.colors),
                                      "colorscale": self.setting("cmap"),
                                      'showscale': showscale,
                                      'colorbar_title': 'Strain' if show_strain else 'Bond length (Ang)'})
        
        self.update_layout(legend_orientation='h')
        
        #tileCombs = itertools.product(*[range(self.setting(tile)) for tile in ("tileX", "tileY", "tileZ")])


# points_per_bond = self.setting("points_per_bond")
# self.show_strain = self.is_strain and self.setting("show_strain")
# colorColumn = "Strain" if self.show_strain else "Bond Length"
# xAxis = self.setting("xAxis"); yAxis = self.setting("yAxis")

# for tiles in tileCombs :
    
#     #Get the translation vector
#     translate = np.array(tiles).dot(self.geom.cell)

#     #Draw bonds
#     self.data = [*self.data, *[{
#                     'type': 'scatter',
#                     'x': np.linspace(bond["init{}".format(xAxis)], bond["final{}".format(xAxis)], points_per_bond) + translate[["X","Y","Z"].index(xAxis)],
#                     'y': np.linspace(bond["init{}".format(yAxis)], bond["final{}".format(yAxis)], points_per_bond) + translate[["X","Y","Z"].index(yAxis)],
#                     'mode': 'markers', 
#                     'name': "{}{}-{}{}".format(bond["From Species"], bond["From"], bond["To Species"], bond["To"]), 
#                     #'line': {"color": "rgba{}".format(cmap(norm(bond["Bond Length"])) ), "width": 3},
#                     'marker': {
#                         "size": 3, 
#                         "color": [bond[colorColumn]]*points_per_bond, 
#                         "coloraxis": "coloraxis"
#                     },
#                     "showlegend": False,
#                     'hoverinfo': "name",
#                     'hovertemplate':'{:.2f} Ang{}'.format(bond["Bond Length"], ". Strain: {:.3f}".format(bond["Strain"]) if self.isStrain else "" ),
#                 } for i, bond in self.df.iterrows() ]]

    # def _after_get_figure(self):

    #     #Add the ticks
    #     self.figure.layout.yaxis.scaleratio = 1
    #     self.figure.layout.yaxis.scaleanchor = "x"
        
    #     colorColumn = "Strain" if self.show_strain else "Bond Length"
    #     cmap = self.setting("cmap")
    #     reverse = "_r" in cmap
    #     cmap = cmap[:-2] if reverse else cmap
    #     self.figure.update_layout(coloraxis = {
    #         'colorbar': {
    #             'title': "Strain" if self.show_strain else "Length (Ang)"
    #         },
    #         'colorscale': cmap,
    #         'reversescale': reverse ,
    #         "cmin": (self.setting("cmin") or self.df[colorColumn].min()) if self.setting("cmid") == None else None,
    #         "cmax": (self.setting("cmax") or self.df[colorColumn].max()) if self.setting("cmid") == None else None,
    #         "cmid": self.setting("cmid"),
    #     }, xaxis_title='X (Ang)', yaxis_title="Y (Ang)")

#Build the dataframe with all the bonds info
        # dfKeys = ("From", "To", "From Species", "To Species", "Bond Length",
        #           "initX", "initY", "initZ", "finalX", "finalY", "finalZ")
        # strainKeys = ("Relaxed Length", "Strain") if self.isStrain else ()

        # dfKeys = (*dfKeys, *strainKeys)
        # bondsDict = {key: [] for key in dfKeys}

        # for at in self.geom:

        #     #If there is a strain reference we take the neighbors of each atom from it
        #     if self.isStrain:
        #         geom = self.relaxedGeom
        #     else:
        #         geom = self.geom

        #     _, neighs = geom.close(at, R=(0.1, self.setting("bond_thresh")))

        #     for neigh in neighs:

        #         bondsDict["From"].append(at)
        #         bondsDict["To"].append(neigh)
        #         bondsDict["From Species"].append(self.geom.atoms[at].symbol)
        #         bondsDict["To Species"].append(
        #             self.geom.atom[neigh % self.geom.na].symbol)
        #         bondsDict["Bond Length"].append(
        #             np.linalg.norm(self.geom[at] - self.geom[neigh]))

        #         if self.isStrain:
        #             relLength = np.linalg.norm(
        #                 self.relaxedGeom[at] - self.relaxedGeom[neigh])
        #             bondsDict["Relaxed Length"].append(relLength)
        #             bondsDict["Strain"].append(
        #                 (bondsDict["Bond Length"][-1] - relLength)/relLength)

        #         bondsDict["initX"].append(self.geom[at][0])
        #         bondsDict["initY"].append(self.geom[at][1])
        #         bondsDict["initZ"].append(self.geom[at][2])
        #         bondsDict["finalX"].append(self.geom[neigh][0])
        #         bondsDict["finalY"].append(self.geom[neigh][1])
        #         bondsDict["finalZ"].append(self.geom[neigh][2])

        # self.df = pd.DataFrame(bondsDict)
