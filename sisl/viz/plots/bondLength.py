import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import itertools

import os
import shutil

import sisl
from ..plot import Plot, MultiplePlot, Animation, PLOTS_CONSTANTS
from ..plotutils import sortOrbitals, initMultiplePlots, copyParams, findFiles, runMultiple, calculateGap
from ..inputFields import TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput

class BondLengthMap(Plot):
    
    '''
    Colorful representation of bond lengths.
    '''

    _plotType = "Bond length"
    
    _requirements = {
        
    }
    
    _parameters = (
        
        SwitchInput(
            key = "geomFromOutput", name = "Geometry from output",
            default = True,
            group = "readdata",
            params = {
                "offLabel": "No",
                "onLabel": "Yes",
            },
            help = "In case the geometry is read from the fdf file, this will determine whether the input or the output geometry is taken.<br>This setting will be ignored if geomFile is passed"
        ),
        
        TextInput(
            key = "geomFile", name = "Path to the geometry file",
            group = "readdata",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write the path to your geometry file here..."
            },
            help = '''This parameter explicitly sets a geometry file. Otherwise, the geometry is attempted to read from the fdf file '''
        ),

        TextInput(
            key = "strainRef", name = "Strain reference geometry",
            default = None,
            group = "readdata",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write the path to your strain reference file here..."
            },
            help = '''The path to a geometry used to calculate strain from.<br>
            This geometry will probably be the relaxed one<br>
            If provided, colors can indicate strain values. Otherwise they are just bond length'''
        ),

        SwitchInput(
            key = "showStrain", name = "Bond display mode",
            default = True,
            params = {
                "offLabel": "Length",
                "onLabel": "Strain"
            },
            help = '''Determines whether, <b>IF POSSIBLE</b>, strain values should be displayed instead of lengths<br>
            If this is set to show strain, but no strain reference is set, <b>it will be ignored</b>
            '''
        ),
        
        FloatInput(
            key = "bondThreshold", name = "Bond length threshold",
            default = 1.7,
            params = {
                "step": 0.01
            },
            help = "Maximum distance between two atoms to draw a bond"
        ),
        
        TextInput(
            key = "cmap", name = "Plotly colormap",
            default = "solar",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write a valid plotly colormap here..."
            },
            help = '''This determines the colormap to be used for the bond lengths display.<br>
            You can see all valid colormaps here: <a>https://plot.ly/python/builtin-colorscales/<a/><br>
            Note that you can reverse a color map by adding _r'''
        ),
        
        IntegerInput(
            key = "tileX", name = "Tile first axis",
            default = 1,
            params = {
                "min": 1
            },
            help = "Number of unit cells to display along the first axis"
        ),
        
        IntegerInput(
            key = "tileY", name = "Tile second axis",
            default = 1,
            params = {
                "min": 1
            },
            help = "Number of unit cells to display along the second axis"
        ),
        
        IntegerInput(
            key = "tileZ", name = "Tile third axis",
            default = 1,
            params = {
                "min": 1
            },
            help = "Number of unit cells to display along the third axis"
        ),
            
        DropdownInput(
            key = "xAxis", name = "Coordinate in X axis",
            default = "X",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Choose the coordinate of the X axis...",
                "options": [
                    {"label": ax, "value": ax} for ax in ("X", "Y", "Z")
                ],
                "isClearable": False,
                "isSearchable": True,
            },
            help = "This is the coordinate that will be shown in the X axis of the plot "
        ),

        DropdownInput(
            key = "yAxis", name = "Coordinate in Y axis",
            default = "Y",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Choose the coordinate of the Y axis...",
                "options": [
                    {"label": ax, "value": ax} for ax in ("X", "Y", "Z")
                ],
                "isClearable": False,
                "isSearchable": True,
            },
            help = "This is the coordinate that will be shown in the Y axis of the plot "
        ),
        
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
        
        IntegerInput(
            key = "pointsPerBond", name = "Points per bond",
            default = 5,
            help = "Number of points that fill a bond <br>More points will make it look more like a line but will slow plot rendering down."
        )
    
    )

    _overwrite_defaults = {
        'xaxis_title': 'X (Ang)', 
        'yaxis_title': "Y (Ang)",
        'yaxis_zeroline': False
    }
    
    @classmethod
    def _defaultAnimation(self, wdir = None, frameNames = None, **kwargs):
        
        geomsFiles = findFiles(wdir, "*.XV", sort = True)

        #def _getFrameNames(self):

            #return [os.path.basename( childPlot.setting("bandsFile")) for childPlot in self.childPlots]

        return BondLengthMap.animated("geomFile", geomsFiles, wdir = wdir, **kwargs)

    def _readSiesOut(self):
        
        geom_keys = {}
        if self.setting("geomFile"):
            geomFile = self.setting("geomFile")
        else:
            geomFile = self.setting("rootFdf")
            geom_keys = {"output": self.setting("geomFromOutput")}
        
        self.geom = self.get_sile(geomFile).read_geometry()
        
        self.isStrain = False
        strainRef_file = self.setting("strainRef")
        if strainRef_file:

            self.relaxedGeom = self.get_sile(strainRef_file).read_geometry()
            self.isStrain = True

            self.relaxedGeom.set_nsc([3,3,3])
        
        #If there isn't a supercell in all directions define it
        self.geom.set_nsc([3,3,3])
        
        #Build the dataframe with all the bonds info
        dfKeys = ("From", "To", "From Species", "To Species", "Bond Length",
            "initX", "initY", "initZ", "finalX", "finalY", "finalZ")
        strainKeys = ("Relaxed Length", "Strain") if self.isStrain else ()

        dfKeys = (*dfKeys, *strainKeys)
        bondsDict = { key: [] for key in dfKeys}

        for at in self.geom:

            #If there is a strain reference we take the neighbors of each atom from it
            if self.isStrain:
                geom = self.relaxedGeom
            else:
                geom = self.geom
            
            _, neighs = geom.close(at, R = (0.1, self.setting("bondThreshold")))

            for neigh in neighs:

                bondsDict["From"].append(at)
                bondsDict["To"].append(neigh)
                bondsDict["From Species"].append(self.geom.atoms[at].symbol)
                bondsDict["To Species"].append(self.geom.atom[neigh % self.geom.na].symbol)
                bondsDict["Bond Length"].append(np.linalg.norm(self.geom[at] - self.geom[neigh]))

                if self.isStrain:
                    relLength = np.linalg.norm(self.relaxedGeom[at] - self.relaxedGeom[neigh])
                    bondsDict["Relaxed Length"].append(relLength)
                    bondsDict["Strain"].append( (bondsDict["Bond Length"][-1] - relLength)/relLength )

                bondsDict["initX"].append(self.geom[at][0])
                bondsDict["initY"].append(self.geom[at][1])
                bondsDict["initZ"].append(self.geom[at][2])
                bondsDict["finalX"].append(self.geom[neigh][0])
                bondsDict["finalY"].append(self.geom[neigh][1])
                bondsDict["finalZ"].append(self.geom[neigh][2])

        self.df = pd.DataFrame(bondsDict)
    
    def _setData(self):
        
        """ #Define a colormap
        cmap = plt.cm.get_cmap(self.setting("cmap"))
        
        #Get the normalizer
        cmin = self.setting("cmin") or self.df["Bond Length"].min()
        cmax = self.setting("cmax") or self.df["Bond Length"].max()
        norm = matplotlib.colors.Normalize(cmin, cmax) """
        
        self.data = []
        tileCombs = itertools.product(*[range(self.setting(tile)) for tile in ("tileX", "tileY", "tileZ")])
        pointsPerBond = self.setting("pointsPerBond")
        self.showStrain = self.isStrain and self.setting("showStrain")
        colorColumn = "Strain" if self.showStrain else "Bond Length"
        xAxis = self.setting("xAxis"); yAxis = self.setting("yAxis")
        
        for tiles in tileCombs :
            
            #Get the translation vector
            translate = np.array(tiles).dot(self.geom.cell)
        
            #Draw bonds
            self.data = [*self.data, *[{
                            'type': 'scatter',
                            'x': np.linspace(bond["init{}".format(xAxis)], bond["final{}".format(xAxis)], pointsPerBond) + translate[["X","Y","Z"].index(xAxis)],
                            'y': np.linspace(bond["init{}".format(yAxis)], bond["final{}".format(yAxis)], pointsPerBond) + translate[["X","Y","Z"].index(yAxis)],
                            'mode': 'markers', 
                            'name': "{}{}-{}{}".format(bond["From Species"], bond["From"], bond["To Species"], bond["To"]), 
                            #'line': {"color": "rgba{}".format(cmap(norm(bond["Bond Length"])) ), "width": 3},
                            'marker': {
                                "size": 3, 
                                "color": [bond[colorColumn]]*pointsPerBond, 
                                "coloraxis": "coloraxis"
                            },
                            "showlegend": False,
                            'hoverinfo': "name",
                            'hovertemplate':'{:.2f} Ang{}'.format(bond["Bond Length"], ". Strain: {:.3f}".format(bond["Strain"]) if self.isStrain else "" ),
                        } for i, bond in self.df.iterrows() ]]

    def _afterGetFigure(self):

        #Add the ticks
        self.figure.layout.yaxis.scaleratio = 1
        self.figure.layout.yaxis.scaleanchor = "x"
        
        colorColumn = "Strain" if self.showStrain else "Bond Length"
        cmap = self.setting("cmap")
        reverse = "_r" in cmap
        cmap = cmap[:-2] if reverse else cmap
        self.figure.update_layout(coloraxis = {
            'colorbar': {
                'title': "Strain" if self.showStrain else "Length (Ang)"
            },
            'colorscale': cmap,
            'reversescale': reverse ,
            "cmin": (self.setting("cmin") or self.df[colorColumn].min()) if self.setting("cmid") == None else None,
            "cmax": (self.setting("cmax") or self.df[colorColumn].max()) if self.setting("cmid") == None else None,
            "cmid": self.setting("cmid"),
        })
        
        self.updateSettings(updateFig = False, xaxis_title = 'X (Ang)', yaxis_title = "Y (Ang)")