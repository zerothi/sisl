import numpy as np
import xarray as xr

import os
import shutil

import sisl
from ...plot import Plot, entry_point
from ...plotutils import run_multiple
from ...input_fields import TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput


class LDOSmap(Plot):
    """
    Generates a heat map with the STS spectra along a path.

    Parameters
    ------------
    %%configurable_settings%%

    """

    _plot_type = "LDOS map"

    _requirements = {
        "siesOut": {
            "files": ["$struct$.DIM", "$struct$.PLD", "*.ion", "$struct$.selected.WFSX"],
            "codes": {

                "denchar": {
                    "reason": "The 'denchar' code is used in this case to generate STS spectra."
                }

            }
        },

    }

    _parameters = (

        RangeSlider(
            key = "Erange", name = "Energy range",
            default = [-2, 4],
            width = "s90%",
            params = {
                "min": -10,
                "max": 10,
                "allowCross": False,
                "step": 0.1,
                "marks": {**{i: str(i) for i in range(-10, 11)}, 0: "Ef", },
            },
            help = "Energy range where the STS spectra are computed."
        ),

        IntegerInput(
            key = "nE", name = "Energy points",
            default = 100,
            params = {
                "min": 1
            },
            help = "The number of energy points that are calculated for each spectra"
        ),

        FloatInput(
            key = "STSEta", name = "Smearing factor (eV)",
            default = 0.05,
            params = {
                "min": 0.01,
                "step": 0.01
            },
            help = """This determines the smearing factor of each STS spectra. You can play with this to modify sensibility in the vertical direction.
                <br> If the smearing value is too high, your map will have a lot of vertical noise"""
        ),

        FloatInput(
            key = "dist_step", name = "Distance step (Ang)",
            default = 0.1,
            params = {
                "min": 0,
                "step": 0.01,
            },
            help = "The step in distance between one point and the next one in the path."
        ),

        ProgramaticInput(
            key = "trajectory", name = "Trajectory",
            default = [],
            help = """You can directly provide a trajectory instead of the corner points.<br>
                    This option has preference over 'points', but can't be used through the GUI.<br>
                    It is useful if you want a non-straight trajectory."""
        ),

        ProgramaticInput(
            key = "widen_func", name = "Widen function",
            default = None,
            help = """You can widen the path with this parameter. 
                    This option has preference over 'widenX', 'widenY' and 'widenZ', but can't be used through the GUI.<br>
                    This must be a function that gets a point of the path and returns a set of points surrounding it (including the point itself).<br>
                    All points of the path must be widened with the same amount of points, otherwise you will get an error."""
        ),

        DropdownInput(
            key = "widen_method", name = "Widen method",
            default = "sum",
            width = "s100% m50% l40%",
            params = {
                "options":  [{"label": "Sum", "value": "sum"}, {"label": "Average", "value": "average"}],
                "isMulti": False,
                "placeholder": "",
                "isClearable": False,
                "isSearchable": True,
            },
            help = "Determines whether values surrounding a point should be summed or averaged"
        ),

        QueriesInput(
            key = "points", name = "Path corners",
            default = [{"x": 0, "y": 0, "z": 0, "atom": None, "active": True}],
            queryForm = [

                *[FloatInput(
                    key = key, name = key.upper(),
                    default = 0,
                    width = "s30%",
                    params = {
                        "step": 0.01
                    }
                ) for key in ("x", "y", "z")],

                DropdownInput(
                    key = "atom", name = "Atom index",
                    default = None,
                    params = {
                        "options":  [],
                        "isMulti": False,
                        "placeholder": "",
                        "isClearable": True,
                        "isSearchable": True,
                    },
                    help = """You can provide an atom index instead of the coordinates<br>
                    If an atom is provided, x, y and z will be interpreted as the supercell indices.<br>
                    That is: atom 23 [x=0,y=0,z=0] is atom 23 in the primary cell, while atom 23 [x=1,y=0,z=0]
                    is the image of atom 23 in the adjacent cell in the direction of x"""
                )
            ],
            help = """Provide the points to generate the path through which STS need to be calculated."""
        ),

        FloatInput(
            key = "cmin", name = "Lower color limit",
            default = 0,
            params = {
                "step": 10*-6
            },
            help = "All points below this value will be displayed as 0."
        ),

        FloatInput(
            key = "cmax", name = "Upper color limit",
            default = 0,
            params = {
                "step": 10*-6
            },
            help = "All points above this value will be displayed as the maximum.<br> Decreasing this value will increase saturation."
        ),

    )

    _layout_defaults = {
        'xaxis_title': "Path coordinate",
        'yaxis_title': "E-Ef (eV)"
    }

    def _getdencharSTSfdf(self, stsPosition, Erange, nE, STSEta):

        return """
            Denchar.PlotSTS .true.
            Denchar.PlotWaveFunctions   .false.
            Denchar.PlotCharge .false.

            %block Denchar.STSposition
                {} {} {}
            %endblock Denchar.STSposition

            Denchar.STSEmin {} eV
            Denchar.STSEmax {} eV
            Denchar.STSEnergyPoints {}
            Denchar.CoorUnits Ang
            Denchar.STSEta {} eV
            """.format(*stsPosition, *(np.array(Erange) + self.fermi), nE, STSEta)

    @entry_point('siesta')
    def _read_siesta_output(self, Erange, nE, STSEta, root_fdf, trajectory, points, dist_step, widen_func):
        """Function that uses denchar to get STSpecra along a path"""

        fdf_sile = self.get_sile(root_fdf)
        root_dir = fdf_sile._directory

        self.geom = fdf_sile.read_geometry(output = True)

        #Find fermi level
        self.fermi = False
        for out_fileName in (self.struct, self.fdf_sile.base_file.replace(".fdf", "")):
            try:
                for line in open(fdf_sile.dir_file(f"{out_fileName}.out")):
                    if "Fermi =" in line:
                        self.fermi = float(line.split()[-1])
                        print("\nFERMI LEVEL FOUND: {} eV\n Energies will be relative to this level (E-Ef)\n".format(self.fermi))
                break
            except FileNotFoundError:
                pass

        if not self.fermi:
            print("\nFERMI LEVEL NOT FOUND IN THE OUTPUT FILE. \nEnergy values will be absolute\n")
            self.fermi = 0

        #Get the path (this also sets some attributes: 'distances', 'pointsByStage', 'totalPoints')
        self._getPath(trajectory, points, dist_step, widen_func)

        #Prepare the array that will store all the spectra
        self.spectra = np.zeros((self.path.shape[0], self.path.shape[1], nE))
        #Other helper arrays
        pathIs = np.linspace(0, self.path.shape[0] - 1, self.path.shape[0])
        Epoints = np.linspace(*(np.array(Erange) + self.fermi), nE)

        #Copy selected WFSX into WFSX if it exists (denchar reads from .WFSX)
        system_label = fdf_sile.get("SystemLabel", default="siesta")
        shutil.copyfile(fdf_sile.dir_file(f"{system_label}.selected.WFSX"),
            fdf_sile.dir_file(f"{system_label}.WFSX"))

        #Get the fdf file and replace include paths so that they work
        with open(root_fdf, "r") as f:
            self.fdfLines = f.readlines()

        for i, line in enumerate(self.fdfLines):
            if "%include" in line and not os.path.isabs(line.split()[-1]):

                self.fdfLines[i] = "%include {}\n".format(os.path.join("../", line.split()[-1]))

        #Denchar needs to be run from the directory where everything is stored
        cwd = os.getcwd()
        os.chdir(root_dir)

        #Inform that the WFSX file is used so that changes in it can be followed
        self.follow(fdf_sile.dir_file(f"{system_label}.WFSX"))

        def getSpectraForPath(argsTuple):

            path, nE, iPath, root_dir, struct, STSflags, args, kwargs = argsTuple

            #Generate a temporal directory so that we don't interfere with the other processes
            tempDir = "{}tempSTS".format(iPath)

            os.makedirs(tempDir, exist_ok = True)
            os.chdir(tempDir)

            tempFdf = os.path.join('{}STS.fdf'.format(struct))
            outputFile = os.path.join('{}.STS'.format(struct))

            #Link all the needed files to this directory
            os.system("ln -s ../*fdf ../*out ../*ion* ../*WFSX ../*DIM ../*PLD . ")

            spectra = []; failedPoints = 0

            for i, point in enumerate(path):

                #Write the fdf
                with open(tempFdf, "w") as fh:
                    fh.writelines(kwargs["fdfLines"])
                    fh.write(STSflags[i])

                #Do the STS calculation for the point
                os.system("denchar < {} > /dev/null".format(tempFdf))

                if i%100 == 0 and i != 0:
                    print("PATH {}. Points calculated: {}".format(int(iPath), i))

                #Retrieve and save the output appropiately
                try:
                    spectrum = np.loadtxt(outputFile)

                    spectra.append(spectrum[:, 1])
                except Exception as e:

                    print("Error calculating the spectra for point {}: \n{}".format(point, e))
                    failedPoints += 1
                    #If any spectrum was read, just fill it with zeros
                    spectra.append(np.zeros(nE))

            if failedPoints:
                print("Path {} finished with {} error{} ({}/{} points succesfully calculated)".format(int(iPath), failedPoints, "s" if failedPoints > 1 else "", len(path) - failedPoints, len(path)))

            os.chdir("..")
            shutil.rmtree(tempDir, ignore_errors=True)

            return spectra

        self.spectra = run_multiple(
            getSpectraForPath,
            self.path,
            nE,
            pathIs,
            root_dir, self.struct,
            #All the strings that need to be added to each file
            [[self._getdencharSTSfdf(point, Erange, nE, STSEta) for point in points] for points in self.path],
            kwargsList = {"root_fdf": root_fdf, "fdfLines": self.fdfLines},
            messageFn = lambda nTasks, nodes: "Calculating {} simultaneous paths in {} nodes".format(nTasks, nodes),
            serial = self.isChildPlot
        )

        self.spectra = np.array(self.spectra)

        #WITH XARRAY
        self.xarr = xr.DataArray(
            name = "LDOSmap",
            data = self.spectra,
            dims = ["iPath", "x", "E"],
            coords = [pathIs, list(range(self.path.shape[1])), Epoints]
        )

        os.chdir(cwd)

        #Update the values for the limits so that they are automatically set
        self.update_settings(run_updates = False, cmin = 0, cmax = 0)

    def _getPath(self, trajectory, points, dist_step, widen_func):

        if list(trajectory):
            #If the user provides a trajectory, we are going to use that without questioning it
            self.path = np.array(trajectory)

            #At the moment these make little sense, but in the future there will be the possibility to add breakpoints
            self.pointsByStage = np.array([len(self.path)])
            self.distances = np.array([np.linalg.norm(self.path[-1] - self.path[0])])
        else:
            #Otherwise, we will calculate the trajectory according to the points provided
            points = []
            for reqPoint in points:

                if reqPoint.get("atom"):
                    translate = np.array([reqPoint.get("x", 0), reqPoint.get("y", 0), reqPoint.get("z", 0)]).dot(self.geom.cell)
                    points.append(self.geom[reqPoint["atom"]] + translate)
                else:
                    points.append([reqPoint["x"], reqPoint["y"], reqPoint["z"]])
            points = np.array(points)

            nCorners = len(points)
            if nCorners < 2:
                raise ValueError("You need more than 1 point to generate a path! You better provide 2 next time...\n")

            #Generate an evenly distributed path along the points provided
            self.path = []
            #This array will store the number of points that each stage has
            self.pointsByStage = np.zeros(nCorners - 1)
            self.distances = np.zeros(nCorners - 1)

            for i, point in enumerate(points[1:]):

                prevPoint = points[i]

                self.distances[i] = np.linalg.norm(point - prevPoint)
                nSteps = int(round(self.distances[i]/dist_step)) + 1

                #Add the trajectory from the previous point to this one to the path
                self.path = [*self.path, *np.linspace(prevPoint, point, nSteps)]

                self.pointsByStage[i] = nSteps

            self.path = np.array(self.path)

        #Then, let's widen the path if the user wants to do it (check also points that surround the path)
        if callable(widen_func):
            self.path = widen_func(self.path)
        else:
            #This is just to normalize path
            self.path = np.expand_dims(self.path, 0)

        #Store the total number of points of the path
        self.nPathPoints = self.path.shape[1]
        self.totalPoints = self.path.shape[0] * self.path.shape[1]
        self.iCorners = self.pointsByStage.cumsum()

    def _set_data(self, widen_method, cmin, cmax, Erange, nE):

        #With xarray
        if widen_method == "sum":
            spectraToPlot = self.xarr.sum(dim = "iPath")
        elif widen_method == "average":
            spectraToPlot = self.xarr.mean(dim = "iPath")

        self.data = [{
            'type': 'heatmap',
            'z': spectraToPlot.transpose("E", "x").values,
            #These limits determine the contrast of the image
            'zmin': cmin,
            'zmax': cmax,
            #Yaxis is the energy axis
            'y': np.linspace(*Erange, nE)}]
