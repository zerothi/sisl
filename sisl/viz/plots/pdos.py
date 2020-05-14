import numpy as np
import pandas as pd

import os
from collections import defaultdict

import sisl
from ..plot import Plot
from ..plotutils import find_files
from ..input_fields import TextInput, FilePathInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeInput, RangeSlider, QueriesInput, ProgramaticInput, Array1dInput, ListInput
from ..input_fields.range import ErangeInput

class PdosPlot(Plot):

    '''
    Plot representation of the projected density of states.
    '''

    #Define all the class attributes
    _plot_type = "PDOS"

    _requirements = {
        "siesOut": {
            "files": ["$struct$.PDOS"]
        }
    }

    _param_groups = (

        {
            "key": "Hparams",
            "name": "Hamiltonian related",
            "icon": "apps",
            "description": "This parameters are meaningful only if you are calculating the PDOS from a Hamiltonian"
        },

    )

    _parameters = (
        
        FilePathInput(
            key = "pdos_file", name = "Path to PDOS file",
            width = "s100% m50% l33%",
            group="dataread",
            params = {
                "placeholder": "Write the path to your PDOS file here...",
            },
            help = '''This parameter explicitly sets a .PDOS file. Otherwise, the PDOS file is attempted to read from the fdf file '''
        ),

        ErangeInput(
            key="Erange",
            default=[-2,2],
            help = "Energy range where PDOS is displayed."
        ),

        Array1dInput(key="kgrid", name="Monkhorst-Pack grid",
            default=None,
            group="Hparams",
            params={
                "shape": (3,)
            },
            help='''The number of kpoints in each reciprocal direction. 
            A Monkhorst-Pack grid will be generated to calculate the PDOS.
            If not provided, it will be set to 3 for the periodic directions
            and 1 for the non-periodic ones.'''
        ),

        Array1dInput(key="kgrid_displ", name="Monkhorst-Pack grid displacement",
            default=[0,0,0],
            group="Hparams",
            help='''Displacement of the Monkhorst-Pack grid'''
        ),
        
        QueriesInput(
            key = "requests", name = "PDOS queries",
            default = [{"active": True, "name": "DOS", "species": None, "atoms": None, "orbitals": None, "spin": None, "normalize": False, "color": "black", "linewidth": 1}],
            help = '''Here you can ask for the specific PDOS that you need. 
                    <br>TIP: Queries can be activated and deactivated.''',
            queryForm = [

                TextInput(
                    key = "name", name = "Name",
                    default = "DOS",
                    width = "s100% m50% l20%",
                    params = {
                        "placeholder": "Name of the line..."
                    },
                ),

                DropdownInput(
                    key = "species", name = "Species",
                    default = None,
                    width = "s100% m50% l40%",
                    params = {
                        "options":  [],
                        "isMulti": True,
                        "placeholder": "",
                        "isClearable": True,
                        "isSearchable": True,  
                    },
                ),

                DropdownInput(
                    key = "atoms", name = "Atoms",
                    default = None,
                    width = "s100% m50% l40%",
                    params = {
                        "options":  [],
                        "isMulti": True,
                        "placeholder": "",
                        "isClearable": True,
                        "isSearchable": True,  
                    },
                ),

                DropdownInput(
                    key = "orbitals", name = "Orbitals",
                    default = None,
                    width = "s100% m50% l50%",
                    params = {
                        "options":  [],
                        "isMulti": True,
                        "placeholder": "",
                        "isClearable": True,
                        "isSearchable": True,  
                    },
                ),

                DropdownInput(
                    key = "spin", name = "Spin",
                    default = None,
                    width = "s100% m50% l25%",
                    params = {
                        "options":  [],
                        "isMulti": False,
                        "placeholder": "",
                        "isClearable": True, 
                    },
                    style = {
                        "width": 200
                    }
                ),

                SwitchInput(
                    key = "normalize", name = "Normalize",
                    default = False,
                    params = {
                        "offLabel": "No",
                        "onLabel": "Yes"
                    }
                ),

                ColorPicker(
                    key = "color", name = "Line color",
                    default = None,
                ),

                FloatInput(
                    key = "linewidth", name = "Line width",
                    default = 1,
                )
            ]
        ),

        DropdownInput(
            key="add_customdata", name="Add customdata",
            default=[],
            params={
                'options': [
                    {"label": key, "value": key}
                    for key in ("iAtom", "Species", "Orbital name", "Spin")
                ],
                'isMulti': True,
                'isClearable': True,
                'isSearchable': True
            },
            help='''Which info about the provenance of each trace should be added in their customdata attribute.
            This is good for post-processing the plot (grouping, filtering...), but it can make the memory requirements
            significantly larger, specially for large systems'''
        )

    )

    _layout_defaults = {
        'xaxis_title': 'Density of states (1/eV)',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'yaxis_title': 'Energy (eV)'
    }

    _shortcuts = {

    }
    
    @classmethod
    def _default_animation(self, wdir = None, frameNames = None, **kwargs):
        
        pdos_files = find_files(wdir, "*.PDOS", sort = True)

        def _getFrameNames(self):

            return [os.path.basename( childPlot.setting("pdos_file")) for childPlot in self.childPlots]

        return PdosPlot.animated("pdos_file", pdos_files, frameNames = _getFrameNames, wdir = wdir, **kwargs)
    
    def _after_init(self):

        self._add_shortcuts()
    
    def _add_shortcuts(self):

        self.add_shortcut(
            "o", "Split on orbitals",
            self.split_DOS, on="orbitals",
            _description="Split the total DOS along the different orbitals"
        )

        self.add_shortcut(
            "s", "Split on species",
            self.split_DOS, on="species",
            _description="Split the total DOS along the different species"
        )
        
        self.add_shortcut(
            "a", "Split on atoms",
            self.split_DOS, on="atoms",
            _description="Split the total DOS along the different atoms"
        )
        
        self.add_shortcut(
            "p", "Split on spin",
            self.split_DOS, on="spin",
            _description="Split the total DOS along the different spin"
        )

    def _read_from_H(self):

        if not hasattr(self, "H"):
            self.setup_hamiltonian()

        # Get the kgrid, or 
        kgrid = self.setting('kgrid')
        if kgrid is None:
            kgrid = [ 3 if nsc > 1 else 1 for nsc in self.H.geom.nsc]
        kgrid_displ = self.setting('kgrid_displ')

        Erange = self.setting("Erange")

        if Erange is None:
            raise Exception('You need to provide an energy range to calculate the PDOS from the Hamiltonian')

        self.E = np.linspace( Erange[0], Erange[-1], 1000) 

        self.mp = sisl.MonkhorstPack(self.H, kgrid)
        self.PDOSinfo = self.mp.apply.average.PDOS(self.E, eta=True)

    def _read_siesta_output(self):

        pdos_file = self.setting("pdos_file") or self.requiredFiles[0]
        #Get the info from the .PDOS file
        self.geom, self.E, self.PDOSinfo = self.get_sile(pdos_file).read_data()

        self.fermi = 0

    def _after_read(self):

        '''

        Gets the information out of the .pdos and processes it into self.PDOSdicts so that it can be accessed by 
        the self.setData() method once the orbitals/atoms to display PDOS are selected.

        The method stores all the information in a pandas dataframe that looks like this:

             |  species  |     1s     |    2s      |
        _____|___________|____________|____________|......
             |           |            |            |
        iAt  |  string   | PDOS array | PDOS array |
        _____|___________|____________|____________|......
             .           .            .            .
             .           .            .            .

        Returns
        ---------
        self.df:
            The dataframe obtained
        self.E:
            The energy values where PDOS is calculated

        '''

        #Get the orbital where each atom starts
        orbitals = self.geom.orbitals.cumsum()[:-1]
        
        #Normalize self.PDOSinfo to do the same treatment for both spin-polarized and spinless simulations
        self.isSpinPolarized = len(self.PDOSinfo.shape) == 3

        #Initialize the dataframe to store all the info
        self.df = pd.DataFrame()

        #Normalize the PDOSinfo array
        self.PDOSinfo = [self.PDOSinfo] if not self.isSpinPolarized else self.PDOSinfo 

        #Loop over all spin components
        for iSpin, spinComponentPDOS in enumerate(self.PDOSinfo):

            #Initialize the dictionary with all the properties of the orbital
            orbProperties = defaultdict(list)     

            #Loop over all orbitals of the basis
            for iAt, iOrb in self.geom.iter_orbitals():

                atom = self.geom.atoms[iAt]
                orb = atom[iOrb]

                orbProperties["iAtom"].append(iAt + 1)
                orbProperties["Species"].append(atom.symbol)
                orbProperties["Atom Z"].append(atom.Z)
                orbProperties["Spin"].append(iSpin)
                orbProperties["Orbital name"].append(orb.name())
                orbProperties["Z shell"].append(getattr(orb, "Z", 1))
                orbProperties["n"].append(getattr(orb, "n", None))
                orbProperties["l"].append(getattr(orb, "l", None))
                orbProperties["m"].append(getattr(orb, "m", None))
                orbProperties["Polarized"].append(getattr(orb, "P", None))
                orbProperties["Initial charge"].append(getattr(orb, "q0", None))
            
            #Append this part of the dataframe (a full spin component)
            self.df = self.df.append( pd.concat([pd.DataFrame(orbProperties), pd.DataFrame(spinComponentPDOS, columns = self.E)], axis=1, sort = False), ignore_index = True)
        
        #"Inform" the queries of the available options
        #First define the function that will modify all the fields of the query form
        def modifier(requestsInput):

            options = {
                "atoms": [{ "label": "{} ({})".format(iAt, self.geom.atoms[iAt - 1].symbol), "value": iAt } 
                    for iAt in self.df["iAtom"].unique()],
                "species": [{ "label": spec, "value": spec } for spec in self.df.Species.unique()],
                "orbitals": [{ "label": orbName, "value": orbName } for orbName in self.df["Orbital name"].unique()],
                "spin": [{ "label": "↑", "value": 0 },{ "label": "↓", "value": 1 }] if self.isSpinPolarized else []
            }

            for key, val in options.items():
                requestsInput.modify_query_param(key, "inputField.params.options", val)

        #And then apply it
        self.modify_param("requests", modifier)

    def _set_data(self):
        '''

        Uses the information processed by the self.read_data() method and converts it into a data object for plotly.

        It stores the data under self.data, so that it can be accessed by posterior methods.

        Arguments
        ---------
        requests: list of [ list of (int, str and dict) ]
            contains all the user's requests for the PDOS display.

            The contributions of all the requests under a request group [ ] will be summed and displayed together.
        
        normalize: bool
            whether the contribution is normalized by the number of atoms.

        Returns
        ---------
        self.data: list of dicts
            contains a dictionary for each bandStruct with all its information.

        '''

        #Get only the energies we are interested in 
        Emin, Emax = self.setting("Erange") or [min(self.E), max(self.E)]
        plotEvals = [Evalue for Evalue in self.E if Emin < Evalue < Emax]

        self.figure.layout.yaxis.range = [Emin, Emax]

        #Inform and abort if there is no data
        if len(plotEvals) == 0:
            print("PDOS Plot error: There is no data for the provided energy range ({}).\n The energy range of the read data is: [{},{}]"
                .format(self.setting("Erange"), min(self.E), max(self.E))
            )

            return self.data

        #If there is data, get it (drop the columns that we don't want)
        self.req_df = self.df.drop([Evalue for Evalue in self.E if Evalue not in plotEvals], axis = 1)
        requests_param = self.get_param("requests")

        #Go request by request and plot the corresponding PDOS contribution
        for request in self.setting("requests"):

            request = self._new_request(**request)

            #Use only the active requests
            if not request["active"]:
                continue

            req_df = self.req_df.copy()

            req_df = requests_param.filter_df(req_df, request,
                [
                    ("atoms", "iAtom"),
                    ("species", "Species"),
                    ("orbitals", "Orbital name"),
                    ("spin", "Spin")
                ]
            )

            if req_df.empty:
                # print("PDOS Plot warning: No PDOS for the following request: {}".format(request.params))
                # PDOS = []
                continue
            else:
                PDOS = req_df[plotEvals].values

                if request.get("normalize"):
                    PDOS = PDOS.mean(axis = 0)
                else:
                    PDOS = PDOS.sum(axis = 0)   
            
            self.add_trace({
                'type': 'scatter',
                'x': PDOS,
                'y': plotEvals ,
                'mode': 'lines', 
                'name': request["name"], 
                'line': {'width' : request["linewidth"], "color": request["color"]},
                "hoverinfo": "name",
                "customdata": [{ key: req_df[key].unique() for key in self.setting("add_customdata")}]
            })
        
        return self.data

    # ----------------------------------
    #        CONVENIENCE METHODS
    # ----------------------------------

    def _matches_request(self, request, query, iReq=None):
        '''
        Checks if a query matches a PDOS request
        '''

        if isinstance(query, (int, str)):
            query = [query]

        if len(query) == 0:
            return True

        return request["name"] in query or iReq in query
    
    def _new_request(self, **kwargs):

        complete_req = self.get_param("requests").complete_query

        return complete_req({"name": str(len(self.settings["requests"])), **kwargs})

    def requests(self, *i_or_names):
        '''
        Gets the requests that match your query

        Parameters
        ----------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `remove_request(*mylist)`
            to spread it and use all items in your list as args.

            If no query is provided, all the requests will be matched
        '''

        return [req for i, req in enumerate(self.setting("requests")) if self._matches_request(req, i_or_names, i)]

    def add_request(self, req = {}, clean=False, **kwargs):
        '''
        Adds a new PDOS request. The new request can be passed as a dict or as a list of keyword arguments.
        The keyword arguments will overwrite what has been passed as a dict if there is conflict.

        Parameters
        ---------
        req: dict, optional
            the new request as a dictionary
        clean: boolean, optional
            whether the plot should be cleaned before drawing the request.
            If `False`, the request will be drawn on top of what is already there.
        **kwargs:
            parameters of the request can be passed as keyword arguments too.
            They will overwrite the values in req
        '''

        request = self._new_request(**{**req, **kwargs})

        try:
            requests = [request] if clean else [*self.settings["requests"], request ]
            self.update_settings(requests=requests)
        except Exception as e:
            print("There was a problem with your new request ({}): \n\n {}".format(request, e))
            self.undo_settings()

        return self

    def remove_requests(self, *i_or_names, all=False, update_fig=True):
        '''
        Removes requests from the PDOS plot

        Parameters
        ------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `remove_requests(*mylist)`
            to spread it and use all items in your list as args
            
            If no query is provided, all the requests will be matched
        '''

        if all:
            requests = []
        else:
            requests = [ req for i, req in enumerate(self.setting("requests")) if not self._matches_request(req, i_or_names, i)]
        
        return self.update_settings(run_updates=update_fig, requests=requests)

    def update_requests(self, *i_or_names, **kwargs):
        '''
        Updates an existing request

        Parameters
        -------
        i_or_names: str or int
            a string (to match the name) or an integer (to match the index)
            this will be used to find the request that you need to update.

            Note that if you have a list of them you can go like `update_requests(*mylist)`
            to spread it and use all items in your list as args
            
            If no query is provided, all the requests will be matched
        **kwargs:
            keyword arguments containing the values that you want to update

        '''

        requests = self.setting("requests")
        for i, request in enumerate(requests):
            if self._matches_request(request, i_or_names, i):
                requests[i] = {**requests[i], **kwargs}

        return self.update_settings(requests=requests)

    def merge_requests(self, *i_or_names, remove=True, clean=False, **kwargs):
        '''
        Merge multiple requests into one.

        Parameters
        ------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `merge_requests(*mylist)`
            to spread it and use all items in your list as args

            If no query is provided, all the requests will be matched
        remove: boolean, optional
            whether the merged requests should be removed.
            If False, they will be kept in the plot
        clean: boolean, optional
            whether all requests should be removed before drawing the merged request
        **kwargs:
            keyword arguments that go directly to the new request.
            
            You can use them to set other attributes to the request. For example:
            `plot.merge_requests(on="orbitals", species=["C"])`
            will split the PDOS on the different orbitals but will take
            only those that belong to carbon atoms.
        '''

        keys = ["atoms", "orbitals", "species", "spin"]

        # Merge all the requests (nice tree I built here, isn't it? :) )
        new_request = {key: [] for key in keys}
        for i, request in enumerate(self.setting("requests")):
            if self._matches_request(request, i_or_names, i):
                for key in keys:
                    if request[key] is not None:
                        new_request[key] = [*new_request[key], *request[key]]
        
        # Remove duplicate values for each key 
        # and if it's an empty list set it to None (empty list returns no PDOS)
        for key in keys:
            new_request[key] = list(set(new_request[key])) or None
        
        # Remove the merged requests if desired
        if remove:
            self.remove_requests(*i_or_names, update_fig=False)
        
        return self.add_request(**new_request, **kwargs, clean=clean)

    def split_requests(self, *i_or_names, on="species", only=None, exclude=None, remove=True, clean=False, **kwargs):
        '''
        Splits the desired requests into multiple requests

        Parameters
        --------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `split_requests(*mylist)`
            to spread it and use all items in your list as args

            If no query is provided, all the requests will be matched
        on: str, {"species", "atoms", "orbitals", "spin"}
            the parameter to split along
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values of the splitting that should not be plotted
        remove:
            whether the splitted requests should be removed.
        clean: boolean, optional
            whether the plot should be cleaned before drawing.
            If False, all the requests that come from the method will
            be drawn on top of what is already there.
        **kwargs:
            keyword arguments that go directly to each request.
            
            This is useful to add extra filters. For example:
            If you had a request called "C":
            `plot.split_request("C", on="orbitals", spin=[0])`
            will split the PDOS on the different orbitals but will take
            only the contributions from spin up.
        '''

        if exclude is None:
            exclude = []
        
        reqs = self.requests(*i_or_names)

        requests = []
        for req in reqs:
            values = req[on]

            #If it's none, it means that is getting all the possible values
            if values is None:
                options = self.get_param("requests", justDict=False).get_param(on, justDict=False)["inputField.params.options"]
                values = [option["value"] for option in options]

            requests = [*requests, *[
                self._new_request(**{**req, on: [value], "name": f'{req["name"]}, {value}', **kwargs})
                for value in values if value not in exclude and (only is None or value in only)
            ]]

        if remove:
            self.remove_requests(*i_or_names, update_fig=False)

        if not clean:
            requests = [ *self.setting("requests"), *requests]

        return self.update_settings(requests=requests)

    def split_DOS(self, on="species", only=None, exclude=None, clean=True, **kwargs):
        '''
        Splits the density of states to the different contributions.

        Parameters
        --------
        on: str, {"species", "atoms", "orbitals", "spin"}
            the parameter to split along
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values that should not be plotted
        clean: boolean, optional
            whether the plot should be cleaned before drawing.
            If False, all the requests that come from the method will
            be drawn on top of what is already there.
        **kwargs:
            keyword arguments that go directly to each request.
            
            This is useful to add extra filters. For example:
            `plot.split_DOS(on="orbitals", species=["C"])`
            will split the PDOS on the different orbitals but will take
            only those that belong to carbon atoms.
        '''

        if exclude is None:
            exclude = []

        # First, we get all available values for the parameter we want to split
        options = self.get_param("requests", justDict=False).get_param(on, justDict=False)["inputField.params.options"]

        # If the parameter is spin but the PDOS is not polarized we will not be providing
        # options to the user, but in fact there is one option: 0
        if on == "spin" and len(options) == 0:
            options = [{"label": 0, "value": 0}]
        
        # Build all the requests that will be passed to the settings of the plot
        requests = [
            self._new_request(**{on: [option["value"]], "name": option["label"], **kwargs})
            for option in options if option["value"] not in exclude and (only is None or option["value"] in only)
        ]

        # If the user doesn't want to clean the plot, we will just add the requests to the existing ones
        if not clean:
            requests = [ *self.setting("requests"), *requests]

        return self.update_settings(requests=requests)
