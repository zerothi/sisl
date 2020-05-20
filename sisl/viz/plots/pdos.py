import numpy as np
from xarray import DataArray

import os
from collections import defaultdict

import sisl
from ..plot import Plot
from ..plotutils import find_files
from ..input_fields import TextInput, FilePathInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeInput, RangeSlider, OrbitalQueries, ProgramaticInput, Array1dInput, ListInput
from ..input_fields.range import ErangeInput

class PdosPlot(Plot):
    '''
    Plot representation of the projected density of states.

    Parameters
    -------------
    pdos_file: str, optional
        This parameter explicitly sets a .PDOS file. Otherwise, the PDOS file
        is attempted to read from the fdf file
    Erange: array-like of shape (2,), optional
        Energy range where PDOS is displayed.
    nE: int, optional
        If calculating the PDOS from a hamiltonian, the number of energy
        points used
    kgrid: array-like, optional
        The number of kpoints in each reciprocal direction.              A
        Monkhorst-Pack grid will be generated to calculate the PDOS.
        If not provided, it will be set to 3 for the periodic directions
        and 1 for the non-periodic ones.
    kgrid_displ: array-like, optional
        Displacement of the Monkhorst-Pack grid
    E0: float, optional
        The energy to which all energies will be referenced (including
        Erange).
    requests: array-like of dict, optional
        Here you can ask for the specific PDOS that you need.
        TIP: Queries can be activated and deactivated.
    reading_order: None, optional
        Order in which the plot tries to read the data it needs.
    root_fdf: str, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
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

        IntegerInput(
            key="nE", name="Number of energy points",
            group="Hparams",
            default=100,
            help='''If calculating the PDOS from a hamiltonian, the number of energy points used'''
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

        FloatInput(key="E0", name="Reference energy",
            default=0,
            help='''The energy to which all energies will be referenced (including Erange).'''
        ),
        
        OrbitalQueries(
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

                'species', 'atoms', 'orbitals', 'spin',

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

        # DropdownInput(
        #     key="add_customdata", name="Add customdata",
        #     default=[],
        #     params={
        #         'options': [
        #             {"label": key, "value": key}
        #             for key in ("iAtom", "Species", "Orbital name", "Spin")
        #         ],
        #         'isMulti': True,
        #         'isClearable': True,
        #         'isSearchable': True
        #     },
        #     help='''Which info about the provenance of each trace should be added in their customdata attribute.
        #     This is good for post-processing the plot (grouping, filtering...), but it can make the memory requirements
        #     significantly larger, specially for large systems'''
        # )

    )

    _layout_defaults = {
        'xaxis_title': 'Density of states (1/eV)',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'yaxis_title': 'Energy (eV)',
        'showlegend': True
    }

    _shortcuts = {

    }
    
    @classmethod
    def _default_animation(self, wdir = None, frame_names = None, **kwargs):
        
        pdos_files = find_files(wdir, "*.PDOS", sort = True)

        def _get_frame_names(self):

            return [os.path.basename( childPlot.setting("pdos_file")) for childPlot in self.childPlots]

        return PdosPlot.animated("pdos_file", pdos_files, frame_names = _get_frame_names, wdir = wdir, **kwargs)
    
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

        self.E = np.linspace(Erange[0], Erange[-1], self.setting("nE")) + self.setting("E0")

        self.mp = sisl.MonkhorstPack(self.H, kgrid, kgrid_displ)
        self.PDOS = self.mp.apply.average.PDOS(self.E, eta=True)

    def _read_siesta_output(self):

        pdos_file = self.setting("pdos_file") or self.requiredFiles[0]
        #Get the info from the .PDOS file
        self.geom, self.E, self.PDOS = self.get_sile(pdos_file).read_data()

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
        
        #Normalize self.PDOS to do the same treatment for both spin-polarized and spinless simulations
        self.isSpinPolarized = len(self.PDOS.shape) == 3

        #Normalize the PDOS array
        self.PDOS = np.array([self.PDOS]) if not self.isSpinPolarized else self.PDOS 

        self.PDOS = DataArray(
            self.PDOS, 
            coords={
                'spin': [0,1] if self.isSpinPolarized else [0],
                'orb': range(0, self.PDOS.shape[1]),
                'E': self.E 
            },
            dims=('spin', 'orb', 'E')
        )

        self.get_param('requests').update_options(self.geom, self.isSpinPolarized)

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
        E0 = self.setting("E0")
        Erange = np.array(self.setting("Erange"))
        if Erange is None:
            Emin, Emax = [min(self.PDOS.E.values), max(self.PDOS.E.values)]
        else:
            Emin, Emax = Erange + E0 

        #Get only the part of the arra
        E_PDOS = self.PDOS.where(
            (self.PDOS.E > Emin) & (self.PDOS.E < Emax), drop=True)

        requests_param = self.get_param("requests")

        #Go request by request and plot the corresponding PDOS contribution
        for request in self.setting("requests"):

            request = self._new_request(**request)

            #Use only the active requests
            if not request["active"]:
                continue

            orb = requests_param.get_orbitals(request)

            if len(orb) == 0:
                # This request does not match any possible orbital
                continue

            req_PDOS = E_PDOS.sel(orb=orb)
            if request['spin'] is not None:
                E_PDOS.sel(spin=request['spin'])

            if request["normalize"]:
                req_PDOS = req_PDOS.mean("orb").mean('spin')
            else:
                req_PDOS = req_PDOS.sum("orb").sum('spin')

            print(request["linewidth"])

            self.add_trace({
                'type': 'scatter',
                'x': req_PDOS.values,
                'y': req_PDOS.E.values - E0,
                'mode': 'lines', 
                'name': request["name"], 
                'line': {'width' : request["linewidth"], "color": request["color"]},
                "hoverinfo": "name",
            })

            self.update_layout(yaxis_range=np.array([Emin - E0, Emax - E0]))
        
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
                values = self.get_param("requests")[on].options
            

            requests = [*requests, *[
                self._new_request(**{**req, on: [value], "name": f'{req["name"]}, {value}', **kwargs})
                for value in values if value not in exclude and (only is None or value in only)
            ]]

            # Use only those requests that make sense
            requests = [req for req in requests if len(self.get_param("requests").get_orbitals(req)) > 0]

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
        options = self.get_param("requests").get_param(on)["inputField.params.options"]

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
