import numpy as np

import sisl
from ..plot import Plot, entry_point
from ..plotutils import find_files, random_color
from ..input_fields import (
    TextInput, SileInput, SwitchInput, ColorPicker, DropdownInput, CreatableDropdown,
    IntegerInput, FloatInput, RangeInput, RangeSlider, OrbitalQueries,
    ProgramaticInput, Array1DInput, ListInput
)
from ..input_fields.range import ErangeInput


class PdosPlot(Plot):
    """
    Plot representation of the projected density of states.

    Parameters
    -------------
    pdos_file: pdosSileSiesta, optional
        This parameter explicitly sets a .PDOS file. Otherwise, the PDOS file
        is attempted to read from the fdf file
    tbt_nc: tbtncSileTBtrans, optional
        This parameter explicitly sets a .TBT.nc file. Otherwise, the PDOS
        file is attempted to read from the fdf file
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
        TIP: Queries can be activated and deactivated.   Each item is a
        dict. Structure of the expected dicts:{         'name':
        'species':          'atoms':          'orbitals':          'spin':
        'normalize':          'color':          'linewidth':          'dash':
        'split_on':          'scale': The final DOS will be multiplied by
        this number. }
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    #Define all the class attributes
    _plot_type = "PDOS"

    _param_groups = (

        {
            "key": "Hparams",
            "name": "Hamiltonian related",
            "icon": "apps",
            "description": "This parameters are meaningful only if you are calculating the PDOS from a Hamiltonian"
        },

    )

    _parameters = (

        SileInput(
            key = "pdos_file", name = "Path to PDOS file",
            dtype=sisl.io.siesta.pdosSileSiesta,
            width = "s100% m50% l33%",
            group="dataread",
            params = {
                "placeholder": "Write the path to your PDOS file here...",
            },
            help = """This parameter explicitly sets a .PDOS file. Otherwise, the PDOS file is attempted to read from the fdf file """
        ),

        SileInput(
            key = "tbt_nc", name = "Path to the TBT.nc file",
            dtype=sisl.io.tbtrans.tbtncSileTBtrans,
            width = "s100% m50% l33%",
            group="dataread",
            params = {
                "placeholder": "Write the path to your TBT.nc file here...",
            },
            help = """This parameter explicitly sets a .TBT.nc file. Otherwise, the PDOS file is attempted to read from the fdf file """
        ),

        ErangeInput(
            key="Erange",
            default=[-2, 2],
            help = "Energy range where PDOS is displayed."
        ),

        IntegerInput(
            key="nE", name="Number of energy points",
            group="Hparams",
            default=100,
            help="""If calculating the PDOS from a hamiltonian, the number of energy points used"""
        ),

        Array1DInput(key="kgrid", name="Monkhorst-Pack grid",
            default=None,
            group="Hparams",
            params={
                "shape": (3,)
            },
            help="""The number of kpoints in each reciprocal direction. 
            A Monkhorst-Pack grid will be generated to calculate the PDOS.
            If not provided, it will be set to 3 for the periodic directions
            and 1 for the non-periodic ones."""
        ),

        Array1DInput(key="kgrid_displ", name="Monkhorst-Pack grid displacement",
            default=[0, 0, 0],
            group="Hparams",
            help="""Displacement of the Monkhorst-Pack grid"""
        ),

        FloatInput(key="E0", name="Reference energy",
            default=0,
            help="""The energy to which all energies will be referenced (including Erange)."""
        ),

        OrbitalQueries(
            key = "requests", name = "PDOS queries",
            default = [{"active": True, "name": "DOS", "species": None, "atoms": None, "orbitals": None, "spin": None, "normalize": False, "color": "black", "linewidth": 1}],
            help = """Here you can ask for the specific PDOS that you need. 
                    <br>TIP: Queries can be activated and deactivated.""",
            queryForm = [

                TextInput(
                    key="name", name="Name",
                    default="DOS",
                    width="s100% m50% l20%",
                    params={
                        "placeholder": "Name of the line..."
                    },
                ),

                'species', 'atoms', 'orbitals', 'spin',

                SwitchInput(
                    key="normalize", name="Normalize",
                    default=False,
                    params={
                        "offLabel": "No",
                        "onLabel": "Yes"
                    }
                ),

                ColorPicker(
                    key="color", name="Line color",
                    default=None,
                ),

                FloatInput(
                    key="linewidth", name="Line width",
                    default=1,
                ),

                DropdownInput(
                    key="dash", name="Line style",
                    default="solid",
                    params={
                        "isMulti": False,
                        "isClearable": False,
                        "isSearchable": True,
                        "options": [{"value": option, "label": option} for option in ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")]
                    }
                ),

                DropdownInput(
                    key="split_on", name="Split",
                    default=None,
                    params={
                        "isMulti": True,
                        "isSearchable": True,
                        "options": [{"value": option, "label": option} for option in ("species", "atoms", "Z", "orbitals", "spin", "n", "l", "m", "zeta")]
                    }
                ),

                FloatInput(
                    key="scale", name="Scale",
                    default=1,
                    params={"min": None},
                    help="The final DOS will be multiplied by this number."
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
        #     help="""Which info about the provenance of each trace should be added in their customdata attribute.
        #     This is good for post-processing the plot (grouping, filtering...), but it can make the memory requirements
        #     significantly larger, specially for large systems"""
        # )

    )

    _layout_defaults = {
        'xaxis_title': 'Density of states [1/eV]',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'yaxis_title': 'Energy [eV]',
        'showlegend': True
    }

    _shortcuts = {

    }

    @classmethod
    def _default_animation(self, wdir = None, frame_names = None, **kwargs):

        pdos_files = find_files(wdir, "*.PDOS.xml", sort = True)

        if not pdos_files:
            pdos_files = find_files(wdir, "*.PDOS", sort = True)

        def _get_frame_names(self):

            return [child_plot.get_setting("pdos_file").name for child_plot in self.child_plots]

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

    @entry_point('siesta output')
    def _read_siesta_output(self, pdos_file):
        """
        Reads the pdos from a SIESTA .PDOS file.
        """
        #Get the info from the .PDOS file
        self.geometry, self.E, self.PDOS = self.get_sile(pdos_file or "pdos_file").read_data()

    @entry_point("TB trans")
    def _read_TBtrans(self, root_fdf, tbt_nc):
        """
        Reads the PDOS from a *.TBT.nc file coming from a TBtrans run.
        """
        #Get the info from the .PDOS file
        tbt_sile = self.get_sile("tbt_nc")
        self.PDOS = tbt_sile.DOS(sum=False).data.T
        self.E = tbt_sile.E

        self.geometry = sisl.get_sile(root_fdf).read_geometry().sub(tbt_sile.a_dev)

    @entry_point('hamiltonian')
    def _read_from_H(self, kgrid, kgrid_displ, Erange, nE, E0):
        """
        Calculates the PDOS from a sisl Hamiltonian.
        """
        if not hasattr(self, "H"):
            self.setup_hamiltonian()

        # Get the kgrid or generate a default grid by checking the interaction between cells
        # This should probably take into account how big the cell is.
        if kgrid is None:
            kgrid = [3 if nsc > 1 else 1 for nsc in self.H.geometry.nsc]

        if Erange is None:
            raise ValueError('You need to provide an energy range to calculate the PDOS from the Hamiltonian')

        self.E = np.linspace(Erange[0], Erange[-1], nE) + E0

        self.mp = sisl.MonkhorstPack(self.H, kgrid, kgrid_displ)

        # Define the available spins
        spin_indices = [0]
        if self.H.spin.is_polarized:
            spin_indices = [0, 1]

        # Calculate the PDOS for all available spins
        PDOS = []
        for spin in spin_indices:
            PDOS.append(self.mp.apply.average.PDOS(self.E, spin=spin, eta=True))
        self.PDOS = np.array(PDOS)

    def _after_read(self):
        """
        Creates the PDOS dataarray and updates the "requests" input field.
        """
        from xarray import DataArray

        # Check if the PDOS contains spin resolution (there should be three dimensions,
        # and the first one should be the spin components)
        self.spin = sisl.Spin.UNPOLARIZED
        if self.PDOS.squeeze().ndim == 3:
            self.spin = {
                2: sisl.Spin.POLARIZED,
                4: sisl.Spin.NONCOLINEAR
            }[self.PDOS.shape[0]]
        self.spin = sisl.Spin(self.spin)

        # Normalize the PDOS array so that we ensure a first dimension for spin even if
        # there is no spin resolution
        if self.PDOS.ndim == 2:
            self.PDOS = np.expand_dims(self.PDOS, axis=0)

        self.get_param('requests').update_options(self.geometry, self.spin)

        self.PDOS = DataArray(
            self.PDOS,
            coords={
                'spin': self.get_param('requests').get_options("spin"),
                'orb': range(self.PDOS.shape[1]),
                'E': self.E
            },
            dims=('spin', 'orb', 'E')
        )

    def _set_data(self, requests, E0, Erange):

        #Get only the energies we are interested in
        Erange = np.array(Erange)
        if Erange is None:
            Emin, Emax = [min(self.PDOS.E.values), max(self.PDOS.E.values)]
        else:
            Emin, Emax = Erange + E0

        #Get only the part of the arra
        E_PDOS = self.PDOS.where(
            (self.PDOS.E > Emin) & (self.PDOS.E < Emax), drop=True)

        requests_param = self.get_param("requests")

        #Go request by request and plot the corresponding PDOS contribution
        for request in requests:
            self._draw_request(request, E_PDOS, requests_param, E0)

        self.update_layout(yaxis_range=np.array([Emin - E0, Emax - E0]))

        return self.data

    def _draw_request(self, request, E_PDOS, requests_param, E0):
        """
        Draws a given PDOS request.

        This has been made a function so that it can call itself recursively
        to support splitting individual requests.

        Parameters
        --------------
        request: dict
            the request to draw
        E_PDOS: DataArray
            the part of the PDOS dataarray that falls in the energy range that we want to draw.
        requests_param: OrbitalQueries
            the requests input field, so that we don't need to retrieve it each time
        E0: float
            the energy reference that we should use for drawing this request.
        """

        request = self._new_request(**request)

        # If the request has an split_on parameter that is not None,
        # we are going to split the request in place. Note that you can also
        # split requests or the full DOS using the `split_requests` and `split_DOS`
        # methods, but this may be more convenient for the GUI.
        if request["split_on"]:

            # We are going to give a different dash style to each obtained request
            dash_options = requests_param["dash"].options
            n_dash_options = len(dash_options)
            def query_gen(i=[-1], **kwargs):
                i[0] += 1
                return self._new_request(**{**kwargs, "dash": dash_options[i[0] % n_dash_options]})

            # And ensure they all have the same color (if the color is None,
            # each request will show up with a different color)
            request["color"] = request["color"] or random_color()

            # Now, get all the requests that emerge from splitting the "parent" request
            # Note that we need to set split_on to None for the new requests, otherwise the
            # cycle would be infinite
            splitted_request = requests_param._split_query(request, on=request["split_on"], split_on=None, query_gen=query_gen, vary="dash")
            # Now that we have them, draw them
            for req in splitted_request:
                self._draw_request(req, E_PDOS, requests_param, E0)
            # Since we have already drawn all the requests, we don't need to do anything else
            # This would not be true if we wanted to represent the "total request" as well, but we
            # don't give that option yet. Just removing the return would draw the total
            return

        # From now on, the code focuses on actually drawing the request

        # Use only the active requests
        if not request["active"]:
            return

        orb = requests_param.get_orbitals(request)

        if len(orb) == 0:
            # This request does not match any possible orbital
            return

        req_PDOS = E_PDOS.sel(orb=orb)
        if request['spin'] is not None:
            req_PDOS = req_PDOS.sel(spin=request['spin'])

        if request["normalize"]:
            req_PDOS = req_PDOS.mean(["orb", "spin"])
        else:
            req_PDOS = req_PDOS.sum(["orb", "spin"])

        self.add_trace({
            'type': 'scatter',
            'x': req_PDOS.values * request["scale"],
            'y': req_PDOS.E.values - E0,
            'mode': 'lines',
            'name': request["name"],
            'line': {'width': request["linewidth"], "color": request["color"], "dash": request["dash"]},
            "hoverinfo": "name",
        })

    # ----------------------------------
    #        CONVENIENCE METHODS
    # ----------------------------------

    def _matches_request(self, request, query, iReq=None):
        """
        Checks if a query matches a PDOS request
        """
        if isinstance(query, (int, str)):
            query = [query]

        if len(query) == 0:
            return True

        return ("name" in request and request.get("name") in query) or iReq in query

    def _new_request(self, **kwargs):

        complete_req = self.get_param("requests").complete_query

        if "spin" not in kwargs and self.spin.is_noncolinear:
            if "spin" not in kwargs.get("split_on", ""):
                kwargs["spin"] = ["sum"]

        return complete_req({"name": str(len(self.settings["requests"])), **kwargs})

    def requests(self, *i_or_names):
        """
        Gets the requests that match your query

        Parameters
        ----------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `remove_request(*mylist)`
            to spread it and use all items in your list as args.

            If no query is provided, all the requests will be matched
        """
        return [req for i, req in enumerate(self.get_setting("requests")) if self._matches_request(req, i_or_names, i)]

    def add_request(self, req = {}, clean=False, **kwargs):
        """
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
        """
        request = self._new_request(**{**req, **kwargs})

        try:
            requests = [request] if clean else [*self.settings["requests"], request]
            self.update_settings(requests=requests)
        except Exception as e:
            print("There was a problem with your new request ({}): \n\n {}".format(request, e))
            self.undo_settings()

        return self

    def remove_requests(self, *i_or_names, all=False, update_fig=True):
        """
        Removes requests from the PDOS plot

        Parameters
        ------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `remove_requests(*mylist)`
            to spread it and use all items in your list as args

            If no query is provided, all the requests will be matched
        """
        if all:
            requests = []
        else:
            requests = [req for i, req in enumerate(self.get_setting("requests", copy=False)) if not self._matches_request(req, i_or_names, i)]

        return self.update_settings(run_updates=update_fig, requests=requests)

    def update_requests(self, *i_or_names, **kwargs):
        """
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

        """
        # We create a new list, otherwise we would be modifying the current one (not good)
        requests = list(self.get_setting("requests", copy=False))
        for i, request in enumerate(requests):
            if self._matches_request(request, i_or_names, i):
                requests[i] = {**request, **kwargs}

        return self.update_settings(requests=requests)

    def _NOT_WORKING_merge_requests(self, *i_or_names, remove=True, clean=False, **kwargs):
        """
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
        """
        keys = ["atoms", "Z", "orbitals", "species", "spin", "n", "l", "m", "zeta"]

        # Merge all the requests (nice tree I built here, isn't it? :) )
        new_request = {key: [] for key in keys}
        for i, request in enumerate(self.get_setting("requests", copy=False)):
            if self._matches_request(request, i_or_names, i):
                for key in keys:
                    if request.get(key, None) is not None:
                        val = request[key]
                        if key == "atoms":
                            val = self.geometry._sanitize_atoms(val)
                        val = np.atleast_1d(val)
                        new_request[key] = [*new_request[key], *val]

        # Remove duplicate values for each key
        # and if it's an empty list set it to None (empty list returns no PDOS)
        for key in keys:
            new_request[key] = list(set(new_request[key])) or None

        # Remove the merged requests if desired
        if remove:
            self.remove_requests(*i_or_names, update_fig=False)

        return self.add_request(**new_request, **kwargs, clean=clean)

    def split_requests(self, *i_or_names, on="species", only=None, exclude=None, remove=True, clean=False, ignore_constraints=False, **kwargs):
        """
        Splits the desired requests into multiple requests

        Parameters
        --------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `split_requests(*mylist)`
            to spread it and use all items in your list as args

            If no query is provided, all the requests will be matched
        on: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"}, or list of str
            the parameter to split along.

            Note that you can combine parameters with a "+" to split along multiple parameters
            at the same time. You can get the same effect also by passing a list. See examples.
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
        ignore_constraints: boolean or array-like, optional
            determines whether constraints (imposed by the request to be splitted) 
            on the parameters that we want to split along should be taken into consideration.

            If `False`: all constraints considered.
            If `True`: no constraints considered.
            If array-like: parameters contained in the list ignore their constraints.
        **kwargs:
            keyword arguments that go directly to each request.

            This is useful to add extra filters. For example:
            If you had a request called "C":
            `plot.split_request("C", on="orbitals", spin=[0])`
            will split the PDOS on the different orbitals but will take
            only the contributions from spin up.

        Examples
        -----------

        >>> plot = H.plot.pdos(requests=[...])
        >>>
        >>> # Split requests 0 and 1 along n and l
        >>> plot.split_requests(0, 1, on="n+l")
        >>> # The same, but this time even if requests 0 or 1 had defined values for "l"
        >>> # just ignore them and use all possible values for l.
        >>> plot.split_requests(0, 1, on="n+l", ignore_constraints=["l"])
        """
        reqs = self.requests(*i_or_names)

        requests = []
        for req in reqs:

            new_requests = self.get_param("requests")._split_query(
                req, on=on, only=only, exclude=exclude, req_gen=self._new_request,
                ignore_constraints=ignore_constraints, **kwargs
            )

            requests.extend(new_requests)

        if remove:
            self.remove_requests(*i_or_names, update_fig=False)

        if not clean:
            requests = [*self.get_setting("requests", copy=False), *requests]

        return self.update_settings(requests=requests)

    def split_DOS(self, on="species", only=None, exclude=None, clean=True, **kwargs):
        """
        Splits the density of states to the different contributions.

        Parameters
        --------
        on: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"}, or list of str
            the parameter to split along.
            Note that you can combine parameters with a "+" to split along multiple parameters
            at the same time. You can get the same effect also by passing a list.
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

        Examples
        -----------

        >>> plot = H.plot.pdos()
        >>>
        >>> # Split the DOS in n and l but show only the DOS from Au
        >>> # Also use "Au $ns" as a template for the name, where $n will
        >>> # be replaced by the value of n.
        >>> plot.split_DOS(on="n+l", species=["Au"], name="Au $ns")
        """
        requests = self.get_param('requests')._generate_queries(
            on=on, only=only, exclude=exclude, query_gen=self._new_request, **kwargs)

        # If the user doesn't want to clean the plot, we will just add the requests to the existing ones
        if not clean:
            requests = [*self.get_setting("requests", copy=False), *requests]

        return self.update_settings(requests=requests)
