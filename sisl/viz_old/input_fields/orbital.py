# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections import defaultdict
import numpy as np

from .basic import OptionsInput

from .queries import QueriesInput
from .atoms import AtomSelect, SpeciesSelect
from .spin import SpinSelect


class OrbitalsNameSelect(OptionsInput):

    _default = {
        "default": None,
        "params": {
            "placeholder": "Select orbitals...",
            "options": [],
            "isMulti": True,
            "isClearable": True,
            "isSearchable": True,
        }
    }

    def update_options(self, geom):

        orbs = set([orb.name()
                    for unique_at in geom.atoms.atom for orb in unique_at])

        self.modify("inputField.params.options",
                    [{"label": orb, "value": orb}
                     for orb in orbs])

        return self


class OrbitalQueries(QueriesInput):
    """
    This class implements an input field that allows you to select orbitals by atom, species, etc...
    """

    _fields = {
        "species": {"field": SpeciesSelect, "name": "Species"},
        "atoms": {"field": AtomSelect, "name": "Atoms"},
        "orbitals": {"field": OrbitalsNameSelect, "name": "Orbitals"},
        "spin": {"field": SpinSelect, "name": "Spin"},
    }

    _keys_to_cols = {
        "atoms": "atom",
        "orbitals": "orbital_name",
    }

    def _build_orb_filtering_df(self, geom):
        import pandas as pd

        orb_props = defaultdict(list)
        del_key = set()
        #Loop over all orbitals of the basis
        for at, iorb in geom.iter_orbitals():

            atom = geom.atoms[at]
            orb = atom[iorb]

            orb_props["atom"].append(at)
            orb_props["Z"].append(atom.Z)
            orb_props["species"].append(atom.symbol)
            orb_props["orbital_name"].append(orb.name())

            for key in ("n", "l", "m", "zeta"):
                val = getattr(orb, key, None)
                if val is None:
                    del_key.add(key)
                orb_props[key].append(val)

        for key in del_key:
            del orb_props[key]

        self.orb_filtering_df = pd.DataFrame(orb_props)

    def update_options(self, geometry, spin=""):
        """
        Updates the options of the orbital queries.

        Parameters
        -----------
        geometry: sisl.Geometry
            the geometry that contains the orbitals that can be selected.
        spin: sisl.Spin, str or int
            It is used to indicate the kind of spin so that the spin selector
            (in case there is one) can display the appropiate options.

        See also
        ---------
        sisl.viz.input_fields.dropdown.SpinSelect
        sisl.physics.Spin
        """
        self.geometry = geometry

        for key in ("species", "atoms", "orbitals"):
            try:
                self.get_query_param(key).update_options(geometry)
            except KeyError:
                pass

        try:
            self.get_query_param('spin').update_options(spin)
        except KeyError:
            pass

        self._build_orb_filtering_df(geometry)

    def get_options(self, key, **kwargs):
        """
        Gets the options for a given key or combination of keys.

        Parameters
        ------------
        key: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"}
            the parameter that you want the options for.

            Note that you can combine them with a "+" to get all the possible combinations.
            You can get the same effect also by passing a list.
            See examples.
        **kwargs:
            keyword arguments that add additional conditions to the query. The values of this
            keyword arguments can be lists, in which case it indicates that you want a value
            that is in the list. See examples.

        Returns
        ----------
        np.ndarray of shape (n_options, [n_keys])
            all the possible options.

            If only one key was provided, it is a one dimensional array. 

        Examples
        -----------

        >>> plot = H.plot.pdos()
        >>> plot.get_param("requests").get_options("l", species="Au")
        >>> plot.get_param("requests").get_options("n+l", atoms=[0,1])
        """
        # Get the tadatframe
        df = self.orb_filtering_df

        # Filter the dataframe according to the constraints imposed by the kwargs,
        # if there are any.
        if kwargs:
            if "atoms" in kwargs:
                kwargs["atoms"] = self.geometry._sanitize_atoms(kwargs["atoms"])
            def _repr(v):
                if isinstance(v, np.ndarray):
                    v = list(v.ravel())
                if isinstance(v, dict):
                    raise Exception(str(v))
                return repr(v)
            query = ' & '.join([f'{self._keys_to_cols.get(k, k)}=={_repr(v)}' for k, v in kwargs.items(
            ) if self._keys_to_cols.get(k, k) in df])
            if query:
                df = df.query(query)

        # If + is in key, it is a composite key. In that case we are going to
        # split it into all the keys that are present and get the options for all
        # of them. At the end we are going to return a list of tuples that will be all
        # the possible combinations of the keys.
        keys = [self._keys_to_cols.get(k, k) for k in key.split("+")]

        # Spin values are not stored in the orbital filtering dataframe. If the options
        # for spin are requested, we need to pop the key out and get the current options
        # for spin from the input field
        spin_in_keys = "spin" in keys
        if spin_in_keys:
            spin_key_i = keys.index("spin")
            keys.remove("spin")
            spin_options = self.get_param("spin").options

            # We might have some constraints on what the spin value can be
            if "spin" in kwargs:
                spin_options = set(spin_options).intersection(kwargs["spin"])

        # Now get the unique options from the dataframe
        if keys:
            options = df.drop_duplicates(subset=keys)[
                keys].values.astype(object)
        else:
            # It might be the only key was "spin", then we are going to fake it
            # to get an options array that can be treated in the same way.
            options = np.array([[]], dtype=object)

        # If "spin" was one of the keys, we are going to incorporate the spin options, taking into
        # account the position (column index) where they are expected to be returned.
        if spin_in_keys and len(spin_options) > 0:
            options = np.concatenate(
                [np.insert(options, spin_key_i, spin, axis=1) for spin in spin_options])

        # Squeeze the options array, just in case there is only one key
        # There's a special case: if there is only one option for that key,
        # squeeze converts it to a number, so we need to make sure there is at least 1d
        if options.shape[1] == 1:
            options = options.squeeze()
            options = np.atleast_1d(options)

        return options

    def get_orbitals(self, query):

        if "atoms" in query:
            query["atoms"] = self.geometry._sanitize_atoms(query["atoms"])

        filtered_df = self.filter_df(
            self.orb_filtering_df, query, self._keys_to_cols)

        return filtered_df.index

    def _split_query(self, query, on, only=None, exclude=None, query_gen=None, ignore_constraints=False, **kwargs):
        """
        Splits a query into multiple queries based on one of its parameters.

        Parameters
        --------
        query: dict
            the query that we want to split
        on: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"}, or list of str
            the parameter to split along.
            Note that you can combine parameters with a "+" to split along multiple parameters
            at the same time. You can get the same effect also by passing a list.
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values of the splitting that should not be plotted.
        query_gen: function, optional
            the request generator. It is a function that takes all the parameters for each
            request that this method has come up with and gets a chance to do some modifications.

            This may be useful, for example, to give each request a color, or a custom name.
        ignore_constraints: boolean or array-like, optional
            determines whether constraints (imposed by the query that you want to split) 
            on the parameters that we want to split along should be taken into consideration.

            If `False`: all constraints considered.
            If `True`: no constraints considered.
            If array-like: parameters contained in the list ignore their constraints.
        **kwargs:
            keyword arguments that go directly to each new request.

            This is useful to add extra filters. For example:

            `self._split_query(request, on="orbitals", spin=[0])`
            will split the request on the different orbitals but will take
            only the contributions from spin up.
        """
        if exclude is None:
            exclude = []

        # Divide the splitting request into all the parameters
        if isinstance(on, str):
            on = on.split("+")

        # Get the current values of the parameters that we want to split the request on
        # because these will be our constraints. If a parameter is set to None or not
        # provided, we have no constraints for that parameter.
        constraints = {}
        if ignore_constraints is not True:

            if ignore_constraints is False:
                ignore_constraints = ()

            for key in filter(lambda key: key not in ignore_constraints, on):
                val = query.get(key, None)
                if val is not None:
                    constraints[key] = val

        # Knowing what are our constraints (which may be none), get the available options
        values = self.get_options("+".join(on), **constraints)

        # We are going to make sure that, even if there was only one parameter to split on,
        # the values are two dimensional. In this way, we can take the same actions for the
        # case when there is only one parameter and the case when there are multiple.
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        # If no function to modify queries was provided we are just going to generate a
        # dummy one that just returns the query as it gets it
        if query_gen is None:
            def query_gen(**kwargs):
                return kwargs

        # We ensure that on is a list even if there is only one parameter, for the same
        # reason we ensured values was 2 dimensional
        if isinstance(on, str):
            on = on.split("+")

        # Define the name that we will give to the new queries, using templating
        # If a splitting parameter is not used by the name, we are going to
        # append it, in order to make names unique and self-explanatory.
        base_name = kwargs.pop("name", query.get("name", ""))
        first_added = True
        for key in on:
            kwargs.pop(key, None)

            if f"${key}" not in base_name:
                base_name += f"{' | ' if first_added else ', '}{key}=${key}"
                first_added = False

        # Now build all the queries
        queries = []
        for i, value in enumerate(values):
            if value not in exclude and (only is None or value in only):

                # Use the name template to generate the name for this query
                name = base_name
                for key, val in zip(on, value):
                    name = name.replace(f"${key}", str(val))

                # And append the new query to the queries
                queries.append(
                    query_gen(**{
                        **query,
                        **{key: [val] for key, val in zip(on, value)},
                        "name": name, **kwargs
                    })
                )

        return queries

    def _generate_queries(self, on, only=None, exclude=None, query_gen=None, **kwargs):
        """
        Automatically generates queries based on the current options.

        Parameters
        --------
        on: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"} or list of str
            the parameter to split along.
            Note that you can combine parameters with a "+" to split along multiple parameters
            at the same time. You can get the same effect also by passing a list.
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values that should not be plotted
        query_gen: function, optional
            the request generator. It is a function that takes all the parameters for each
            request that this method has come up with and gets a chance to do some modifications.

            This may be useful, for example, to give each request a color, or a custom name.
        **kwargs:
            keyword arguments that go directly to each request.

            This is useful to add extra filters. For example:
            `plot._generate_requests(on="orbitals", species=["C"])`
            will split the PDOS on the different orbitals but will take
            only those that belong to carbon atoms.
        """
        return self._split_query({}, on=on, only=only, exclude=exclude, query_gen=query_gen, **kwargs)
