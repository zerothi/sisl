# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from ..plot import Plot


class OrbitalGroupsPlot(Plot):
    """Contains methods to manipulate an input accepting groups of orbitals.

    Plots that need this functionality should inherit from this class.
    """

    _orbital_manager_key: str = "orbital_manager"
    _orbital_groups_input_key: str = "groups"

    def _matches_group(self, group, query, iReq=None):
        """Checks if a query matches a group."""
        if isinstance(query, (int, str)):
            query = [query]

        if len(query) == 0:
            return True

        return ("name" in group and group.get("name") in query) or iReq in query

    def groups(self, *i_or_names):
        """Gets the groups that match your query

        Parameters
        ----------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `remove_group(*mylist)`
            to spread it and use all items in your list as args.

            If no query is provided, all the groups will be matched
        """
        return [
            req
            for i, req in enumerate(self.get_input(self._orbital_groups_input_key))
            if self._matches_group(req, i_or_names, i)
        ]

    def add_group(self, group={}, clean=False, **kwargs):
        """Adds a new orbitals group.

        The new group can be passed as a dict or as keyword arguments.
        The keyword arguments will overwrite what has been passed as a dict if there is conflict.

        Parameters
        ---------
        group: dict, optional
            the new group as a dictionary
        clean: boolean, optional
            whether the plot should be cleaned before drawing the group.
            If `False`, the group will be drawn on top of what is already there.
        **kwargs:
            parameters of the group can be passed as keyword arguments too.
            They will overwrite the values in req
        """
        group = {**group, **kwargs}

        groups = (
            [group]
            if clean
            else [*self.get_input(self._orbital_groups_input_key), group]
        )
        return self.update_inputs(**{self._orbital_groups_input_key: groups})

    def remove_groups(self, *i_or_names, all=False):
        """Removes orbital groups.

        Parameters
        ------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `remove_groups(*mylist)`
            to spread it and use all items in your list as args

            If no query is provided, all the groups will be matched
        """
        if all:
            groups = []
        else:
            groups = [
                req
                for i, req in enumerate(self.get_input(self._orbital_groups_input_key))
                if not self._matches_group(req, i_or_names, i)
            ]

        return self.update_inputs(**{self._orbital_groups_input_key: groups})

    def update_groups(self, *i_or_names, **kwargs):
        """Updates existing groups.

        Parameters
        -------
        i_or_names: str or int
            a string (to match the name) or an integer (to match the index)
            this will be used to find the group that you need to update.

            Note that if you have a list of them you can go like `update_groups(*mylist)`
            to spread it and use all items in your list as args

            If no query is provided, all the groups will be matched
        **kwargs:
            keyword arguments containing the values that you want to update

        """
        # We create a new list, otherwise we would be modifying the current one (not good)
        groups = list(self.get_input(self._orbital_groups_input_key))
        for i, group in enumerate(groups):
            if self._matches_group(group, i_or_names, i):
                groups[i] = {**group, **kwargs}

        return self.update_inputs(**{self._orbital_groups_input_key: groups})

    def split_groups(
        self,
        *i_or_names,
        on="species",
        only=None,
        exclude=None,
        remove=True,
        clean=False,
        ignore_constraints=False,
        **kwargs,
    ):
        """Splits the orbital groups into multiple groups.

        Parameters
        --------
        *i_or_names: str, int
            a string (to match the name) or an integer (to match the index),
            You can pass as many as you want.

            Note that if you have a list of them you can go like `split_groups(*mylist)`
            to spread it and use all items in your list as args

            If no query is provided, all the groups will be matched
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
            whether the splitted groups should be removed.
        clean: boolean, optional
            whether the plot should be cleaned before drawing.
            If False, all the groups that come from the method will
            be drawn on top of what is already there.
        ignore_constraints: boolean or array-like, optional
            determines whether constraints (imposed by the group to be splitted)
            on the parameters that we want to split along should be taken into consideration.

            If `False`: all constraints considered.
            If `True`: no constraints considered.
            If array-like: parameters contained in the list ignore their constraints.
        **kwargs:
            keyword arguments that go directly to each group.

            This is useful to add extra filters. For example:
            If you had a group called "C":
            `plot.split_group("C", on="orbitals", spin=[0])`
            will split the PDOS on the different orbitals but will take
            only the contributions from spin up.

        Examples
        -----------

        >>> # Split groups 0 and 1 along n and l
        >>> plot.split_groups(0, 1, on="n+l")
        >>> # The same, but this time even if groups 0 or 1 had defined values for "l"
        >>> # just ignore them and use all possible values for l.
        >>> plot.split_groups(0, 1, on="n+l", ignore_constraints=["l"])
        """
        queries_manager = getattr(self.nodes, self._orbital_manager_key).get()

        old_groups = self.get_input(self._orbital_groups_input_key)

        if len(i_or_names) == 0:
            groups = queries_manager.generate_queries(
                split=on, only=only, exclude=exclude, **kwargs
            )
        else:
            reqs = self.groups(*i_or_names)

            groups = []
            for req in reqs:
                new_groups = queries_manager._split_query(
                    req,
                    on=on,
                    only=only,
                    exclude=exclude,
                    ignore_constraints=ignore_constraints,
                    **kwargs,
                )

                groups.extend(new_groups)

            if remove:
                old_groups = [
                    req
                    for i, req in enumerate(old_groups)
                    if not self._matches_group(req, i_or_names, i)
                ]

        if not clean:
            groups = [*old_groups, *groups]

        return self.update_inputs(**{self._orbital_groups_input_key: groups})

    def split_orbs(self, on="species", only=None, exclude=None, clean=True, **kwargs):
        """
        Splits the orbitals into different groups.

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
            `plot.split_orbs(on="orbitals", species=["C"])`
            will split on the different orbitals but will take
            only those that belong to carbon atoms.
        """
        return self.split_groups(
            on=on, only=only, exclude=exclude, clean=clean, **kwargs
        )
