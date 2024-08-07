# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
from functools import partial

from sisl._internal import set_module

from ..atom import AtomInput
from ._variable import Variable
from ._yaml_reader import parse_variable, read_yaml

__all__ = ["AtomPseudo"]


_log = logging.getLogger(__name__)
_spdfgh = "spdfgh"


@set_module("sisl_toolbox.siesta.minimizer")
class AtomPseudo(AtomInput):
    def get_variables(self, dict_or_yaml, nodes=()):
        """Convert a dictionary or yaml file input to variables usable by the minimizer"""
        if not isinstance(dict_or_yaml, dict):
            dict_or_yaml = read_yaml(dict_or_yaml)
        if isinstance(nodes, str):
            nodes = [nodes]
        for node in nodes:
            dict_or_yaml = dict_or_yaml[node]
        return self._get_variables_dict(dict_or_yaml)

    def _get_variables_dict(self, dic):
        """Parse a dictionary adding potential variables to the minimize model"""
        tag = self.atom.tag

        # with respect to the basis
        def update_orb(old, new, orb, key):
            """Update an orbital's radii"""
            setattr(orb, f"_{key}", new)

        # Define other options
        def update(old, new, d, key, idx=None):
            """An updater for a dictionary with optional keys"""
            if idx is None:
                d[key] = new
            else:
                d[key][idx] = new

        # returned variables
        V = []

        def add_variable(var):
            nonlocal V
            if var.value is not None:
                _log.info(f"{self.__class__.__name__} adding {var}")
            if isinstance(var, Variable):
                V.append(var)

        # get default options for pseudo
        pseudo = dic.get("pseudo", {})
        add_variable(
            parse_variable(
                pseudo.get("log-radii"),
                unit="Ang",
                name=f"{tag}.logr",
                update_func=partial(update, d=self.opts, key="logr"),
            )
        )

        add_variable(
            parse_variable(
                pseudo.get("core-correction"),
                0.0,
                unit="Ang",
                name=f"{tag}.core",
                update_func=partial(update, d=self.opts, key="rcore"),
            )
        )

        # parse depending on shells in the atom
        for orb in self.atom:
            nl = f"{orb.n}{_spdfgh[orb.l]}"
            pseudo = dic.get(nl, {}).get("pseudo")
            if pseudo is None:
                _log.info(f"{self.__class__.__name__} skipping node: {nl}.pseudo")
                continue

            # Now parse this one
            add_variable(
                parse_variable(
                    pseudo.get("cutoff"),
                    orb.R,
                    unit="Ang",
                    name=f"{tag}.{nl}.r",
                    update_func=partial(update_orb, orb=orb, key="R"),
                )
            )

            add_variable(
                parse_variable(
                    pseudo.get("charge"),
                    orb.q0,
                    name=f"{tag}.{nl}.q",
                    update_func=partial(update_orb, orb=orb, key="q0"),
                )
            )

        return V
