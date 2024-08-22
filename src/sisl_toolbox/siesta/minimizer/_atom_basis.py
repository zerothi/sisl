# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
from functools import partial

import sisl as si
from sisl._internal import set_module
from sisl.utils import NotNonePropertyDict

from ._variable import Variable
from ._yaml_reader import parse_variable, read_yaml

__all__ = ["AtomBasis"]


_log = logging.getLogger(__name__)
_Ang2Bohr = si.units.convert("Ang", "Bohr")
_eV2Ry = si.units.convert("eV", "Ry")


@set_module("sisl_toolbox.siesta.minimizer")
class AtomBasis:
    """Basis block format for Siesta"""

    def __init__(self, atom, opts=None):
        # opts = {(n, l): # or n=1, l=0 1s
        #             {"soft": (prefactor, inner-rad),
        #              "split": split_norm_value,
        #              "charge": (Z, screen, delta),
        #              "filter": cutoff,
        #              "pol" : npol,
        #             }
        #         "ion_charge": ionic_charge,
        #         "type": "split"|"splitgauss"|"nodes"|"nonodes"|"filteret"
        # }
        # All arguments should be in Ang, eV
        self.atom = atom
        assert isinstance(atom, si.Atom)
        if "." in self.atom.tag:
            raise ValueError("The atom 'tag' must not contain a '.'!")

        if opts is None:
            self.opts = dict()
        else:
            self.opts = opts
        if not isinstance(self.opts, dict):
            raise ValueError(
                f"{self.__class__.__name__} must get `opts` as a dictionary argument"
            )

        # Assert that we have options corresonding to the orbitals present
        for key in self.opts.keys():
            try:
                n, l = key
            except Exception:
                continue
            found = False
            for orb in self.atom:
                if orb.n == n and orb.l == l:
                    found = True
            if not found:
                raise ValueError(
                    "Options passed for n={n} l={l}, but no orbital with that signiture is present?"
                )

        # ensure each orbital has an option associated
        for (n, l), orbs in self.yield_nl_orbs():
            self.opts.setdefault((n, l), {})

    @classmethod
    def from_dict(cls, dic):
        """Return an `AtomBasis` from a dictionary

        Parameters
        ----------
        dic : dict
        """
        from sisl_toolbox.siesta.atom._atom import _shell_order

        element = dic["element"]
        tag = dic.get("tag")
        mass = dic.get("mass", None)

        # get default options for pseudo
        opts = NotNonePropertyDict()

        basis = dic.get("basis", {})
        opts["ion_charge"] = parse_variable(basis.get("ion-charge")).value
        opts["type"] = basis.get("type")

        def get_radius(orbs, zeta):
            for orb in orbs:
                if orb.zeta == zeta:
                    return orb.R
            raise ValueError("Could parse the negative R value")

        orbs = []
        for nl in dic:
            if nl not in _shell_order:
                continue

            n, l = int(nl[0]), "spdfgh".index(nl[1])
            # Now we are sure we are dealing with valence shells
            basis = dic[nl].get("basis", {})

            opt_nl = NotNonePropertyDict()
            orbs_nl = []

            # Now read through the entries
            for key, entry in basis.items():
                if key in ("charge-confinement", "charge-conf"):
                    opt_nl["charge"] = [
                        parse_variable(entry.get("charge")).value,
                        parse_variable(entry.get("yukawa"), unit="1/Ang").value,
                        parse_variable(entry.get("width"), unit="Ang").value,
                    ]
                elif key in ("soft-confinement", "soft-conf"):
                    opt_nl["soft"] = [
                        parse_variable(entry.get("V0"), unit="eV").value,
                        parse_variable(entry.get("ri"), unit="Ang").value,
                    ]
                elif key in ("filter",):
                    opt_nl["filter"] = parse_variable(entry, unit="eV").value
                elif key in ("split-norm", "split"):
                    opt_nl["split"] = parse_variable(entry).value
                elif key in ("polarization", "pol"):
                    opt_nl["pol"] = parse_variable(entry).value
                elif key.startswith("zeta"):
                    # cutoff of zeta
                    zeta = int(key[4:])
                    R = parse_variable(entry, unit="Ang").value
                    if R < 0:
                        R *= -get_radius(orbs_nl, zeta - 1)
                    orbs_nl.append(si.AtomicOrbital(n=n, l=l, m=0, zeta=zeta, R=R))

            if len(orbs_nl) > 0:
                opts[(n, l)] = opt_nl
                orbs.extend(orbs_nl)

        atom = si.Atom(element, orbs, mass=mass, tag=tag)
        return cls(atom, opts)

    @classmethod
    def from_yaml(cls, file, nodes=()):
        """Parse the yaml file"""
        from ._yaml_reader import read_yaml

        return cls.from_dict(read_yaml(file, nodes))

    @classmethod
    def from_block(cls, block):
        """Return an `Atom` for a specified basis block

        Parameters
        ----------
        block : list or str
           the PAO.basis block (as read from an fdf file).
           Should be a list of lines.
        """
        if isinstance(block, str):
            block = block.splitlines()
        else:
            # store local list
            block = list(block)

        def blockline():
            nonlocal block
            out = ""
            while len(out) == 0:
                out = block.pop(0).split("#")[0].strip()
            return out

        # define global opts
        opts = {}

        specie = blockline()
        specie = specie.split()
        if len(specie) == 4:
            # we have Symbol, nl, type, ionic_charge
            symbol, nl, opts["type"], opts["ion_charge"] = specie
        elif len(specie) == 3:
            # we have Symbol, nl, type
            # or
            # we have Symbol, nl, ionic_charge
            symbol, nl, opt = specie
            try:
                opts["ion_charge"] = float(opt)
            except Exception:
                opts["type"] = opt
        elif len(specie) == 2:
            # we have Symbol, nl
            symbol, nl = specie

        # now loop orbitals
        orbs = []
        for _ in range(int(nl)):
            # we have 2 or 3 lines
            nl_line = blockline()
            rc_line = blockline()
            # check if we have contraction in the line
            # This is not perfect, but should grab
            # contration lines rather than next orbital line.
            # This is because the first n=<integer> should never
            # contain a ".", whereas the contraction *should*.
            if len(block) > 0:
                if "." in block[0].split()[0]:
                    contract_line = blockline()

            # remove n=
            nl_line = nl_line.replace("n=", "").split()

            # first 3 are n, l, Nzeta
            n = int(nl_line.pop(0))
            l = int(nl_line.pop(0))
            nzeta = int(nl_line.pop(0))
            # assign defaults
            nlopts = {}

            while len(nl_line) > 0:
                opt = nl_line.pop(0)
                if opt == "P":
                    try:
                        npol = int(nl_line[0])
                        nl_line.pop(0)
                        nlopts["pol"] = npol
                    except Exception:
                        nlopts["pol"] = 1
                elif opt == "S":
                    nlopts["split"] = float(nl_line.pop(0))
                elif opt == "F":
                    nlopts["filter"] = float(nl_line.pop(0)) / _eV2Ry
                elif opt == "E":
                    # 1 or 2 values
                    V0 = float(nl_line.pop(0)) / _eV2Ry
                    try:
                        ri = float(nl_line[0]) / _Ang2Bohr
                        nl_line.pop(0)
                    except Exception:
                        # default to None (uses siesta default)
                        ri = None
                    nlopts["soft"] = [V0, ri]
                elif opt == "Q":
                    # 1, 2 or 3 values
                    charge = float(nl_line.pop(0))
                    try:
                        # this is in Bohr-1
                        yukawa = float(nl_line[0]) * _Ang2Bohr
                        nl_line.pop(0)
                    except Exception:
                        # default to None (uses siesta default)
                        yukawa = None
                    try:
                        width = float(nl_line[0]) / _Ang2Bohr
                        nl_line.pop(0)
                    except Exception:
                        # default to None (uses siesta default)
                        width = None
                    nlopts["charge"] = [charge, yukawa, width]

            # now we have everything to build the orbitals etc.
            for izeta, rc in enumerate(map(float, rc_line.split()), 1):
                if rc > 0:
                    rc /= _Ang2Bohr
                elif rc < 0 and izeta > 1:
                    rc *= -orbs[-1].R
                elif rc == 0 and izeta > 1:
                    # this is ok, the split-norm will be used to
                    # calculate the radius
                    pass
                else:
                    raise ValueError(
                        f"Could not parse the PAO.Basis block for the zeta ranges {rc_line}."
                    )
                orb = si.AtomicOrbital(n=n, l=l, m=0, zeta=izeta, R=rc)
                nzeta -= 1
                orbs.append(orb)

            # In case the final orbitals hasn't been defined.
            # They really should be defined in this one, but sometimes it may be
            # useful to leave the rc's definitions out.
            rc = orbs[-1].R
            for izeta in range(nzeta):
                orb = si.AtomicOrbital(n=n, l=l, m=0, zeta=orbs[-1].zeta + 1, R=rc)
                orbs.append(orb)
            opts[(n, l)] = nlopts

        # Now create the atom
        atom = si.Atom(symbol, orbs)
        return cls(atom, opts)

    def yield_nl_orbs(self):
        """An iterator with each different ``n, l`` pair returned with a list of zeta-shells"""
        orbs = {}
        for orb in self.atom:
            # build a dictionary
            # {(n=2, l=1): [Z1, Z2],
            #  (n=2, l=0): [Z1, Z2, Z3]
            # }
            orbs.setdefault((orb.n, orb.l), []).append(orb)
        yield from orbs.items()

    def basis(self):
        """Get basis block lines (as list)"""

        block = []

        # Determine number of unique n,l combinations
        nl = sum(1 for _ in self.yield_nl_orbs())

        line = f"{self.atom.symbol} {nl}"
        if "type" in self.opts:
            line += f" {self.opts['type']}"
        if "ion_charge" in self.opts:
            line += f" {self.opts['ion_charge']:.10f}"
        block.append(line)

        # Now add basis lines
        for (n, l), orbs in self.yield_nl_orbs():
            l = orbs[0].l

            # get options for this shell
            opts = self.opts[(n, l)]
            line = f"n={n} {l} {len(orbs)}"
            for key, value in opts.items():
                if key == "pol":
                    # number of polarization orbitals
                    if value > 0:
                        line += " P"
                        if value > 1:
                            line += f" {value}"
                elif key == "soft":
                    V0, ri = value
                    # V0, inner-radius
                    # a confinement potential below 1meV makes no sense
                    if V0 > 0.001:
                        line += f" E {V0*_eV2Ry:.10f}"
                        if not ri is None:
                            if ri > 0:
                                # explicit radius, convert to Bohr
                                ri *= _Ang2Bohr
                            line += f" {ri:.10f}"

                elif key == "split":
                    # split-norm value
                    split_norm = value
                    line += f" S {split_norm:.10f}"
                elif key == "filter":
                    # filter cutoff
                    filter_cutoff = value
                    line += f" F {filter_cutoff*_eV2Ry:.10f}"
                elif key == "charge":
                    # Z = charge (in e)
                    # Yukawa screening parameter (in Bohr-1)
                    # delta = singularity regularization parameter
                    Z, yukawa, width = value
                    if abs(Z) > 0.005:
                        # only consider charges above 0.005 electrons
                        line += f" Q {Z:.10f}"
                        if not yukawa is None:
                            line += f" {yukawa/_Ang2Bohr:.10f}"
                        if not width is None:
                            line += f" {width*_Ang2Bohr:.10f}"
                else:
                    raise ValueError(f"Unknown option for n={n} l={l}: {key}")
            block.append(line)

            # now add rcs
            # sort according to Zeta (should generally be, but just
            # to be sure)
            orbs_sorted = sorted(orbs, key=lambda orb: orb.zeta)

            line = " ".join(map(lambda orb: f"{orb.R*_Ang2Bohr:.10f}", orbs_sorted))
            block.append(line)
            # We don't need the 1's, they are contraction factors
            # and we simply keep them the default values
            # line = " ".join(map(lambda orb: "1.0000", orbs))
            # block.append(line)
        return block

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
        def update_orb(old, new, orb):
            """Update an orbital's radius"""
            orb._R = new

        # Define other options
        def update(old, new, d, key, index=None):
            """An updater for a dictionary with optional keys"""
            if index is None:
                d[key] = new
            else:
                d[key][index] = new

        # returned variables
        V = []

        def add_variable(var):
            nonlocal V
            if var.value is not None:
                _log.info(f"{self.__class__.__name__} adding {var}")
            if isinstance(var, Variable):
                if var.name in V:
                    return
                V.append(var)

        # get default options for pseudo
        basis = dic.get("basis", {})
        add_variable(
            parse_variable(
                basis.get("ion-charge"),
                name=f"{tag}.ion-q",
                update_func=partial(update, d=self.opts, key="ion_charge"),
            )
        )

        # parse depending on shells in the atom
        spdf = "spdfgh"
        for orb in self.atom:
            n, l = orb.n, orb.l
            nl = f"{n}{spdf[l]}"
            basis = dic.get(nl, {}).get("basis")
            if basis is None:
                _log.info(f"{self.__class__.__name__} skipping node: {nl}.basis")
                continue

            for flag in ("charge-confinement", "charge-conf"):
                # Now parse this one
                d = basis.get(flag, {})
                for var in [
                    parse_variable(
                        d.get("charge"),
                        name=f"{tag}.{nl}.charge.q",
                        update_func=partial(
                            update, d=self.opts[(n, l)], key="charge", index=0
                        ),
                    ),
                    parse_variable(
                        d.get("yukawa"),
                        unit="1/Ang",
                        name=f"{tag}.{nl}.charge.yukawa",
                        update_func=partial(
                            update, d=self.opts[(n, l)], key="charge", index=1
                        ),
                    ),
                    parse_variable(
                        d.get("width"),
                        unit="Ang",
                        name=f"{tag}.{nl}.charge.width",
                        update_func=partial(
                            update, d=self.opts[(n, l)], key="charge", index=2
                        ),
                    ),
                ]:
                    add_variable(var)

            for flag in ("soft-confinement", "soft-conf"):
                # Now parse this one
                d = basis.get(flag, {})
                for var in [
                    parse_variable(
                        d.get("V0"),
                        unit="eV",
                        name=f"{tag}.{nl}.soft.V0",
                        update_func=partial(
                            update, d=self.opts[(n, l)], key="soft", index=0
                        ),
                    ),
                    parse_variable(
                        d.get("ri"),
                        unit="Ang",
                        name=f"{tag}.{nl}.soft.ri",
                        update_func=partial(
                            update, d=self.opts[(n, l)], key="soft", index=1
                        ),
                    ),
                ]:
                    add_variable(var)

            add_variable(
                parse_variable(
                    basis.get("filter"),
                    unit="eV",
                    name=f"{tag}.{nl}.filter",
                    update_func=partial(update, d=self.opts[(n, l)], key="filter"),
                )
            )

            for flag in ("split-norm", "split"):
                add_variable(
                    parse_variable(
                        basis.get(flag),
                        name=f"{tag}.{nl}.split",
                        update_func=partial(update, d=self.opts[(n, l)], key="split"),
                    )
                )

            add_variable(
                parse_variable(
                    basis.get(f"zeta{orb.zeta}"),
                    name=f"{tag}.{nl}.z{orb.zeta}",
                    update_func=partial(update_orb, orb=orb),
                )
            )
        return V
