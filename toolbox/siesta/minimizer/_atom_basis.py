from functools import partial

import sisl as si

from ._variable import UpdateVariable


__all__ = ["AtomBasis"]


_Ang2Bohr = si.units.convert("Ang", "Bohr")
_eV2Ry = si.units.convert("eV", "Ry")


class AtomBasis:
    """Basis block format for Siesta """

    def __init__(self, atom, opts=None):
        # opts = {(n, l): # or n=1, l=0 1s
        #             {"soft": (prefactor, inner-rad),
        #              "split": split_norm_value,
        #              "charge": (Z, screen, delta),
        #              "filter": cutoff,
        #              "pol" : npol,
        #             }
        #         "charge": ionic_charge,
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
            raise ValueError(f"{self.__class__.__name__} must get `opts` as a dictionary argument")

        # Assert that we have options corresonding to the orbitals present
        for key in self.opts.keys():
            try:
                n, l = key
            except:
                continue
            found = False
            for orb in self.atom:
                if orb.n == n and orb.l == l:
                    found = True
            if not found:
                raise ValueError("Options passed for n={n} l={l}, but no orbital with that signiture is present?")

        # ensure each orbital has an option associated
        for (n, l), orbs in self.yield_nl_orbs():
            self.opts.setdefault((n, l), {})

    @classmethod
    def from_block(cls, block):
        """ Return an `Atom` for a specified basis block

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
                out = block.pop(0).split('#')[0]
            return out

        # define global opts
        opts = {}

        specie = blockline()
        specie = specie.split()
        if len(specie) == 4:
            # we have Symbol, nl, type, ionic_charge
            symbol, nl, opts["type"], opts["charge"] = specie
        elif len(specie) == 3:
            # we have Symbol, nl, type
            # or
            # we have Symbol, nl, ionic_charge
            symbol, nl, opt = specie
            try:
                opts["charge"] = float(opt)
            except:
                opts["type"] = opt
        elif len(specie) == 2:
            # we have Symbol, nl
            symbol, nl = specie
            type = None

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
            if '.' in block[0].split()[0]:
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
                    except:
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
                    except:
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
                    except:
                        # default to None (uses siesta default)
                        yukawa = None
                    try:
                        width = float(nl_line[0]) / _Ang2Bohr
                        nl_line.pop(0)
                    except:
                        # default to None (uses siesta default)
                        width = None
                    nlopts["charge"] = [charge, yukawa, width]

            # now we have everything to build the orbitals etc.
            for izeta, rc in enumerate(map(float, rc_line.split())):
                orb = si.AtomicOrbital(n=n, l=l, m=0, zeta=izeta+1, R=rc / _Ang2Bohr)
                nzeta -= 1
                orbs.append(orb)
            assert nzeta == 0
            opts[(n, l)] = nlopts

        # Now create the atom
        atom = si.Atom(symbol, orbs)
        return cls(atom, opts)

    def yield_nl_orbs(self):
        """ An iterator with each different ``n, l`` pair returned with a list of zeta-shells """
        orbs = {}
        for orb in self.atom:
            # build a dictionary
            # {(n=2, l=1): [Z1, Z2],
            #  (n=2, l=0): [Z1, Z2, Z3]
            # }
            orbs.setdefault((orb.n, orb.l), []).append(orb)
        yield from orbs.items()

    def basis(self):
        """ Get basis block lines (as list)"""

        block = []

        # Determine number of unique n,l combinations
        nl = sum(1 for _ in self.yield_nl_orbs())

        line = f"{self.atom.symbol} {nl}"
        if "type" in self.opts:
            line += f" {self.opts['type']}"
        if "charge" in self.opts:
            line += f" {self.opts['charge']:.10f}"
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

            line = " ".join(map(lambda orb: f"{orb.R*_Ang2Bohr:.10f}",
                                orbs_sorted))
            block.append(line)
            # We don't need the 1's, they are contraction factors
            # and we simply keep them the default values
            #line = " ".join(map(lambda orb: "1.0000", orbs))
            #block.append(line)
        return block

    def get_variables(self, dict_or_yaml):
        """ Convert a dictionary or yaml file input to variables usable by the minimizer """
        if not isinstance(dict_or_yaml, dict):
            # Then it must be a yaml file
            import yaml
            yaml_dict = yaml.load(open(dict_or_yaml, 'r'), Loader=yaml.CLoader)
            dict_or_yaml = yaml_dict[self.atom.tag]
        return self._get_variables_dict(dict_or_yaml)

    def _get_variables_dict(self, dic):
        """ Parse a dictionary adding potential variables to the minimize model """
        symbol = self.atom.tag

        # Now loop and find coincidences in the dictionary
        # with respect to the basis
        def update_orb(old, new, orb):
            """ Update an orbital's radii """
            orb._R = new

        # Define other options
        def update(old, new, d, key, idx=None):
            """ An updater for a dictionary with optional keys """
            if idx is None:
                d[key] = new
            else:
                d[key][idx] = new

        def parse_nl(self, n, l, orbs, dic):
            V = []

            if len(dic) == 0:
                # quick return if empty
                return V

            # Now parse dictionary for nl
            # Loop keys to figure out what is there
            for key, value in dic.items():
                # we don't need to parse POLARIZATION
                # since that is implicitly handled
                if key == "filter":
                    # Extract values
                    v0 = value["initial"]
                    bounds = value["bounds"]
                    delta = value["delta"]
                    # Ensure it is defined
                    self.opts[(n, l)][key] = v0
                    var = UpdateVariable(f"{symbol}.n{n}l{l}.filter", v0, bounds,
                                         partial(update, d=self.opts[(n, l)], key=key),
                                         delta=delta)
                    V.append(var)

                elif key == "split":
                    # Extract values
                    v0 = value["initial"]
                    bounds = value["bounds"]
                    delta = value["delta"]
                    # Ensure it is defined
                    self.opts[(n, l)][key] = v0
                    var = UpdateVariable(f"{symbol}.n{n}l{l}.split", v0, bounds,
                                         partial(update, d=self.opts[(n, l)], key=key),
                                         delta=delta)
                    V.append(var)

                elif key == "soft":
                    # Soft confinement is comprised of two variables
                    #  1. V0, potential hight
                    # *2. turn-on radius for the potential
                    # function is defined in ri < r < rc
                    # and has shape:
                    #   V0 * e^(- (rc - ri)/(r - ri)) / (rc - r)

                    if "ri" in value:
                        def_ri = value["ri"]["initial"]
                    else:
                        def_ri = self.opts[(n, l)].get(key, [0, None])[1]

                    # Extract values
                    V0 = value["V0"]
                    v0 = V0["initial"]
                    bounds = V0["bounds"]
                    delta = V0["delta"]
                    # Ensure it is defined
                    self.opts[(n, l)][key] = [v0, def_ri]
                    var = UpdateVariable(f"{symbol}.n{n}l{l}.soft.V0", v0, bounds,
                                         partial(update, d=self.opts[(n, l)], key=key, idx=0),
                                         delta=delta)
                    V.append(var)

                    if "ri" in value:
                        ri = value["ri"]
                        v0 = ri["initial"]
                        bounds = ri["bounds"]
                        delta = ri["delta"]
                        var = UpdateVariable(f"{symbol}.n{n}l{l}.soft.ri", v0, bounds,
                                             partial(update, d=self.opts[(n, l)], key=key, idx=1),
                                             delta=delta)
                        V.append(var)

                elif key == "charge":
                    # Charge confinement is comprised of three variables
                    #  1. Q, charge value for confinement potential
                    # *2. Yukawa screening length (inverse length), lambda
                    # *3. width of the potential
                    # function is defined in r < rc
                    # and has shape:
                    #   Q * e^(- lambda * r) / (r**2 + width ** 2) ** 0.5

                    if "yukawa" in value:
                        def_yukawa = value["yukawa"]["initial"]
                    else:
                        def_yukawa = self.opts[(n, l)].get(key, [0, None, None])[1]
                    if "width" in value:
                        def_width = value["width"]["initial"]
                    else:
                        def_width = self.opts[(n, l)].get(key, [0, None, None])[2]

                        # Extract values
                    Q = value["Q"]
                    v0 = Q["initial"]
                    bounds = Q["bounds"]
                    delta = Q["delta"]
                    # Ensure it is defined
                    self.opts[(n, l)][key] = [v0, def_yukawa, def_width]
                    var = UpdateVariable(f"{symbol}.n{n}l{l}.charge.Q", v0, bounds,
                                         partial(update, d=self.opts[(n, l)], key=key, idx=0),
                                         delta=delta)
                    V.append(var)

                    if "yukawa" in value:
                        yukawa = value["yukawa"]
                        v0 = yukawa["initial"]
                        bounds = yukawa["bounds"]
                        delta = yukawa["delta"]
                        var = UpdateVariable(f"{symbol}.n{n}l{l}.charge.yukawa", v0, bounds,
                                             partial(update, d=self.opts[(n, l)], key=key, idx=1),
                                             delta=delta)
                        V.append(var)

                    if "width" in value:
                        width = value["width"]
                        v0 = width["initial"]
                        bounds = width["bounds"]
                        delta = width["delta"]
                        var = UpdateVariable(f"{symbol}.n{n}l{l}.charge.width", v0, bounds,
                                             partial(update, d=self.opts[(n, l)], key=key, idx=2),
                                             delta=delta)
                        V.append(var)

                elif key.startswith("zeta"):
                    zeta = int(key[4:])
                    found = False
                    for orb in orbs:
                        if orb.zeta == zeta:
                            found = True
                            # create variable for this
                            # 6 Ang should be more than enough
                            v0 = value["initial"]
                            bounds = value["bounds"]
                            delta = value["delta"]
                            # ensure we start correctly
                            orb._R = v0
                            var = UpdateVariable(f"{symbol}.n{n}l{l}.z{zeta}", v0, bounds,
                                                 partial(update_orb, orb=orb),
                                                 delta=delta)
                            V.append(var)
                    if not found:
                        raise ValueError("Could not find zeta value in the orbitals?")
                else:
                    raise ValueError(f"Could not find something useful to do with: {key} = {value}, a typo perhaps?")

            return V

        V = []
        for (n, l), orbs in self.yield_nl_orbs():
            V.extend(parse_nl(self, n, l, orbs,
                              dic.get(f"{n}" + "spdfg"[l], {})))

        # Now add atomic charge
        charge = dic.get("charge", {})
        if len(charge) > 0:
            v0 = charge["initial"]
            bounds = charge["bounds"]
            delta = charge["delta"]
            var = UpdateVariable(f"{symbol}.charge", v0, bounds,
                                 partial(update, d=self.opts, key="charge"),
                                 delta=delta)
            V.append(var)

        return V
