# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from .sile import SileORCA
from ..sile import add_sile, sile_fh_open

from sisl.utils import PropertyDict
from sisl._internal import set_module


__all__ = ['txtSileORCA']


@set_module("sisl.io.orca")
class txtSileORCA(SileORCA):
    """ Output property txt file from ORCA """

    @sile_fh_open()
    def read_energy(self, all=False):
        """ Reads the energy specification from ORCA property txt file and returns 
        energy dictionary in units of eV (and related info from the block)

        Parameters
        ----------
        all: bool, optional
            return a list of dictionaries from each step

        Returns
        -------
        PropertyDict : all data from the "DFT_Energy" segment of ORCA property output
        """

        def readE(itt):
            # read the DFT_Energy block
            f = self.step_to("$ DFT_Energy", reread=False)[0]
            if not f:
                return None
            next(itt) # description
            next(itt) # geom. index
            next(itt) # prop. index
            line = next(itt)
            E = PropertyDict()
            while "----" not in line:
                v = line.split()
                value = float(v[-1])
                v.append([]) # ensure at least four entries
                if v[3] == "Electrons":
                    if v[2] == "Alpha":
                        E["elec_alpha"] = value
                    elif v[2] == "Beta":
                        E["elec_beta"] = value
                    else:
                        E["elec_total"] = value
                elif v[0] == "Exchange":
                    E["exchange"] = value
                elif v[0] == "Correlation":
                    if v[2] == "NL":
                        E["correlation_nl"] = value
                    else:
                        E["correlation"] = value
                elif v[0] == "Exchange-Correlation":
                    E["exchange-correlation"] = value
                elif v[0] == "Embedding":
                    E["embedding"] = value
                elif v[1] == "DFT":
                    E["total_energy"] = value
                line = next(itt)
            return E

        itt = iter(self)
        E = []
        e = readE(itt)
        while e is not None:
            E.append(e)
            e = readE(itt)

        if all:
            return E
        if len(E) > 0:
            return E[-1]
        return None

add_sile('txt', txtSileORCA, gzip=True)
