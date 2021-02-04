from functools import partial
from subprocess import CompletedProcess
import logging

import numpy as np

from sisl.io import tableSile
from sisl.io.siesta import fdfSileSiesta
from sisl._array import arrayi, zerosd, arangei

from ._runner import AndRunner
from ._minimize import *


_log = logging.getLogger("sisl_toolbox.siesta.pseudo")


class MinimizeSiesta(BaseMinimize): # no inheritance!
    """ A minimize minimizer for siesta (PP and basis or what-ever)

    It is important that a this gets initialized with ``runner`` and ``metric``
    keyword arguments.
    """

    def __init__(self, runner, metric, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runner = runner
        self.metric = metric

    def get_constraints(self, factor=0.95):
        """ Return contraints for the zeta channels """
        # Now we define the constraints of the orbitals.
        def unpack(name):
            try:
                # split at name
                symbol, name, zeta = name.split(".")
                n = int(name[1])
                l = int(name[3])
                zeta = int(zeta[1:])
                return symbol, n, l, zeta
            except:
                return None, None, None, None

        orb_R = {} # (n,l) = {1: idx-nlzeta=1, 2: idx-nlzeta=2}
        for i, v in enumerate(self.variables):
            symbol, n, l, zeta = unpack(v.name)
            if symbol is None:
                continue
            (orb_R
             .setdefault(symbol, {})
             .setdefault((n, l), {})
             .update({zeta: i})
            )

        def assert_bounds(i1, i2):
            v1 = self.variables[i1]
            v2 = self.variables[i2]
            b1 = v1.bounds
            b2 = v2.bounds
            if not np.allclose(b1, b2):
                raise ValueError("Bounds for zeta must be the same due to normalization")

        # get two lists of neighbouring zeta's
        # Our constraint is that zeta cutoffs should be descending.
        zeta1, zeta2 = [], []
        # v now contains a dictionary with indices for the zeta orbitals
        for atom in orb_R.values():
            for z_idx in atom.values():
                for i in range(2, max(z_idx.keys()) + 1):
                    # this will request zeta-indices in order (zeta1, zeta2, ...)
                    zeta1.append(z_idx[i-1])
                    zeta2.append(z_idx[i])
                    assert_bounds(zeta1[-1], zeta2[-1])

        zeta1 = arrayi(zeta1)
        zeta2 = arrayi(zeta2)

        # now create constraint
        def fun_factory(factor, zeta1, zeta2):
            def fun(v):
                # an inequality constraint must return a non-negative
                # zeta1.R * `factor` - zeta2.R >= 0.
                return v[zeta1] * factor - v[zeta2]
            return fun

        def jac_factory(factor, zeta1, zeta2):
            def jac(v):
                out = zerosd([len(zeta1), len(v)])
                idx = arangei(len(zeta1))
                # derivative of
                # zeta1.R * `factor` - zeta2.R >= 0.
                out[idx, zeta1] = factor
                out[idx, zeta2] = -1
                return out
            return jac

        constr = []
        constr.append({
            "type": "ineq",
            "fun": fun_factory(factor, zeta1, zeta2),
            "jac": jac_factory(factor, zeta1, zeta2),
        })

        return constr

    def candidates(self, delta=1e-2, target=None, sort="max"):
        """ Compare samples and find candidates within a delta-metric of `delta`

        Candidiates are ordered around the basis-set sizes.
        This means that *zeta* variables are the only ones used for figuring out
        the candidates.

        Parameters
        ----------
        delta : float, optional
           only consider sampled metrics that lie within ``target +- delta``
        target : float, optional
           target metric value to search around. This may be useful in situations
           where all basis sets are very large and that accuracy isn't really needed.
           Defaults to the minimum metric.
        sort : {max, l1, l2}, callable
           How to sort the basis set ranges. If a callable it should return the
           indices that pivots the data to sorted candidates.
           The callable should accept two arrays, ``(x, y)`` and all variables will
           be passed (not only basis-set ranges).
        """
        # Retrieve all data points within the minim
        x = np.array(self.data.x)
        y = np.array(self.data.y)
        if target is None:
            idx_target = np.argmin(y)
        else:
            idx_target = np.argmin(np.fabs(y - target))
        xtarget = x[idx_target]
        ytarget = y[idx_target]

        # Now find all valid samples
        valid = np.logical_and(ytarget - delta <= y,
                               y <= ytarget + delta).nonzero()[0]

        # Reduce to candidate points
        x_valid = x[valid]
        y_valid = y[valid]

        # Figure out which variables are *basis ranges*
        idx_R = []
        for idx, v in enumerate(self.variables):
            # I think this should be enough for a zeta value
            if ".z" in v.name:
                idx_R.append(idx)

        if len(idx_R) > 0:
            # only use these indices to find minimum candidates
            if isinstance(sort, str):
                if sort == "max":
                    idx_increasing = np.argsort(x_valid[:, idx_R].max(axis=1))
                elif sort == "l1":
                    idx_increasing = np.argsort(x_valid[:, idx_R].sum(axis=1))
                elif sort == "l2":
                    # no need for sqrt (does nothing for sort)
                    idx_increasing = np.argsort((x_valid[:, idx_R] ** 2).sum(axis=1))
                else:
                    raise ValueError(f"{self.__class__.__name__}.candidates got an unknown value for 'sort={sort}', must be one of [max,l1,l2].")
            else:
                # it really has to be callable ;)
                idx_increasing = sort(x_valid, y_valid)

            x_valid = x_valid[idx_increasing]
            y_valid = y_valid[idx_increasing]
        elif callable(sort):
            idx_increasing = sort(x_valid, y_valid)
            x_valid = x_valid[idx_increasing]
            y_valid = y_valid[idx_increasing]

        # Return the candidates
        candidates = PropertyDict()
        candidates.x = x_valid
        candidates.y = y_valid
        candidates.x_target = xtarget
        candidates.y_target = ytarget

        return candidates

    # Redefine call
    def __call__(self, variables):
        _log.info(f"{self.__class__.__name__} variables: {variables}")

        # Run the runner
        _log.debug(f"{self.__class__.__name__} running runners")
        rets = self.runner.run()
        if not isinstance(self.runner, AndRunner):
            rets = [rets]
        for ret in rets:
            if isinstance(ret, CompletedProcess):
                # this is a shell, check return code
                if ret.returncode != 0:
                    _log.warning(f"{self.__class__.__name__} ({ret.args}) failed")

        # Calculate metric
        _log.debug(f"{self.__class__.__name__} running metrics")
        metric = self.metric.metric(variables)
        _log.info(f"{self.__class__.__name__} final metric: {metric}")
        return metric


class LocalMinimizeSiesta(LocalMinimize, MinimizeSiesta):
    pass


class DualAnnealingMinimizeSiesta(DualAnnealingMinimize, MinimizeSiesta):
    pass
