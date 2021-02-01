import logging
import sisl.io.siesta as io_siesta

from ._path import path_abs, path_rel_or_abs
from ._metric import Metric


__all__ = ["SiestaMetric", "EnergyMetric", "EigenvalueMetric", "StressMetric"]


_log = logging.getLogger("sisl_toolbox.siesta.pseudo")


def _siesta_out_accept(out):
    if not isinstance(out, io_siesta.outSileSiesta):
        out = io_siesta.outSileSiesta(self.out)
    accept = out.completed()
    if accept:
        with out:
            # We do accept
            # KBproj: WARNING: Cut off radius for the KB projector too big
            # We do not accept:
            # KBproj: WARNING: KB projector does not decay to zero
            accept = not out.step_to("KB projector does not decay to zero")[0]
    return accept


class SiestaMetric(Metric):
    """ Generic Siesta metric

    Since in some cases siesta may crash we need to have *failure* metrics
    that returns if siesta fails to run.
    """

    def __init__(self, failure=0.):
        self.failure = failure


class EnergyMetric(SiestaMetric):
    """ Metric is the energy (default total), read from the output file

    Parameters
    ----------
    out : str, Path
       the output from a Siesta run
    failure : float, optional
       in case the output does not contain anything runner fails, then we should return a "fake" metric.
    energy: str, optional
       which energy to minimize, default total, can be anything `sisl.io.siesta.outSileSiesta` returns
    """

    def __init__(self, out, failure=0., energy='total'):
        super().__init__(failure)
        self.out = path_rel_or_abs(out)
        self.energy = energy

    def metric(self, variables):
        """ Read the energy from the out file in `path` """
        out = io_siesta.outSileSiesta(self.out)
        if _siesta_out_accept(out):
            metric = out.read_energy()[self.energy]
            _log.debug(f"metric.energy [{self.out}:{self.energy}] success {metric}")
        else:
            metric = self.failure
            _log.warning(f"metric.energy [{self.out}:{self.energy}] fail {metric}")
        return metric


class EigenvalueMetric(SiestaMetric):
    """ Compare eigenvalues between two calculations and return the difference as the metric """

    def __init__(self, eig_file, eig_ref, dist=None, align_valence=False, failure=0.):
        """ Store the reference eigenvalues along the distribution (if any) """
        super().__init__(failure)
        self.eig_file = path_rel_or_abs(eig_file)
        # we copy to ensure users don't change these
        self.eig_ref = eig_ref.copy()
        if dist is None:
            self.dist = 1.
        elif callable(dist):
            self.dist = dist(eig_ref)
        else:
            try:
                a = eig_ref * dist
            except:
                raise ValueError(f"{self.__class__.__name__} was passed `dist` which was not "
                                 "broadcastable to `eig_ref`. Please ensure compatibility.")
            self.dist = dist.copy()

        # whether we should align the valence band edges
        # only for semi-conductors
        self.align_valence = align_valence

    def metric(self, variables):
        """ Compare eigenvalues with a reference eigenvalue set, scaled by dist """
        try:
            eig = io_siesta.eigSileSiesta(self.eig_file).read_data()
            eig = eig[:, :, :self.eig_ref.shape[2]]

            if self.align_valence:
                # align data at the valence band (
                eig -= eig[eig < 0.].max()

            # Calculate the metric, also average around k-points
            metric = (((eig - self.eig_ref) * self.dist) ** 2).sum() ** 0.5 / eig.shape[1]
            _log.debug(f"metric.eigenvalue [{self.eig_file}] success {metric}")
        except:
            metric = self.failure
            _log.warning(f"metric.eigenvalue [{self.eig_file}] fail {metric}")
        return metric


class StressMetric(SiestaMetric):
    """ Metric is the stress tensor, read from the output file

    Parameters
    ----------
    out : str, Path
       output from a Siesta run
    stress_op : func, optional
       function which transforms the stress to a single number (the metric).
       By default it sums the diagonal stress components.
    failure : float, optional
       in case the output does not contain anything runner fails, then we should return a "fake" metric.
    """

    def __init__(self, out, stress_op=None, failure=2.):
        super().__init__(failure)
        self.out = path_rel_or_abs(out)
        if stress_op is None:
            def stress_op(stress):
                return np.diag(stress).sum()
        if not callable(stress_op):
            raise ValueError(f"{self.__class__.__name__} requires stress_op to be callable")
        self.stress_op = stress_op

    def metric(self, variables):
        """ Convert the stress-tensor to a single metric that should be minimized """
        out = io_siesta.outSileSiesta(self.out)
        if _siesta_out_accept(out):
            metric = self.stress_op(out.read_stress())
            _log.debug(f"metric.stress [{self.out}] success {metric}")
        else:
            metric = self.failure
            _log.warning(f"metric.stress [{self.out}] fail {metric}")
        return metric
