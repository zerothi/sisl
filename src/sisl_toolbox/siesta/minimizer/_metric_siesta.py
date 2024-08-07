# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
from numbers import Number

import numpy as np

import sisl.io.siesta as io_siesta
from sisl._internal import set_module
from sisl.io import get_sile
from sisl.utils import direction

from ._metric import Metric
from ._path import path_rel_or_abs

__all__ = [
    "SiestaMetric",
    "EnergyMetric",
    "EigenvalueMetric",
    "ForceMetric",
    "StressMetric",
]


_log = logging.getLogger(__name__)


def _siesta_out_accept(out):
    if not isinstance(out, io_siesta.stdoutSileSiesta):
        out = io_siesta.stdoutSileSiesta(out)
    accept = out.completed()
    if accept:
        with out:
            # We do not accept:
            # KBproj: WARNING: KB projector does not decay to zero
            accept = not out.step_to("KB projector does not decay to zero")[0]
    if accept:
        for l in (0, 1, 2):
            with out:
                # We do not accept
                # KBproj: WARNING: Cut off radius for the KB projector too big
                accept &= not out.step_to(f"KBproj: WARNING: Rc({l})=")[0]

    return accept


@set_module("sisl_toolbox.siesta.minimizer")
class SiestaMetric(Metric):
    """Generic Siesta metric

    Since in some cases siesta may crash we need to have *failure* metrics
    that returns if siesta fails to run.
    """

    def __init__(self, failure=0.0):
        if isinstance(failure, Number):

            def func(metric, fail):
                if fail:
                    return failure
                return metric

            self.failure = func
        elif callable(failure):
            self.failure = failure
        else:
            raise ValueError(
                f"{self.__class__.__name__} could not initialize failure, not number or callable"
            )


@set_module("sisl_toolbox.siesta.minimizer")
class EigenvalueMetric(SiestaMetric):
    """Compare eigenvalues between two calculations and return the difference as the metric"""

    def __init__(self, eig_file, eig_ref, dist=None, align_valence=False, failure=0.0):
        """Store the reference eigenvalues along the distribution (if any)"""
        super().__init__(failure)
        self.eig_file = path_rel_or_abs(eig_file)
        # we copy to ensure users don't change these
        self.eig_ref = eig_ref.copy()
        if dist is None:
            self.dist = 1.0
        elif callable(dist):
            self.dist = dist(eig_ref)
        else:
            try:
                eig_ref * dist
            except Exception:
                raise ValueError(
                    f"{self.__class__.__name__} was passed `dist` which was not "
                    "broadcastable to `eig_ref`. Please ensure compatibility."
                )
            self.dist = dist.copy()

        # whether we should align the valence band edges
        # only for semi-conductors
        self.align_valence = align_valence

    def metric(self, variables):
        """Compare eigenvalues with a reference eigenvalue set, scaled by dist"""
        try:
            eig = io_siesta.eigSileSiesta(self.eig_file).read_data()
            eig = eig[:, :, : self.eig_ref.shape[2]]

            if self.align_valence:
                # align data at the valence band (
                eig -= eig[eig < 0.0].max()

            # Calculate the metric, also average around k-points
            metric = (((eig - self.eig_ref) * self.dist) ** 2).sum() ** 0.5 / eig.shape[
                1
            ]
            metric = self.failure(metric, False)
            _log.debug(f"metric.eigenvalue [{self.eig_file}] success {metric}")
        except Exception:
            metric = self.failure(0.0, True)
            _log.warning(f"metric.eigenvalue [{self.eig_file}] fail {metric}")
        return metric


@set_module("sisl_toolbox.siesta.minimizer")
class EnergyMetric(SiestaMetric):
    """Metric is the energy (default basis.enthalpy), read from the output file

    Alternatively the metric could be any operation of the energies that is returned.

    Parameters
    ----------
    out : str, Path
       the output from a Siesta run
    energy : callable, str, optional
       an operation to post-process the energy.
       If a `str` it will use the given energy, otherwise the function should accept a single
       dictionary (output from: `sisl.io.siesta.stdoutSileSiesta.read_energy`) and convert that
       to a single energy metric
    failure : float, optional
       in case the output does not contain anything runner fails, then we should return a "fake" metric.
    """

    def __init__(self, out, energy="basis.enthalpy", failure=0.0):
        super().__init__(failure)
        self.out = path_rel_or_abs(out)
        if isinstance(energy, str):
            energy_str = energy.split(".")

            def energy(energy_dict):
                """energy metric"""
                for sub in energy_str[:-1]:
                    energy_dict = energy_dict[sub]
                return energy_dict[energy_str[-1]]

            # TODO fix documentation to     f""" {'.'.join(energy_str)} metric """

        if not callable(energy):
            raise ValueError(
                f"{self.__class__.__name__} requires energy to be callable or str"
            )
        self.energy = energy

    def metric(self, variables):
        """Read the energy from the out file in `path`"""
        out = io_siesta.stdoutSileSiesta(self.out)
        if _siesta_out_accept(out):
            metric = self.failure(self.energy(out.read_energy()), False)
            _log.debug(f"metric.energy [{self.out}] success {metric}")
        else:
            metric = self.failure(0.0, True)
            _log.warning(f"metric.energy [{self.out}] fail {metric}")
        return metric


@set_module("sisl_toolbox.siesta.minimizer")
class ForceMetric(SiestaMetric):
    """Metric is the force (default maximum), read from the FA file

    Alternatively the metric could be any operation on the forces.

    Parameters
    ----------
    file : str, Path
       the file from which to read the forces
    force : {'abs.max', 'l2.max'} or callable, optional
       an operation to post-process the energy.
       If a `str` it will use the given numpy operation on the flattened force array.
       dictionary (output from: `sisl.io.siesta.*.read_force`) and convert that
       to a single metric
    failure : float, optional
       in case the output does not contain anything runner fails, then we should return a "fake" metric.
    """

    def __init__(self, file, force="abs.max", failure=0.0):
        super().__init__(failure)
        self.file = path_rel_or_abs(file)
        if isinstance(force, str):
            force_op = force.split(".")

            def force(forces):
                """force metric"""
                out = forces
                for op in force_op:
                    if op == "l2":
                        out = (out**2).sum(-1) ** 0.5
                    else:
                        out = getattr(np, op)(out)
                return out

            # TODO fix documentation f""" {force_op} metric """
        if not callable(force):
            raise ValueError(
                f"{self.__class__.__name__} requires force to be callable or str"
            )
        self.force = force

    def metric(self, variables):
        """Read the force from the `self.file` in `path`"""
        try:
            force = self.force(get_sile(self.file).read_force())
            metric = self.failure(force, False)
            _log.debug(f"metric.force [{self.file}] success {metric}")
        except Exception:
            metric = self.failure(0.0, True)
            _log.debug(f"metric.force [{self.file}] fail {metric}")
        return metric


@set_module("sisl_toolbox.siesta.minimizer")
class StressMetric(SiestaMetric):
    """Metric is the stress tensor, read from the output file

    Parameters
    ----------
    out : str, Path
       output from a Siesta run
    stress : callable, optional
       function which transforms the stress to a single number (the metric).
       By default it sums the diagonal stress components.
    failure : float, optional
       in case the output does not contain anything runner fails, then we should return a "fake" metric.
    """

    def __init__(self, out, stress="ABC", failure=2.0):
        super().__init__(failure)
        self.out = path_rel_or_abs(out)
        if isinstance(stress, str):
            stress_directions = list(map(direction, stress))

            def stress(stress_matrix):
                """stress metric"""
                return stress_matrix[stress_directions, stress_directions].sum()

            # TODO fix documentation f""" {stress_directions} metric """
        if not callable(stress):
            raise ValueError(
                f"{self.__class__.__name__} requires stress to be callable"
            )
        self.stress = stress

    def metric(self, variables):
        """Convert the stress-tensor to a single metric that should be minimized"""
        out = io_siesta.stdoutSileSiesta(self.out)
        if _siesta_out_accept(out):
            stress = self.stress(out.read_stress[-1]())
            metric = self.failure(stress, False)
            _log.debug(f"metric.stress [{self.out}] success {metric}")
        else:
            metric = self.failure(0.0, True)
            _log.warning(f"metric.stress [{self.out}] fail {metric}")
        return metric
