# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

import sisl as si
from sisl_toolbox.models._base import ReferenceDispatch

from ._base import GrapheneModel

__all__ = ["GrapheneHamiltonian"]


class GrapheneHamiltonian(GrapheneModel):
    # copy the dispatcher method
    ref = GrapheneModel.ref.copy()


class SimpleDispatch(ReferenceDispatch):
    """This implements the simple nearest neighbor TB model"""

    def dispatch(self, t: float = -2.7, a: float = 1.42, orthogonal: bool = False):
        """The simplest tight-binding model for graphene"""
        # Define the graphene lattice
        da = 0.0005
        C = si.Atom(6, si.AtomicOrbital(n=2, l=1, m=0, R=a + da))
        graphene = si.geom.graphene(a, C, orthogonal=orthogonal)
        # Define the Hamiltonian
        H = si.Hamiltonian(graphene)
        H.construct([(da, a + da), (0, t)])
        return H


GrapheneHamiltonian.ref.register("simple", SimpleDispatch)


class Hancock2010Dispatch(ReferenceDispatch):
    """Implementing reference models from 10.1103/PhysRevB.81.245402"""

    doi = "10.1103/PhysRevB.81.245402"

    def dispatch(self, set: str = "A", a: float = 1.42, orthogonal: bool = False):
        """Tight-binding model based on 10.1103/PhysRevB.81.245402"""
        distance = self._obj.distance
        da = 0.0005
        H_orthogonal = True
        # U = 2.0

        R = tuple(distance(i, a) + da for i in range(4))
        if set == "A":
            # same as simple
            t = (0, -2.7)
            # U = 0.
        elif set == "B":
            # same as simple
            t = (0, -2.7)
        elif set == "C":
            t = (0, -2.7, -0.2)
        elif set == "D":
            t = (0, -2.7, -0.2, -0.18)
        elif set == "E":
            # same as D, but specific for GNR
            t = (0, -2.7, -0.2, -0.18)
        elif set == "F":
            # same as D, but specific for GNR
            t = [(0, 1), (-2.7, 0.11), (-0.09, 0.045), (-0.27, 0.065)]
            H_orthogonal = False
        elif set == "G":
            # same as D, but specific for GNR
            t = [(0, 1), (-2.97, 0.073), (-0.073, 0.018), (-0.33, 0.026)]
            # U = 0.
            H_orthogonal = False
        else:
            raise ValueError(
                f"Set specification for {self.doi} does not exist, should be one of [A-G]"
            )

        # Reduce size of R
        R = R[: len(t)]

        # Currently we do not carry over U, since it is not specified for the
        # sisl objects....

        # Define the graphene lattice
        C = si.Atom(6, si.AtomicOrbital(n=2, l=1, m=0, R=R[-1]))
        graphene = si.geom.graphene(a, C, orthogonal=orthogonal)

        nsc = graphene.find_nsc(axes=[0, 1])
        graphene.set_nsc(nsc)

        # Define the Hamiltonian
        H = si.Hamiltonian(graphene, orthogonal=H_orthogonal)
        H.construct([R, t])

        return H


GrapheneHamiltonian.ref.register("Hancock2010", Hancock2010Dispatch)
GrapheneHamiltonian.ref.register(Hancock2010Dispatch.doi, Hancock2010Dispatch)


class Ishii2010Dispatch(ReferenceDispatch):
    r"""Implementing reference model from 10.1103/PhysRevLett.104.116801

    Instead of using the :math:`\lambda_0` as parameter name, we use ``t`` for the
    coupling strength.
    """

    doi = "10.1103/PhysRevLett.104.116801"

    def dispatch(self, t: float = -2.7, a: float = 1.42, orthogonal: bool = False):
        """Tight-binding model based on 10.1103/PhysRevLett.104.116801"""
        distance = self._obj.distance
        da = 0.0005

        R = (distance(0, a) + da, distance(1, a) + da)

        def construct(H, ia, atoms, atoms_xyz=None):
            idx_t01, rij_t01 = H.geometry.close(
                ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz, ret_rij=True
            )
            H[ia, idx_t01[0]] = 0.0
            H[ia, idx_t01[1]] = t * (a / rij_t01[1]) ** 2

        # Define the graphene lattice
        C = si.Atom(6, si.AtomicOrbital(n=2, l=1, m=0, R=R[-1]))
        graphene = si.geom.graphene(a, C, orthogonal=orthogonal)

        # Define the Hamiltonian
        H = si.Hamiltonian(graphene)
        H.construct(construct)

        return H


GrapheneHamiltonian.ref.register("Ishii2010", Ishii2010Dispatch)
GrapheneHamiltonian.ref.register(Ishii2010Dispatch.doi, Ishii2010Dispatch)


class Cummings2019Dispatch(ReferenceDispatch):
    """Implementing reference model from 10.1021/acs.nanolett.9b03112"""

    doi = "10.1021/acs.nanolett.9b03112"

    def dispatch(
        self,
        t: tuple[float, float] = (-2.414, -0.168),
        beta: tuple[float, float] = (-1.847, -3.077),
        a: float = 1.42,
        orthogonal: bool = False,
    ):
        """Tight-binding model based on 10.1021/acs.nanolett.9b03112"""
        distance = self._obj.distance
        da = 0.0005

        R = (distance(0, a) + da, distance(1, a) + da, distance(2, a) + da)

        def construct(H, ia, atoms, atoms_xyz=None):
            idx_t012, rij_t012 = H.geometry.close(
                ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz, ret_rij=True
            )
            H[ia, idx_t012[0]] = 0.0
            H[ia, idx_t012[1]] = t[0] * np.exp(beta[0] * (rij_t012[1] - R[1]))
            H[ia, idx_t012[2]] = t[1] * np.exp(beta[1] * (rij_t012[2] - R[2]))

        # Define the graphene lattice
        C = si.Atom(6, si.AtomicOrbital(n=2, l=1, m=0, R=R[-1]))
        graphene = si.geom.graphene(a, C, orthogonal=orthogonal)

        # Define the Hamiltonian
        H = si.Hamiltonian(graphene)
        H.construct(construct)

        return H


GrapheneHamiltonian.ref.register("Cummings2019", Cummings2019Dispatch)
GrapheneHamiltonian.ref.register(Cummings2019Dispatch.doi, Cummings2019Dispatch)


class Wu2011Dispatch(ReferenceDispatch):
    """Implementing reference model from 10.1007/s11671-010-9791-y"""

    doi = "10.1007/s11671-010-9791-y"

    def dispatch(self, a: float = 1.42, orthogonal: bool = False):
        """Tight-binding model based on 10.1007/s11671-010-9791-y"""
        distance = self._obj.distance
        da = 0.0005

        R = (
            distance(0, a) + da,
            distance(1, a) + da,
            distance(2, a) + da,
            distance(3, a) + da,
        )
        # Define the graphene lattice
        C = si.Atom(6, si.AtomicOrbital(n=2, l=1, m=0, R=R[-1]))
        graphene = si.geom.graphene(a, C, orthogonal=orthogonal)

        # Define the Hamiltonian
        H = si.Hamiltonian(graphene, orthogonal=False)
        t = [(-0.45, 1), (-2.78, 0.117), (-0.15, 0.004), (-0.095, 0.002)]
        H.construct([R, t])

        return H


GrapheneHamiltonian.ref.register("Wu2011", Wu2011Dispatch)
GrapheneHamiltonian.ref.register(Wu2011Dispatch.doi, Wu2011Dispatch)
