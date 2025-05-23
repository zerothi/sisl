# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Block-tri-diagonal electrode handling"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

import sisl as si
import sisl.io.siesta.binaries
import sisl.io.tbtrans
from sisl._indices import indices, indices_only
from sisl._internal import set_module
from sisl.linalg import solve
from sisl.typing import KPoint
from sisl.utils.misc import PropertyDict

from ._help import expand_btd, expand_orbs, get_expand

__all__ = ["PivotSelfEnergy", "DownfoldSelfEnergy"]


@set_module("sisl_toolbox.btd")
class PivotSelfEnergy(si.physics.SelfEnergy):
    """Container for the self-energy object

    This may either be a

    - `tbtsencSileTBtrans`
    - `tbtgfSileTBtrans`
    - `sisl.physics.SelfEnergy`
    """

    def __init__(
        self,
        name: str,
        se,
        pivot=None,
        bloch: Optional[Union[si.Bloch, Tuple[int, int, int]]] = None,
    ):

        # Name of electrode
        self.name = name

        # File containing the self-energy
        # This may be either of:
        #  tbtsencSileTBtrans
        #  tbtgfSileTBtrans
        #  SelfEnergy object (for direct calculation)
        self._se = se

        # Store the pivoting for faster indexing
        # Ensure we have a correct `pivot` argument
        if pivot is None:
            if not isinstance(se, si.io.tbtrans.tbtsencSileTBtrans):
                raise ValueError(
                    f"{self.__class__.__name__} must be passed a sisl.io.tbtrans.tbtsencSileTBtrans. "
                    "Otherwise use the DownfoldSelfEnergy object with appropriate arguments."
                )
            pivot = se

        if bloch is None:
            # necessary to get bloch-expansion of the electrode
            # In case the `pivot` holds that information, lets use it
            bloch = pivot.bloch(name)

        if not isinstance(bloch, si.Bloch):
            bloch = si.Bloch(bloch)

        if len(bloch) == 1:

            def _bloch(func, k, *args, **kwargs):
                """Simple no-up wrapper"""
                return func(*args, k=k, **kwargs)

        else:
            _bloch = bloch

        # Store the Bloch-expansion function
        self._bloch = _bloch

        # Recall that device region geometries uses the Bloch-expanded
        # electrodes. So we have to also figure out the full matrix
        # for the electrode calculation.
        # So we take out the number of orbitals in the electrode.
        # The electrode atoms for a pivoting matrix is the *actual*
        # number of orbitals *after* Bloch-expansion.
        n_orbs = pivot.read_geometry().sub(pivot.a_elec(name)).no

        if isinstance(se, sisl.io.tbtrans.tbtsencSileTBtrans):
            # We can't figure out what it is.
            raise NotImplementedError
        elif isinstance(se, sisl.io.siesta.binaries._gfSileSiesta):
            # The GF files automatically handle the bloch expansion
            # *AND* it also doubles based on spin-configuration
            # It likely shouldn't
            len_H = se._no_u
        elif isinstance(se, si.physics.SelfEnergy):
            len_H = len(se) * len(bloch)
        else:
            raise ValueError("Unknown 'se' argument.")

        # Determine whether the pivoting elements should be accounted for
        # by the spin-dimensions.
        # E.g. if this is a non-collinear spin configuration, we should double
        # everything.
        expand = get_expand(len_H, n_orbs)

        # Pivoting indices for the self-energy in the full system
        self.pvt = expand_orbs(pivot.pivot(name), expand).reshape(-1, 1)

        # Pivoting indices for the self-energy in the device region
        self.pvt_dev = expand_orbs(pivot.pivot(name, in_device=True), expand).reshape(
            -1, 1
        )

        # Retrieve BTD matrices for the corresponding electrode
        self.btd = expand_btd(pivot.btd(name), expand)

        # Get the individual matrices
        cbtd = np.cumsum(self.btd)
        # self.pvt_btd = np.concatenate(pvt_btd).reshape(-1, 1)
        # self.pvt_btd_sort = _a.arangei(o)

        if isinstance(se, si.io.tbtrans.tbtsencSileTBtrans):

            def se_func(*args, **kwargs):
                return self._se.self_energy(self.name, *args, **kwargs)

            def broad_func(*args, **kwargs):
                return self._se.broadening_matrix(self.name, *args, **kwargs)

        else:

            def se_func(*args, **kwargs):
                return self._se.self_energy(*args, **kwargs)

            def broad_func(*args, **kwargs):
                return self._se.broadening_matrix(*args, **kwargs)

        self._se_func = se_func
        self._broad_func = broad_func

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{{no: {len(self)}}}"

    def __len__(self) -> int:
        """Length of the self-energy once it has been downfolded into the device."""
        return len(self.pvt_dev)

    def self_energy(self, E: complex, k: KPoint = (0, 0, 0), *args, **kwargs):
        """Return self-energy for given parameters."""
        return self._bloch(self._se_func, k, *args, E=E, **kwargs)

    def broadening_matrix(self, E: complex, k: KPoint = (0, 0, 0), *args, **kwargs):
        """Return broadening matrix for given parameters."""
        return self._bloch(self._broad_func, k, *args, E=E, **kwargs)


@set_module("sisl_toolbox.btd")
class DownfoldSelfEnergy(PivotSelfEnergy):
    def __init__(
        self,
        name: str,
        se,
        pivot,
        Hdevice: si.Hamiltonian,
        eta_device: float = 0.0,
        bulk: bool = True,
        bloch: Optional = None,
    ):

        # Default initialize from the super (PivotSelfEnergy)
        # This also determines whether the thing requires expansion
        # due to pivoting stemming from a diagonal spin calculation.
        super().__init__(name, se, pivot, bloch)

        self._eta_device = eta_device

        # To re-create the downfoldable self-energies we need a few things:
        # pivot == for pivoting indices and BTD downfolding region
        # se == SelfEnergy for calculating self-energies and broadening matrix
        # Hdevice == device H for downfolding the electrode self-energy
        # bulk == whether the electrode self-energy argument should be passed bulk
        #         or not
        # name == just the name

        # storage data
        self._data = PropertyDict()
        data = self._data
        data.bulk = bulk

        # Create BTD indices
        # Necessary for the down-folding
        data.btd_cum0 = np.append(0, np.cumsum(self.btd))

        # Now figure out all the atoms in the downfolding region
        # pivot is the orbitals reaching into the device
        pvt = pivot.pivot(self.name)
        # TODO , this kind-of makes PivotSelfEnergy obsolete...
        # We cheat here, the super class does the calculation.
        # The parent calls 'pivot.pivot' and stores it in 'pvt' (after
        # expanding). So here we can get the exact expansion
        expand = get_expand(len(self.pvt), len(pvt))
        # note that the last orbitals in pivot_down is the returned self-energies
        # that we want to calculate in this class

        # The orbital indices in self.H.device.geometry
        # which transfers the orbitals to the downfolding region

        # Now we need to figure out the pivoting indices from the sub-set
        # geometry

        data.H = PropertyDict()
        # Store the electrode Hamiltonian used for creating the bulk part
        data.H.electrode = self._se.spgeom0

        # The *device* is now the shrunk Hamiltonian, only containing the
        # down-folding atoms for the electrode + the device part.
        # NOTE, this geometry/Hamiltonian, is *not* sorted according
        # to the down-folding scheme (unique=True).
        atoms_down = Hdevice.o2a(
            expand_orbs(pivot.pivot_down(name), expand), unique=True
        )
        data.H.device = Hdevice.sub(atoms_down)

        orbitals_down = Hdevice.a2o(atoms_down, all=True)
        orbitals_elec = Hdevice.a2o(pivot.a_elec(name), all=True)
        # Store the place of the self-energies in `data.H.device`
        # I.e. before any pivoting!
        data.elec = indices_only(orbitals_down, orbitals_elec).reshape(-1, 1)

        # the pivoting in the downfolding, in [0:len(pivot_down)]
        data.pvt_down_down = expand_orbs(
            pivot.pivot_down(name, in_down=True), expand
        ).reshape(-1, 1)

    def __str__(self) -> str:
        eta = None
        try:
            eta = self._se.eta
        except Exception:
            pass
        se = str(self._se).replace("\n", "\n ")
        return f"{self.__class__.__name__}{{no: {len(self)}, blocks: {len(self.btd)}, eta: {eta}, eta_device: {self._eta_device},\n {se}\n}}"

    def _check_Ek(self, E: complex, k: KPoint, **kwargs):
        """Check whether we have already runned this exact energy point.

        If not, then setup the data set for the next iteration.
        """
        if hasattr(self._data, "E"):
            if np.allclose(self._data.E, E) and np.allclose(self._data.k, k):
                # we have already prepared the calculation
                return True

        self._data.E = E
        self._data.Ed = E
        self._data.Eb = E
        if np.isrealobj(E):
            self._data.Ed = E + 1j * self._eta_device
            try:
                self._data.Eb = E + 1j * self._se.eta
            except Exception:
                pass
        self._data.k = np.asarray(k, dtype=np.float64)

        return False

    def _prepare(
        self, E: complex, k: KPoint = (0, 0, 0), dtype=np.complex128, **kwargs
    ):
        if self._check_Ek(E, k, **kwargs):
            return

        # Prepare the matrices
        data = self._data
        H = data.H

        Ed = data.Ed
        Eb = data.Eb
        data.SeH = H.device.Sk(k, dtype=dtype) * Ed - H.device.Pk(
            k, dtype=dtype, **kwargs
        )
        if data.bulk:

            def hsk(k, **kwargs):
                Helec = H.electrode
                # constructor for the H and S part
                return Helec.Sk(k, format="array", dtype=dtype) * Eb - Helec.Hk(
                    k, format="array", dtype=dtype, **kwargs
                )

            old = data.SeH[data.elec, data.elec.T].copy()
            data.SeH[data.elec, data.elec.T] = self._bloch(hsk, k, **kwargs)

    def self_energy(
        self, E: complex, k: KPoint = (0, 0, 0), dtype=np.complex128, *args, **kwargs
    ):
        self._prepare(E, k, dtype, **kwargs)
        data = self._data
        se = super().self_energy(E, k=k, *args, dtype=dtype, **kwargs)

        # now put it in the matrix
        M = data.SeH.copy()
        # the electrode, is not pivoted, i.e. the indices are not changed.
        M[data.elec, data.elec.T] -= se
        # Pivot the matrix to the downfolding order
        M = M[data.pvt_down_down, data.pvt_down_down.T]

        def gM(M, idx1, idx2):
            return M[idx1, :][:, idx2].toarray()

        Mr = 0
        cbtd = data.btd_cum0
        sli = slice(cbtd[0], cbtd[1])
        for b in range(1, len(self.btd)):
            sli1 = slice(cbtd[b], cbtd[b + 1])

            # Do the downfolding of the self-energies
            Mr = gM(M, sli1, sli) @ solve(
                gM(M, sli, sli) - Mr,
                gM(M, sli, sli1),
                overwrite_a=True,
                overwrite_b=True,
            )
            sli = sli1
        return Mr

    def broadening_matrix(self, *args, **kwargs):
        return self.se2broadening(self.self_energy(*args, **kwargs))
