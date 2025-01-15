# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" Eigenchannel calculator for any number of electrodes

Developer: Nick Papior
Contact: nickpapior <at> gmail.com
sisl-version: >=0.11.0
tbtrans-version: >=siesta-4.1.5

This eigenchannel calculater uses TBtrans output to calculate the eigenchannels
for N-terminal systems. In the future this will get transferred to the TBtrans code
but for now this may be used for arbitrary geometries.

It requires two inputs and has several optional flags.

- The siesta.TBT.nc file which contains the geometry that is to be calculated for
  The reason for using the siesta.TBT.nc file is the ease of use:

    The siesta.TBT.nc contains electrode atoms and device atoms. Hence it
    becomes easy to read in the electrode atomic positions.
    Note that since you'll always do a 0 V calculation this isn't making
    any implications for the requirement of the TBT.nc file.
"""
from numbers import Integral
from pathlib import Path

import numpy as np
import scipy.sparse as ssp
from scipy.sparse.linalg import svds

import sisl as si
from sisl import _array as _a
from sisl._internal import set_module
from sisl.linalg import (
    cholesky,
    eigh,
    eigh_destroy,
    inv_destroy,
    signsqrt,
    solve,
    svd_destroy,
)
from sisl.messages import warn
from sisl.utils.misc import PropertyDict

arangei = _a.arangei
indices_only = si._indices.indices_only
indices = si._indices.indices
conj = np.conj

__all__ = ["PivotSelfEnergy", "DownfoldSelfEnergy", "DeviceGreen"]


def dagger(M):
    return conj(M.T)


def _scat_state_svd(A, **kwargs):
    """Calculating the SVD of matrix A for the scattering state

    Parameters
    ----------
    A : numpy.ndarray
       matrix to obtain SVD from
    scale : bool or float, optional
       whether to scale matrix `A` to be above ``1e-12`` or by a user-defined number
    lapack_driver : str, optional
       driver queried from `scipy.linalg.svd`
    """
    scale = kwargs.get("scale", False)
    # Scale matrix by a factor to lie in [1e-12; inf[
    if isinstance(scale, bool):
        if scale:
            _ = np.floor(np.log10(np.absolute(A).min())).astype(int)
            if _ < -12:
                scale = 10 ** (-12 - _)
            else:
                scale = False
    if scale:
        A *= scale

    # Numerous accounts of SVD algorithms using gesdd results
    # in poor results when min(M, N) >= 26 (block size).
    # This may be an error in the D&C algorithm.
    # Here we resort to precision over time, but user may decide.
    driver = kwargs.get("driver", "gesvd").lower()
    if driver in ("arpack", "lobpcg", "sparse"):
        if driver == "sparse":
            driver = "arpack"  # scipy default

        # filter out keys for scipy.sparse.svds
        svds_kwargs = {
            key: kwargs[key] for key in ("k", "ncv", "tol", "v0") if key in kwargs
        }
        # do not calculate vt
        svds_kwargs["return_singular_vectors"] = "u"
        svds_kwargs["solver"] = driver
        if "k" not in svds_kwargs:
            k = A.shape[1] // 2
            if k < 3:
                k = A.shape[1] - 1
            svds_kwargs["k"] = k

        A, DOS, _ = svds(A, **svds_kwargs)

    else:
        # it must be a lapack driver:
        A, DOS, _ = svd_destroy(A, full_matrices=False, lapack_driver=driver)
    del _
    if scale:
        DOS /= scale

    # A note of caution.
    # The DOS values are not actual DOS values.
    # In fact the DOS should be calculated as:
    #   DOS * <i| S(k) |i>
    # to account for the overlap matrix. For orthogonal basis sets
    # this DOS eigenvalue is correct.
    return DOS**2 / (2 * np.pi), A


@set_module("sisl_toolbox.btd")
class PivotSelfEnergy(si.physics.SelfEnergy):
    """Container for the self-energy object

    This may either be a `tbtsencSileTBtrans`, a `tbtgfSileTBtrans` or a sisl.SelfEnergy objectfile
    """

    def __init__(self, name, se, pivot=None):
        # Name of electrode
        self.name = name

        # File containing the self-energy
        # This may be either of:
        #  tbtsencSileTBtrans
        #  tbtgfSileTBtrans
        #  SelfEnergy object (for direct calculation)
        self._se = se

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

        # Store the pivoting for faster indexing
        if pivot is None:
            if not isinstance(se, si.io.tbtrans.tbtsencSileTBtrans):
                raise ValueError(
                    f"{self.__class__.__name__} must be passed a sisl.io.tbtrans.tbtsencSileTBtrans. "
                    "Otherwise use the DownfoldSelfEnergy method with appropriate arguments."
                )
            pivot = se

        # Pivoting indices for the self-energy for the device region
        # but with respect to the full system size
        self.pvt = pivot.pivot(name).reshape(-1, 1)

        # Pivoting indices for the self-energy for the device region
        # but with respect to the device region only
        self.pvt_dev = pivot.pivot(name, in_device=True).reshape(-1, 1)

        # the pivoting in the downfolding region (with respect to the full
        # system size)
        self.pvt_down = pivot.pivot_down(name).reshape(-1, 1)

        # Retrieve BTD matrices for the corresponding electrode
        self.btd = pivot.btd(name)

        # Get the individual matrices
        cbtd = np.cumsum(self.btd)
        pvt_btd = []
        o = 0
        for i in cbtd:
            # collect the pivoting indices for the downfolding
            pvt_btd.append(self.pvt_down[o:i, 0])
            o += i
        # self.pvt_btd = np.concatenate(pvt_btd).reshape(-1, 1)
        # self.pvt_btd_sort = arangei(o)

        self._se_func = se_func
        self._broad_func = broad_func

    def __str__(self):
        return f"{self.__class__.__name__}{{no: {len(self)}}}"

    def __len__(self):
        return len(self.pvt_dev)

    def self_energy(self, *args, **kwargs):
        return self._se_func(*args, **kwargs)

    def broadening_matrix(self, *args, **kwargs):
        return self._broad_func(*args, **kwargs)


@set_module("sisl_toolbox.btd")
class DownfoldSelfEnergy(PivotSelfEnergy):
    def __init__(
        self, name, se, pivot, Hdevice, eta_device=0, bulk=True, bloch=(1, 1, 1)
    ):
        super().__init__(name, se, pivot)

        if np.allclose(bloch, 1):

            def _bloch(func, k, *args, **kwargs):
                return func(*args, k=k, **kwargs)

            self._bloch = _bloch
        else:
            self._bloch = si.Bloch(bloch)

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
        self._data.bulk = bulk

        # Retain the device for only the downfold region
        # a_down is sorted!
        a_elec = pivot.a_elec(self.name)

        # Now figure out all the atoms in the downfolding region
        # pivot_down is the electrode + all orbitals including the orbitals
        # reaching into the device
        pivot_down = pivot.pivot_down(self.name)
        # note that the last orbitals in pivot_down is the returned self-energies
        # that we want to calculate in this class

        geometry = pivot.geometry
        # Figure out the full device part of the downfolding region
        # this will still be sorted
        down_atoms = geometry.o2a(pivot_down, unique=True).astype(np.int32, copy=False)
        # this will also be sorted
        down_orbitals = geometry.a2o(down_atoms, all=True).astype(np.int32, copy=False)

        # The orbital indices in self.H.device.geometry
        # which transfers the orbitals to the downfolding region

        # Now we need to figure out the pivoting indices from the sub-set
        # geometry

        self._data.H = PropertyDict()
        self._data.H.electrode = se.spgeom0
        self._data.H.device = Hdevice.sub(down_atoms)
        # geometry_down = self._data.H.device.geometry

        # Now we retain the positions of the electrode orbitals in the
        # non pivoted structure for inserting the self-energy
        # Once the down-folded matrix is formed we can pivot it
        # in the BTD class
        # The self-energy is inserted in the non-pivoted matrix
        o_elec = geometry.a2o(a_elec, all=True).astype(np.int32, copy=False)
        pvt = indices(down_orbitals, o_elec)
        self._data.elec = pvt[pvt >= 0].reshape(-1, 1)
        pvt = indices(down_orbitals, pivot_down)
        self._data.dev = pvt[pvt >= 0].reshape(-1, 1)

        # Create BTD indices
        self._data.cumbtd = np.append(0, np.cumsum(self.btd))

    def __str__(self):
        eta = None
        try:
            eta = self._se.eta
        except Exception:
            pass
        se = str(self._se).replace("\n", "\n ")
        return f"{self.__class__.__name__}{{no: {len(self)}, blocks: {len(self.btd)}, eta: {eta}, eta_device: {self._eta_device},\n {se}\n}}"

    def __len__(self):
        return len(self._data.dev)

    def _check_Ek(self, E, k):
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

    def _prepare(self, E, k=(0, 0, 0)):
        if self._check_Ek(E, k):
            return

        # Prepare the matrices
        data = self._data
        H = data.H

        Ed = data.Ed
        Eb = data.Eb
        data.SeH = H.device.Sk(k, dtype=np.complex128) * Ed - H.device.Hk(
            k, dtype=np.complex128
        )
        if data.bulk:

            def hsk(k, **kwargs):
                # constructor for the H and S part
                return H.electrode.Sk(k, **kwargs) * Eb - H.electrode.Hk(k, **kwargs)

            data.SeH[data.elec, data.elec.T] = self._bloch(
                hsk, k, format="array", dtype=np.complex128
            )

    def self_energy(self, E, k=(0, 0, 0), *args, **kwargs):
        self._prepare(E, k)
        data = self._data
        se = self._bloch(super().self_energy, k, *args, E=E, **kwargs)

        # now put it in the matrix
        M = data.SeH.copy()
        M[data.elec, data.elec.T] -= se

        # transfer to BTD
        pvt = data.dev
        cbtd = data.cumbtd

        def gM(M, idx1, idx2):
            return M[pvt[idx1], pvt[idx2].T].toarray()

        Mr = 0
        sli = slice(cbtd[0], cbtd[1])
        for b in range(1, len(self.btd)):
            sli1 = slice(cbtd[b], cbtd[b + 1])

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


@set_module("sisl_toolbox.btd")
class BlockMatrixIndexer:
    def __init__(self, bm):
        self._bm = bm

    def __len__(self):
        return len(self._bm.blocks)

    def __iter__(self):
        """Loop contained indices in the BlockMatrix"""
        yield from self._bm._M.keys()

    def __delitem__(self, key):
        if not isinstance(key, tuple):
            raise ValueError(
                f"{self.__class__.__name__} index deletion must be done with a tuple."
            )
        del self._bm._M[key]

    def __contains__(self, key):
        if not isinstance(key, tuple):
            raise ValueError(
                f"{self.__class__.__name__} index checking must be done with a tuple."
            )
        return key in self._bm._M

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            raise ValueError(
                f"{self.__class__.__name__} index retrieval must be done with a tuple."
            )
        M = self._bm._M.get(key)
        if M is None:
            i, j = key
            # the data-type is probably incorrect.. :(
            return np.zeros([self._bm.blocks[i], self._bm.blocks[j]])
        return M

    def __setitem__(self, key, M):
        if not isinstance(key, tuple):
            raise ValueError(
                f"{self.__class__.__name__} index setting must be done with a tuple."
            )

        s = (self._bm.blocks[key[0]], self._bm.blocks[key[1]])
        assert (
            M.shape == s
        ), f"Could not assign matrix of shape {M.shape} into matrix of shape {s}"
        self._bm._M[key] = M


@set_module("sisl_toolbox.btd")
class BlockMatrix:
    """Container class that holds a block matrix"""

    def __init__(self, blocks):
        self._M = {}
        self._blocks = blocks

    @property
    def blocks(self):
        return self._blocks

    def toarray(self):
        BI = self.block_indexer
        nb = len(BI)
        # stack stuff together
        return np.concatenate(
            [np.concatenate([BI[i, j] for i in range(nb)], axis=0) for j in range(nb)],
            axis=1,
        )

    def tobtd(self):
        """Return only the block tridiagonal part of the matrix"""
        ret = self.__class__(self.blocks)
        sBI = self.block_indexer
        rBI = ret.block_indexer
        nb = len(sBI)
        for j in range(nb):
            for i in range(max(0, j - 1), min(j + 2, nb)):
                rBI[i, j] = sBI[i, j]
        return ret

    def tobd(self):
        """Return only the block diagonal part of the matrix"""
        ret = self.__class__(self.blocks)
        sBI = self.block_indexer
        rBI = ret.block_indexer
        nb = len(sBI)
        for i in range(nb):
            rBI[i, i] = sBI[i, i]
        return ret

    def diagonal(self):
        BI = self.block_indexer
        return np.concatenate([BI[b, b].diagonal() for b in range(len(BI))])

    @property
    def block_indexer(self):
        return BlockMatrixIndexer(self)


@set_module("sisl_toolbox.btd")
class DeviceGreen:
    r"""Block-tri-diagonal Green function calculator

    This class enables the extraction and calculation of some important
    quantities not currently accessible in TBtrans.

    For instance it may be used to calculate scattering states from
    the Green function.
    Once scattering states have been calculated one may also calculate
    the eigenchannels.

    Both calculations are very efficient and uses very little memory
    compared to the full matrices normally used.

    Consider a regular 2 electrode setup with transport direction
    along the 3rd lattice vector. Then the following example may
    be used to calculate the eigen-channels.

    The below short-form of reading all variables should cover most variables
    encountered in the FDF file.

    .. code-block:: python

       G = DeviceGreen.from_fdf("RUN.fdf")

       # Calculate the scattering state from the left electrode
       # and then the eigen channels to the right electrode
       state = G.scattering_state("Left", E=0.1)
       eig_channel = G.eigenchannel(state, "Right")

    The above ``DeviceGreen.from_fdf`` is a short-hand for something
    like the below (it actually does more than that, so prefer the `from_fdf`):

    .. code-block:: python

       import sisl
       from sisl_toolbox.btd import *
       # First read in the required data
       H_elec = sisl.Hamiltonian.read("ELECTRODE.nc")
       H = sisl.Hamiltonian.read("DEVICE.nc")
       # remove couplings along the self-energy direction
       # to ensure no fake couplings.
       H.set_nsc(c=1)

       # Read in a single tbtrans output which contains the BTD matrices
       # and instructs this class how it should pivot the matrix to obtain
       # a BTD matrix.
       tbt = sisl.get_sile("siesta.TBT.nc")

       # Define the self-energy calculators which will downfold the
       # self-energies into the device region.
       # Since a downfolding will be done it requires the device Hamiltonian.
       H_elec.shift(tbt.mu("Left"))
       left = DownfoldSelfEnergy("Left", s.RecursiveSI(H_elec, "-C", eta=tbt.eta("Left"),
                                 tbt, H)
       H_elec.shift(tbt.mu("Right"))
       left = DownfoldSelfEnergy("Right", s.RecursiveSI(H_elec, "+C", eta=tbt.eta("Right"),
                                 tbt, H)

       G = DeviceGreen(H, [left, right], tbt)


    Notes
    -----

    Currently one cannot use these classes to calculate the
    scattering-states/eigenchannels for the spin-down component of a polarized
    calculation. One has to explicitly remove the spin-up component of the Hamiltonians
    before doing the calculations.
    """

    # TODO we should speed this up by overwriting A with the inverse once
    #      calculated. We don't need it at that point.
    #      That would probably require us to use a method to retrieve
    #      the elements which determines if it has been calculated or not.

    def __init__(self, H, elecs, pivot, eta=0.0):
        """Create Green function with Hamiltonian and BTD matrix elements"""
        self.H = H

        # Store electrodes (for easy retrieval of the SE)
        # There may be no electrodes
        self.elecs = elecs
        # self.elecs_pvt = [pivot.pivot(el.name).reshape(-1, 1)
        #                  for el in elecs]
        self.elecs_pvt_dev = [
            pivot.pivot(el.name, in_device=True).reshape(-1, 1) for el in elecs
        ]

        self.pvt = pivot.pivot()
        self.btd = pivot.btd()

        # global device eta
        self.eta = eta

        # Create BTD indices
        self.btd_cum0 = np.empty([len(self.btd) + 1], dtype=self.btd.dtype)
        self.btd_cum0[0] = 0
        self.btd_cum0[1:] = np.cumsum(self.btd)
        self.reset()

    def __str__(self):
        ret = f"{self.__class__.__name__}{{no: {len(self)}, blocks: {len(self.btd)}, eta: {self.eta:.3e}"
        for elec in self.elecs:
            e = str(elec).replace("\n", "\n  ")
            ret = f"{ret},\n {elec.name}:\n  {e}"
        return f"{ret}\n}}"

    @classmethod
    def from_fdf(cls, fdf, prefix="TBT", use_tbt_se=False, eta=None):
        """Return a new `DeviceGreen` using information gathered from the fdf

        Parameters
        ----------
        fdf : str or fdfSileSiesta
           fdf file to read the parameters from
        prefix : {"TBT", "TS"}
           which prefix to use, if TBT it will prefer TBT prefix, but fall back
           to TS prefixes.
           If TS, only these prefixes will be used.
        use_tbt_se : bool, optional
           whether to use the TBT.SE.nc files for self-energies
           or calculate them on the fly.
        """
        if not isinstance(fdf, si.BaseSile):
            fdf = si.io.siesta.fdfSileSiesta(fdf)

        # Now read the values needed
        slabel = fdf.get("SystemLabel", "siesta")
        # Test if the TBT output file exists:
        tbt = None
        for end in ("TBT.nc", "TBT_UP.nc", "TBT_DN.nc"):
            if Path(f"{slabel}.{end}").is_file():
                tbt = f"{slabel}.{end}"
        if tbt is None:
            raise FileNotFoundError(
                f"{cls.__name__}.from_fdf could "
                f"not find file {slabel}.[TBT|TBT_UP|TBT_DN].nc"
            )
        tbt = si.get_sile(tbt)
        is_tbtrans = prefix.upper() == "TBT"

        # Read the device H, only valid for TBT stuff
        for hs_ext in ("TS.HSX", "TSHS", "HSX"):
            if Path(f"{slabel}.{hs_ext}").exists():
                # choose a sane default (if it exists!)
                hs_default = f"{slabel}.{hs_ext}"
                break
        else:
            hs_default = f"{slabel}.TSHS"
        Hdev = si.get_sile(fdf.get("TBT.HS", hs_default)).read_hamiltonian()

        def get_line(line):
            """Parse lines in the %block constructs of fdf's"""
            key, val = line.split(" ", 1)
            return key.lower().strip(), val.split("#", 1)[0].strip()

        def read_electrode(elec_prefix):
            """Parse the electrode information and return a dictionary with content"""
            from sisl.unit.siesta import unit_convert

            ret = PropertyDict()

            if is_tbtrans:

                def block_get(dic, key, default=None, unit=None):
                    ret = dic.get(f"tbt.{key}", dic.get(key, default))
                    if unit is None or not isinstance(ret, str):
                        return ret
                    ret, un = ret.split()
                    return float(ret) * unit_convert(un, unit)

            else:

                def block_get(dic, key, default=None, unit=None):
                    ret = dic.get(key, default)
                    if unit is None or not isinstance(ret, str):
                        return ret
                    ret, un = ret.split()
                    return float(ret) * unit_convert(un, unit)

            tbt_prefix = f"TBT.{elec_prefix}"
            ts_prefix = f"TS.{elec_prefix}"

            block = fdf.get(f"{ts_prefix}")
            Helec = fdf.get(f"{ts_prefix}.HS")
            bulk = fdf.get("TS.Elecs.Bulk", True)
            eta = fdf.get("TS.Elecs.Eta", 1e-3, unit="eV")
            bloch = [1, 1, 1]
            for i in range(3):
                bloch[i] = fdf.get(f"{ts_prefix}.Bloch.A{i+1}", 1)
            if is_tbtrans:
                block = fdf.get(f"{tbt_prefix}", block)
                Helec = fdf.get(f"{tbt_prefix}.HS", Helec)
                bulk = fdf.get("TBT.Elecs.Bulk", bulk)
                eta = fdf.get("TBT.Elecs.Eta", eta, unit="eV")
                for i in range(3):
                    bloch[i] = fdf.get(f"{tbt_prefix}.Bloch.A{i+1}", bloch[i])

            # Convert to key value based function
            dic = {key: val for key, val in map(get_line, block)}

            # Retrieve data
            for key in ("hs", "hs-file", "tshs", "tshs-file"):
                Helec = block_get(dic, key, Helec)
            if Helec:
                Helec = si.get_sile(Helec).read_hamiltonian()
            else:
                raise ValueError(
                    f"{cls.__name__}.from_fdf could not find "
                    f"electrode HS in block: {prefix} ??"
                )

            # Get semi-infinite direction
            semi_inf = None
            for suf in ("-direction", "-dir", ""):
                semi_inf = block_get(dic, f"semi-inf{suf}", semi_inf)
            if semi_inf is None:
                raise ValueError(
                    f"{cls.__name__}.from_fdf could not find "
                    f"electrode semi-inf-direction in block: {prefix} ??"
                )
            # convert to sisl infinite
            semi_inf = semi_inf.lower()
            semi_inf = semi_inf[0] + {"a1": "a", "a2": "b", "a3": "c"}.get(
                semi_inf[1:], semi_inf[1:]
            )
            # Check that semi_inf is a recursive one!
            if semi_inf not in ("-a", "+a", "-b", "+b", "-c", "+c"):
                raise NotImplementedError(
                    f"{cls.__name__} does not implement other "
                    "self energies than the recursive one."
                )

            bulk = bool(block_get(dic, "bulk", bulk))
            # loop for 0
            for i, sufs in enumerate([("a", "a1"), ("b", "a2"), ("c", "a3")]):
                for suf in sufs:
                    bloch[i] = block_get(dic, f"bloch-{suf}", bloch[i])

            bloch = [
                int(b)
                for b in block_get(
                    dic, "bloch", f"{bloch[0]} {bloch[1]} {bloch[2]}"
                ).split()
            ]

            ret.eta = block_get(dic, "eta", eta, unit="eV")
            # manual shift of the fermi-level
            dEf = block_get(dic, "delta-Ef", 0.0, unit="eV")
            # shift electronic structure here, we store it in the returned
            # dictionary, for information, but it shouldn't be used.
            Helec.shift(dEf)
            ret.dEf = dEf
            # add a fraction of the bias in the coupling elements of the
            # E-C region, only meaningful for
            ret.V_fraction = block_get(dic, "V-fraction", 0.0)
            if ret.V_fraction > 0.0:
                warn(
                    f"{cls.__name__}.from_fdf(electrode={elec}) found a non-zero V-fraction value. "
                    "This is currently not implemented."
                )
            ret.Helec = Helec
            ret.bloch = bloch
            ret.semi_inf = semi_inf
            ret.bulk = bulk
            return ret

        # Loop electrodes and read in and construct data
        if isinstance(use_tbt_se, bool):
            if use_tbt_se:
                use_tbt_se = tbt.elecs
            else:
                use_tbt_se = []
        elif isinstance(use_tbt_se, str):
            use_tbt_se = [use_tbt_se]

        elec_data = {}
        eta_dev = 1e123
        for elec in tbt.elecs:
            # read from the block
            data = read_electrode(f"Elec.{elec}")
            elec_data[elec] = data

            # read from the TBT file (to check if the user has changed the input file)
            elec_eta = tbt.eta(elec)
            if not np.allclose(elec_eta, data.eta):
                warn(
                    f"{cls.__name__}.from_fdf(electrode={elec}) found inconsistent "
                    f"imaginary eta from the fdf vs. TBT output, will use fdf value.\n"
                    f"  {tbt} = {eta} eV\n  {fdf} = {data.eta} eV"
                )

            bloch = tbt.bloch(elec)
            if not np.allclose(bloch, data.bloch):
                warn(
                    f"{cls.__name__}.from_fdf(electrode={elec}) found inconsistent "
                    f"Bloch expansions from the fdf vs. TBT output, will use fdf value.\n"
                    f"  {tbt} = {bloch}\n  {fdf} = {data.bloch}"
                )

            eta_dev = min(data.eta, eta_dev)

        # Correct by a factor 1/10 to minimize smearing for device states.
        # We want the electrode to smear.
        eta_dev /= 10

        # Now we can estimate the device eta value.
        # It is based on the electrode values
        eta_dev_tbt = tbt.eta()
        if is_tbtrans:
            eta_dev = fdf.get("TBT.Contours.Eta", eta_dev, unit="eV")
        else:
            eta_dev = fdf.get("TS.Contours.nEq.Eta", eta_dev, unit="eV")

        if eta is not None:
            # use passed option
            eta_dev = eta

        elif not np.allclose(eta_dev, eta_dev_tbt):
            warn(
                f"{cls.__name__}.from_fdf found inconsistent "
                f"imaginary eta from the fdf vs. TBT output, will use fdf value.\n"
                f"  {tbt} = {eta_dev_tbt} eV\n  {fdf} = {eta_dev} eV"
            )

        elecs = []
        for elec in tbt.elecs:
            mu = tbt.mu(elec)
            data = elec_data[elec]

            if elec in use_tbt_se:
                if Path(f"{slabel}.TBT.SE.nc").is_file():
                    tbtse = si.get_sile(f"{slabel}.TBT.SE.nc")
                else:
                    raise FileNotFoundError(
                        f"{cls.__name__}.from_fdf "
                        f"could not find file {slabel}.TBT.SE.nc "
                        "but it was requested by 'use_tbt_se'!"
                    )

            # shift according to potential
            data.Helec.shift(mu)
            data.mu = mu
            se = si.RecursiveSI(data.Helec, data.semi_inf, eta=data.eta)

            # Limit connections of the device along the semi-inf directions
            # TODO check whether there are systems where it is important
            # we do all set_nsc before passing it for each electrode.
            kw = {"abc"[se.semi_inf]: 1}
            Hdev.set_nsc(**kw)

            if elec in use_tbt_se:
                elec_se = PivotSelfEnergy(elec, tbtse)
            else:
                elec_se = DownfoldSelfEnergy(
                    elec,
                    se,
                    tbt,
                    Hdev,
                    eta_device=eta_dev,
                    bulk=data.bulk,
                    bloch=data.bloch,
                )

            elecs.append(elec_se)

        return cls(Hdev, elecs, tbt, eta_dev)

    def reset(self):
        """Clean any memory used by this object"""
        self._data = PropertyDict()

    def __len__(self):
        return len(self.pvt)

    def _elec(self, elec):
        """Convert a string electrode to the proper linear index"""
        if isinstance(elec, str):
            for iel, el in enumerate(self.elecs):
                if el.name == elec:
                    return iel
        elif isinstance(elec, PivotSelfEnergy):
            return self._elec(elec.name)
        return elec

    def _elec_name(self, elec):
        """Convert an electrode index or str to the name of the electrode"""
        if isinstance(elec, str):
            return elec
        elif isinstance(elec, PivotSelfEnergy):
            return elec.name
        return self.elecs[elec].name

    def _check_Ek(self, E, k):
        if hasattr(self._data, "E"):
            if np.allclose(self._data.E, E) and np.allclose(self._data.k, k):
                # we have already prepared the calculation
                return True

        # while resetting is not necessary, it can
        # save a lot of memory since some arrays are not
        # temporarily stored twice.
        self.reset()
        self._data.E = E
        self._data.Ec = E
        if np.isrealobj(E):
            self._data.Ec = E + 1j * self.eta
        self._data.k = np.asarray(k, dtype=np.float64)

        return False

    def _prepare_se(self, E, k):
        if self._check_Ek(E, k):
            return
        E = self._data.E
        k = self._data.k

        # Create all self-energies (and store the Gamma's)
        se = []
        gamma = []
        for elec in self.elecs:
            # Insert values
            SE = elec.self_energy(E, k)
            se.append(SE)
            gamma.append(elec.se2broadening(SE))
        self._data.se = se
        self._data.gamma = gamma

    def _prepare_tgamma(self, E, k, cutoff):
        if self._check_Ek(E, k) and hasattr(self._data, "tgamma"):
            if abs(cutoff - self._data.tgamma_cutoff) < 1e-13:
                return

        # ensure we have the self-energies
        self._prepare_se(E, k)

        # Get the sqrt of the level broadening matrix
        def eigh_sqrt(gam, cutoff):
            eig, U = eigh(gam)
            idx = (eig >= cutoff).nonzero()[0]
            eig = np.emath.sqrt(eig[idx])
            return eig * U.T[idx].T

        tgamma = []
        for gam in self._data.gamma:
            tgamma.append(eigh_sqrt(gam, cutoff))

        self._data.tgamma = tgamma
        self._data.tgamma_cutoff = cutoff

    def _prepare(self, E, k):
        if self._check_Ek(E, k) and hasattr(self._data, "A"):
            return

        data = self._data
        E = data.E
        # device region: E + 1j*eta
        Ec = data.Ec
        k = data.k

        # Prepare the Green function calculation
        inv_G = self.H.Sk(k, dtype=np.complex128) * Ec - self.H.Hk(
            k, dtype=np.complex128
        )

        # Now reduce the sparse matrix to the device region (plus do the pivoting)
        inv_G = inv_G[self.pvt, :][:, self.pvt]

        # Create all self-energies (and store the Gamma's)
        gamma = []
        for elec in self.elecs:
            # Insert values
            SE = elec.self_energy(E, k)
            inv_G[elec.pvt_dev, elec.pvt_dev.T] -= SE
            gamma.append(elec.se2broadening(SE))
        del SE
        data.gamma = gamma

        nb = len(self.btd)
        nbm1 = nb - 1

        # Now we have all needed to calculate the inverse parts of the Green function
        A = [None] * nb
        B = [0] * nb
        C = [0] * nb

        # Now we can calculate everything
        cbtd = self.btd_cum0

        sl0 = slice(cbtd[0], cbtd[1])
        slp = slice(cbtd[1], cbtd[2])
        # initial matrix A and C
        iG = inv_G[sl0, :].tocsc()
        A[0] = iG[:, sl0].toarray()
        C[1] = iG[:, slp].toarray()
        for b in range(1, nbm1):
            # rotate slices
            sln = sl0
            sl0 = slp
            slp = slice(cbtd[b + 1], cbtd[b + 2])
            iG = inv_G[sl0, :].tocsc()

            B[b - 1] = iG[:, sln].toarray()
            A[b] = iG[:, sl0].toarray()
            C[b + 1] = iG[:, slp].toarray()
        # and final matrix A and B
        iG = inv_G[slp, :].tocsc()
        A[nbm1] = iG[:, slp].toarray()
        B[nbm1 - 1] = iG[:, sl0].toarray()

        # clean-up, not used anymore
        del inv_G

        data.A = A
        data.B = B
        data.C = C

        # Now do propagation forward, tilde matrices
        tX = [0] * nb
        tY = [0] * nb
        # \tilde Y
        tY[1] = solve(A[0], C[1])
        # \tilde X
        tX[-2] = solve(A[-1], B[-2])
        for n in range(2, nb):
            p = nb - n - 1
            # \tilde Y
            tY[n] = solve(A[n - 1] - B[n - 2] @ tY[n - 1], C[n], overwrite_a=True)
            # \tilde X
            tX[p] = solve(A[p + 1] - C[p + 2] @ tX[p + 1], B[p], overwrite_a=True)

        data.tX = tX
        data.tY = tY

    def _matrix_to_btd(self, M):
        BM = BlockMatrix(self.btd)
        BI = BM.block_indexer
        c = self.btd_cum0
        nb = len(BI)
        if ssp.issparse(M):
            for jb in range(nb):
                for ib in range(max(0, jb - 1), min(jb + 2, nb)):
                    BI[ib, jb] = M[c[ib] : c[ib + 1], c[jb] : c[jb + 1]].toarray()
        else:
            for jb in range(nb):
                for ib in range(max(0, jb - 1), min(jb + 2, nb)):
                    BI[ib, jb] = M[c[ib] : c[ib + 1], c[jb] : c[jb + 1]]
        return BM

    def Sk(self, *args, **kwargs):
        is_btd = False
        if "format" in kwargs:
            if kwargs["format"].lower() == "btd":
                is_btd = True
                del kwargs["format"]

        pvt = self.pvt.reshape(-1, 1)
        M = self.H.Sk(*args, **kwargs)[pvt, pvt.T]
        if is_btd:
            return self._matrix_to_btd(M)
        return M

    def Hk(self, *args, **kwargs):
        is_btd = False
        if "format" in kwargs:
            if kwargs["format"].lower() == "btd":
                is_btd = True
                del kwargs["format"]

        pvt = self.pvt.reshape(-1, 1)
        M = self.H.Hk(*args, **kwargs)[pvt, pvt.T]
        if is_btd:
            return self._matrix_to_btd(M)
        return M

    def _get_blocks(self, idx):
        # Figure out which blocks are requested
        block1 = (idx.min() < self.btd_cum0[1:]).nonzero()[0][0]
        block2 = (idx.max() < self.btd_cum0[1:]).nonzero()[0][0]
        if block1 == block2:
            blocks = [block1]
        else:
            blocks = [b for b in range(block1, block2 + 1)]
        return blocks

    def green(self, E, k=(0, 0, 0), format="array"):
        r"""Calculate the Green function for a given `E` and `k` point

        The Green function is calculated as:

        .. math::
            \mathbf G(E,\mathbf k) = \big[\mathbf S(\mathbf k) E - \mathbf H(\mathbf k)
                  - \sum \boldsymbol \Sigma(E,\mathbf k)\big]^{-1}

        Parameters
        ----------
        E : float
           the energy to calculate at, may be a complex value.
        k : array_like, optional
           k-point to calculate the Green function at
        format : {"array", "btd", "bm", "bd", "sparse"}
           return the matrix in a specific format

           - array: a regular numpy array (full matrix)
           - btd: a block-matrix object with only the diagonals and first off-diagonals
           - bm: a block-matrix object with diagonals and all off-diagonals
           - bd: a block-matrix object with only diagonals (no off-diagonals)
           - sparse: a sparse-csr matrix for the sparse elements as found in the Hamiltonian
        """
        self._prepare(E, k)
        format = format.lower()
        if format == "dense":
            format = "array"
        func = getattr(self, f"_green_{format}", None)
        if func is None:
            raise ValueError(
                f"{self.__class__.__name__}.green format not valid input [array|sparse|bm|btd|bd]"
            )
        return func()

    def _green_array(self):
        """Calculate the Green function on a full np.array matrix"""
        n = len(self.pvt)
        G = np.empty([n, n], dtype=self._data.A[0].dtype)

        btd = self.btd
        nb = len(btd)
        nbm1 = nb - 1
        sumbs = 0
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b, bs in enumerate(btd):
            sl0 = slice(sumbs, sumbs + bs)

            # Calculate diagonal part
            if b == 0:
                G[sl0, sl0] = inv_destroy(A[b] - C[b + 1] @ tX[b])
            elif b == nbm1:
                G[sl0, sl0] = inv_destroy(A[b] - B[b - 1] @ tY[b])
            else:
                G[sl0, sl0] = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])

            # Do above
            next_sum = sumbs
            slp = sl0
            for a in range(b - 1, -1, -1):
                # Calculate all parts above
                sla = slice(next_sum - btd[a], next_sum)
                G[sla, sl0] = -tY[a + 1] @ G[slp, sl0]
                slp = sla
                next_sum -= btd[a]

            sl0 = slice(sumbs, sumbs + bs)

            # Step block
            sumbs += bs

            # Do below
            next_sum = sumbs
            slp = sl0
            for a in range(b + 1, nb):
                # Calculate all parts above
                sla = slice(next_sum, next_sum + btd[a])
                G[sla, sl0] = -tX[a - 1] @ G[slp, sl0]
                slp = sla
                next_sum += btd[a]

        return G

    def _green_btd(self):
        """Calculate the Green function only in the BTD matrix elements.

        Stored in a `BlockMatrix` class."""
        G = BlockMatrix(self.btd)
        BI = G.block_indexer
        nb = len(BI)
        nbm1 = nb - 1
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b in range(nb):
            # Calculate diagonal part
            if b == 0:
                G11 = inv_destroy(A[b] - C[b + 1] @ tX[b])
            elif b == nbm1:
                G11 = inv_destroy(A[b] - B[b - 1] @ tY[b])
            else:
                G11 = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])

            BI[b, b] = G11
            # do above
            if b > 0:
                BI[b - 1, b] = -tY[b] @ G11
            # do below
            if b < nbm1:
                BI[b + 1, b] = -tX[b] @ G11

        return G

    def _green_bm(self):
        """Calculate the full Green function.

        Stored in a `BlockMatrix` class."""
        G = self._green_btd()
        BI = G.block_indexer
        nb = len(BI)
        nbm1 = nb - 1

        tX = self._data.tX
        tY = self._data.tY
        for b in range(nb):
            G0 = BI[b, b]
            for bb in range(b, 0, -1):
                G0 = -tY[bb] @ G0
                BI[bb - 1, b] = G0
            G0 = BI[b, b]
            for bb in range(b, nbm1):
                G0 = -tX[bb] @ G0
                BI[bb + 1, b] = G0

        return G

    def _green_bd(self):
        """Calculate the Green function only along the diagonal block matrices.

        Stored in a `BlockMatrix` class."""
        G = BlockMatrix(self.btd)
        BI = G.block_indexer
        nb = len(BI)
        nbm1 = nb - 1

        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        BI[0, 0] = inv_destroy(A[0] - C[1] @ tX[0])
        for b in range(1, nbm1):
            # Calculate diagonal part
            BI[b, b] = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])
        BI[nbm1, nbm1] = inv_destroy(A[nbm1] - B[nbm1 - 1] @ tY[nbm1])

        return G

    def _green_sparse(self):
        """Calculate the Green function only in where the sparse H and S are non-zero.

        Stored in a `scipy.sparse.csr_matrix` class."""
        # create a sparse matrix
        G = self.H.Sk(format="csr", dtype=self._data.A[0].dtype)
        # pivot the matrix
        G = G[self.pvt, :][:, self.pvt]

        # Get row and column entries
        ncol = np.diff(G.indptr)
        row = (ncol > 0).nonzero()[0]
        # Now we have [0 0 0 0 1 1 1 1 2 2 ... no-1 no-1]
        row = np.repeat(row.astype(np.int32, copy=False), ncol[row])
        col = G.indices

        def get_idx(row, col, row_b, col_b=None):
            if col_b is None:
                col_b = row_b
            idx = (row_b[0] <= row).nonzero()[0]
            idx = idx[row[idx] < row_b[1]]
            idx = idx[col_b[0] <= col[idx]]
            return idx[col[idx] < col_b[1]]

        btd = self.btd
        nb = len(btd)
        nbm1 = nb - 1
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        sumbsn, sumbs, sumbsp = 0, 0, 0
        for b, bs in enumerate(btd):
            sumbsp = sumbs + bs
            if b < nbm1:
                bsp = btd[b + 1]

            # Calculate diagonal part
            if b == 0:
                GM = inv_destroy(A[b] - C[b + 1] @ tX[b])
            elif b == nbm1:
                GM = inv_destroy(A[b] - B[b - 1] @ tY[b])
            else:
                GM = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])

            # get all entries where G is non-zero
            idx = get_idx(row, col, (sumbs, sumbsp))
            G.data[idx] = GM[row[idx] - sumbs, col[idx] - sumbs]

            # check if we should do block above
            if b > 0:
                idx = get_idx(row, col, (sumbsn, sumbs), (sumbs, sumbsp))
                if len(idx) > 0:
                    G.data[idx] = -(tY[b] @ GM)[row[idx] - sumbsn, col[idx] - sumbs]

            # check if we should do block below
            if b < nbm1:
                idx = get_idx(row, col, (sumbsp, sumbsp + bsp), (sumbs, sumbsp))
                if len(idx) > 0:
                    G.data[idx] = -(tX[b] @ GM)[row[idx] - sumbsp, col[idx] - sumbs]

            bsn = bs
            sumbsn = sumbs
            sumbs += bs

        return G

    def _green_diag_block(self, idx):
        """Calculate the Green function only on specific (neighboring) diagonal block matrices.

        Stored in a `np.array` class."""
        nb = len(self.btd)
        nbm1 = nb - 1

        # Find parts we need to calculate
        blocks = self._get_blocks(idx)
        assert (
            len(blocks) <= 2
        ), f"{self.__class__.__name__} green(diagonal) requires maximally 2 blocks"
        if len(blocks) == 2:
            assert (
                blocks[0] + 1 == blocks[1]
            ), f"{self.__class__.__name__} green(diagonal) requires spanning only 2 blocks"

        n = self.btd[blocks].sum()
        G = np.empty([n, len(idx)], dtype=self._data.A[0].dtype)

        btd = self.btd
        c = self.btd_cum0
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b in blocks:
            # Find the indices in the block
            i = idx[c[b] <= idx]
            i = i[i < c[b + 1]].astype(np.int32)

            b_idx = indices_only(arangei(c[b], c[b + 1]), i)

            if b == blocks[0]:
                sl = slice(0, btd[b])
                c_idx = arangei(len(b_idx))
            else:
                sl = slice(btd[blocks[0]], btd[blocks[0]] + btd[b])
                c_idx = arangei(len(idx) - len(b_idx), len(idx))

            if b == 0:
                G[sl, c_idx] = inv_destroy(A[b] - C[b + 1] @ tX[b])[:, b_idx]
            elif b == nbm1:
                G[sl, c_idx] = inv_destroy(A[b] - B[b - 1] @ tY[b])[:, b_idx]
            else:
                G[sl, c_idx] = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])[
                    :, b_idx
                ]

            if len(blocks) == 1:
                break

            # Now calculate the thing (below/above)
            if b == blocks[0]:
                # Calculate below
                slp = slice(btd[b], btd[b] + btd[blocks[1]])
                G[slp, c_idx] = -tX[b] @ G[sl, c_idx]
            else:
                # Calculate above
                slp = slice(0, btd[blocks[0]])
                G[slp, c_idx] = -tY[b] @ G[sl, c_idx]

        return blocks, G

    def _green_column(self, idx):
        """Calculate the full Green function column for a subset of columns.

        Stored in a `np.array` class."""
        # To calculate the full Gf for specific column indices
        # These indices should maximally be spanning 2 blocks
        nb = len(self.btd)
        nbm1 = nb - 1

        # Find parts we need to calculate
        blocks = self._get_blocks(idx)
        assert (
            len(blocks) <= 2
        ), f"{self.__class__.__name__}.green(column) requires maximally 2 blocks"
        if len(blocks) == 2:
            assert (
                blocks[0] + 1 == blocks[1]
            ), f"{self.__class__.__name__}.green(column) requires spanning only 2 blocks"

        n = len(self)
        G = np.empty([n, len(idx)], dtype=self._data.A[0].dtype)

        c = self.btd_cum0
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b in blocks:
            # Find the indices in the block
            i = idx[c[b] <= idx]
            i = i[i < c[b + 1]].astype(np.int32)

            b_idx = indices_only(arangei(c[b], c[b + 1]), i)

            if b == blocks[0]:
                c_idx = arangei(len(b_idx))
            else:
                c_idx = arangei(len(idx) - len(b_idx), len(idx))

            sl = slice(c[b], c[b + 1])
            if b == 0:
                G[sl, c_idx] = inv_destroy(A[b] - C[b + 1] @ tX[b])[:, b_idx]
            elif b == nbm1:
                G[sl, c_idx] = inv_destroy(A[b] - B[b - 1] @ tY[b])[:, b_idx]
            else:
                G[sl, c_idx] = inv_destroy(A[b] - B[b - 1] @ tY[b] - C[b + 1] @ tX[b])[
                    :, b_idx
                ]

            if len(blocks) == 1:
                break

            # Now calculate above/below

            sl = slice(c[b], c[b + 1])
            if b == blocks[0] and b < nb - 1:
                # Calculate below
                slp = slice(c[b + 1], c[b + 2])
                G[slp, c_idx] = -tX[b] @ G[sl, c_idx]
            elif b > 0:
                # Calculate above
                slp = slice(c[b - 1], c[b])
                G[slp, c_idx] = -tY[b] @ G[sl, c_idx]

        # Now we can calculate the Gf column above
        b = blocks[0]
        slp = slice(c[b], c[b + 1])
        for b in range(blocks[0] - 1, -1, -1):
            sl = slice(c[b], c[b + 1])
            G[sl, :] = -tY[b + 1] @ G[slp, :]
            slp = sl

        # All blocks below
        b = blocks[-1]
        slp = slice(c[b], c[b + 1])
        for b in range(blocks[-1] + 1, nb):
            sl = slice(c[b], c[b + 1])
            G[sl, :] = -tX[b - 1] @ G[slp, :]
            slp = sl

        return G

    def spectral(
        self,
        elec,
        E,
        k=(0, 0, 0),
        format: str = "array",
        method: str = "column",
        herm: bool = True,
    ):
        r"""Calculate the spectral function for a given `E` and `k` point from a given electrode

        The spectral function is calculated as:

        .. math::
            \mathbf A_{\mathfrak{e}}(E,\mathbf k) = \mathbf G(E,\mathbf k)\boldsymbol\Gamma_{\mathfrak{e}}(E,\mathbf k)
                   \mathbf G^\dagger(E,\mathbf k)

        Parameters
        ----------
        elec : str or int
           the electrode to calculate the spectral function from
        E : float
           the energy to calculate at, may be a complex value.
        k : array_like, optional
           k-point to calculate the spectral function at
        format : {"array", "btd", "bm", "bd"}
           return the matrix in a specific format

           - array: a regular numpy array (full matrix)
           - btd: a block-matrix object with only the diagonals and first off-diagonals
           - bm: a block-matrix object with diagonals and all off-diagonals
           - bd: same as btd, since they are already calculated
        method : {"column", "propagate"}
           which method to use for calculating the spectral function.
           Depending on the size of the BTD blocks one may be faster than the
           other. For large systems you are recommended to time the different methods
           and stick with the fastest one, they are numerically identical.
        herm:
           The spectral function is a Hermitian matrix, by default (True), the methods
           that can utilize the Hermitian property only calculates the lower triangular
           part of :math:`\mathbf A`, and then copies the Hermitian to the upper part.
           By setting this to `False` the entire matrix is explicitly calculated.
        """
        # the herm flag is considered useful for testing, there is no need to
        # play with it. So it isn't documented.

        elec = self._elec(elec)
        self._prepare(E, k)
        format = format.lower()
        method = method.lower()
        if format == "dense":
            format = "array"
        elif format == "bd":
            # the bd also returns the off-diagonal ones since
            # they are needed to calculate the diagonal terms anyway.
            format = "btd"
        func = getattr(self, f"_spectral_{method}_{format}", None)
        if func is None:
            raise ValueError(
                f"{self.__class__.__name__}.spectral combination of format+method not recognized {format}+{method}."
            )
        return func(elec, herm)

    def _spectral_column_array(self, elec, herm):
        """Spectral function from a column array (`herm` not used)"""
        G = self._green_column(self.elecs_pvt_dev[elec].ravel())
        # Now calculate the full spectral function
        return G @ self._data.gamma[elec] @ dagger(G)

    def _spectral_column_bm(self, elec, herm):
        """Spectral function from a column array

        Returns a `BlockMatrix` class with all elements calculated.

        Parameters
        ----------
        herm: bool
           if true, only calculate the lower triangular part, and copy
           the Hermitian part to the upper triangular part.
           Else, calculate the full matrix via MM.
        """
        G = self._green_column(self.elecs_pvt_dev[elec].ravel())
        nb = len(self.btd)

        Gam = self._data.gamma[elec]

        # Now calculate the full spectral function
        btd = BlockMatrix(self.btd)
        BI = btd.block_indexer

        c = self.btd_cum0

        if herm:
            # loop columns
            for jb in range(nb):
                slj = slice(c[jb], c[jb + 1])
                Gj = Gam @ dagger(G[slj, :])
                for ib in range(jb):
                    sli = slice(c[ib], c[ib + 1])
                    BI[ib, jb] = G[sli, :] @ Gj
                    BI[jb, ib] = BI[ib, jb].T.conj()
                BI[jb, jb] = G[slj, :] @ Gj

        else:
            # loop columns
            for jb in range(nb):
                slj = slice(c[jb], c[jb + 1])
                Gj = Gam @ dagger(G[slj, :])
                for ib in range(nb):
                    sli = slice(c[ib], c[ib + 1])
                    BI[ib, jb] = G[sli, :] @ Gj

        return btd

    def _spectral_column_btd(self, elec, herm):
        """Spectral function from a column array

        Returns a `BlockMatrix` class with only BTD blocks calculated.

        Parameters
        ----------
        herm: bool
           if true, only calculate the lower triangular part, and copy
           the Hermitian part to the upper triangular part.
           Else, calculate the full matrix via MM.
        """
        G = self._green_column(self.elecs_pvt_dev[elec].ravel())
        nb = len(self.btd)

        Gam = self._data.gamma[elec]

        # Now calculate the full spectral function
        btd = BlockMatrix(self.btd)
        BI = btd.block_indexer

        c = self.btd_cum0
        if herm:
            # loop columns
            for jb in range(nb):
                slj = slice(c[jb], c[jb + 1])
                Gj = Gam @ dagger(G[slj, :])
                for ib in range(max(0, jb - 1), jb):
                    sli = slice(c[ib], c[ib + 1])
                    BI[ib, jb] = G[sli, :] @ Gj
                    BI[jb, ib] = BI[ib, jb].T.conj()
                BI[jb, jb] = G[slj, :] @ Gj

        else:
            # loop columns
            for jb in range(nb):
                slj = slice(c[jb], c[jb + 1])
                Gj = Gam @ dagger(G[slj, :])
                for ib in range(max(0, jb - 1), min(jb + 2, nb)):
                    sli = slice(c[ib], c[ib + 1])
                    BI[ib, jb] = G[sli, :] @ Gj

        return btd

    def _spectral_propagate_array(self, elec, herm):
        nb = len(self.btd)
        nbm1 = nb - 1

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self.elecs_pvt_dev[elec].ravel())
        nblocks = len(blocks)
        A = A @ self._data.gamma[elec] @ dagger(A)

        # Allocate space for the full matrix
        S = np.empty([len(self), len(self)], dtype=A.dtype)

        c = self.btd_cum0
        S[c[blocks[0]] : c[blocks[-1] + 1], c[blocks[0]] : c[blocks[-1] + 1]] = A
        del A

        # now loop backwards
        tX = self._data.tX
        tY = self._data.tY

        def gs(ib, jb):
            return slice(c[ib], c[ib + 1]), slice(c[jb], c[jb + 1])

        if herm:
            # above left
            for jb in range(blocks[0], -1, -1):
                for ib in range(jb, 0, -1):
                    A = -tY[ib] @ S[gs(ib, jb)]
                    S[gs(ib - 1, jb)] = A
                    S[gs(jb, ib - 1)] = A.T.conj()
                # calculate next diagonal
                if jb > 0:
                    S[gs(jb - 1, jb - 1)] = -S[gs(jb - 1, jb)] @ dagger(tY[jb])

            if nblocks == 2:
                # above
                for ib in range(blocks[1], 1, -1):
                    A = -tY[ib - 1] @ S[gs(ib - 1, blocks[1])]
                    S[gs(ib - 2, blocks[1])] = A
                    S[gs(blocks[1], ib - 2)] = A.T.conj()
                # below
                for ib in range(blocks[0], nbm1 - 1):
                    A = -tX[ib + 1] @ S[gs(ib + 1, blocks[0])]
                    S[gs(ib + 2, blocks[0])] = A
                    S[gs(blocks[0], ib + 2)] = A.T.conj()

            # below right
            for jb in range(blocks[-1], nb):
                for ib in range(jb, nbm1):
                    A = -tX[ib] @ S[gs(ib, jb)]
                    S[gs(ib + 1, jb)] = A
                    S[gs(jb, ib + 1)] = A.T.conj()
                # calculate next diagonal
                if jb < nbm1:
                    S[gs(jb + 1, jb + 1)] = -S[gs(jb + 1, jb)] @ dagger(tX[jb])

        else:
            for jb in range(blocks[0], -1, -1):
                # above
                for ib in range(jb, 0, -1):
                    S[gs(ib - 1, jb)] = -tY[ib] @ S[gs(ib, jb)]
                # calculate next diagonal
                if jb > 0:
                    S[gs(jb - 1, jb - 1)] = -S[gs(jb - 1, jb)] @ dagger(tY[jb])
                # left
                for ib in range(jb, 0, -1):
                    S[gs(jb, ib - 1)] = -S[gs(jb, ib)] @ dagger(tY[ib])

            if nblocks == 2:
                # above and left
                for ib in range(blocks[1], 1, -1):
                    S[gs(ib - 2, blocks[1])] = -tY[ib - 1] @ S[gs(ib - 1, blocks[1])]
                    S[gs(blocks[1], ib - 2)] = -S[gs(blocks[1], ib - 1)] @ dagger(
                        tY[ib - 1]
                    )
                # below and right
                for ib in range(blocks[0], nbm1 - 1):
                    S[gs(ib + 2, blocks[0])] = -tX[ib + 1] @ S[gs(ib + 1, blocks[0])]
                    S[gs(blocks[0], ib + 2)] = -S[gs(blocks[0], ib + 1)] @ dagger(
                        tX[ib + 1]
                    )

            # below right
            for jb in range(blocks[-1], nb):
                for ib in range(jb, nbm1):
                    S[gs(ib + 1, jb)] = -tX[ib] @ S[gs(ib, jb)]
                # calculate next diagonal
                if jb < nbm1:
                    S[gs(jb + 1, jb + 1)] = -S[gs(jb + 1, jb)] @ dagger(tX[jb])
                # right
                for ib in range(jb, nbm1):
                    S[gs(jb, ib + 1)] = -S[gs(jb, ib)] @ dagger(tX[ib])

        return S

    def _spectral_propagate_bm(self, elec, herm):
        btd = self.btd
        nb = len(btd)
        nbm1 = nb - 1

        BM = BlockMatrix(self.btd)
        BI = BM.block_indexer

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self.elecs_pvt_dev[elec].ravel())
        nblocks = len(blocks)
        A = A @ self._data.gamma[elec] @ dagger(A)

        BI[blocks[0], blocks[0]] = A[: btd[blocks[0]], : btd[blocks[0]]]
        if len(blocks) > 1:
            BI[blocks[0], blocks[1]] = A[: btd[blocks[0]], btd[blocks[0]] :]
            BI[blocks[1], blocks[0]] = A[btd[blocks[0]] :, : btd[blocks[0]]]
            BI[blocks[1], blocks[1]] = A[btd[blocks[0]] :, btd[blocks[0]] :]

        # now loop backwards
        tX = self._data.tX
        tY = self._data.tY

        if herm:
            # above left
            for jb in range(blocks[0], -1, -1):
                for ib in range(jb, 0, -1):
                    A = -tY[ib] @ BI[ib, jb]
                    BI[ib - 1, jb] = A
                    BI[jb, ib - 1] = A.T.conj()
                # calculate next diagonal
                if jb > 0:
                    BI[jb - 1, jb - 1] = -BI[jb - 1, jb] @ dagger(tY[jb])

            if nblocks == 2:
                # above
                for ib in range(blocks[1], 1, -1):
                    A = -tY[ib - 1] @ BI[ib - 1, blocks[1]]
                    BI[ib - 2, blocks[1]] = A
                    BI[blocks[1], ib - 2] = A.T.conj()
                # below
                for ib in range(blocks[0], nbm1 - 1):
                    A = -tX[ib + 1] @ BI[ib + 1, blocks[0]]
                    BI[ib + 2, blocks[0]] = A
                    BI[blocks[0], ib + 2] = A.T.conj()

            # below right
            for jb in range(blocks[-1], nb):
                for ib in range(jb, nbm1):
                    A = -tX[ib] @ BI[ib, jb]
                    BI[ib + 1, jb] = A
                    BI[jb, ib + 1] = A.T.conj()
                # calculate next diagonal
                if jb < nbm1:
                    BI[jb + 1, jb + 1] = -BI[jb + 1, jb] @ dagger(tX[jb])

        else:
            for jb in range(blocks[0], -1, -1):
                # above
                for ib in range(jb, 0, -1):
                    BI[ib - 1, jb] = -tY[ib] @ BI[ib, jb]
                # calculate next diagonal
                if jb > 0:
                    BI[jb - 1, jb - 1] = -BI[jb - 1, jb] @ dagger(tY[jb])
                # left
                for ib in range(jb, 0, -1):
                    BI[jb, ib - 1] = -BI[jb, ib] @ dagger(tY[ib])

            if nblocks == 2:
                # above and left
                for ib in range(blocks[1], 1, -1):
                    BI[ib - 2, blocks[1]] = -tY[ib - 1] @ BI[ib - 1, blocks[1]]
                    BI[blocks[1], ib - 2] = -BI[blocks[1], ib - 1] @ dagger(tY[ib - 1])
                # below and right
                for ib in range(blocks[0], nbm1 - 1):
                    BI[ib + 2, blocks[0]] = -tX[ib + 1] @ BI[ib + 1, blocks[0]]
                    BI[blocks[0], ib + 2] = -BI[blocks[0], ib + 1] @ dagger(tX[ib + 1])

            # below right
            for jb in range(blocks[-1], nb):
                for ib in range(jb, nbm1):
                    BI[ib + 1, jb] = -tX[ib] @ BI[ib, jb]
                # calculate next diagonal
                if jb < nbm1:
                    BI[jb + 1, jb + 1] = -BI[jb + 1, jb] @ dagger(tX[jb])
                # right
                for ib in range(jb, nbm1):
                    BI[jb, ib + 1] = -BI[jb, ib] @ dagger(tX[ib])

        return BM

    def _spectral_propagate_btd(self, elec, herm):
        btd = self.btd
        nb = len(btd)
        nbm1 = nb - 1

        BM = BlockMatrix(self.btd)
        BI = BM.block_indexer

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self.elecs_pvt_dev[elec].ravel())
        A = A @ self._data.gamma[elec] @ dagger(A)

        BI[blocks[0], blocks[0]] = A[: btd[blocks[0]], : btd[blocks[0]]]
        if len(blocks) > 1:
            BI[blocks[0], blocks[1]] = A[: btd[blocks[0]], btd[blocks[0]] :]
            BI[blocks[1], blocks[0]] = A[btd[blocks[0]] :, : btd[blocks[0]]]
            BI[blocks[1], blocks[1]] = A[btd[blocks[0]] :, btd[blocks[0]] :]

        # now loop backwards
        tX = self._data.tX
        tY = self._data.tY

        if herm:
            # above
            for b in range(blocks[0], 0, -1):
                A = -tY[b] @ BI[b, b]
                BI[b - 1, b] = A
                BI[b - 1, b - 1] = -A @ dagger(tY[b])
                BI[b, b - 1] = A.T.conj()
            # right
            for b in range(blocks[-1], nbm1):
                A = -BI[b, b] @ dagger(tX[b])
                BI[b, b + 1] = A
                BI[b + 1, b + 1] = -tX[b] @ A
                BI[b + 1, b] = A.T.conj()

        else:
            # above
            for b in range(blocks[0], 0, -1):
                dtY = dagger(tY[b])
                A = -tY[b] @ BI[b, b]
                BI[b - 1, b] = A
                BI[b - 1, b - 1] = -A @ dtY
                BI[b, b - 1] = -BI[b, b] @ dtY
            # right
            for b in range(blocks[-1], nbm1):
                A = -BI[b, b] @ dagger(tX[b])
                BI[b, b + 1] = A
                BI[b + 1, b + 1] = -tX[b] @ A
                BI[b + 1, b] = -tX[b] @ BI[b, b]

        return BM

    def _scattering_state_reduce(self, elec, DOS, U, cutoff):
        """U on input is a fortran-index as returned from eigh or svd"""
        # Select only the first N components where N is the
        # number of orbitals in the electrode (there can't be
        # any more propagating states anyhow).
        N = len(self._data.gamma[elec])

        # sort and take N highest values
        idx = np.argsort(-DOS)[:N]

        if cutoff > 0:
            # also retain values with large negative DOS.
            # These should correspond to states with large weight, but in some
            # way unphysical. The DOS *should* be positive.
            # Here we do the normalization depending on the number of orbitals
            # that is touched. This is important to make a uniformly defined
            # cutoff that does not depend on the device size.
            idx1 = (np.fabs(DOS[idx]) >= cutoff * U.shape[0]).nonzero()[0]
            idx = idx[idx1]

        return DOS[idx], U[:, idx]

    def scattering_state(
        self,
        elec,
        E,
        k=(0, 0, 0),
        cutoff=0.0,
        method: str = "svd:gamma",
        *args,
        **kwargs,
    ):
        r"""Calculate the scattering states for a given `E` and `k` point from a given electrode

        The scattering states are the eigen states of the spectral function:

        .. math::
            \mathbf A_{\mathfrak e}(E,\mathbf k) \mathbf u_i = 2\pi a_i \mathbf u_i

        where :math:`a_i` is the DOS carried by the :math:`i`'th scattering
        state.

        Parameters
        ----------
        elec : str or int
           the electrode to calculate the spectral function from
        E : float
           the energy to calculate at, may be a complex value.
        k : array_like, optional
           k-point to calculate the spectral function at
        cutoff : float, optional
           cutoff the returned scattering states at some DOS value. Any scattering states
           with normalized eigenvalues lower than `cutoff` are discarded.
           The normalization is done by dividing the eigenvalue with the number of orbitals
           in the device region. This normalization ensures the same cutoff value has roughly
           the same meaning for different size devices.
           Values above or close to 1e-5 should be used with care.
        method : {"svd:gamma", "svd:A", "full"}
           which method to use for calculating the scattering states.
           Use only the ``full`` method for testing purposes as it is extremely slow
           and requires a substantial amount of memory.
           The ``svd:gamma`` is the fastests while retaining complete precision.
           The ``svd:A`` may be even faster for very large systems with
           very little loss of precision, since it diagonalizes :math:`\mathbf A` in
           the subspace of the electrode `elec` and reduces the propagated part of the spectral
           matrix.
        cutoff_elec : float, optional
           Only used for ``method=svd:A``. The initial block of the spectral function is
           diagonalized and only eigenvectors with eigenvalues ``>=cutoff_elec`` are retained,
           thus reducing the initial propagated modes. The normalization explained for `cutoff`
           also applies here.

        Returns
        -------
        scat : StateCElectron
           the scattering states from the spectral function. The ``scat.state`` contains
           the scattering state vectors (eigenvectors of the spectral function).
           ``scat.c`` contains the DOS of the scattering states scaled by :math:`1/(2\pi)`
           so ensure correct density of states.
           One may recreate the spectral function with ``scat.outer(matrix=scat.c * 2 * pi)``.
        """
        elec = self._elec(elec)
        self._prepare(E, k)
        method = method.lower().replace(":", "_")
        func = getattr(self, f"_scattering_state_{method}", None)
        if func is None:
            raise ValueError(
                f"{self.__class__.__name__}.scattering_state method is not [full,svd,propagate]"
            )
        return func(elec, cutoff, *args, **kwargs)

    def _scattering_state_full(self, elec, cutoff=0.0, **kwargs):
        # We know that scattering_state has called prepare!
        A = self.spectral(elec, self._data.E, self._data.k, **kwargs)

        # add something to the diagonal (improves diag precision for small states)
        np.fill_diagonal(A, A.diagonal() + 0.1)

        # Now diagonalize A
        DOS, A = eigh_destroy(A)
        # backconvert diagonal
        DOS -= 0.1
        # TODO check with overlap convert with correct magnitude (Tr[A] / 2pi)
        DOS /= 2 * np.pi
        DOS, A = self._scattering_state_reduce(elec, DOS, A, cutoff)

        data = self._data
        info = dict(
            method="full", elec=self._elec_name(elec), E=data.E, k=data.k, cutoff=cutoff
        )

        # always have the first state with the largest values
        return si.physics.StateCElectron(A.T, DOS, self, **info)

    def _scattering_state_svd_gamma(self, elec, cutoff=0.0, **kwargs):
        A = self._green_column(self.elecs_pvt_dev[elec].ravel())

        # This calculation uses the cholesky decomposition of Gamma
        # combined with SVD of the A column
        A = A @ cholesky(self._data.gamma[elec], lower=True)

        # Perform svd
        DOS, A = _scat_state_svd(A, **kwargs)
        DOS, A = self._scattering_state_reduce(elec, DOS, A, cutoff)

        data = self._data
        info = dict(
            method="svd:Gamma",
            elec=self._elec_name(elec),
            E=data.E,
            k=data.k,
            cutoff=cutoff,
        )

        # always have the first state with the largest values
        return si.physics.StateCElectron(A.T, DOS, self, **info)

    def _scattering_state_svd_a(self, elec, cutoff=0, **kwargs):
        # Parse the cutoff value
        # Here we may use 2 values, one for cutting off the initial space
        # and one for the returned space.
        cutoff = np.array(cutoff).ravel()
        if cutoff.size != 2:
            cutoff0 = cutoff1 = cutoff[0]
        else:
            cutoff0, cutoff1 = cutoff[0], cutoff[1]
        cutoff0 = kwargs.get("cutoff_elec", cutoff0)

        # First we need to calculate diagonal blocks of the spectral matrix
        # This is basically the same thing as calculating the Gf column
        # But only in the 1/2 diagonal blocks of Gf
        blocks, u = self._green_diag_block(self.elecs_pvt_dev[elec].ravel())

        # Calculate the spectral function only for the blocks that host the
        # scattering matrix
        u = u @ self._data.gamma[elec] @ dagger(u)

        # add something to the diagonal (improves diag precision)
        np.fill_diagonal(u, u.diagonal() + 0.1)

        # Calculate eigenvalues
        DOS, u = eigh_destroy(u)
        # backconvert diagonal
        DOS -= 0.1
        # TODO check with overlap convert with correct magnitude (Tr[A] / 2pi)
        DOS /= 2 * np.pi

        # Remove states for cutoff and size
        # Since there cannot be any addition of states later, we
        # can do the reduction here.
        # This will greatly increase performance for very wide systems
        # since the number of contributing states is generally a fraction
        # of the total electrode space.
        DOS, u = self._scattering_state_reduce(elec, DOS, u, cutoff0)
        # Back-convert to retain scale of the vectors before SVD
        # and also take the sqrt to ensure u u^dagger returns
        # a sensible value, the 2*pi factor ensures the *original* scale.
        u *= signsqrt(DOS * 2 * np.pi)

        nb = len(self.btd)
        cbtd = self.btd_cum0

        # Create full U
        A = np.empty([len(self), u.shape[1]], dtype=u.dtype)

        sl = slice(cbtd[blocks[0]], cbtd[blocks[0] + 1])
        A[sl, :] = u[: self.btd[blocks[0]], :]
        if len(blocks) > 1:
            sl = slice(cbtd[blocks[1]], cbtd[blocks[1] + 1])
            A[sl, :] = u[self.btd[blocks[0]] :, :]
        del u

        # Propagate A in the full BTD matrix
        t = self._data.tY
        sl = slice(cbtd[blocks[0]], cbtd[blocks[0] + 1])
        for b in range(blocks[0], 0, -1):
            sln = slice(cbtd[b - 1], cbtd[b])
            A[sln] = -t[b] @ A[sl]
            sl = sln

        t = self._data.tX
        sl = slice(cbtd[blocks[-1]], cbtd[blocks[-1] + 1])
        for b in range(blocks[-1], nb - 1):
            slp = slice(cbtd[b + 1], cbtd[b + 2])
            A[slp] = -t[b] @ A[sl]
            sl = slp

        # Perform svd
        # TODO check with overlap convert with correct magnitude (Tr[A] / 2pi)
        DOS, A = _scat_state_svd(A, **kwargs)

        DOS, A = self._scattering_state_reduce(elec, DOS, A, cutoff1)

        # Now we have the full u, create it and transpose to get it in C indexing
        data = self._data
        info = dict(
            method="svd:A",
            elec=self._elec_name(elec),
            E=data.E,
            k=data.k,
            cutoff_elec=cutoff0,
            cutoff=cutoff1,
        )
        return si.physics.StateCElectron(A.T, DOS, self, **info)

    def scattering_matrix(self, elec_from, elec_to, E, k=(0, 0, 0), cutoff=0.0):
        r""" Calculate the scattering matrix (S-matrix) between `elec_from` and `elec_to`

        The scattering matrix is calculated as

        .. math::
               \mathbf S_{\mathfrak e_{\mathrm{to}}\mathfrak e_{\mathrm{from}} }(E, \mathbf) = -\delta_{\alpha\beta} + i
               \tilde{\boldsymbol\Gamma}_{\mathfrak e_{\mathrm{to}}}
               \mathbf G
               \tilde{\boldsymbol\Gamma}_{\mathfrak e_{\mathrm{from}}}

        Here :math:`\tilde{\boldsymbol\Gamma}` is defined as:

        .. math::
            \boldsymbol\Gamma(E,\mathbf k) \mathbf u_i &= \lambda_i \mathbf u_i
            \\
            \tilde{\boldsymbol\Gamma}(E,\mathbf k) &= \operatorname{diag}\{ \sqrt{\boldsymbol\lambda} \} \mathbf u

        Once the scattering matrices have been calculated one can calculate the full transmission
        function

        .. math::
              T_{\mathfrak e_{\mathrm{from}}\mathfrak e_{\mathrm{to}} }(E, \mathbf k) = \operatorname{Tr}\big[
              \mathbf S_{\mathfrak e_{\mathrm{to}}\mathfrak e_{\mathrm{from}} }^\dagger
              \mathbf S_{\mathfrak e_{\mathrm{to}}\mathfrak e_{\mathrm{from}} }\big]


        Parameters
        ----------
        elec_from : str or int
           the electrode where the scattering matrix originates from
        elec_to : str or int or list of
           where the scattering matrix ends in.
        E : float
           the energy to calculate at, may be a complex value.
        k : array_like, optional
           k-point to calculate the scattering matrix at
        cutoff : float, optional
           cutoff the eigen states of the broadening matrix that are below this value.
           I.e. only :math:`\lambda` values above this value will be used.
           A too high value will remove too many eigen states and results will be wrong.
           A small value improves precision at the cost of bigger matrices.

        Returns
        -------
        scat_matrix : numpy.ndarray or tuple of numpy.ndarray
           for each `elec_to` a scattering matrix will be returned. Its dimensions will be
           depending on the `cutoff` value at the cost of precision.
        """
        # Calculate the full column green function
        elec_from = self._elec(elec_from)

        is_single = False
        if isinstance(elec_to, (Integral, str, PivotSelfEnergy)):
            is_single = True
            elec_to = [elec_to]
        # convert to indices
        elec_to = [self._elec(e) for e in elec_to]

        # Prepare calculation @ E and k
        self._prepare(E, k)
        self._prepare_tgamma(E, k, cutoff)

        # Get full G in column of 'from'
        G = self._green_column(self.elecs_pvt_dev[elec_from].ravel())

        # the \tilde \Gamma functions
        tG = self._data.tgamma

        # Now calculate the S matrices
        def calc_S(elec_from, jtgam_from, elec_to, tgam_to, G):
            pvt = self.elecs_pvt_dev[elec_to].ravel()
            g = G[pvt, :]
            ret = dagger(tgam_to) @ g @ jtgam_from
            if elec_from == elec_to:
                min_n = np.arange(min(ret.shape))
                np.add.at(ret, (min_n, min_n), -1)
            return ret

        tgam_from = 1j * tG[elec_from]
        S = tuple(calc_S(elec_from, tgam_from, elec, tG[elec], G) for elec in elec_to)

        if is_single:
            return S[0]
        return S

    def eigenchannel(self, state, elec_to, ret_coeff=False):
        r""" Calculate the eigenchannel from scattering states entering electrodes `elec_to`

        The energy and k-point is inferred from the `state` object as returned from
        `scattering_state`.

        The eigenchannels are the eigen states of the transmission matrix in the
        DOS weighted scattering states:

        .. math::
            \mathbf A_{\mathfrak e_{\mathrm{from}} }(E,\mathbf k) \mathbf u_i &= 2\pi a_i \mathbf u_i
            \\
            \mathbf t_{\mathbf u} &= \sum \langle \mathbf u | \boldsymbol\Gamma_{ \mathfrak e_{\mathrm{to}} }  | \mathbf u\rangle

        where the eigenvectors of :math:`\mathbf t_{\mathbf u}` are the coefficients of the
        DOS weighted scattering states (:math:`\sqrt{2\pi a_i} u_i`) for the individual eigen channels.
        The eigenvalues are the transmission eigenvalues. Further details may be found in :cite:`Paulsson2007`.

        Parameters
        ----------
        state : sisl.physics.StateCElectron
            the scattering states as obtained from `scattering_state`
        elec_to : str or int (list or not)
            which electrodes to consider for the transmission eigenchannel
            decomposition (the sum in the above equation)
        ret_coeff : bool, optional
            return also the scattering state coefficients

        Returns
        -------
        T_eig : sisl.physics.StateCElectron
            the transmission eigenchannels, the ``T_eig.c`` contains the transmission
            eigenvalues.
        coeff : sisl.physics.StateElectron
            coefficients of `state` that creates the transmission eigenchannels
            Only returned if `ret_coeff` is True. There is a one-to-one correspondance
            between ``coeff`` and ``T_eig`` (with a prefactor of :math:`\sqrt{2\pi}`).
            This is equivalent to the ``T_eig`` states in the scattering state basis.
        """
        self._prepare_se(state.info["E"], state.info["k"])
        if isinstance(elec_to, (Integral, str, PivotSelfEnergy)):
            elec_to = [elec_to]
        # convert to indices
        elec_to = [self._elec(e) for e in elec_to]

        # The sign shouldn't really matter since the states should always
        # have a finite DOS, however, for completeness sake we retain the sign.
        # We scale the vectors by sqrt(DOS/2pi).
        # This is because the scattering states from self.scattering_state
        # stores eig(A) / 2pi.
        sqDOS = signsqrt(state.c).reshape(-1, 1)
        # Retrive the scattering states `A` and apply the proper scaling
        # We need this scaling for the eigenchannel construction anyways.
        A = state.state * sqDOS

        # create shorthands
        elec_pvt_dev = self.elecs_pvt_dev
        G = self._data.gamma

        # Create the first electrode
        el = elec_to[0]
        idx = elec_pvt_dev[el].ravel()

        # Retrive the scattering states `A` and apply the proper scaling
        # We need this scaling for the eigenchannel construction anyways.
        u = A[:, idx]
        # the summed transmission matrix
        Ut = u.conj() @ G[el] @ u.T

        # same for other electrodes
        for el in elec_to[1:]:
            idx = elec_pvt_dev[el].ravel()
            u = A[:, idx]
            Ut += u.conj() @ G[el] @ u.T

        # TODO currently a factor depends on what is used
        #      in `scattering_states`, so go check there.
        #      The resulting Ut should have a factor: 1 / 2pi ** 0.5
        #      When the states DOS values (`state.c`) has the factor 1 / 2pi
        #      then `u` has the correct magnitude and all we need to do is to add the factor 2pi
        # diagonalize the transmission matrix tt
        tt, Ut = eigh_destroy(Ut)
        tt *= 2 * np.pi

        info = {**state.info, "elec_to": tuple(self._elec_name(e) for e in elec_to)}

        # Backtransform U to form the eigenchannels
        teig = si.physics.StateCElectron(Ut[:, ::-1].T @ A, tt[::-1], self, **info)
        if ret_coeff:
            return teig, si.physics.StateElectron(Ut[:, ::-1].T, self, **info)
        return teig
