# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import scipy.sparse as ssp
from scipy.sparse.linalg import svds

import sisl as si
from sisl import _array as _a
from sisl._indices import indices_only
from sisl._internal import set_module
from sisl.linalg import (
    cholesky,
    eigh,
    eigh_destroy,
    inv_destroy,
    signsqrt,
    solve,
    sqrth,
    svd_destroy,
)
from sisl.messages import info, warn
from sisl.typing import KPoint
from sisl.typing._core import SileLike
from sisl.utils.misc import PropertyDict

from ._btd import *
from ._electrode import *
from ._help import *

__all__ = ["DeviceGreen"]


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
            scale = np.floor(np.log10(np.absolute(A).min())).astype(int)
            if scale < -12:
                scale = 10 ** (-12 - scale)
            else:
                scale = False
    if scale:
        A *= scale

    ret_uv = kwargs.get("ret_uv", False)

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
        svds_kwargs["return_singular_vectors"] = True if ret_uv else "u"
        svds_kwargs["solver"] = driver
        if "k" not in svds_kwargs:
            k = A.shape[1] // 2
            if k < 3:
                k = A.shape[1] - 1
            svds_kwargs["k"] = k

        A, DOS, B = svds(A, **svds_kwargs)

    else:
        # it must be a lapack driver:
        A, DOS, B = svd_destroy(A, full_matrices=False, lapack_driver=driver)

    if kwargs.get("ret_uv", False):
        A = (A, B)
    else:
        del B

    if scale:
        DOS /= scale

    # A note of caution.
    # The DOS values are not actual DOS values.
    # In fact the DOS should be calculated as:
    #   DOS * <i| S(k) |i>
    # to account for the overlap matrix. For orthogonal basis sets
    # this DOS eigenvalue is correct.
    return DOS * DOS.conj() / (2 * np.pi), A


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

    Sometimes one wishes to investigate more details in the calculation process
    to discern importance of the eigenvalue separations.

    When calculating scattering states/matrices one can
    reduce the complexity by removing eigen/singular values.
    By default we use the `cutoff` values as a relative cutoff value for
    the values. I.e. keeping ``value / value.max() > cutoff``.
    However, sometimes the relative value is a bad metric since there are
    still important values close to unity value. Consider e.g. an array of
    values of ``[1e5, 1e4, 1, 0.5, 1e-4]``. In this case we would require the
    ``[1, 0.5]`` values as important, but this would only be grabbed by a relative
    cutoff value of ``1e-6`` which in some other cases are a too high value.

    Instead of providing `cutoff` values as `float` values, one can also
    pass a function that takes in an array of values. It should return the
    indices of the values it wishes to retain.

    The below is equivalent to a cutoff value of ``1e-4``, or values
    above 0.01.
    >>> def cutoff_func(V):
    >>>     return np.logical_or(V / V.max() > 1e-4, V > 1e-2).nonzero()[0]

    Passing functions for cutting off values can be useful because one
    can also debug the values and see what's happening.
    """

    # TODO we should speed this up by overwriting A with the inverse once
    #      calculated. We don't need it at that point.
    #      That would probably require us to use a method to retrieve
    #      the elements which determines if it has been calculated or not.

    def __init__(self, H: si.Hamiltonian, elecs, pivot, eta: float = 0.0):
        """Create Green function with Hamiltonian and BTD matrix elements"""
        self.H = H

        # Store electrodes (for easy retrieval of the SE)
        # There may be no electrodes
        self.elecs = elecs

        # In case the pivot scheme does match the spin-orbit case
        # We should fix things
        expand = get_expand(len(H), pivot.no_u)

        if expand > 1:
            info(
                f"The pivoting information in {pivot!s} does not "
                "match the number of expanded orbitals in the Hamiltonian.\n"
                f"Will expand all necessary arrays with {expand} for compatibility."
            )

        # the pivoting table for the device region
        self.pvt = expand_orbs(pivot.pivot(), expand)
        # the BTD blocks (in the pivoted space) for the device region
        self.btd = expand_btd(pivot.btd(), expand)

        # global device eta
        self.eta = eta

        # Create BTD indices
        self.btd_cum0 = np.empty([len(self.btd) + 1], dtype=self.btd.dtype)
        self.btd_cum0[0] = 0
        self.btd_cum0[1:] = np.cumsum(self.btd)
        self.clear()

    def __str__(self) -> str:
        ret = f"{self.__class__.__name__}{{no: {len(self)}, blocks: {len(self.btd)}, eta: {self.eta:.3e}"
        for elec in self.elecs:
            e = str(elec).replace("\n", "\n  ")
            ret = f"{ret},\n {elec.name}:\n  {e}"
        return f"{ret}\n}}"

    @classmethod
    def from_fdf(
        cls,
        fdf: SileLike,
        prefix: Literal["TBT", "TS"] = "TBT",
        use_tbt_se: bool = False,
        eta: Optional[float] = None,
        **kwargs,
    ) -> Self:
        """Return a new `DeviceGreen` using information gathered from the fdf file.

        Parameters
        ----------
        fdf :
           fdf file to read the parameters from
        prefix :
           which prefix to use, if TBT it will prefer TBT prefix, but fall back
           to TS prefixes.
           If TS, only these prefixes will be used.
        use_tbt_se :
           whether to use the TBT.SE.nc files for self-energies
           or calculate them on the fly.
        eta :
            force a specific eta value
        kwargs :
            passed to the class instantiating.
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
        for hs_ext in ("TS.HSX", "TSHS", "HSX", "nc"):
            if Path(f"{slabel}.{hs_ext}").is_file():
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
            # dictionary, for information, but it shouldn't be used!
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
        eta_dev = 1e123  # just a very large number so we default to the smallest one
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
                    f"  {tbt} = {elec_eta} eV\n  {fdf} = {data.eta} eV"
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
            eta_key = "TBT.Contours.Eta"
        else:
            eta_key = "TS.Contours.nEq.Eta"
        eta_dev_found = fdf.type(eta_key)
        # work-around to ensure we don't default to something that
        # isn't in the TBT.nc file
        if eta_dev_found:
            eta_dev = fdf.get(eta_key, eta_dev, unit="eV")
        else:
            eta_dev = eta_dev_tbt

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

        return cls(Hdev, elecs, tbt, eta=eta_dev, **kwargs)

    def clear(self, *keys) -> None:
        """Clean any memory used by this object"""
        if keys:
            for key in keys:
                try:
                    del self._data[key]
                except KeyError:
                    pass  # ok that key does not exist
        else:
            self._data = PropertyDict()

    def __len__(self) -> int:
        """Length of Green function matrix."""
        return len(self.pvt)

    def _elec(self, elec) -> Union[int, Sequence[int]]:
        """Convert a string electrode to the proper linear index"""
        if isinstance(elec, str):
            for iel, el in enumerate(self.elecs):
                if el.name == elec:
                    return iel
        elif isinstance(elec, PivotSelfEnergy):
            return self._elec(elec.name)
        elif isinstance(elec, (tuple, list)):
            return [self._elec(e) for e in elec]
        return elec

    def _serialize_elecs(self, elecs, omit_elecs: Optional = None):
        """Convert a list/str/int of elecs into a list of ints"""
        is_all = elecs is None  # TODO change this to something more sensible
        if is_all:
            elecs = list(range(len(self.elecs)))
        is_single = not isinstance(elecs, (tuple, list))
        if is_single:
            # ensure it is a list
            elecs = [elecs]

        # convert to list of ints
        elecs = self._elec(elecs)
        if is_all and omit_elecs is not None:
            _, omit_elecs = self._serialize_elecs(omit_elecs)
            elecs = list(filter(lambda elec: not elec in omit_elecs, elecs))
            is_single = len(elecs) == 1

        return is_single, elecs

    def _elec_name(self, elec) -> str:
        """Convert an electrode index or str to the name of the electrode"""
        if isinstance(elec, str):
            return elec
        elif isinstance(elec, PivotSelfEnergy):
            return elec.name
        return self.elecs[elec].name

    def _pivot_matrix(self, M):
        """Pivot's a (full) matrix, i.e. not one that is already reduced."""
        return M[self.pvt, :][:, self.pvt]

    def _as_cutoff_func(self, cutoff):
        """Ensure the cutoff value is transformed into a cut-off function"""

        # Removing values is hard, because there may be loads
        # of values very close to each other.
        # I.e. a list of:
        # [1.1e-4, 1.0e-4, 1e-5]
        # Then they are *all* important to capture the physics.
        # Hence, the cutoff is a *relative* cutoff to the highest value.
        # I.e. if the cutoff is 1e-3.
        # Then all values which are within a factor of 1000 from the highest
        # value (absolute), will be retained.
        # This is much more stable for things with low DOS.
        # Perhaps there should be some way to retrieve these values, to
        # actually check if it makes physical sense.
        if callable(cutoff):
            return cutoff

        def cutoff_func(v):
            nonlocal cutoff
            rel_v = v / np.max(v)
            return (rel_v >= cutoff).nonzero()[0]

        return cutoff_func

    def _check_Ek(self, E: complex, k: KPoint, **kwargs) -> bool:
        """Check whether the stored quantities has already been calculated

        It does this by checking the internal data-structures stored `E` and `k`
        values.
        """
        if hasattr(self._data, "E"):
            if np.allclose(self._data.E, E) and np.allclose(self._data.k, k):
                # we have already prepared the calculation
                return True

        # while resetting is not necessary, it can
        # save a lot of memory since some arrays are not
        # temporarily stored twice.
        self.clear()
        self._data.kwargs = kwargs
        self._data.E = E
        # the imaginary value in the device region
        self._data.Ec = E
        if np.isrealobj(E):
            self._data.Ec = E + 1j * self.eta
        self._data.k = np.asarray(k, dtype=np.float64)

        return False

    def _prepare_se(self, E: complex, k: KPoint, dtype, **kwargs) -> None:
        """Pre-calculate all self-energies (and store the Gamma matrices as well)."""
        if self._check_Ek(E, k, **kwargs):
            if hasattr(self._data, "se"):
                return
        else:
            self.clear("A", "tgamma")

        E = self._data.E
        k = self._data.k

        # Create all self-energies (and store the Gamma's)
        se = []
        gamma = []
        for elec in self.elecs:
            # Insert values
            SE = elec.self_energy(E, k=k, dtype=dtype, **kwargs)
            se.append(SE)
            gamma.append(elec.se2broadening(SE))

        self._data.se = se
        self._data.gamma = gamma

    def _prepare_tgamma(self, E: complex, k: KPoint, dtype, cutoff, **kwargs) -> None:
        if self._check_Ek(E, k, **kwargs):
            if hasattr(self._data, "tgamma"):
                if hash(cutoff) == self._data.tgamma_cutoff:
                    return
        else:
            self.clear("A", "se")

        # ensure we have the self-energies
        self._prepare(E, k, dtype, **kwargs)

        tgamma = []

        # See Sanz, Mach-Zender paper
        # Get the sqrt of the level broadening matrix
        def eigh_sqrt(gam):
            nonlocal cutoff
            eig, U = eigh(gam)
            idx = cutoff(eig)
            eig = np.emath.sqrt(eig[idx])
            U = U[:, idx]
            return eig * U

        for gam in self._data.gamma:
            tgamma.append(eigh_sqrt(gam))

        self._data.tgamma = tgamma
        self._data.tgamma_cutoff = hash(cutoff)

    def _prepare(self, E: complex, k: KPoint, dtype, **kwargs) -> None:
        r"""Pre-calculate the needed quantities for Green function calculation

        It calculates:

        - self-energies (and consequently the Gamma matrices)
        - Splits the inverse-Green function into blocks according
          to the article.

          .. math::

             \mathbf A_i, \mathbf B_i, \mathbf C_i, \tilde\mathbf X_i, \tilde\mathbf Y_i
        """
        if self._check_Ek(E, k, **kwargs):
            if hasattr(self._data, "A"):
                return
        else:
            self.clear("tgamma", "se")

        data = self._data

        E = data.E
        # device region: E + 1j*eta
        Ec = data.Ec
        k = data.k

        # Prepare the Green function calculation
        invG = self.H.Sk(k, dtype=dtype) * Ec - self.H.Hk(k, dtype=dtype, **kwargs)
        invG = self._pivot_matrix(invG).tolil()

        # Create all self-energies (and store the Gamma's)
        if hasattr(data, "se"):
            for elec, SE in zip(self.elecs, data.se):
                pvt = elec.pvt_dev
                invG[pvt, pvt.T] -= SE
        else:
            gamma = []
            for elec in self.elecs:
                pvt = elec.pvt_dev
                # Insert values
                SE = elec.self_energy(E, k=k, dtype=dtype, **kwargs)
                invG[pvt, pvt.T] -= SE
                gamma.append(elec.se2broadening(SE))
            del SE
            data.gamma = gamma
        # convert to csr format (that's how will mostly convert it)
        invG = invG.tocsr()

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
        iG = invG[sl0, :].tocsc()
        A[0] = iG[:, sl0].toarray()
        C[1] = iG[:, slp].toarray()
        for b in range(1, nbm1):
            # rotate slices
            sln = sl0
            sl0 = slp
            slp = slice(cbtd[b + 1], cbtd[b + 2])
            iG = invG[sl0, :].tocsc()

            B[b - 1] = iG[:, sln].toarray()
            A[b] = iG[:, sl0].toarray()
            C[b + 1] = iG[:, slp].toarray()
        # and final matrix A and B
        iG = invG[slp, :].tocsc()
        A[nbm1] = iG[:, slp].toarray()
        B[nbm1 - 1] = iG[:, sl0].toarray()

        # clean-up, not used anymore
        del invG, iG

        # store in the data field
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

        # store tilde-matrices, now we can fast re-calculate everything as needed.
        data.tX = tX
        data.tY = tY

    def _matrix_to_btd(self, M, format: str = "array") -> BlockMatrix:
        """Convert a matrix `M` into a BTD matrix.

        Parameters
        ----------
        M :
            the matrix to convert to a BTD matrix form
        sparse :
            whether each block in the BTD matrix may be sparse or not.
        """
        BM = BlockMatrix(self.btd)
        BI = BM.block_indexer
        c = self.btd_cum0
        nb = len(BI)

        def cast(M):
            return M

        if ssp.issparse(M):

            def cast(M):
                return getattr(M, f"to{format}")()

        for jb in range(nb):
            for ib in range(max(0, jb - 1), min(jb + 2, nb)):
                BI[ib, jb] = cast(M[c[ib] : c[ib + 1], c[jb] : c[jb + 1]])

        return BM

    def Sk(self, *args, **kwargs):
        """Return the overlap matrix in the pivoted device region"""
        is_btd = False
        if "format" in kwargs:
            if kwargs["format"].lower() == "btd":
                is_btd = True
                del kwargs["format"]

        M = self._pivot_matrix(self.H.Sk(*args, **kwargs))
        if is_btd:
            return self._matrix_to_btd(M)
        return M

    def Hk(self, *args, **kwargs):
        """Return the Hamiltonian matrix in the pivoted device region"""
        is_btd = False
        if "format" in kwargs:
            if kwargs["format"].lower() == "btd":
                is_btd = True
                del kwargs["format"]

        M = self._pivot_matrix(self.H.Hk(*args, **kwargs))
        if is_btd:
            return self._matrix_to_btd(M)
        return M

    def _get_blocks(self, idx):
        """Returns the blocks that an index belongs to"""
        block1 = (idx.min() < self.btd_cum0[1:]).nonzero()[0][0]
        block2 = (idx.max() < self.btd_cum0[1:]).nonzero()[0][0]
        if block1 == block2:
            blocks = [block1]
        else:
            blocks = [b for b in range(block1, block2 + 1)]
        return blocks

    def green(
        self,
        E: complex,
        k: KPoint = (0, 0, 0),
        format="array",
        dtype=np.complex128,
        **kwargs,
    ) -> Union[np.ndarray, BlockMatrix]:
        r"""Calculate the Green function for a given `E` and `k` point

        The Green function is calculated as:

        .. math::
            \mathbf G(E,\mathbf k) = \big[\mathbf S(\mathbf k) E - \mathbf H(\mathbf k)
                  - \sum \boldsymbol \Sigma(E,\mathbf k)\big]^{-1}

        Parameters
        ----------
        E :
            the energy to calculate at, may be a complex value.
        k :
            k-point to calculate the Green function at
        format : {"array", "btd", "bm", "bd", "sparse"}
            return the matrix in a specific format

            - array: a regular numpy array (full matrix)
            - btd: a block-matrix object with only the diagonals and first off-diagonals
            - bm: a block-matrix object with diagonals and all off-diagonals
            - bd: a block-matrix object with only diagonals (no off-diagonals)
            - sparse: a sparse-csr matrix for the sparse elements as found in the Hamiltonian
        dtype :
            the data-type of the array.

        Returns
        -------
        np.ndarray or BlockMatrix
            the Green function matrix, the format depends on `format`.
        """
        self._prepare(E, k, dtype, **kwargs)
        format = format.lower()
        if format == "dense":
            format = "array"
        func = getattr(self, f"_green_{format}", None)
        if func is None:
            raise ValueError(
                f"{self.__class__.__name__}.green format not valid input [array|sparse|bm|btd|bd]"
            )
        return func()

    def _green_array(self) -> np.ndarray:
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

    def _green_btd(self) -> BlockMatrix:
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

    def _green_bm(self) -> BlockMatrix:
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

    def _green_bd(self) -> BlockMatrix:
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
        """Calculate the Green function only where the sparse H and S are non-zero.

        Stored in a `scipy.sparse.csr_matrix` class."""
        # create a sparse matrix
        G = self.H.Sk(format="csr", dtype=self._data.A[0].dtype)
        # pivot the matrix
        G = self._pivot_matrix(G)

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

    def _green_diag_block(self, cols):
        """Calculate the Green function only on specific (neighboring) diagonal block matrices.

        Stored in a `np.array` class."""
        ncols = len(cols)
        nb = len(self.btd)
        nbm1 = nb - 1

        # Find parts we need to calculate
        blocks = self._get_blocks(cols)
        assert (
            len(blocks) <= 2
        ), f"{self.__class__.__name__} green(diagonal) requires maximally 2 blocks"
        if len(blocks) == 2:
            assert (
                blocks[0] + 1 == blocks[1]
            ), f"{self.__class__.__name__} green(diagonal) requires spanning only 2 blocks"

        n = self.btd[blocks].sum()
        G = np.empty([n, ncols], dtype=self._data.A[0].dtype)

        btd = self.btd
        c = self.btd_cum0
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b in blocks:
            # Find the indices in the block
            i = cols[c[b] <= cols]
            i = i[i < c[b + 1]].astype(np.int32)

            b_idx = indices_only(_a.arangei(c[b], c[b + 1]), i)

            if b == blocks[0]:
                sl = slice(0, btd[b])
                c_idx = _a.arangei(len(b_idx))
            else:
                sl = slice(btd[blocks[0]], btd[blocks[0]] + btd[b])
                c_idx = _a.arangei(ncols - len(b_idx), ncols)

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

    def _green_column(self, cols) -> np.ndarray:
        """Calculate the full Green function column for a subset of columns.

        Stored in a `np.array` class."""
        # To calculate the full Gf for specific column indices
        # These indices should maximally be spanning 2 blocks
        cols = cols.ravel()
        ncols = len(cols)
        nb = len(self.btd)
        nbm1 = nb - 1

        # Find parts we need to calculate
        blocks = self._get_blocks(cols)
        assert (
            len(blocks) <= 2
        ), f"{self.__class__.__name__}.green(column) requires maximally 2 blocks"
        if len(blocks) == 2:
            assert (
                blocks[0] + 1 == blocks[1]
            ), f"{self.__class__.__name__}.green(column) requires spanning only 2 blocks"

        n = len(self)
        G = np.empty([n, ncols], dtype=self._data.A[0].dtype)

        c = self.btd_cum0
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b in blocks:
            # Find the indices in the block
            i = cols[c[b] <= cols]
            i = i[i < c[b + 1]].astype(np.int32)

            b_idx = indices_only(_a.arangei(c[b], c[b + 1]), i)

            if b == blocks[0]:
                c_idx = _a.arangei(len(b_idx))
            else:
                c_idx = _a.arangei(ncols - len(b_idx), ncols)

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
        E: complex,
        elec,
        k: KPoint = (0, 0, 0),
        format: str = "array",
        method: Literal["column", "propagate"] = "column",
        herm: bool = True,
        dtype=np.complex128,
        **kwargs,
    ) -> Union[np.ndarray, BlockMatrix]:
        r"""Calculate the spectral function for a given `E` and `k` point from a given electrode

        The spectral function is calculated as:

        .. math::
            \mathbf A_{\mathfrak{e}}(E,\mathbf k) = \mathbf G(E,\mathbf k)\boldsymbol\Gamma_{\mathfrak{e}}(E,\mathbf k)
                   \mathbf G^\dagger(E,\mathbf k)

        Parameters
        ----------
        E :
           the energy to calculate at, may be a complex value.
        elec : str or int
           the electrode to calculate the spectral function from
        k :
           k-point to calculate the spectral function at
        format : {"array", "btd", "bm", "bd"}
           return the matrix in a specific format

           - array: a regular numpy array (full matrix)
           - bm: in block-matrix form (full matrix)
           - btd: a block-matrix object with only the diagonals and first off-diagonals
           - bd: same as btd, since the off-diagonals are already calculated
        method :
           which method to use for calculating the spectral function.
           Depending on the size of the BTD blocks one may be faster than the
           other. For large systems you are recommended to time the different methods
           and stick with the fastest one, they are numerically identical.
        herm:
           The spectral function is a Hermitian matrix, by default (True), the methods
           that can utilize the Hermitian property only calculates the lower triangular
           part of :math:`\mathbf A`, and then copies the Hermitian to the upper part.
           By setting this to `False` the entire matrix is explicitly calculated.

        Returns
        -------
        np.ndarray or BlockMatrix
            the spectral function for a given electrode in the format
            as specified by `format`. Note that some formats does not calculate
            the entire spectral function matrix.
        """
        # the herm flag is considered useful for testing, there is no need to
        # play with it. So it isn't documented.

        elec = self._elec(elec)
        self._prepare(E, k, dtype, **kwargs)
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

    def _spectral_column_array(self, elec, herm: bool) -> np.ndarray:
        """Spectral function from a column array (`herm` not used)"""
        G = self._green_column(self.elecs[elec].pvt_dev.ravel())
        # Now calculate the full spectral function
        return G @ self._data.gamma[elec] @ np.conj(G.T)

    def _spectral_column_bm(self, elec, herm: bool) -> BlockMatrix:
        """Spectral function from a column array

        Returns a `BlockMatrix` class with all elements calculated.

        Parameters
        ----------
        herm:
           if true, only calculate the lower triangular part, and copy
           the Hermitian part to the upper triangular part.
           Else, calculate the full matrix via MM.
        """
        G = self._green_column(self.elecs[elec].pvt_dev.ravel())
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
                Gj = Gam @ np.conj(G[slj, :].T)
                for ib in range(jb):
                    sli = slice(c[ib], c[ib + 1])
                    BI[ib, jb] = G[sli, :] @ Gj
                    BI[jb, ib] = BI[ib, jb].T.conj()
                BI[jb, jb] = G[slj, :] @ Gj

        else:
            # loop columns
            for jb in range(nb):
                slj = slice(c[jb], c[jb + 1])
                Gj = Gam @ np.conj(G[slj, :].T)
                for ib in range(nb):
                    sli = slice(c[ib], c[ib + 1])
                    BI[ib, jb] = G[sli, :] @ Gj

        return btd

    def _spectral_column_btd(self, elec, herm: bool) -> BlockMatrix:
        """Spectral function from a column array

        Returns a `BlockMatrix` class with only BTD blocks calculated.

        Parameters
        ----------
        herm:
           if true, only calculate the lower triangular part, and copy
           the Hermitian part to the upper triangular part.
           Else, calculate the full matrix via MM.
        """
        G = self._green_column(self.elecs[elec].pvt_dev.ravel())
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
                Gj = Gam @ np.conj(G[slj, :].T)
                for ib in range(max(0, jb - 1), jb):
                    sli = slice(c[ib], c[ib + 1])
                    BI[ib, jb] = G[sli, :] @ Gj
                    BI[jb, ib] = BI[ib, jb].T.conj()
                BI[jb, jb] = G[slj, :] @ Gj

        else:
            # loop columns
            for jb in range(nb):
                slj = slice(c[jb], c[jb + 1])
                Gj = Gam @ np.conj(G[slj, :].T)
                for ib in range(max(0, jb - 1), min(jb + 2, nb)):
                    sli = slice(c[ib], c[ib + 1])
                    BI[ib, jb] = G[sli, :] @ Gj

        return btd

    def _spectral_propagate_array(self, elec, herm: bool) -> np.ndarray:
        nb = len(self.btd)
        nbm1 = nb - 1

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self.elecs[elec].pvt_dev.ravel())
        nblocks = len(blocks)
        A = A @ self._data.gamma[elec] @ np.conj(A.T)

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
                    S[gs(jb - 1, jb - 1)] = -S[gs(jb - 1, jb)] @ np.conj(tY[jb].T)

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
                    S[gs(jb + 1, jb + 1)] = -S[gs(jb + 1, jb)] @ np.conj(tX[jb].T)

        else:
            for jb in range(blocks[0], -1, -1):
                # above
                for ib in range(jb, 0, -1):
                    S[gs(ib - 1, jb)] = -tY[ib] @ S[gs(ib, jb)]
                # calculate next diagonal
                if jb > 0:
                    S[gs(jb - 1, jb - 1)] = -S[gs(jb - 1, jb)] @ np.conj(tY[jb].T)
                # left
                for ib in range(jb, 0, -1):
                    S[gs(jb, ib - 1)] = -S[gs(jb, ib)] @ np.conj(tY[ib].T)

            if nblocks == 2:
                # above and left
                for ib in range(blocks[1], 1, -1):
                    S[gs(ib - 2, blocks[1])] = -tY[ib - 1] @ S[gs(ib - 1, blocks[1])]
                    S[gs(blocks[1], ib - 2)] = -S[gs(blocks[1], ib - 1)] @ np.conj(
                        tY[ib - 1].T
                    )
                # below and right
                for ib in range(blocks[0], nbm1 - 1):
                    S[gs(ib + 2, blocks[0])] = -tX[ib + 1] @ S[gs(ib + 1, blocks[0])]
                    S[gs(blocks[0], ib + 2)] = -S[gs(blocks[0], ib + 1)] @ np.conj(
                        tX[ib + 1].T
                    )

            # below right
            for jb in range(blocks[-1], nb):
                for ib in range(jb, nbm1):
                    S[gs(ib + 1, jb)] = -tX[ib] @ S[gs(ib, jb)]
                # calculate next diagonal
                if jb < nbm1:
                    S[gs(jb + 1, jb + 1)] = -S[gs(jb + 1, jb)] @ np.conj(tX[jb].T)
                # right
                for ib in range(jb, nbm1):
                    S[gs(jb, ib + 1)] = -S[gs(jb, ib)] @ np.conj(tX[ib].T)

        return S

    def _spectral_propagate_bm(self, elec, herm: bool) -> BlockMatrix:
        btd = self.btd
        nb = len(btd)
        nbm1 = nb - 1

        BM = BlockMatrix(self.btd)
        BI = BM.block_indexer

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self.elecs[elec].pvt_dev.ravel())
        nblocks = len(blocks)
        A = A @ self._data.gamma[elec] @ np.conj(A.T)

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
                    BI[jb - 1, jb - 1] = -BI[jb - 1, jb] @ np.conj(tY[jb].T)

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
                    BI[jb + 1, jb + 1] = -BI[jb + 1, jb] @ np.conj(tX[jb].T)

        else:
            for jb in range(blocks[0], -1, -1):
                # above
                for ib in range(jb, 0, -1):
                    BI[ib - 1, jb] = -tY[ib] @ BI[ib, jb]
                # calculate next diagonal
                if jb > 0:
                    BI[jb - 1, jb - 1] = -BI[jb - 1, jb] @ np.conj(tY[jb].T)
                # left
                for ib in range(jb, 0, -1):
                    BI[jb, ib - 1] = -BI[jb, ib] @ np.conj(tY[ib].T)

            if nblocks == 2:
                # above and left
                for ib in range(blocks[1], 1, -1):
                    BI[ib - 2, blocks[1]] = -tY[ib - 1] @ BI[ib - 1, blocks[1]]
                    BI[blocks[1], ib - 2] = -BI[blocks[1], ib - 1] @ np.conj(
                        tY[ib - 1].T
                    )
                # below and right
                for ib in range(blocks[0], nbm1 - 1):
                    BI[ib + 2, blocks[0]] = -tX[ib + 1] @ BI[ib + 1, blocks[0]]
                    BI[blocks[0], ib + 2] = -BI[blocks[0], ib + 1] @ np.conj(
                        tX[ib + 1].T
                    )

            # below right
            for jb in range(blocks[-1], nb):
                for ib in range(jb, nbm1):
                    BI[ib + 1, jb] = -tX[ib] @ BI[ib, jb]
                # calculate next diagonal
                if jb < nbm1:
                    BI[jb + 1, jb + 1] = -BI[jb + 1, jb] @ np.conj(tX[jb].T)
                # right
                for ib in range(jb, nbm1):
                    BI[jb, ib + 1] = -BI[jb, ib] @ np.conj(tX[ib].T)

        return BM

    def _spectral_propagate_btd(self, elec, herm: bool) -> BlockMatrix:
        btd = self.btd
        nb = len(btd)
        nbm1 = nb - 1

        BM = BlockMatrix(self.btd)
        BI = BM.block_indexer

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self.elecs[elec].pvt_dev.ravel())
        A = A @ self._data.gamma[elec] @ np.conj(A.T)

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
                BI[b - 1, b - 1] = -A @ np.conj(tY[b].T)
                BI[b, b - 1] = A.T.conj()
            # right
            for b in range(blocks[-1], nbm1):
                A = -BI[b, b] @ np.conj(tX[b].T)
                BI[b, b + 1] = A
                BI[b + 1, b + 1] = -tX[b] @ A
                BI[b + 1, b] = A.T.conj()

        else:
            # above
            for b in range(blocks[0], 0, -1):
                dtY = np.conj(tY[b].T)
                A = -tY[b] @ BI[b, b]
                BI[b - 1, b] = A
                BI[b - 1, b - 1] = -A @ dtY
                BI[b, b - 1] = -BI[b, b] @ dtY
            # right
            for b in range(blocks[-1], nbm1):
                A = -BI[b, b] @ np.conj(tX[b].T)
                BI[b, b + 1] = A
                BI[b + 1, b + 1] = -tX[b] @ A
                BI[b + 1, b] = -tX[b] @ BI[b, b]

        return BM

    def _coefficient_state_reduce(self, elec, coeff: np.ndarray, U: np.ndarray, cutoff):
        """U on input is a fortran-index as returned from eigh or svd

        Also sorts"""
        # Select only the first N components where N is the
        # number of orbitals in the electrode (there can't be
        # any more propagating states anyhow).
        N = len(self._data.gamma[elec])

        if callable(cutoff):
            idx = cutoff(coeff)
            if len(idx) == 0:
                # always retain at least 2 eigen-values
                # Otherwise we will sometimes get 0 states...
                idx = np.argsort(-np.fabs(coeff))[: min(N, 2)]

            # sort it
            idx = idx[np.argsort(-coeff[idx])]
        else:
            idx = np.argsort(-coeff)

        # reduce idx to max N elements
        idx = idx[:N]

        return coeff[idx], U[:, idx]

    def scattering_state(
        self,
        E: complex,
        elec,
        k: KPoint = (0, 0, 0),
        cutoff: Union[float, Callable] = 0.0,
        method: Literal["svd:gamma", "svd:A", "eig"] = "svd:gamma",
        dtype=np.complex128,
        *args,
        **kwargs,
    ) -> si.physics.StateCElectron:
        r"""Calculate the scattering states for a given `E` and `k` point from a given electrode

        The scattering states are the eigen states of the spectral function:

        .. math::
            \mathbf A_{\mathfrak e}(E,\mathbf k) \mathbf u_i = 2\pi a_i \mathbf u_i

        where :math:`a_i` is the DOS carried by the :math:`i`'th scattering
        state.

        Parameters
        ----------
        E :
           the energy to calculate at, may be a complex value.
        elec : str or int
           the electrode to calculate the spectral function from
        k :
           k-point to calculate the spectral function at
        cutoff :
           Cut off the returned scattering states at some DOS value. Any scattering states
           with relative eigenvalues (to the largest eigenvalue), lower than `cutoff` are discarded.
           For example, we keep according to :math:`\epsilon_i/\max(\epsilon_i) > \mathrm{cutoff}`.
           Values above or close to 0.1 should be used with care.
           Can be a function, see the details of this class.
        method :
           which method to use for calculating the scattering states.
           Use only the ``eig`` method for testing purposes as it is extremely slow
           and requires a substantial amount of memory.
           The ``svd:gamma`` is the fastests while retaining complete precision.
           The ``svd:A`` may be even faster for very large systems with
           very little loss of precision, since it diagonalizes :math:`\mathbf A` in
           the subspace of the electrode `elec` and reduces the propagated part of the spectral
           matrix.
        cutoff_elec : float, optional
           Only used for ``method=svd:A``. The initial block of the spectral function is
           diagonalized and only eigenvectors with relative eigenvalues
           ``>=cutoff_elec`` are retained.
           thus reducing the initial propagated modes. The normalization explained for `cutoff`
           also applies here.
           Can be a function, see the details of this class.

        Returns
        -------
        sisl.physics.electron.StateCElectron
           the scattering states from the spectral function. The ``scat.state`` contains
           the scattering state vectors (eigenvectors of the spectral function).
           ``scat.c`` contains the DOS of the scattering states scaled by :math:`1/(2\pi)`
           so ensure correct density of states.
           One may recreate the spectral function with ``scat.outer(matrix=scat.c * 2 * pi)``.
        """
        elec = self._elec(elec)
        self._prepare(E, k, dtype)
        method = method.lower().replace(":", "_")
        func = getattr(self, f"_scattering_state_{method}", None)
        if func is None:
            raise ValueError(
                f"{self.__class__.__name__}.scattering_state method is not [full,svd,propagate]"
            )
        return func(method, elec, cutoff, *args, **kwargs)

    def _scattering_state_eig(self, _method, elec, cutoff, **kwargs):
        # We know that scattering_state has called prepare!
        A = self.spectral(elec, self._data.E, self._data.k, **kwargs)
        cutoff = self._as_cutoff_func(cutoff)

        # add something to the diagonal (improves diag precision for small states)
        idx = np.arange(len(A))
        CONST = 0.1
        np.add.at(A, (idx, idx), CONST)
        del idx

        # Now diagonalize A
        DOS, A = eigh_destroy(A)
        # backconvert diagonal
        DOS -= CONST
        # TODO check with overlap convert with correct magnitude (Tr[A] / 2pi)
        DOS /= 2 * np.pi
        DOS, A = self._coefficient_state_reduce(elec, DOS, A, cutoff)

        data = self._data
        info = dict(
            method="full", elec=self._elec_name(elec), E=data.E, k=data.k, cutoff=cutoff
        )

        # always have the first state with the largest values
        return si.physics.StateCElectron(A.T, DOS, self, **info)

    def _scattering_state_svd_gamma(self, _method, elec, cutoff, **kwargs):
        A = self._green_column(self.elecs[elec].pvt_dev.ravel())
        cutoff = self._as_cutoff_func(cutoff)

        # This calculation uses the cholesky decomposition of Gamma
        # combined with SVD of the A column
        if "sqrth" in _method:
            Gam_sqrt = sqrth(self._data.gamma[elec])
        else:
            try:
                Gam_sqrt = cholesky(self._data.gamma[elec], lower=True)
            except np.linalg.LinAlgError:
                # TODO log/warn about reverting to sqrth
                warn(
                    f"{self.__class__.__name__}.scattering_state(svd:gamma) failed "
                    "Cholesky; reverting to Hermitian sqrt."
                )
                Gam_sqrt = sqrth(self._data.gamma[elec])
        A = A @ Gam_sqrt

        # Perform svd
        DOS, A = _scat_state_svd(A, **kwargs)
        DOS, A = self._coefficient_state_reduce(elec, DOS, A, cutoff)

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

    def _scattering_state_svd_a(
        self, _method, elec, cutoff: Union[float, Tuple[float, float]] = 0, **kwargs
    ):
        # Parse the cutoff value
        # Here we may use 2 values, one for cutting off the initial space
        # and one for the returned space.
        if not isinstance(cutoff, Sequence):
            cutoff = (cutoff, cutoff)
        cutoff0, cutoff1 = cutoff[0], cutoff[1]
        cutoff0 = kwargs.get("cutoff_elec", cutoff0)
        cutoff0 = self._as_cutoff_func(cutoff0)
        cutoff1 = self._as_cutoff_func(cutoff1)

        # First we need to calculate diagonal blocks of the spectral matrix
        # This is basically the same thing as calculating the Gf column
        # But only in the 1/2 diagonal blocks of Gf
        blocks, u = self._green_diag_block(self.elecs[elec].pvt_dev.ravel())
        # Calculate the spectral function only for the blocks that host the
        # scattering matrix
        u = u @ self._data.gamma[elec] @ np.conj(u.T)

        # add something to the diagonal (improves diag precision)
        idx = np.arange(len(u))
        CONST = 0.1
        np.add.at(u, (idx, idx), CONST)
        del idx

        # Calculate eigenvalues
        DOS, u = eigh_destroy(u)
        # backconvert diagonal
        DOS -= CONST
        # TODO check with overlap convert with correct magnitude (Tr[A] / 2pi)
        DOS /= 2 * np.pi

        # Remove states for cutoff and size
        # Since there cannot be any addition of states later, we
        # can do the reduction here.
        # This will greatly increase performance for very wide systems
        # since the number of contributing states is generally a fraction
        # of the total electrode space.
        DOS, u = self._coefficient_state_reduce(elec, DOS, u, cutoff0)
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
        DOS, A = self._coefficient_state_reduce(elec, DOS, A, cutoff1)

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

    def transmission(
        self,
        E: complex,
        elec_from,
        elec_to=None,
        k: KPoint = (0, 0, 0),
        dtype=np.complex128,
    ) -> Union[float, tuple[float, ...]]:
        r"""Calculate the transmission between an electrode, and one or more other electrodes

        The transmission function is calculated as:

        .. math::
            \mathcal T_{\mathfrak e\to\mathfrak e'}(E,\mathbf k) =
            \boldsymbol \Gamma_{\mathfrak e'}(E,\mathbf k)
            \mathbf G(E,\mathbf k)
            \boldsymbol\Gamma_{\mathfrak e}(E,\mathbf k)
            \mathbf G^\dagger(E,\mathbf k)

        Parameters
        ----------
        E :
           the energy to calculate at, may be a complex value.
        elec_from : str or int
           the electrode to calculate the transmission *from*.
        elec_to : str or int or list of
           the electrode(s) to calculate the transmission *to*.
        k :
           k-point to calculate the transmission at.
        """
        # Calculate the full column green function
        elec_from = self._elec(elec_from)
        is_single, elec_to = self._serialize_elecs(elec_to, elec_from)

        # Prepare calculation @ E and k
        self._prepare(E, k, dtype)

        # Get full G in column of 'from'
        G = self._green_column(self.elecs[elec_from].pvt_dev)

        # The gamma matrices
        Gam = self._data.gamma

        # Now calculate the transmission
        def calc(Gam_from, elec_to, Gam_to, G):
            pvt = self.elecs[elec_to].pvt_dev.ravel()
            g = G[pvt, :]
            A = g @ Gam_from @ dagger(g)

            # Return the trace of the final quadruple product
            return (Gam_to.ravel() @ A.T.ravel()).real

        T = tuple(calc(Gam[elec_from], elec, Gam[elec], G) for elec in elec_to)

        if is_single:
            return T[0]
        return T

    def scattering_matrix(
        self,
        E: complex,
        elec_from,
        elec_to=None,
        k: KPoint = (0, 0, 0),
        cutoff: float = 1e-4,
        dtype=np.complex128,
    ) -> Union[si.physics.StateElectron, tuple[si.physics.StateElectron, ...]]:
        r""" Calculate the scattering matrix (S-matrix) between `elec_from` and `elec_to`

        The scattering matrix is calculated as

        .. math::
               \mathbf S_{\mathfrak e'\mathfrak e}(E, \mathbf) = -\delta_{\alpha\beta} + i
               \tilde{\boldsymbol\Gamma}_{\mathfrak e'}
               \mathbf G
               \tilde{\boldsymbol\Gamma}_{\mathfrak e}

        Here :math:`\tilde{\boldsymbol\Gamma}` is defined as:

        .. math::
            \boldsymbol\Gamma(E,\mathbf k) \mathbf u_i &= \lambda_i \mathbf u_i
            \\
            \tilde{\boldsymbol\Gamma}(E,\mathbf k) &= \operatorname{diag}\{ \sqrt{\boldsymbol\lambda} \} \mathbf u

        Once the scattering matrices have been calculated one can calculate the full transmission
        function

        .. math::
              \mathcal T_{\mathfrak e\to\mathfrak e'}(E, \mathbf k) = \operatorname{Tr}\big[
              \mathbf S_{\mathfrak e'\mathfrak e }^\dagger
              \mathbf S_{\mathfrak e'\mathfrak e }\big]

        The scattering matrix approach can be found in details in
        :cite:`Sanz2023-gv`.


        Parameters
        ----------
        E :
           the energy to calculate at, may be a complex value.
        elec_from : str or int
           the electrode where the scattering matrix originates from
        elec_to : str or int or list of
           where the scattering matrix ends in.
        k :
           k-point to calculate the scattering matrix at
        cutoff :
           cutoff eigen states of the broadening matrix.
           The cutoff is based on a relative fraction of the maximum eigen value.
           that are below this value.
           For example, we keep according to :math:`\lambda_i/\max(\lambda_i) > \mathrm{cutoff}`.
           A too high value will remove too many eigen states and results will be wrong.
           A small value improves precision at the cost of bigger matrices.
           The :math:`\Gamma` matrix should be positive definite, however, due
           to the imaginary part of the self-energies it tends to only be *close*
           to positive definite.

        Returns
        -------
        sisl.physics.electron.StateElectron or tuple[sisl.physics.electron.StateElectron,...]
           for each `elec_to` a scattering matrix will be returned. Its dimensions will be
           depending on the `cutoff` value at the cost of precision.
        """
        # Calculate the full column green function
        elec_from = self._elec(elec_from)
        is_single, elec_to = self._serialize_elecs(elec_to, elec_from)

        cutoff = self._as_cutoff_func(cutoff)
        # Prepare calculation @ E and k
        self._prepare(E, k, dtype)
        self._prepare_tgamma(E, k, dtype, cutoff)

        # Get full G in column of 'from'
        G = self._green_column(self.elecs[elec_from].pvt_dev.ravel())

        # the \tilde \Gamma functions
        tG = self._data.tgamma

        data = self._data
        info = dict(elec=self._elec_name(elec_from), E=data.E, k=data.k, cutoff=cutoff)

        # Now calculate the S matrices
        def calc(elec_from, jtgam_from, elec_to, tgam_to, G):
            pvt = self.elecs[elec_to].pvt_dev.ravel()
            g = G[pvt, :]
            ret = dagger(tgam_to) @ g @ jtgam_from
            if elec_from == elec_to:
                idx = np.arange(min(ret.shape))
                np.add.at(ret, (idx, idx), -1)
            return si.physics.StateElectron(
                ret.T, self, **info, elec_to=self._elec_name(elec_to)
            )

        jtgam_from = 1j * tG[elec_from]
        S = tuple(calc(elec_from, jtgam_from, elec, tG[elec], G) for elec in elec_to)

        if is_single:
            return S[0]
        return S

    def eigenchannel_from_scattering_matrix(
        self,
        scat_matrix: si.physics.StateElectron,
        ret_out: bool = False,
    ) -> Union[si.physics.StateCElectron, tuple[si.physics.StateCElectron, ...]]:
        r"""Calculate the eigenchannel from a scattering matrix

        The energy and k-point is inferred from the `state_matrix` object as returned from
        `scattering_matrix`.

        The eigenchannels are the SVD of the scattering matrix in the
        DOS weighted scattering states:

        Parameters
        ----------
        state_matrix :
            the scattering matrix as obtained from `scattering_matrix`

        Returns
        -------
        T_eig_in : sisl.physics.electron.StateCElectron
            the transmission eigenchannels as seen from the incoming state, the ``T_eig.c`` contains the transmission
            eigenvalues.
        T_eig_out: sisl.physics.electron.StateCElectron
            the transmission eigenchannels as seen from the outgoing state, the ``T_eig.c`` contains the transmission
            eigenvalues.
            Only returned if `ret_out` is true.
        """
        tt_eig, U = _scat_state_svd(state_matrix.state.T, ret_uv=ret_out)
        # Here there is a wrong pre-factor of 2pi
        tt_eig *= 2 * np.pi

        info = {**state_matrix.info}
        SCE = si.physics.StateCElectron
        if ret_out:
            U_in = SCE(U[0].T, tt_eig, self, **info)
            # note: V^dagger is returned from svd, so only conj
            U_out = SCE(np.conj(U[1]), tt_eig, self, **info)
            return U_in, U_out
        else:
            U = SCE(U.T, tt_eig, self, **info)
            return U

    def eigenchannel(
        self, state: si.physics.StateCElectron, elec_to=None, ret_coeff: bool = False
    ) -> Union[
        si.physics.StateCElectron,
        tuple[si.physics.StateCElectron, si.physics.StateElectron],
    ]:
        r""" Calculate the eigenchannel from scattering states entering electrodes `elec_to`

        The energy and k-point is inferred from the `state` object as returned from
        `scattering_state`.

        The eigenchannels are the eigenstates of the transmission matrix in the
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
        state :
            the scattering states as obtained from `scattering_state`
        elec_to : str or int (list or not)
            which electrodes to consider for the transmission eigenchannel
            decomposition (the sum in the above equation).
            Defaults to all but the origin electrode.
        ret_coeff :
            return also the scattering state coefficients

        Returns
        -------
        T_eig : sisl.physics.electron.StateCElectron
            the transmission eigenchannels, the ``T_eig.c`` contains the transmission
            eigenvalues.
        coeff : sisl.physics.electron.StateElectron
            coefficients of `state` that creates the transmission eigenchannels
            Only returned if `ret_coeff` is True. There is a one-to-one correspondance
            between ``coeff`` and ``T_eig`` (with a prefactor of :math:`\sqrt{2\pi}`).
            This is equivalent to the ``T_eig`` states in the scattering state basis.
        """
        self._prepare_se(state.info["E"], state.info["k"], state.dtype)
        _, elec_to = self._serialize_elecs(elec_to, state.info["elec"])

        # The sign shouldn't really matter since the states should always
        # have a finite DOS, however, for completeness sake we retain the sign.
        # We scale the vectors by sqrt(DOS/2pi).
        # This is because the scattering states from self.scattering_state
        # stores eig(A) / 2pi.
        sqDOS = signsqrt(state.c)
        # Retrieve the scattering states `A` and apply the proper scaling
        # We need this scaling for the eigenchannel construction anyways.
        A = state.state.T * sqDOS

        # create short hands
        G = self._data.gamma

        # Create the first electrode
        el = elec_to[0]
        idx = self.elecs[el].pvt_dev.ravel()

        u = A[idx]
        # the summed transmission matrix
        Ut = u.conj().T @ G[el] @ u

        # same for other electrodes
        for el in elec_to[1:]:
            idx = self.elecs[el].pvt_dev.ravel()
            u = A[idx]
            Ut += u.conj().T @ G[el] @ u

        # TODO currently a factor depends on what is used
        #      in `scattering_states`, so go check there.
        #      The state.c contains a factor /(2pi) meaning
        #      that we should remove that factor here.
        # diagonalise the transmission matrix tt to get the eigenchannels
        teig, Ut = eigh_destroy(Ut)
        # Reorder Ut to have them descending
        Ut = Ut[:, ::-1]
        # remove factor /2pi
        teig = teig[::-1] * 2 * np.pi

        info = {**state.info, "elec_to": tuple(self._elec_name(e) for e in elec_to)}

        # Backtransform A in the basis of Ut to form the eigenchannels
        teig = si.physics.StateCElectron((A @ Ut).T, teig, self, **info)
        if ret_coeff:
            return teig, si.physics.StateElectron(Ut.T, self, **info)
        return teig
