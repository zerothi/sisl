# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Optional, Self

import numpy as np
import numpy.typing as npt
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
from ._btd import (
    _blocks_in_range,
    _btd_blocks_sub,
    _btd_memory,
    _device_permute,
    _graph,
    _permute_a2o,
    _split_pivot,
)
from ._electrode import *
from ._electrode import (
    _check_placement,
    _fdf_device,
    _fdf_downfold,
    _fdf_electrode,
    _restrict_nsc_elecs,
    _TransportFdf,
)
from ._help import *
from ._help import _check_atoms, _split_space, _to_orbs

__all__ = ["DeviceGreen"]


# shift added to the diagonal of the spectral function before diagonalization,
# it improves the precision for states with small DOS and is subtracted from
# the eigenvalues afterwards
_SPECTRAL_DIAG_SHIFT = 0.1

# keyword arguments consumed by the scattering state methods, see `_scat_state_svd`
# and `DeviceGreen._scattering_state_svd_a`
_SCAT_STATE_KWARGS = ("scale", "ret_uv", "driver", "ncv", "tol", "v0", "cutoff_elec")


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


def _fdf_permute_method(value: str) -> tuple[str, str]:
    """Convert a TBtrans permutation method to the space and a `reduce_bandwidth` method"""
    space, method = _split_pivot(value)
    method = {"rev-cm": "rcm"}.get(method, method)
    if method not in ("gps", "ggps", "cm", "rcm"):
        warn(f"permutation method {value!r} is not implemented, using GPS")
        method = "gps"
    return space, method


class _DeviceElectrode(NamedTuple):
    """An electrode placed in the permuted device region of `DeviceGreen`"""

    #: the electrode
    elec: Electrode
    #: positions of the self-energy in the permuted device region (sorted)
    permute: np.ndarray
    #: permutes the self-energy rows and columns to match `permute`
    sort: np.ndarray
    #: whether the electrode region is replaced by the bulk electrode matrix
    bulk: bool


def _elec_index(elecs: Sequence[Electrode], elec, clsname: str) -> int:
    """Index of `elec` in `elecs`

    `elec` is an index, a name (case-insensitive), or an electrode object
    (also the electrode of a `DownfoldElectrode` in `elecs`).
    """
    if isinstance(elec, Electrode):
        for ielec, el in enumerate(elecs):
            while el is not None:
                if el is elec:
                    return ielec
                el = getattr(el, "electrode", None)
        if elec.name is None:
            raise KeyError(f"{clsname} does not contain the (unnamed) electrode {elec}")
        elec = elec.name
    if isinstance(elec, str):
        for ielec, el in enumerate(elecs):
            if el.name is not None and el.name.lower() == elec.lower():
                return ielec
        names = [el.name for el in elecs]
        raise KeyError(
            f"{clsname} has no electrode {elec!r}, available electrodes: {names}"
        )
    return elec


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

    A device can be created from any Hamiltonian, here a 1D chain along
    the first lattice vector, with the electrodes on the first and last atom:

    .. code-block:: python

       import sisl
       from sisl_toolbox.btd import *

       chain = sisl.Geometry(
           [0, 0, 0], sisl.Atom(1, R=1.1), lattice=sisl.Lattice([1, 10, 10], nsc=[3, 1, 1])
       )
       H_elec = sisl.Hamiltonian(chain)
       H_elec.construct([(0.1, 1.1), (0.0, -1.0)])
       H = H_elec.tile(10, 0)

       # the bulk electrodes extend along -a (left) and +a (right)
       left = BlochElectrode(sisl.RecursiveSI(H_elec, "-A"), name="Left")
       right = BlochElectrode(sisl.RecursiveSI(H_elec, "+A"), name="Right")
       G = DeviceGreen(H, [(left, [0]), (right, [9])])
       T = G.transmission(0.5, "Left", "Right")

    Consider a regular 2 electrode setup with transport direction
    along the 3rd lattice vector. Then the following example may
    be used to calculate the eigen-channels.

    The below short-form of reading all variables should cover most variables
    encountered in the FDF file.

    .. code-block:: python

       G = DeviceGreen.from_fdf("RUN.fdf")

       # Calculate the scattering state from the left electrode
       # and then the eigen channels to the right electrode
       state = G.scattering_state(0.1, "Left")
       eig_channel = G.eigenchannel(state, "Right")

    ``DeviceGreen.from_fdf`` only reads the fdf file (and the Hamiltonian files
    it refers to). The downfolding regions and the permutation tables are calculated.

    To re-use the downfolding regions and permutation tables of a TBtrans calculation,
    use ``DeviceGreen.from_tbt``, which requires the electrode Hamiltonians
    and their semi-infinite directions (these are not stored in the TBtrans output):

    .. code-block:: python

       import sisl
       from sisl_toolbox.btd import *

       H = sisl.Hamiltonian.read("DEVICE.nc")
       tbt = sisl.get_sile("siesta.TBT.nc")
       G = DeviceGreen.from_tbt(
           tbt,
           H,
           {
               "Left": (sisl.Hamiltonian.read("LEFT.nc"), "-C"),
               "Right": (sisl.Hamiltonian.read("RIGHT.nc"), "+C"),
           },
       )

    Both are short-hands for creating the electrodes (`BlochElectrode`), downfolding
    them into the device region (`DownfoldElectrode`) and passing them, together with
    the orbitals they are placed on, to `DeviceGreen`.


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

    def __init__(
        self,
        H: si.Hamiltonian,
        elecs: Sequence[
            Electrode
            | tuple[Electrode, npt.ArrayLike | Mapping[str, npt.ArrayLike]]
            | tuple[Electrode, npt.ArrayLike | Mapping[str, npt.ArrayLike], bool]
        ],
        permute: (
            PermuteBTD | npt.ArrayLike | str | Mapping[str, PermuteBTD | npt.ArrayLike]
        ) = "gps",
        eta: float | None = None,
    ):
        r"""Create Green function with Hamiltonian and BTD matrix elements

        Parameters
        ----------
        H :
            the Hamiltonian containing the device region.
            Couplings along the semi-infinite directions of the electrodes are
            removed (on a copy), the self-energies replace them.
        elecs :
            the electrodes, each a tuple of the electrode and
            its indices in `H`, which corresponds to the rows of the self-energy matrix.
            The indices are atoms, ``{"atoms": idx}`` (equivalent), or
            ``{"orbitals": idx}`` (not accounting for spin).
            Use orbitals when the self-energy is not on all orbitals of
            its atoms, e.g. for a `DownfoldElectrode`.
            An optional third element (``bulk``, defaults to false) replaces the
            device Hamiltonian of the electrode orbitals by the electrode
            Hamiltonian (`Electrode.bulk_matrix`), otherwise the self-energy
            is put on the device Hamiltonian.
            An electrode may also be given alone, when its orbitals (`Electrode.orbs`)
            are set (e.g. from `Electrode.from_fdf`), ``bulk`` is then
            ``info.bulk`` (if the electrode has a bulk matrix).
        permute :
            the permutation table and BTD blocks of the device region,
            a `PermuteBTD` or an array (the permutation table only).
            The indices are atoms, ``{"atoms": permute}`` (equivalent), or
            ``{"orbitals": permute}`` (not accounting for spin).
            Atoms are expanded to all their orbitals (in order), and the block
            sizes to the number of orbitals in each block.
            For an array, the BTD blocks are found with
            `~sisl_toolbox.btd.permute.btd_blocks` (for atoms or orbitals).
            If a `str`, the full `H` is the device region and it is passed
            as the ``method`` argument to `btd_partition`, ``[atom+|orb+]<method>``
            partitions the atoms (default) or orbitals, e.g. ``"orb+gps"``.
            For ``"electrode"`` the (at most 2) electrodes are placed
            at the start and end of the BTD matrix.
            For an array and a `str` the electrodes are dense blocks
            in the partitioning, see `btd_partition`.
        eta :
            imaginary part of the device region (for real energies).
            Defaults to :math:`\eta/10` for the smallest :math:`\eta`
            of the electrodes (`Electrode.eta`).

        Warns
        -----
        SislWarning
            when an electrode does not seem to be placed correctly in `H`:
            its orbitals have a different order (or relative positions) than the electrode,
            or `H` couples it to device orbitals on its semi-infinite side
            (wrong semi-infinite direction, or not placed at the edge of the device).
            Electrodes downfolded with `DownfoldElectrode` are checked
            when they are created.
        """

        def ensure_this_elec(elec):
            if isinstance(elec, si.physics.SelfEnergy):
                # automatic upscaling with no Name!
                return BlochElectrode(elec)
            return elec

        def as_tuple(elec):
            """Convert an electrode to the tuple form ``(elec, orbitals, *bulk)``"""
            elec = ensure_this_elec(elec)
            if not isinstance(elec, Electrode):
                elec, idx, *bulk = elec
                return (
                    ensure_this_elec(elec),
                    _to_orbs(H, idx, "electrode indices"),
                    *bulk,
                )
            if elec.orbs is None:
                raise ValueError(
                    f"{self.__class__.__name__} electrode {elec.name} has no orbitals, "
                    "pass it as (electrode, atoms)."
                )
            bulk = elec.has_bulk and bool(elec.info.get("bulk", False))
            return (elec, elec.orbs, bulk)

        elecs = [as_tuple(elec) for elec in elecs]
        names = [elec[0].name.lower() for elec in elecs if elec[0].name is not None]
        if len(set(names)) != len(names):
            raise ValueError(
                f"{self.__class__.__name__} got electrodes with the same name "
                f"(case-insensitive): {names}"
            )

        # The electrode self-energies replace the couplings along the
        # semi-infinite directions
        H_in = H
        H = _restrict_nsc_elecs(H, [elec[0] for elec in elecs])
        if H is not H_in:
            info(
                f"{self.__class__.__name__} removes the couplings of H along the "
                f"semi-infinite directions of the electrodes, nsc={H_in.nsc} -> {H.nsc}."
            )
        self.H = H

        # The partitioning is done for atoms or orbitals (space)
        if isinstance(permute, str):
            space, permute = _split_pivot(permute)
        else:
            space, permute = _split_space(permute, "permute")
            if space == "atoms":
                idx = permute.permute if isinstance(permute, PermuteBTD) else permute
                _check_atoms(H, np.asarray(idx).ravel(), "permute indices")

        if not isinstance(permute, PermuteBTD):
            # The electrode self-energies are dense blocks in the device region
            if space == "atoms":
                fix = lambda atoms: np.unique(H.o2a(atoms))
            else:
                fix = lambda atoms: np.asarray(atoms).ravel()
            dense = [fix(elec[1]) for elec in elecs]
            if space == "atoms":
                csr = H.toSparseAtom()
            else:
                csr = H

        if isinstance(permute, str):
            kwargs = {}
            if permute.startswith("electrode"):
                # Only fix the start electrode (name or index), defaults
                # to the first electrode. The user can swap them.
                permute, _, start = permute.partition("=")
                if start.isdigit():
                    start = int(start)
                ielec = _elec_index(
                    [elec[0] for elec in elecs], start or 0, self.__class__.__name__
                )
                kwargs = {"start": dense[ielec]}

            permute = btd_partition(csr, method=permute, dense=dense, **kwargs)
        elif not isinstance(permute, PermuteBTD):
            permute = np.asarray(permute).ravel()
            permute = PermuteBTD(permute, _btd_blocks_sub(csr, permute, dense=dense))
        if space == "atoms":
            permute = _permute_a2o(H.geometry, permute)

        # Retrieve the (spin) expansion of the blocks/orbitals.
        # The permutation tables are orbital indices, without spin.
        expand = get_expand(len(H), H.no)

        if expand > 1:
            info(
                f"{self.__class__.__name__} expands the permutation table and blocks "
                f"by {expand} to account for the spin-configuration of the Hamiltonian."
            )

        # the permutation table for the device region
        self.permute = expand_orbs(permute.permute, expand)
        # the BTD blocks (in the permuted space) for the device region
        self.blocks = expand_blocks(permute.blocks, expand)

        # the electrode orbitals in the permuted device region
        # inv_permute[orbital in H] = index in the permuted device region
        inv_permute = np.full(len(H), -1, dtype=self.permute.dtype)
        inv_permute[self.permute] = np.arange(len(self.permute))
        # The positions are sorted (required by the block algorithms),
        # and the self-energies are permuted accordingly, see `_DeviceElectrode`.
        # There may be no electrodes
        self._elecs = []
        for elec, orbs, *bulk in elecs:
            bulk = bool(bulk[0]) if bulk else False
            if bulk and not elec.has_bulk:
                raise ValueError(
                    f"{self.__class__.__name__} electrode {elec.name} is bulk, "
                    f"but {elec.__class__.__name__} has no bulk matrix."
                )
            permute = inv_permute[expand_orbs(np.asarray(orbs).ravel(), expand)]
            if np.any(permute < 0):
                raise ValueError(
                    f"{self.__class__.__name__} electrode {elec.name} has orbitals "
                    "outside the device region."
                )
            if len(permute) != len(elec):
                raise ValueError(
                    f"{self.__class__.__name__} electrode {elec.name} has size {len(elec)}, "
                    f"but its orbitals gives {len(permute)}"
                )
            sort = np.argsort(permute)
            self._elecs.append(_DeviceElectrode(elec, permute[sort], sort, bulk))
            if not isinstance(elec, DownfoldElectrode):
                _check_placement(
                    H,
                    elec,
                    orbs,
                    self.permute[::expand] // expand,
                    self.__class__.__name__,
                )

        # The downfolded orbitals are removed from the calculation, they
        # must not be part of the device region, nor other downfolding regions
        is_device = np.zeros(H.no, dtype=bool)
        is_device[self.permute // expand] = True
        count = np.zeros(H.no, dtype=np.int32)
        downs = [
            el.elec for el in self._elecs if isinstance(el.elec, DownfoldElectrode)
        ]
        for elec in downs:
            if is_device[elec.orbs_down].any():
                raise ValueError(
                    f"{self.__class__.__name__} electrode {elec.name} has a "
                    "downfolding region overlapping the device region."
                )
            count[elec.orbs_down] += 1
        if (count > 1).any():
            names = [elec.name for elec in downs if (count[elec.orbs_down] > 1).any()]
            raise ValueError(
                f"{self.__class__.__name__} electrodes {names} have overlapping "
                "downfolding regions."
            )

        # Each electrode is a dense block, which must maximally span
        # 2 (neighboring) blocks.
        blocks_end = np.cumsum(self.blocks)
        for el in self._elecs:
            b0, b1 = np.searchsorted(blocks_end, el.permute[[0, -1]], side="right")
            if b1 - b0 > 1:
                raise ValueError(
                    f"{self.__class__.__name__} electrode {el.elec.name} spans the "
                    f"BTD blocks {b0}-{b1}, it may maximally span 2 neighboring blocks. "
                    "Use btd_partition(..., dense=...) with the electrode orbitals."
                )

        # global device eta
        if eta is None:
            etas = [elec.eta for elec in self.elecs if elec.eta is not None]
            if not etas:
                raise ValueError(
                    f"{self.__class__.__name__} could not determine eta from "
                    "the electrodes, please pass eta explicitly."
                )
            eta = min(etas) / 10
        self.eta = eta

        # Create BTD indices
        self.blocks_cum0 = np.empty([len(self.blocks) + 1], dtype=self.blocks.dtype)
        self.blocks_cum0[0] = 0
        self.blocks_cum0[1:] = np.cumsum(self.blocks)
        self.clear()

    @property
    def elecs(self) -> list[Electrode]:
        """The electrodes"""
        return [el.elec for el in self._elecs]

    def __str__(self) -> str:
        MB = _btd_memory(self.blocks, np.complex128)
        suffix = "MB"
        if MB > 1024:
            MB /= 1024
            suffix = "GB"

        ret = f"{self.__class__.__name__}{{no: {len(self)}, blocks: {len(self.blocks)}, btd_mem: {MB:.1f}{suffix}, eta: {self.eta:.3e}"
        for ielec, elec in enumerate(self.elecs):
            e = str(elec).replace("\n", "\n  ")
            ret = f"{ret},\n {self._elec_name(ielec)}:\n  {e}"
        return f"{ret}\n}}"

    @classmethod
    def _from_downfold(
        cls,
        H: si.Hamiltonian,
        downfolds,
        permute,
        eta: float,
        bulk: bool | Mapping[str, bool],
        **kwargs,
    ) -> Self:
        """Create from electrodes and their downfolding regions

        Parameters
        ----------
        downfolds :
            for each electrode a tuple of ``(electrode, orbs_down, blocks_down, orbs_elec)``,
            see `DownfoldElectrode`.
            If ``orbs_down`` is None, the electrode is not downfolded, and placed
            directly on ``orbs_elec`` (in the device region).
        bulk :
            whether the electrode Hamiltonian is used in the electrode region
            (in the downfolding or device region).
            If a `bool`, it is the default for electrodes without ``info.bulk``
            (from `Electrode.from_fdf`).
            Otherwise a mapping from the electrode names.
        """
        elecs = []
        for elec, orbs_down, blocks_down, orbs_elec in downfolds:
            if isinstance(bulk, Mapping):
                elec_bulk = bulk[elec.name]
            else:
                elec_bulk = elec.info.get("bulk", bulk)
            if orbs_down is None:
                elecs.append((elec, {"orbitals": orbs_elec}, elec_bulk))
                continue
            down = DownfoldElectrode(
                elec,
                H,
                {"orbitals": orbs_elec},
                {"orbitals": PermuteBTD(orbs_down, blocks_down)},
                eta_device=eta,
                bulk=elec_bulk,
            )
            elecs.append((down, {"orbitals": down.orbs}))
        return cls(H, elecs, permute, eta=eta, **kwargs)

    @classmethod
    def from_tbt(
        cls,
        tbt: SileLike,
        H: si.Hamiltonian,
        elecs: Mapping[str, Electrode | tuple[si.Hamiltonian, str]],
        eta: Optional[float] = None,
        bulk: bool | Mapping[str, bool] = True,
        **kwargs,
    ) -> Self:
        """Return a new `DeviceGreen` using the regions and permutation tables of a TBtrans calculation

        The device region, the downfolding regions and their permutation tables are
        read from `tbt`. The electrode Hamiltonians are not stored in `tbt`, and
        must be passed in `elecs`.

        Parameters
        ----------
        tbt :
           the TBtrans output file (``*.TBT.nc``)
        H :
           the Hamiltonian of the full system (as used in TBtrans).
           Couplings along the semi-infinite directions of the electrodes are
           removed (on a copy).
        elecs :
           all electrodes in `tbt`, by their names. Either

           - an `Electrode`, used as-is
           - a tuple of the electrode Hamiltonian and its semi-infinite direction
             (e.g. ``(H_left, "-C")``). The chemical potential, imaginary part
             and Bloch expansion are read from `tbt` (the Hamiltonian is copied).
        eta :
           imaginary part of the device region, defaults to the TBtrans value.
        bulk :
           whether the electrodes are bulk in the downfolding region, either
           for all electrodes, or a mapping from the electrode names.
           A `bool` is overwritten by ``info.bulk`` of the electrodes (if present).
        kwargs :
           passed to the class instantiating.
        """
        if isinstance(tbt, (str, Path)):
            tbt = si.get_sile(tbt)
        clsname = f"{cls.__name__}.from_tbt"

        missing = [name for name in tbt.elecs if name not in elecs]
        unknown = [name for name in elecs if name not in tbt.elecs]
        if missing or unknown:
            raise ValueError(
                f"{clsname} requires the electrodes {tbt.elecs}, "
                f"missing {missing}, unknown {unknown}"
            )

        bloch_elecs = []
        for name in tbt.elecs:
            elec = elecs[name]
            if isinstance(elec, Electrode):
                if isinstance(elec, BlochElectrode) and not np.allclose(
                    tbt.bloch(name), elec.bloch.bloch
                ):
                    warn(
                        f"{clsname}(electrode={name}) has inconsistent Bloch "
                        f"expansions, {tbt} = {tbt.bloch(name)}, "
                        f"electrode = {elec.bloch.bloch}"
                    )
                if elec.eta is not None and not np.allclose(tbt.eta(name), elec.eta):
                    warn(
                        f"{clsname}(electrode={name}) has inconsistent imaginary "
                        f"eta, {tbt} = {tbt.eta(name)} eV, electrode = {elec.eta} eV"
                    )
            else:
                Helec, semi_inf = elec
                Helec = Helec.copy()
                Helec.shift(tbt.mu(name))
                elec = BlochElectrode(
                    si.RecursiveSI(Helec, semi_inf, eta=tbt.eta(name)),
                    bloch=tbt.bloch(name),
                    name=name,
                )
            bloch_elecs.append(elec)

        H = _restrict_nsc_elecs(H, bloch_elecs)
        if eta is None:
            eta = tbt.eta()

        downfolds = []
        for name, elec in zip(tbt.elecs, bloch_elecs):
            # The downfolding region, in the BTD order of TBtrans.
            # The last block is the electrode self-energy in the device region.
            orbs_down = tbt.pivot_down(name)
            blocks_down = tbt.btd(name)
            if not np.array_equal(orbs_down[-blocks_down[-1] :], tbt.pivot(name)):
                raise ValueError(
                    f"{clsname}(electrode={name}) found inconsistent "
                    f"down-folding and device orbitals in {tbt}."
                )
            orbs_elec = H.a2o(tbt.a_elec(name), all=True)
            downfolds.append((elec, orbs_down, blocks_down, orbs_elec))

        permute = {"orbitals": PermuteBTD(tbt.pivot(), tbt.btd())}
        return cls._from_downfold(H, downfolds, permute, eta, bulk, **kwargs)

    @classmethod
    def from_fdf(
        cls,
        fdf: SileLike,
        prefix: Literal["TBT", "TS"] = "TBT",
        eta: Optional[float] = None,
        use_gf: bool = False,
        **kwargs,
    ) -> Self:
        """Return a new `DeviceGreen` using only the information in the fdf file

        The Hamiltonians (device and electrodes) are read from the files
        in the fdf file (relative to the fdf file).
        The device region, the downfolding regions and their permutation tables
        are calculated, i.e. they need not be the same as the ones TBtrans uses.

        The following fdf flags are used (``TBT`` prefixed flags take precedence
        over ``TS`` prefixed flags, when `prefix` is ``TBT``):

        - ``TS.Elecs`` (block): the electrode names, each electrode is read as in
          `Electrode.from_fdf`, and downfolded through the regions found by
          `downfold_partition`.
          Defaults to ``Left`` and ``Right`` (as TranSiesta/TBtrans).
          The electrode atoms are determined by ``electrode-position``,
          either the first atom ``<int>``, or the last atom ``end <int>``
          (negative values counts from the end, ``-1`` is the last atom).
        - ``TS.Atoms.Buffer``: atoms removed from the calculation
        - ``TBT.Atoms.Device``: atoms of the device region, defaults to all atoms
          not in the buffer atoms, nor in the electrodes.
          The device region may contain electrode atoms (not buffer atoms).
          Electrodes fully in the device region are not downfolded,
          electrodes partially in the device region are partially downfolded
          (only their atoms outside the device region are downfolded).
        - ``TBT.HS``: the device Hamiltonian, defaults to ``<SystemLabel>.TSHS``
        - ``TBT.Contours.Eta`` (or ``TS.Contours.nEq.Eta``): imaginary part of the
          device region, defaults to the smallest electrode imaginary part divided by 10.
        - ``TBT.BTD.Pivot.Device`` (or ``TBT.BTD.Pivot``, ``TS.BTD.Pivot``): the bandwidth
          reduction method of the device, one of ``[atom+|orb+](GPS|GGPS|CM|rev-CM)``,
          defaults to ``atom+GPS``. ``atom+`` (default) partitions the atoms, ``orb+``
          the orbitals.
        - ``TBT.BTD.Pivot.Elec.<name>`` (or ``TBT.BTD.Pivot.Elecs``): only the
          ``atom+``/``orb+`` prefix is used, whether the downfolding region is
          found for atoms (default) or orbitals, see `downfold_partition`.

        Parameters
        ----------
        fdf :
           fdf file to read the parameters from
        prefix :
           which prefix to use, if TBT it will prefer TBT prefix, but fall back
           to TS prefixes.
           If TS, only these prefixes will be used.
        eta :
            force a specific eta value
        use_gf :
            whether the electrode self-energies are read from their ``GF`` files,
            see `Electrode.from_fdf`. Required for real-space electrodes.
        kwargs :
            passed to the class instantiating.
        """
        tfdf = _TransportFdf(fdf, prefix)
        clsname = f"{cls.__name__}.from_fdf"

        setup = _fdf_device(tfdf, clsname)
        H = setup.H
        elecs = [
            _fdf_electrode(
                tfdf, opts, None, use_gf, f"{clsname}(electrode={opts.name})"
            )
            for opts in setup.opts
        ]
        eta_dev = setup.eta if eta is None else eta

        # Calculate the downfolding regions (in orbitals)
        parts = _fdf_downfold(tfdf, setup)
        downfolds = []
        for elec, orbs_elec, part in zip(elecs, setup.elec_orbs, parts):
            if part is None:
                # in the device region
                downfolds.append((elec, None, None, orbs_elec))
            else:
                downfolds.append((elec, part.permute, part.blocks, orbs_elec))

        # Permutation table of the device, the blocks are calculated in __init__
        pivot = tfdf.fdf.get("TS.BTD.Pivot", "atom+GPS")
        if tfdf.is_tbtrans:
            pivot = tfdf.fdf.get("TBT.BTD.Pivot", pivot)
            pivot = tfdf.fdf.get("TBT.BTD.Pivot.Device", pivot)
        space, method = _fdf_permute_method(pivot)
        dense = [
            orbs_elec if orbs_down is None else orbs_down[-blocks_down[-1] :]
            for _, orbs_down, blocks_down, orbs_elec in downfolds
        ]
        if space == "atoms":
            csr = H.toSparseAtom()
            device = setup.device_atoms
            dense = [np.unique(H.o2a(d)) for d in dense]
        else:
            csr = H
            device = setup.device_orbs
        permute = {space: _device_permute(_graph(csr), device, dense, method)}

        return cls._from_downfold(H, downfolds, permute, eta_dev, True, **kwargs)

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
        return len(self.permute)

    def _elec(self, elec) -> int | Sequence[int]:
        """Convert an electrode (name, index or object) to the proper linear index"""
        if isinstance(elec, (tuple, list)):
            return [self._elec(e) for e in elec]
        return _elec_index(self.elecs, elec, self.__class__.__name__)

    def _serialize_elecs(self, elecs, omit_elecs=None):
        """Convert a list/str/int of elecs into a list of ints"""
        is_all = elecs is None  # TODO change this to something more sensible
        if is_all:
            elecs = list(range(len(self._elecs)))
        is_single = not isinstance(elecs, (tuple, list))
        if is_single:
            # ensure it is a list
            elecs = [elecs]

        # convert to list of ints
        elecs = self._elec(elecs)
        if is_all and omit_elecs is not None:
            _, omit_elecs = self._serialize_elecs(omit_elecs)
            elecs = [elec for elec in elecs if elec not in omit_elecs]
            is_single = len(elecs) == 1

        return is_single, elecs

    def _elec_name(self, elec) -> str | int:
        """Name of the electrode (name, index or object), its index if it has no name"""
        ielec = self._elec(elec)
        name = self._elecs[ielec].elec.name
        return ielec if name is None else name

    def _permute_matrix(self, M):
        """Permutes a (full) matrix, i.e. not one that is already reduced."""
        return M[self.permute, :][:, self.permute]

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
            rel_v = v / np.max(v)
            return (rel_v >= cutoff).nonzero()[0]

        return cutoff_func

    def _check_Ek(self, E: complex, k: KPoint, dtype, **kwargs) -> bool:
        """Check whether the stored quantities has already been calculated

        It does this by checking the internal data-structures stored `E`, `k`,
        `dtype` and `kwargs` values.
        """
        data = self._data
        if hasattr(data, "E"):
            if (
                np.allclose(data.E, E)
                and np.allclose(data.k, k)
                and data.dtype == dtype
                and data.kwargs == kwargs
            ):
                # we have already prepared the calculation
                return True

        # while resetting is not necessary, it can
        # save a lot of memory since some arrays are not
        # temporarily stored twice.
        self.clear()
        self._data.dtype = dtype
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
        if self._check_Ek(E, k, dtype, **kwargs):
            if hasattr(self._data, "se"):
                return

        E = self._data.E
        k = self._data.k

        # Create all self-energies (and store the Gamma's)
        se = []
        gamma = []
        for el in self._elecs:
            sort = el.sort.reshape(-1, 1)
            # Insert values
            SE = el.elec.self_energy(E, k=k, dtype=dtype, **kwargs)[sort, sort.T]
            se.append(SE)
            gamma.append(el.elec.se2broadening(SE))

        self._data.se = se
        self._data.gamma = gamma

    def _prepare_tgamma(self, E: complex, k: KPoint, dtype, cutoff, **kwargs) -> None:
        r"""Pre-calculate the :math:`\tilde{\boldsymbol\Gamma}` matrices for `cutoff`

        `cutoff` is the value passed by the user (a value or a function).
        It is stored (not its hash) to ensure that a function's id
        cannot be reused by another function.
        """
        if self._check_Ek(E, k, dtype, **kwargs):
            if hasattr(self._data, "tgamma"):
                prev = self._data.tgamma_cutoff
                if callable(cutoff) or callable(prev):
                    # functions can only be compared by identity
                    same = cutoff is prev
                else:
                    same = np.array_equal(cutoff, prev)
                if same:
                    return

        # ensure we have the self-energies
        self._prepare(E, k, dtype, **kwargs)

        tgamma = []
        cutoff_func = self._as_cutoff_func(cutoff)

        # See Sanz, Mach-Zender paper
        # Get the sqrt of the level broadening matrix
        def eigh_sqrt(gam):
            eig, U = eigh(gam)
            idx = cutoff_func(eig)
            eig = np.emath.sqrt(eig[idx])
            U = U[:, idx]
            return eig * U

        for gam in self._data.gamma:
            tgamma.append(eigh_sqrt(gam))

        self._data.tgamma = tgamma
        self._data.tgamma_cutoff = cutoff

    def _prepare(self, E: complex, k: KPoint, dtype, **kwargs) -> None:
        r"""Pre-calculate the needed quantities for Green function calculation

        It calculates:

        - self-energies (and consequently the Gamma matrices)
        - Splits the inverse-Green function into blocks according
          to the article.

          .. math::

             \mathbf A_i, \mathbf B_i, \mathbf C_i, \tilde\mathbf X_i, \tilde\mathbf Y_i
        """
        if self._check_Ek(E, k, dtype, **kwargs):
            if hasattr(self._data, "A"):
                return

        data = self._data

        E = data.E
        # device region: E + 1j*eta
        Ec = data.Ec
        k = data.k

        # Prepare the Green function calculation
        invG = self.H.Sk(k, dtype=dtype) * Ec - self.H.Hk(k, dtype=dtype, **kwargs)
        invG = self._permute_matrix(invG).tolil()

        # Replace the electrode regions by the bulk electrode matrices
        for el in self._elecs:
            if el.bulk:
                permute = el.permute.reshape(-1, 1)
                sort = el.sort.reshape(-1, 1)
                invG[permute, permute.T] = el.elec.bulk_matrix(
                    E, k, dtype=dtype, **kwargs
                )[sort, sort.T]

        # Create all self-energies (and store the Gamma's)
        if hasattr(data, "se"):
            for el, SE in zip(self._elecs, data.se):
                permute = el.permute.reshape(-1, 1)
                invG[permute, permute.T] -= SE
        else:
            gamma = []
            for el in self._elecs:
                permute = el.permute.reshape(-1, 1)
                sort = el.sort.reshape(-1, 1)
                # Insert values
                SE = el.elec.self_energy(E, k=k, dtype=dtype, **kwargs)[sort, sort.T]
                invG[permute, permute.T] -= SE
                gamma.append(el.elec.se2broadening(SE))
            del SE
            data.gamma = gamma
        # convert to csr format (that's how will mostly convert it)
        invG = invG.tocsr()

        nb = len(self.blocks)
        nbm1 = nb - 1

        # Now we have all needed to calculate the inverse parts of the Green function
        A = [None] * nb
        B = [0] * nb
        C = [0] * nb

        # Now we can calculate everything
        cblocks = self.blocks_cum0

        sl0 = slice(cblocks[0], cblocks[1])
        slp = slice(cblocks[1], cblocks[2])
        # initial matrix A and C
        iG = invG[sl0, :].tocsc()
        A[0] = iG[:, sl0].toarray()
        C[1] = iG[:, slp].toarray()
        for b in range(1, nbm1):
            # rotate slices
            sln = sl0
            sl0 = slp
            slp = slice(cblocks[b + 1], cblocks[b + 2])
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

    def _matrix_to_btd(self, M, format: str = "array") -> BlockArray:
        """Convert a matrix `M` into a BTD matrix.

        Parameters
        ----------
        M :
            the matrix to convert to a BTD matrix form
        format :
            the format of the blocks when `M` is a sparse matrix,
            any format ``X`` for which ``M.toX()`` exists.
        """
        BM = BlockArray((self.blocks, self.blocks), dtype=M.dtype)
        BI = BM.block_indexer
        c = self.blocks_cum0
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

    def _permute_k(self, func: Callable, *args, **kwargs):
        """Call ``func(*args, **kwargs)`` and permute the result to the device region

        ``format="btd"`` returns the matrix as a `BlockArray`, any other
        `format` is passed to `func`.
        """
        is_btd = False
        if "format" in kwargs:
            if kwargs["format"].lower() == "btd":
                is_btd = True
                del kwargs["format"]

        M = self._permute_matrix(func(*args, **kwargs))
        if is_btd:
            return self._matrix_to_btd(M)
        return M

    def Sk(self, *args, **kwargs):
        """Return the overlap matrix in the permuted device region"""
        return self._permute_k(self.H.Sk, *args, **kwargs)

    def Hk(self, *args, **kwargs):
        """Return the Hamiltonian matrix in the permuted device region"""
        return self._permute_k(self.H.Hk, *args, **kwargs)

    def green(
        self,
        E: complex,
        k: KPoint = (0, 0, 0),
        format="array",
        dtype=np.complex128,
        **kwargs,
    ) -> np.ndarray | BlockArray:
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
            - btd: a block-array object with only the diagonals and first off-diagonals
            - bm: a block-array object with diagonals and all off-diagonals
            - bd: a block-array object with only diagonals (no off-diagonals)
            - sparse: a sparse-csr matrix for the sparse elements as found in the Hamiltonian
        dtype :
            the data-type of the array.

        Returns
        -------
        np.ndarray or BlockArray
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
        n = len(self.permute)
        G = np.empty([n, n], dtype=self._data.A[0].dtype)

        blocks = self.blocks
        nb = len(blocks)
        nbm1 = nb - 1
        sumbs = 0
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        for b, bs in enumerate(blocks):
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
                sla = slice(next_sum - blocks[a], next_sum)
                G[sla, sl0] = -tY[a + 1] @ G[slp, sl0]
                slp = sla
                next_sum -= blocks[a]

            sl0 = slice(sumbs, sumbs + bs)

            # Step block
            sumbs += bs

            # Do below
            next_sum = sumbs
            slp = sl0
            for a in range(b + 1, nb):
                # Calculate all parts above
                sla = slice(next_sum, next_sum + blocks[a])
                G[sla, sl0] = -tX[a - 1] @ G[slp, sl0]
                slp = sla
                next_sum += blocks[a]

        return G

    def _green_btd(self) -> BlockArray:
        """Calculate the Green function only in the BTD matrix elements.

        Stored in a `BlockArray` class."""
        G = BlockArray((self.blocks, self.blocks), dtype=self._data.A[0].dtype)
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

    def _green_bm(self) -> BlockArray:
        """Calculate the full Green function.

        Stored in a `BlockArray` class."""
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

    def _green_bd(self) -> BlockArray:
        """Calculate the Green function only along the diagonal block matrices.

        Stored in a `BlockArray` class."""
        G = BlockArray((self.blocks, self.blocks), dtype=self._data.A[0].dtype)
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

        Stored in a `scipy.sparse.csr_array` class."""
        # create a sparse matrix
        G = self.H.Sk(format="csr", dtype=self._data.A[0].dtype)
        # permute the matrix
        G = self._permute_matrix(G)

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

        blocks = self.blocks
        nb = len(blocks)
        nbm1 = nb - 1
        A = self._data.A
        B = self._data.B
        C = self._data.C
        tX = self._data.tX
        tY = self._data.tY
        sumbsn, sumbs, sumbsp = 0, 0, 0
        for b, bs in enumerate(blocks):
            sumbsp = sumbs + bs
            if b < nbm1:
                bsp = blocks[b + 1]

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
        nb = len(self.blocks)
        nbm1 = nb - 1

        # Find parts we need to calculate
        blocks = _blocks_in_range(self.blocks_cum0, cols)
        assert (
            len(blocks) <= 2
        ), f"{self.__class__.__name__} green(diagonal) requires maximally 2 blocks"
        if len(blocks) == 2:
            assert (
                blocks[0] + 1 == blocks[1]
            ), f"{self.__class__.__name__} green(diagonal) requires spanning only 2 blocks"

        n = self.blocks[blocks].sum()
        G = np.empty([n, ncols], dtype=self._data.A[0].dtype)

        sizes = self.blocks
        c = self.blocks_cum0
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
                sl = slice(0, sizes[b])
                c_idx = _a.arangei(len(b_idx))
            else:
                sl = slice(sizes[blocks[0]], sizes[blocks[0]] + sizes[b])
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
                slp = slice(sizes[b], sizes[b] + sizes[blocks[1]])
                G[slp, c_idx] = -tX[b] @ G[sl, c_idx]
            else:
                # Calculate above
                slp = slice(0, sizes[blocks[0]])
                G[slp, c_idx] = -tY[b] @ G[sl, c_idx]

        return blocks, G

    def _green_column(self, cols) -> np.ndarray:
        """Calculate the full Green function column for a subset of columns.

        Stored in a `np.array` class."""
        # To calculate the full Gf for specific column indices
        # These indices should maximally be spanning 2 blocks
        cols = cols.ravel()
        ncols = len(cols)
        nb = len(self.blocks)
        nbm1 = nb - 1

        # Find parts we need to calculate
        blocks = _blocks_in_range(self.blocks_cum0, cols)
        assert (
            len(blocks) <= 2
        ), f"{self.__class__.__name__}.green(column) requires maximally 2 blocks"
        if len(blocks) == 2:
            assert (
                blocks[0] + 1 == blocks[1]
            ), f"{self.__class__.__name__}.green(column) requires spanning only 2 blocks"

        n = len(self)
        G = np.empty([n, ncols], dtype=self._data.A[0].dtype)

        c = self.blocks_cum0
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
    ) -> np.ndarray | BlockArray:
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
           - btd: a block-array object with only the diagonals and first off-diagonals
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
        np.ndarray or BlockArray
            the spectral function for a given electrode in the format
            as specified by `format`. Note that some formats does not calculate
            the entire spectral function matrix.
        """
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
        G = self._green_column(self._elecs[elec].permute)
        # Now calculate the full spectral function
        return G @ self._data.gamma[elec] @ np.conj(G.T)

    def _spectral_column_bm(self, elec, herm: bool) -> BlockArray:
        """Spectral function from a column array

        Returns a `BlockArray` class with all elements calculated.

        Parameters
        ----------
        herm:
           if true, only calculate the lower triangular part, and copy
           the Hermitian part to the upper triangular part.
           Else, calculate the full matrix via MM.
        """
        G = self._green_column(self._elecs[elec].permute)
        nb = len(self.blocks)

        Gam = self._data.gamma[elec]

        # Now calculate the full spectral function
        btd = BlockArray((self.blocks, self.blocks), dtype=np.result_type(G, Gam))
        BI = btd.block_indexer

        c = self.blocks_cum0

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

    def _spectral_column_btd(self, elec, herm: bool) -> BlockArray:
        """Spectral function from a column array

        Returns a `BlockArray` class with only BTD blocks calculated.

        Parameters
        ----------
        herm:
           if true, only calculate the lower triangular part, and copy
           the Hermitian part to the upper triangular part.
           Else, calculate the full matrix via MM.
        """
        G = self._green_column(self._elecs[elec].permute)
        nb = len(self.blocks)

        Gam = self._data.gamma[elec]

        # Now calculate the full spectral function
        btd = BlockArray((self.blocks, self.blocks), dtype=np.result_type(G, Gam))
        BI = btd.block_indexer

        c = self.blocks_cum0
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
        nb = len(self.blocks)
        nbm1 = nb - 1

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self._elecs[elec].permute)
        nblocks = len(blocks)
        A = A @ self._data.gamma[elec] @ np.conj(A.T)

        # Allocate space for the full matrix
        S = np.empty([len(self), len(self)], dtype=A.dtype)

        c = self.blocks_cum0
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

    def _spectral_propagate_bm(self, elec, herm: bool) -> BlockArray:
        sizes = self.blocks
        nb = len(sizes)
        nbm1 = nb - 1

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self._elecs[elec].permute)
        nblocks = len(blocks)
        A = A @ self._data.gamma[elec] @ np.conj(A.T)

        BM = BlockArray((self.blocks, self.blocks), dtype=A.dtype)
        BI = BM.block_indexer

        BI[blocks[0], blocks[0]] = A[: sizes[blocks[0]], : sizes[blocks[0]]]
        if len(blocks) > 1:
            BI[blocks[0], blocks[1]] = A[: sizes[blocks[0]], sizes[blocks[0]] :]
            BI[blocks[1], blocks[0]] = A[sizes[blocks[0]] :, : sizes[blocks[0]]]
            BI[blocks[1], blocks[1]] = A[sizes[blocks[0]] :, sizes[blocks[0]] :]

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

    def _spectral_propagate_btd(self, elec, herm: bool) -> BlockArray:
        sizes = self.blocks
        nb = len(sizes)
        nbm1 = nb - 1

        # First we need to calculate diagonal blocks of the spectral matrix
        blocks, A = self._green_diag_block(self._elecs[elec].permute)
        A = A @ self._data.gamma[elec] @ np.conj(A.T)

        BM = BlockArray((self.blocks, self.blocks), dtype=A.dtype)
        BI = BM.block_indexer

        BI[blocks[0], blocks[0]] = A[: sizes[blocks[0]], : sizes[blocks[0]]]
        if len(blocks) > 1:
            BI[blocks[0], blocks[1]] = A[: sizes[blocks[0]], sizes[blocks[0]] :]
            BI[blocks[1], blocks[0]] = A[sizes[blocks[0]] :, : sizes[blocks[0]]]
            BI[blocks[1], blocks[1]] = A[sizes[blocks[0]] :, sizes[blocks[0]] :]

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
        *,
        cutoff: float | Callable = 0.0,
        method: Literal["svd:gamma", "svd:A", "eig"] = "svd:gamma",
        dtype=np.complex128,
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
        # split the arguments for the scattering state methods from the ones
        # passed to the matrix and self-energy calculations
        method_kwargs = {
            key: kwargs.pop(key) for key in _SCAT_STATE_KWARGS if key in kwargs
        }
        self._prepare(E, k, dtype, **kwargs)
        method_name = method
        method = method.lower().replace(":", "_")
        func = getattr(self, f"_scattering_state_{method}", None)
        if func is None:
            raise ValueError(
                f"{self.__class__.__name__}.scattering_state got unknown method={method_name!r}, "
                "must be one of [svd:gamma, svd:A, eig]"
            )
        return func(method, elec, cutoff, **method_kwargs)

    def _scattering_state_eig(self, _method, elec, cutoff, **kwargs):
        # We know that scattering_state has called prepare!
        data = self._data
        A = self.spectral(data.E, elec, data.k, dtype=data.dtype, **data.kwargs)
        cutoff = self._as_cutoff_func(cutoff)

        # add something to the diagonal (improves diag precision for small states)
        idx = np.arange(len(A))
        np.add.at(A, (idx, idx), _SPECTRAL_DIAG_SHIFT)
        del idx

        # Now diagonalize A
        DOS, A = eigh_destroy(A)
        # backconvert diagonal
        DOS -= _SPECTRAL_DIAG_SHIFT
        # TODO check with overlap convert with correct magnitude (Tr[A] / 2pi)
        DOS /= 2 * np.pi
        DOS, A = self._coefficient_state_reduce(elec, DOS, A, cutoff)

        info = dict(
            method="full",
            elec=self._elec_name(elec),
            E=data.E,
            k=data.k,
            kwargs=data.kwargs,
            cutoff=cutoff,
        )

        # always have the first state with the largest values
        return si.physics.StateCElectron(A.T, DOS, self, **info)

    def _scattering_state_svd_gamma(self, _method, elec, cutoff, **kwargs):
        A = self._green_column(self._elecs[elec].permute)
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
            kwargs=data.kwargs,
            cutoff=cutoff,
        )

        # always have the first state with the largest values
        return si.physics.StateCElectron(A.T, DOS, self, **info)

    def _scattering_state_svd_a(
        self, _method, elec, cutoff: float | tuple[float, float] = 0, **kwargs
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
        blocks, u = self._green_diag_block(self._elecs[elec].permute)
        # Calculate the spectral function only for the blocks that host the
        # scattering matrix
        u = u @ self._data.gamma[elec] @ np.conj(u.T)

        # add something to the diagonal (improves diag precision)
        idx = np.arange(len(u))
        np.add.at(u, (idx, idx), _SPECTRAL_DIAG_SHIFT)
        del idx

        # Calculate eigenvalues
        DOS, u = eigh_destroy(u)
        # backconvert diagonal
        DOS -= _SPECTRAL_DIAG_SHIFT
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

        nb = len(self.blocks)
        cblocks = self.blocks_cum0

        # Create full U
        A = np.empty([len(self), u.shape[1]], dtype=u.dtype)

        sl = slice(cblocks[blocks[0]], cblocks[blocks[0] + 1])
        A[sl, :] = u[: self.blocks[blocks[0]], :]
        if len(blocks) > 1:
            sl = slice(cblocks[blocks[1]], cblocks[blocks[1] + 1])
            A[sl, :] = u[self.blocks[blocks[0]] :, :]
        del u

        # Propagate A in the full BTD matrix
        t = self._data.tY
        sl = slice(cblocks[blocks[0]], cblocks[blocks[0] + 1])
        for b in range(blocks[0], 0, -1):
            sln = slice(cblocks[b - 1], cblocks[b])
            A[sln] = -t[b] @ A[sl]
            sl = sln

        t = self._data.tX
        sl = slice(cblocks[blocks[-1]], cblocks[blocks[-1] + 1])
        for b in range(blocks[-1], nb - 1):
            slp = slice(cblocks[b + 1], cblocks[b + 2])
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
            kwargs=data.kwargs,
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
        **kwargs,
    ) -> float | tuple[float, ...]:
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
        dtype :
            the data-type of the matrices.
        **kwargs :
            passed to the Hamiltonian and self-energy calculations (e.g. ``spin``).
        """
        # Calculate the full column green function
        elec_from = self._elec(elec_from)
        is_single, elec_to = self._serialize_elecs(elec_to, elec_from)

        # Prepare calculation @ E and k
        self._prepare(E, k, dtype, **kwargs)

        # Get full G in column of 'from'
        G = self._green_column(self._elecs[elec_from].permute)

        # The gamma matrices
        Gam = self._data.gamma

        # Now calculate the transmission
        def calc(Gam_from, elec_to, Gam_to, G):
            permute = self._elecs[elec_to].permute
            g = G[permute, :]
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
        **kwargs,
    ) -> si.physics.StateElectron | tuple[si.physics.StateElectron, ...]:
        r""" Calculate the scattering matrix (S-matrix) between `elec_from` and `elec_to`

        The scattering matrix is calculated as

        .. math::
               \mathbf S_{\mathfrak e'\mathfrak e}(E, \mathbf k) = -\delta_{\alpha\beta} + i
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
        dtype :
            the data-type of the matrices.
        **kwargs :
            passed to the Hamiltonian and self-energy calculations (e.g. ``spin``).

        Returns
        -------
        sisl.physics.electron.StateElectron or tuple[sisl.physics.electron.StateElectron,...]
           for each `elec_to` a scattering matrix will be returned. Its dimensions will be
           depending on the `cutoff` value at the cost of precision.
        """
        # Calculate the full column green function
        elec_from = self._elec(elec_from)
        is_single, elec_to = self._serialize_elecs(elec_to, elec_from)

        # Prepare calculation @ E and k
        self._prepare(E, k, dtype, **kwargs)
        self._prepare_tgamma(E, k, dtype, cutoff, **kwargs)

        # Get full G in column of 'from'
        G = self._green_column(self._elecs[elec_from].permute)

        # the \tilde \Gamma functions
        tG = self._data.tgamma

        data = self._data
        info = dict(
            elec=self._elec_name(elec_from),
            E=data.E,
            k=data.k,
            kwargs=data.kwargs,
            cutoff=cutoff,
        )

        # Now calculate the S matrices
        def calc(elec_from, jtgam_from, elec_to, tgam_to, G):
            permute = self._elecs[elec_to].permute
            g = G[permute, :]
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
    ) -> si.physics.StateCElectron | tuple[si.physics.StateCElectron, ...]:
        r"""Calculate the eigenchannel from a scattering matrix

        The energy and k-point is inferred from the `scat_matrix` object as returned from
        `scattering_matrix`.

        The eigenchannels are the SVD of the scattering matrix in the
        DOS weighted scattering states:

        Parameters
        ----------
        scat_matrix :
            the scattering matrix as obtained from `scattering_matrix`
        ret_out :
            also return the eigenchannels as seen from the outgoing state

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
        tt_eig, U = _scat_state_svd(scat_matrix.state.T, ret_uv=ret_out)
        # Here there is a wrong pre-factor of 2pi
        tt_eig *= 2 * np.pi

        info = {**scat_matrix.info}
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
    ) -> (
        si.physics.StateCElectron
        | tuple[si.physics.StateCElectron, si.physics.StateElectron]
    ):
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
        self._prepare_se(
            state.info["E"], state.info["k"], state.dtype, **state.info["kwargs"]
        )
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
        idx = self._elecs[el].permute

        u = A[idx]
        # the summed transmission matrix
        Ut = u.conj().T @ G[el] @ u

        # same for other electrodes
        for el in elec_to[1:]:
            idx = self._elecs[el].permute
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
