# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from pathlib import Path

import numpy as np

import sisl._array as _a

# Import the geometry object
from sisl import Atom, Geometry, Lattice, SparseOrbitalBZSpin
from sisl._core.sparse import _ncol_to_indptr

# Import sile objects
from sisl._internal import set_module
from sisl.messages import deprecate_argument, warn
from sisl.unit.siesta import unit_convert

from ..siesta._help import (
    _csr_from_sc_off,
    _csr_to_siesta,
    _mat_siesta2sisl,
    _mat_sisl2siesta,
    _siesta_sc_off,
)
from ..sile import SileError, add_sile, sile_raise_write
from .sile import SileCDFTBtrans

__all__ = ["deltancSileTBtrans"]


Bohr2Ang = unit_convert("Bohr", "Ang")
Ry2eV = unit_convert("Ry", "eV")
eV2Ry = unit_convert("eV", "Ry")


# The delta nc file
@set_module("sisl.io.tbtrans")
class deltancSileTBtrans(SileCDFTBtrans):
    r"""TBtrans :math:`\delta` file object

    The :math:`\delta` file object is an extension enabled in `TBtrans`_ which
    allows changing the Hamiltonian in transport problems.

    .. math::
        \mathbf H'(\mathbf k) = \mathbf H(\mathbf k) +
            \delta\mathbf H(E, \mathbf k) + \delta\boldsymbol\Sigma(E, \mathbf k)

    This file may either be used directly as the :math:`\delta\mathbf H` or the
    :math:`\delta\boldsymbol\Sigma`.

    When writing :math:`\delta` terms using `write_delta` one may add ``k`` or ``E`` arguments
    to make the :math:`\delta` dependent on ``k`` and/or ``E``.

    Refer to the TBtrans manual on how to use this feature.

    Examples
    --------
    >>> H = Hamiltonian(geom.graphene(), dtype=np.complex128)
    >>> H[0, 0] = 1j
    >>> dH = get_sile('deltaH.dH.nc', 'w')
    >>> dH.write_delta(H)
    >>> H[1, 1] = 1.
    >>> dH.write_delta(H, k=[0, 0, 0]) # Gamma only
    >>> H[0, 0] += 1.
    >>> dH.write_delta(H, E=1.) # only at 1 eV
    >>> H[1, 1] += 1.j
    >>> dH.write_delta(H, E=1., k=[0, 0, 0]) # only at 1 eV and Gamma-point
    """

    @classmethod
    def merge(cls, fname, *deltas, **kwargs):
        """Merge several delta files into one Sile which contains the sum of the content

        In cases where implementors use several different delta files it is necessary
        to merge them into a single delta file before use in TBtrans.
        This method does exactly that.

        Notes
        -----
        The code checks whether `fname` is different from all `deltas` and that
        all `deltas` are the same class.

        Parameters
        ----------
        fname : str, Path
          the output name of the merged file
        *deltas : deltancSileTBtrans, str, Path
          all the delta files that should be merged
        **kwargs :
          arguments passed directly to the init of ``cls(fname, **kwargs)``
        """
        file = Path(fname)
        deltas_obj = []
        for delta in deltas:
            if isinstance(delta, (str, Path)):
                delta = cls(delta, mode="r")
            deltas_obj.append(delta)
            if delta.__class__ != cls:
                raise ValueError(
                    f"{cls.__name__}.merge requires all files to be the same class."
                )
            if delta.file == file:
                raise ValueError(
                    f"{cls.__name__}.merge requires that the output file is different from all arguments."
                )

        # be sure to overwrite the input with objects
        deltas = deltas_obj

        out = cls(fname, mode="w", **kwargs)

        # Now create and simultaneously check for the same arguments
        geom = deltas[0].read_geometry()
        for delta in deltas[1:]:
            if not geom.equal(delta.read_geometry()):
                raise ValueError(
                    f"{cls.__name__}.merge requires that the input files all contain the same geometry."
                )

        # Now we are ready to write
        out.write_geometry(geom)

        # Now loop all different sparse stuff

        # Level 1
        deltas_lvl = []
        m = 0
        for delta in deltas:
            if delta.has_level(1):
                m = m + delta.read_delta()
                deltas_lvl.append(delta)
        if len(deltas_lvl) > 0:
            out.write_delta(m)

        # Level 2
        deltas_lvl = []
        ks = []
        for delta in deltas:
            if delta.has_level(2):
                ks.append(delta._get_lvl(2).variables["kpt"][:])
                deltas_lvl.append(delta)
        ks = np.concatenate(ks)
        ks = np.unique(ks, axis=0)
        for k in ks:
            m = 0
            for delta in deltas_lvl:
                try:
                    # it could be that the k does not exist in this delta
                    m = m + delta.read_delta(k=k)
                except ValueError:
                    pass
            out.write_delta(m, k=k)

        # Level 3
        deltas_lvl = []
        Es = []
        for delta in deltas:
            if delta.has_level(3):
                Es.append(delta._get_lvl(3).variables["E"][:] * Ry2eV)
                deltas_lvl.append(delta)
        Es = np.concatenate(Es)
        Es = np.unique(Es)
        for E in Es:
            m = 0
            for delta in deltas_lvl:
                try:
                    # it could be that the E does not exist in this delta
                    m = m + delta.read_delta(E=E)
                except ValueError:
                    pass
            out.write_delta(m, E=E)

        # Level 4
        deltas_lvl = []
        ks, Es = [], []
        for delta in deltas:
            if delta.has_level(4):
                lvl = delta._get_lvl(4)
                ks.append(lvl.variables["kpt"][:])
                Es.append(lvl.variables["E"][:] * Ry2eV)
                deltas_lvl.append(delta)
        ks = np.concatenate(ks)
        ks = np.unique(ks, axis=0)
        Es = np.concatenate(Es)
        Es = np.unique(Es)

        for k in ks:
            for E in Es:
                m = 0
                for delta in deltas_lvl:
                    try:
                        # it could be that the E does not exist in this delta
                        m = m + delta.read_delta(E=E, k=k)
                    except ValueError:
                        pass
                out.write_delta(m, E=E, k=k)

    def has_level(self, ilvl):
        """Query whether the file has level `ilvl` content

        Parameters
        ----------
        ilvl : int
           the level to be queried, one of 1, 2, 3 or 4
        """
        return f"LEVEL-{ilvl}" in self.groups

    def read_lattice(self):
        """Returns the `Lattice` object from this file"""
        cell = _a.arrayd(np.copy(self._value("cell"))) * Bohr2Ang
        cell.shape = (3, 3)

        nsc = self._value("nsc")
        lattice = Lattice(cell, nsc=nsc)
        try:
            lattice.sc_off = self._value("isc_off")
        except Exception:
            # This is ok, we simply do not have the supercell offsets
            pass

        return lattice

    def read_geometry(self, *args, **kwargs):
        """Returns the `Geometry` object from this file"""
        lattice = self.read_lattice()

        xyz = _a.arrayd(np.copy(self._value("xa"))) * Bohr2Ang
        xyz.shape = (-1, 3)

        # Create list with correct number of orbitals
        lasto = _a.arrayi(np.copy(self._value("lasto")))
        nos = np.diff(lasto, prepend=0)

        if "atom" in kwargs:
            # The user "knows" which atoms are present
            atms = kwargs["atom"]
            # Check that all atoms have the correct number of orbitals.
            # Otherwise we will correct them
            for i in range(len(atms)):
                if atms[i].no != nos[i]:
                    atms[i] = Atom(atms[i].Z, [-1] * nos[i], tag=atms[i].tag)

        else:
            # Default to Hydrogen atom with nos[ia] orbitals
            # This may be counterintuitive but there is no storage of the
            # actual species
            atms = [Atom("H", [-1] * o) for o in nos]

        # Create and return geometry object
        geom = Geometry(xyz, atms, lattice=lattice)

        return geom

    @deprecate_argument("sc", "lattice", "use lattice= instead of sc=", "0.15", "0.17")
    def write_lattice(self, lattice):
        """Creates the NetCDF file and writes the supercell information"""
        sile_raise_write(self)

        # Create initial dimensions
        self._crt_dim(self, "one", 1)
        self._crt_dim(self, "n_s", np.prod(lattice.nsc))
        self._crt_dim(self, "xyz", 3)

        # Create initial geometry
        v = self._crt_var(self, "nsc", "i4", ("xyz",))
        v.info = "Number of supercells in each unit-cell direction"
        v[:] = lattice.nsc[:]
        v = self._crt_var(self, "isc_off", "i4", ("n_s", "xyz"))
        v.info = "Index of supercell coordinates"
        v[:] = lattice.sc_off[:, :]
        v = self._crt_var(self, "cell", "f8", ("xyz", "xyz"))
        v.info = "Unit cell"
        v.unit = "Bohr"
        v[:] = lattice.cell[:, :] / Bohr2Ang

        # Create designation of the creation
        self.method = "sisl"

    def write_geometry(self, geometry):
        """Creates the NetCDF file and writes the geometry information"""
        sile_raise_write(self)

        # Create initial dimensions
        self.write_lattice(geometry.lattice)
        self._crt_dim(self, "no_s", np.prod(geometry.nsc) * geometry.no)
        self._crt_dim(self, "no_u", geometry.no)
        self._crt_dim(self, "na_u", geometry.na)

        # Create initial geometry
        v = self._crt_var(self, "lasto", "i4", ("na_u",))
        v.info = "Last orbital of equivalent atom"
        v = self._crt_var(self, "xa", "f8", ("na_u", "xyz"))
        v.info = "Atomic coordinates"
        v.unit = "Bohr"

        # Save stuff
        self.variables["xa"][:] = geometry.xyz / Bohr2Ang

        bs = self._crt_grp(self, "BASIS")
        b = self._crt_var(bs, "basis", "i4", ("na_u",))
        b.info = "Basis of each atom by ID"

        orbs = _a.emptyi([geometry.na])

        for ia, a, isp in geometry.iter_species():
            b[ia] = isp + 1
            orbs[ia] = a.no
            if a.tag in bs.groups:
                # Assert the file sizes
                if bs.groups[a.tag].Number_of_orbitals != a.no:
                    raise ValueError(
                        f"File {self.file} "
                        "has erroneous data in regards of "
                        "of the alreay stored dimensions."
                    )
            else:
                ba = bs.createGroup(a.tag)
                ba.ID = np.int32(isp + 1)
                ba.Atomic_number = np.int32(a.Z)
                ba.Mass = a.mass
                ba.Label = a.tag
                ba.Element = a.symbol
                ba.Number_of_orbitals = np.int32(a.no)

        # Store the lasto variable as the remaining thing to do
        self.variables["lasto"][:] = _a.cumsumi(orbs)

    def _get_lvl_k_E(self, **kwargs):
        """Return level, k and E indices, in that order.

        The indices are negative if a new index needs to be created.
        """
        # Determine the type of dH we are storing...
        k = kwargs.get("k", None)
        if k is not None:
            k = _a.asarrayd(k).flatten()
        E = kwargs.get("E", None)

        if (k is None) and (E is None):
            ilvl = 1
        elif (k is not None) and (E is None):
            ilvl = 2
        elif (k is None) and (E is not None):
            ilvl = 3
            # Convert to Rydberg
            E = E * eV2Ry
        elif (k is not None) and (E is not None):
            ilvl = 4
            # Convert to Rydberg
            E = E * eV2Ry

        try:
            lvl = self._get_lvl(ilvl)
        except Exception:
            return ilvl, -1, -1

        # Now determine the energy and k-indices
        iE = -1
        if ilvl in (3, 4):
            if lvl.variables["E"].size != 0:
                Es = _a.arrayd(lvl.variables["E"][:])
                iE = np.argmin(np.abs(Es - E))
                if abs(Es[iE] - E) > 0.0001:
                    iE = -1

        ik = -1
        if ilvl in (2, 4):
            if lvl.variables["kpt"].size != 0:
                kpt = _a.arrayd(lvl.variables["kpt"][:])
                kpt.shape = (-1, 3)
                ik = np.argmin(np.abs(kpt - k[None, :]).sum(axis=1))
                if not np.allclose(kpt[ik, :], k, atol=0.0001):
                    ik = -1

        return ilvl, ik, iE

    def _get_lvl(self, ilvl):
        if self.has_level(ilvl):
            return self._crt_grp(self, f"LEVEL-{ilvl}")
        raise ValueError(f"Level {ilvl} does not exist in {self.file}.")

    def _add_lvl(self, ilvl):
        """Simply adds and returns a group if it does not exist it will be created"""
        slvl = f"LEVEL-{ilvl}"
        if slvl in self.groups:
            lvl = self._crt_grp(self, slvl)
        else:
            lvl = self._crt_grp(self, slvl)
            if ilvl in (2, 4):
                self._crt_dim(lvl, "nkpt", None)
                self._crt_var(
                    lvl,
                    "kpt",
                    "f8",
                    ("nkpt", "xyz"),
                    attrs={"info": "k-points for delta values", "unit": "b**-1"},
                )
            if ilvl in (3, 4):
                self._crt_dim(lvl, "ne", None)
                self._crt_var(
                    lvl,
                    "E",
                    "f8",
                    ("ne",),
                    attrs={"info": "Energy points for delta values", "unit": "Ry"},
                )

        return lvl

    def write_delta(self, delta, **kwargs):
        r"""Writes a :math:`\delta` term to the file

        This term may be of

        - level-1: no E or k dependence
        - level-2: k-dependent
        - level-3: E-dependent
        - level-4: k- and E-dependent


        Parameters
        ----------
        delta : SparseOrbitalBZSpin
           the model to be saved in the NC file
        k : array_like, optional
           a specific k-point :math:`\delta` term. I.e. only save the :math:`\delta` term for
           the given k-point. May be combined with `E` for a specific k and energy point.
        E : float, optional
           an energy dependent :math:`\delta` term. I.e. only save the :math:`\delta` term for
           the given energy. May be combined with `k` for a specific k and energy point.

        Notes
        -----
        The input options for `TBtrans`_ determine whether this is a self-energy term
        or a Hamiltonian term.
        """
        delta = delta.copy()
        if delta._csr.nnz == 0:
            raise SileError(
                f"{self!s}.write_overlap cannot write a zero element sparse matrix!"
            )

        # convert to siesta thing and store
        _csr_to_siesta(delta.geometry, delta._csr, diag=False)
        # delta should always write sorted matrices
        delta._csr.finalize(sort=True)

        _mat_sisl2siesta(delta)

        # Ensure that the geometry is written
        self.write_geometry(delta.geometry)

        self._crt_dim(self, "spin", delta.spin.size(delta.dtype))

        # Determine the type of delta we are storing...
        k = kwargs.get("k", None)
        E = kwargs.get("E", None)

        ilvl, ik, iE = self._get_lvl_k_E(**kwargs)
        lvl = self._add_lvl(ilvl)

        # Append the sparsity pattern
        # Create basis group
        if "n_col" in lvl.variables:
            if len(lvl.dimensions["nnzs"]) != delta._csr.nnz:
                raise ValueError(
                    "The sparsity pattern stored in delta *MUST* be equivalent for "
                    "all delta entries [nnz]."
                )
            if np.any(lvl.variables["n_col"][:] != delta._csr.ncol[:]):
                raise ValueError(
                    "The sparsity pattern stored in delta *MUST* be equivalent for "
                    "all delta entries [n_col]."
                )
            if np.any(lvl.variables["list_col"][:] != delta._csr.col[:] + 1):
                raise ValueError(
                    "The sparsity pattern stored in delta *MUST* be equivalent for "
                    "all delta entries [list_col]."
                )
            if np.any(
                lvl.variables["isc_off"][:]
                != _siesta_sc_off(delta.geometry.lattice.nsc)
            ):
                raise ValueError(
                    "The sparsity pattern stored in delta *MUST* be equivalent for "
                    "all delta entries [sc_off]."
                )
        else:
            self._crt_dim(lvl, "nnzs", delta._csr.nnz)
            v = self._crt_var(lvl, "n_col", "i4", ("no_u",))
            v.info = "Number of non-zero elements per row"
            v[:] = delta._csr.ncol[:]
            v = self._crt_var(
                lvl,
                "list_col",
                "i4",
                ("nnzs",),
                chunksizes=(delta._csr.nnz,),
                **self._cmp_args,
            )
            v.info = "Supercell column indices in the sparse format"
            v[:] = delta._csr.col[:] + 1  # correct for fortran indices
            v = self._crt_var(lvl, "isc_off", "i4", ("n_s", "xyz"))
            v.info = "Index of supercell coordinates"
            v[:] = _siesta_sc_off(delta.geometry.lattice.nsc)

        warn_E = True
        if ilvl in (3, 4):
            if iE < 0:
                # We need to add the new value
                iE = lvl.variables["E"].shape[0]
                lvl.variables["E"][iE] = E * eV2Ry
                warn_E = False

        warn_k = True
        if ilvl in (2, 4):
            if ik < 0:
                ik = lvl.variables["kpt"].shape[0]
                lvl.variables["kpt"][ik, :] = k
                warn_k = False

        if ilvl == 4 and warn_k and warn_E and False:
            # As soon as we have put the second k-point and the first energy
            # point, this warning will proceed...
            # I.e. even though the variable has not been set, it will WARN
            # Hence we out-comment this for now...
            # warn(f"Overwriting k-point {ik} and energy point {iE} correction.")
            pass
        elif ilvl == 3 and warn_E:
            warn(f"Overwriting energy point {iE} correction.")
        elif ilvl == 2 and warn_k:
            warn(f"Overwriting k-point {ik} correction.")

        if ilvl == 1:
            dim = ("spin", "nnzs")
            sl = [slice(None)] * 2
            csize = [1] * 2
        elif ilvl == 2:
            dim = ("nkpt", "spin", "nnzs")
            sl = [slice(None)] * 3
            sl[0] = ik
            csize = [1] * 3
        elif ilvl == 3:
            dim = ("ne", "spin", "nnzs")
            sl = [slice(None)] * 3
            sl[0] = iE
            csize = [1] * 3
        elif ilvl == 4:
            dim = ("nkpt", "ne", "spin", "nnzs")
            sl = [slice(None)] * 4
            sl[0] = ik
            sl[1] = iE
            csize = [1] * 4

        # Number of non-zero elements
        csize[-1] = delta._csr.nnz

        if delta.spin.kind > delta.spin.POLARIZED:
            raise ValueError(
                f"{self.__class__.__name__}.write_delta only allows spin-polarized "
                f"delta values, got {delta.spin!s}"
            )

        if delta.dtype.kind == "c":
            v1 = self._crt_var(
                lvl,
                "Redelta",
                "f8",
                dim,
                chunksizes=csize,
                attrs={"info": "Real part of delta", "unit": "Ry"},
                **self._cmp_args,
            )
            v2 = self._crt_var(
                lvl,
                "Imdelta",
                "f8",
                dim,
                chunksizes=csize,
                attrs={"info": "Imaginary part of delta", "unit": "Ry"},
                **self._cmp_args,
            )
            for i in range(delta.spin.size(delta.dtype)):
                sl[-2] = i
                v1[sl] = delta._csr._D[:, i].real * eV2Ry
                v2[sl] = delta._csr._D[:, i].imag * eV2Ry

        else:
            v = self._crt_var(
                lvl,
                "delta",
                "f8",
                dim,
                chunksizes=csize,
                attrs={"info": "delta", "unit": "Ry"},
                **self._cmp_args,
            )
            for i in range(delta.spin.size(delta.dtype)):
                sl[-2] = i
                v[sl] = delta._csr._D[:, i] * eV2Ry

    def _r_class(self, cls, **kwargs):
        """Reads a class model from a file"""

        # Ensure that the geometry is written
        geom = self.read_geometry()

        # Determine the type of delta we are storing...
        ilvl, ik, iE = self._get_lvl_k_E(**kwargs)

        # Get the level
        lvl = self._get_lvl(ilvl)

        if iE < 0 and ilvl in (3, 4):
            E = kwargs.get("E", None)
            raise ValueError(f"Energy {E} eV does not exist in the file.")
        if ik < 0 and ilvl in (2, 4):
            raise ValueError("k-point requested does not exist in the file.")

        if ilvl == 1:
            sl = [slice(None)] * 2
        elif ilvl == 2:
            sl = [slice(None)] * 3
            sl[0] = ik
        elif ilvl == 3:
            sl = [slice(None)] * 3
            sl[0] = iE
        elif ilvl == 4:
            sl = [slice(None)] * 4
            sl[0] = ik
            sl[1] = iE

        # Now figure out what data-type the delta is.
        if "Redelta" in lvl.variables:
            # It *must* be a complex valued Hamiltonian
            is_complex = True
            dtype = np.complex128
        elif "delta" in lvl.variables:
            is_complex = False
            dtype = np.float64

        # Get number of spins
        nspin = len(self.dimensions["spin"])

        # Now create the sparse matrix stuff (we re-create the
        # array, hence just allocate the smallest amount possible)
        C = cls(geom, nspin, nnzpr=1, dtype=dtype, orthogonal=True)

        C._csr.ncol = _a.arrayi(lvl.variables["n_col"][:])
        # Update maximum number of connections (in case future stuff happens)
        C._csr.ptr = _ncol_to_indptr(C._csr.ncol)
        C._csr.col = _a.arrayi(lvl.variables["list_col"][:]) - 1

        # Copy information over
        C._csr._nnz = len(C._csr.col)
        C._csr._D = np.empty([C._csr.ptr[-1], nspin], dtype)
        if is_complex:
            for ispin in range(nspin):
                sl[-2] = ispin
                C._csr._D[:, ispin].real = lvl.variables["Redelta"][sl] * Ry2eV
                C._csr._D[:, ispin].imag = lvl.variables["Imdelta"][sl] * Ry2eV
        else:
            for ispin in range(nspin):
                sl[-2] = ispin
                C._csr._D[:, ispin] = lvl.variables["delta"][sl] * Ry2eV

        # Convert from isc to sisl isc
        _csr_from_sc_off(C.geometry, lvl.variables["isc_off"][:, :], C._csr)

        _mat_siesta2sisl(C)
        C = C.astype(dtype=kwargs.get("dtype", dtype), copy=False)

        return C

    def read_delta(self, **kwargs):
        """Reads a delta model from the file"""
        return self._r_class(SparseOrbitalBZSpin, **kwargs)


add_sile("delta.nc", deltancSileTBtrans)
add_sile("dH.nc", deltancSileTBtrans)
add_sile("dSE.nc", deltancSileTBtrans)
