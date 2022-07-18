# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from dataclasses import asdict, is_dataclass
from numbers import Number
from typing import Sequence, Optional, Union
import numpy as np
from xarray import DataArray

import sisl
from sisl.geometry import Geometry
from sisl.physics.distribution import get_distribution
from sisl.viz.input_fields import SpinIndexSelect, OrbitalQueries
from sisl.viz.nodes.data_sources.atom_data import AtomData
from sisl.viz.nodes.node import Node
from ..data_sources import DataSource, FileDataSIESTA, HamiltonianDataSource
from .orbital import reduce_orbital_data

try:
    import pathos
    _do_parallel_calc = True
except:
    _do_parallel_calc = False

class OrbitalData:
    pass

class PDOSData(OrbitalData):
    """Holds PDOS Data in a custom xarray DataArray.
    
    The point of this class is to normalize the data coming from different sources
    so that functions can use it without worrying where the data came from. 
    """

    def __init__(self, PDOS: np.ndarray, geometry: Geometry, E: Sequence[float], E_units: str = 'eV', spin: Optional[Union[sisl.Spin, str, int]] = None, extra_attrs: dict = {}):
        """

        Parameters
        ----------
        PDOS: numpy.ndarray of shape ([nSpin], nE, nOrb)
            The Projected Density Of States, orbital resolved. The array can have 2 or 3 dimensions,
            since the spin dimension is optional. The spin class of the calculation that produced the data
            is inferred from the spin dimension:
                If there is no spin dimension or nSpin == 1, the calculation is spin unpolarized.
                If nSpin == 2, the calculation is spin polarized.
                If nSpin == 4, the calculation is assumed to be with noncolinear spin.
        geometry: sisl.Geometry
            The geometry to which the data corresponds. It must have as many orbitals as the PDOS data.
        E: numpy.ndarray of shape (nE,)
            The energies to which the data corresponds.
        E_units: str, optional
            The units of the energy. Defaults to 'eV'.
        extra_attrs: dict
            A dictionary of extra attributes to be added to the DataArray. One of the attributes that 
        """
        # Understand what the spin class is for this data.
        data_spin = sisl.Spin.UNPOLARIZED
        if PDOS.squeeze().ndim == 3:
            data_spin = {
                2: sisl.Spin.POLARIZED,
                4: sisl.Spin.NONCOLINEAR
            }[PDOS.shape[0]]
        data_spin = sisl.Spin(data_spin)

        # If no spin specification was passed, then assume the spin is what we inferred from the data.
        # Otherwise, make sure the spin specification is consistent with the data.
        if spin is None:
            spin = data_spin
        else:
            spin = sisl.Spin(spin)
            if data_spin.is_diagonal:
                assert spin == data_spin
            else:
                assert not spin.is_diagonal

        # Unpolarized spin data doesn't have a spin dimension.
        if spin.is_unpolarized and PDOS.ndim == 3:
            PDOS = PDOS[0]

        # Check that the number of orbitals in the geometry and the data match.
        orb_dim = PDOS.ndim - 2
        if geometry is not None:
            if geometry.no != PDOS.shape[orb_dim]:
                raise ValueError(f"The geometry provided contains {geometry.no} orbitals, while we have PDOS information of {PDOS.shape[orb_dim]}.")
        
        # Build the standardized dataarray, with everything needed to understand it.
        E_units = extra_attrs.pop("E_units", "eV")
        coords = [("orb", range(PDOS.shape[orb_dim])), ("E", E, {"units": E_units})]
        if not spin.is_unpolarized:
            spin_options = SpinIndexSelect.get_spin_options(spin)
            coords = [("spin", spin_options), *coords]

        attrs = {"spin": spin, "geometry": geometry, "units": f"1/{E_units}", **extra_attrs}

        self._data = DataArray(
            PDOS, 
            coords=coords, 
            name="PDOS",
            attrs=attrs
        )

    def __getattr__(self, key):
        return getattr(self._data, key)

    def __dir__(self):
        return dir(self._data)

class PDOSDataNode(DataSource):

    @classmethod
    def register(cls, func):
        return cls.from_func(func)

    _get = PDOSData

@PDOSDataNode.register
def PDOSDataSIESTA(fdf=None, pdos_file=None) -> PDOSData:
    """Gets the PDOS from a SIESTA PDOS file"""
    pdos_file = FileDataSIESTA(fdf=fdf, path=pdos_file, cls=sisl.io.pdosSileSiesta)
    # Get the info from the .PDOS file
    geometry, E, PDOS = pdos_file.read_data()

    return PDOSData(PDOS, geometry, E)

@PDOSDataNode.register
def PDOSDataH(H, kgrid=None, kgrid_displ=(0, 0, 0), Erange=(-2, 2),
    E0=0, nE=100, distribution=get_distribution("gaussian")) -> PDOSData:
    """Calculates the PDOS from a sisl Hamiltonian."""
        
    H = HamiltonianDataSource(H=H)

    # Get the kgrid or generate a default grid by checking the interaction between cells
    # This should probably take into account how big the cell is.
    kgrid = kgrid
    if kgrid is None:
        kgrid = [3 if nsc > 1 else 1 for nsc in H.geometry.nsc]

    Erange = Erange
    if Erange is None:
        raise ValueError('You need to provide an energy range to calculate the PDOS from the Hamiltonian')

    E = np.linspace(Erange[0], Erange[-1], nE) + E0

    bz = sisl.MonkhorstPack(H, kgrid, kgrid_displ)

    # Define the available spins
    spin_indices = [0]
    if H.spin.is_polarized:
        spin_indices = [0, 1]

    # Calculate the PDOS for all available spins
    PDOS = []
    for spin in spin_indices:
        with bz.apply(pool=_do_parallel_calc) as parallel:
            spin_PDOS = parallel.average.eigenstate(
                spin=spin,
                wrap=lambda eig: eig.PDOS(E, distribution=distribution)
            )

        PDOS.append(spin_PDOS)

    if not H.spin.is_diagonal:
        PDOS = PDOS[0]

    PDOS = np.array(PDOS)

    return PDOSData(PDOS, H.geometry, E, spin=H.spin, extra_attrs={'bz': bz})

@PDOSDataNode.register
def PDOSDataTBTrans(fdf=None, tbt_nc=None) -> PDOSData:
    """Reads the PDOS from a *.TBT.nc file coming from a TBtrans run."""
    tbt_nc = FileDataSIESTA(fdf=fdf, path=tbt_nc, cls=sisl.io.tbtncSileTBtrans)
    PDOS = tbt_nc.DOS(sum=False).T
    E = tbt_nc.E

    read_geometry_kwargs = {}
    # Try to get the basis information from the fdf, if possible
    try:
        fdf = FileDataSIESTA(path=fdf)
        read_geometry_kwargs["atom"] = fdf.read_geometry(output=True).atoms
    except: #(FileNotFoundError, TypeError):
        pass

    # Read the geometry from the TBT.nc file and get only the device part
    geometry = tbt_nc.read_geometry(**read_geometry_kwargs).sub(tbt_nc.a_dev)

    return PDOSData(PDOS, geometry, E)

@PDOSDataNode.register
def PDOSDataWFSX(H=None, fdf=None, wfsx_file=None, Erange=(-2, 2), 
    nE=100, E0=0, distribution=get_distribution('gaussian')) -> PDOSData:
    """Generates the PDOS values from a file containing eigenstates."""
    # Read the hamiltonian. We need it because we need the overlap matrix.
    H = HamiltonianDataSource(H=H)
    geometry = getattr(H, "geometry", None)
    if geometry is None:
        fdf = FileDataSIESTA(path=fdf)
        geometry = fdf.read_geometry(output=True)

    # Get the wfsx file
    wfsx_sile = FileDataSIESTA(
        fdf=fdf, path=wfsx_file, cls=sisl.io.wfsxSileSiesta, parent=H
    )

    # Read the sizes of the file, which contain the number of spin channels
    # and the number of orbitals and the number of k points.
    sizes = wfsx_sile.read_sizes()
    # Check that spin sizes of hamiltonian and wfsx file match
    assert H.spin.size == sizes.nspin, \
        f"Hamiltonian has spin size {H.spin.size} while file has spin size {sizes.nspin}"
    # Get the size of the spin channel. The size returned might be 8 if it is a spin-orbit
    # calculation, but we need only 4 spin channels (total, x, y and z), same as with non-colinear
    nspin = min(4, sizes.nspin)

    # Get the energies for which we need to calculate the PDOS.
    Erange = Erange
    E = np.linspace(Erange[0], Erange[-1], nE) + E0

    # Initialize the PDOS array
    PDOS = np.zeros((nspin, sizes.no_u, E.shape[0]), dtype=np.float64)

    # Loop through eigenstates in the WFSX file and add their contribution to the PDOS.
    # Note that we pass the hamiltonian as the parent here so that the overlap matrix
    # for each point can be calculated by eigenstate.PDOS()
    for eigenstate in wfsx_sile.yield_eigenstate():
        spin = eigenstate.info.get("spin", 0)
        if nspin == 4:
            spin = slice(None)

        PDOS[spin] += eigenstate.PDOS(E, distribution=distribution) * eigenstate.info.get("weight", 1)
    
    return PDOSData(PDOS, geometry, E, spin=H.spin)

class PDOSProcessor(Node):
    pass

def get_request_sanitizer(PDOS_data):
    # Get the requests parameter, which will be needed to retrieve available options
    # and get the list of orbitals that correspond to a given request.
    requests_param = OrbitalQueries()

    requests_param.update_options(PDOS_data.geometry, PDOS_data.spin)

    def _sanitize_request(request):
        # First, complete the request
        complete_req = requests_param.complete_query

        # If it's a non-colinear or spin orbit spin class, the default spin will be total,
        # since averaging/summing over "x","y","z" does not make sense.
        default_spin = None
        if not PDOS_data.spin.is_diagonal:
            default_spin = ["total"]
        
        # Get the complete request and make sure it is a dict.
        request = complete_req({"spin": default_spin, **request})
        if is_dataclass(request):
            request = asdict(request)

        # Determine the reduce function from the "reduce" passed and the scale factor. 
        def _reduce_func(*args, **kwargs):
            reduce_ = request['reduce']
            if isinstance(reduce_, str):
                reduce_ = getattr(np, reduce_)
            
            return reduce_(*args, **kwargs) * request.get("scale", 1)

        # Finally, return the sanitized request, converting the request (contains "species", "n", "l", etc...)
        # into a list of orbitals. 
        return {
            **request,
            "orbitals": requests_param.get_orbitals(request),
            "reduce_func": _reduce_func,
            "width": request['size'],
        }

    return _sanitize_request

@PDOSProcessor.from_func
def get_PDOS_requests(PDOS_data, requests, E0=0, Erange=None, drop_empty: bool = True):

    # Shift the energies
    E_PDOS = PDOS_data.assign_coords(E=PDOS_data.E - E0)
    # Select a given energy range
    if Erange is not None:
        # Get the energy range that has been asked for.
        Emin, Emax = Erange
        E_PDOS = PDOS_data.sel(E=slice(Emin, Emax))

    # Get the function that is going to convert our request to something that can actually
    # select orbitals from the xarray object.
    _sanitize_request = get_request_sanitizer(PDOS_data)

    PDOS = reduce_orbital_data(
        E_PDOS, requests, orb_dim="orb", spin_dim="spin", sanitize_group=_sanitize_request,
        group_vars=('color', 'width', 'dash'), groups_dim="request", drop_empty=drop_empty,
        spin_reduce=np.sum,
    )

    return PDOS

class AtomPDOS(AtomData):

    def __init__(self, pdos_data, atoms=None, E=None, request_kwargs={}, **kwargs):
        super().__init__(pdos_data=pdos_data, atoms=atoms, E=E, request_kwargs=request_kwargs, **kwargs)
    
    def _get(self, pdos_data, atoms=None, E=None, request_kwargs={}):
        # Unless E is a slice, we will match energies according to the
        # nearest one available.
        if E is None:
            data = pdos_data
        elif isinstance(E, slice):
            data = pdos_data.sel(E=E)
        else:
            # Ensure that when we select, we always keep the Energy dimension
            if isinstance(E, Number):
                E = (E, )
            data = pdos_data.sel(E=E, method="nearest")

        # Get the PDOS for all energies. The array is of shape
        # (n_atoms, n_energies).
        atom_data = get_PDOS_requests(
            data, requests=[{
                "atoms": atoms, **request_kwargs, "split_on": "atoms"
            }]
        ).values

        # Ensure the array has an energy dimension and reduce it.
        return atom_data.reshape((atom_data.shape[0], -1)).sum(axis=-1)
  