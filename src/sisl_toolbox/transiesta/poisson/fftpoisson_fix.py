# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
r""" Hartree correction for FFT-Poisson solver for arbitrary electrode positions

Developer: Nick Papior
Contact: nickpapior <at> gmail.com
sisl-version: >=0.9.3

This Poisson solver uses pyamg to calculate an initial guess for the Poisson
solution to correct the FFT solution. It does this by setting up boundary
conditions on electrodes and then solving the Hartree potential using multi-grid
solvers.

It requires two inputs and has several optional flags.

- The siesta.TBT.nc file which contains the geometry that is to be calculated for
  The reason for using the siesta.TBT.nc file is the ease of use:

    The siesta.TBT.nc contains electrode atoms and device atoms. Hence it
    becomes easy to read in the electrode atomic positions.
    Note that since you'll always do a 0 V calculation it is easy to
    do a 0 bias calculation first, then create a guess using this script,
    and finally do bias calculations.

- The grid size of the simulation grid, this needs to be commensurate with the
  actual Siesta grid used.

This script is a command-line utility with several options (please refer to
--help). There are a few important flags you should know about:

  --tolerance [tol] specify the tolerance of the solution, the tighter the longer solution time
  --pyamg-shape [nx ny nz] shape for which the solution is calculated (impacts speed)
  --shape [nx ny nz] final shape of the solution, if pyamg-shape is not the same the solution will be interpolated (order=2)
  --dtype [f|d] the data-type used to solve the Poisson equation
  --out [file] any sisl compatible grid file, please at least do --out V.TSV.nc which is compatible with TranSiesta.

This tool requires the following packages:
- pyamg

Known problems:
- The pyamg solver requires quite a bit of memory, you should preferentially select
  the largest grid (up to the actual grid size you want) possible.
- The Neumann implementation in the boundary conditions is not correct, hence
  it may never converge (or produce nan's). If this happens please try another
  boundary condition.
- It may not always converge which requires some fine-tuning of the tolerances,
  secondly it may converge too fast so the solution is not really good.
"""
from typing import Tuple, List, Optional, Dict
from typing_extensions import Annotated

import argparse as argp
from enum import Enum
import os
import sys
from pathlib import Path

import numpy as np

import sisl as si

from sisl_toolbox.cli._cli_arguments import CLIOption

__all__ = ['pyamg_solve', 'solve_poisson', 'fftpoisson_fix']


# Base-script name
_script = Path(sys.argv[0]).name

_DEBUG = os.environ.get("SISL_TS_FFT_DEBUG", "False")
# Determine from env-var whether we should use debug mode
_DEBUG = _DEBUG.lower() in ("true", "t", "1", "y", "yes", "on")


def pyamg_solve(A, b, tolerance=1e-12, accel=None, title=""):
    import pyamg
    print(f"\nSetting up pyamg solver... {title}")
    ml = pyamg.aggregation.smoothed_aggregation_solver(A, max_levels=1000)
    del A
    print(ml)
    residuals = []
    def callback(x):
        # residuals calculated in the solve function is a pre-conditioned residual
        #residuals.append(np.linalg.norm(b - A.dot(x)) ** 0.5)
        print("    {:4d}  residual = {:.5e}   x0-residual = {:.5e}".format(len(residuals) - 1, residuals[-1], residuals[-1] / residuals[0]))
    x = ml.solve(b, tol=tolerance, callback=callback, residuals=residuals,
                 accel=accel, cycle='W', maxiter=1e7)
    print('Done solving the Poisson equation!')
    return x


def solve_poisson(geometry, shape, radius="empirical",
                  dtype=np.float64, tolerance=1e-8,
                  accel=None, boundary_fft=True,
                  device_val=None, plot_boundary=False,
                  box=False, boundary=None, **elecs_V):
    """ Solve Poisson equation """
    error = False
    elecs = []
    for name in geometry.names:
        if ('+' in name) or (name in ["Buffer", "Device"]):
            continue

        # This is actually an electrode
        elecs.append(name)
        error = error or (name not in elecs_V)

    if len(elecs) == 0:
        raise ValueError(f"{_script}: Could not find any electrodes in the geometry.")

    error = error or len(elecs) != len(elecs_V)
    if error:
        for name in elecs:
            if not name in elecs_V:
                print(f" missing electrode bias: {name}")
        raise ValueError(f"{_script}: Missing electrode arguments for specifying the bias.")

    if boundary is None:
        bc = [[si.Grid.PERIODIC, si.Grid.PERIODIC] for _ in range(3)]
    else:
        bc = []
        def bc2bc(s):
            return {'periodic': 'PERIODIC', 'p': 'PERIODIC', si.Grid.PERIODIC: 'PERIODIC',
                    'dirichlet': 'DIRICHLET', 'd': 'DIRICHLET', si.Grid.DIRICHLET: 'DIRICHLET',
                    'neumann': 'NEUMANN', 'n': 'NEUMANN', si.Grid.NEUMANN: 'NEUMANN',
            }.get(s.lower(), s.upper())
        for bottom, top in boundary:
            bc.append([getattr(si.Grid, bc2bc(bottom)), getattr(si.Grid, bc2bc(top))])
        if len(bc) != 3:
            raise ValueError(f"{_script}: Requires a 3x2 list input for the boundary conditions.")

    def _create_shape_tree(xyz, A, B=None):
        """ Takes two lists A and B which returns a shape with a binary nesting

        This makes further index handling much faster.
        """
        if B is None or len(B) == 0:
            return _create_shape_tree(xyz, *np.array_split(A, 2))

        AA, BB = None, None
        if len(A) == 1:
            AA = si.Sphere(radius, xyz[A[0]])
            if len(B) == 0:
                return AA

        if len(B) == 1:
            BB = si.Sphere(radius, xyz[B[0]])
            if len(A) == 0:
                return BB

        # Quick return if these are the final ones
        if AA and BB:
            return AA | BB

        if not AA:
            AA = _create_shape_tree(xyz, *np.array_split(A, 2))
        if not BB:
            BB = _create_shape_tree(xyz, *np.array_split(B, 2))

        return AA | BB

    # Create grid
    grid = si.Grid(shape, geometry=geometry, bc=bc, dtype=dtype)
    class _fake:
        @property
        def shape(self):
            return shape
        @property
        def dtype(self):
            return dtype

    # Fake the grid to reduce memory requirement
    grid.grid = _fake()

    # Construct matrices we need to specify the boundary conditions on
    A, b = grid.topyamg()

    # Short-hand notation
    xyz = geometry.xyz

    if not device_val is None:
        print(f"\nApplying device potential = {device_val}")
        idx = geometry.names["Device"]
        device = _create_shape_tree(xyz, idx)
        idx = grid.index_truncate(grid.index(device))
        idx = grid.pyamg_index(idx)
        grid.pyamg_fix(A, b, idx, device_val)

    # Apply electrode constants
    print("\nApplying electrode potentials")
    for i, elec in enumerate(elecs):
        V = elecs_V[elec]
        print(f"  - {elec} = {V}")

        idx = geometry.names[elec]
        elec_shape = _create_shape_tree(xyz, idx)

        idx = grid.index_truncate(grid.index(elec_shape))
        idx = grid.pyamg_index(idx)
        grid.pyamg_fix(A, b, idx, V)
    del idx, elec_shape

    # Now we have initialized both A and b with correct boundary conditions
    # Lets solve the Poisson equation!
    if box:
        # No point in solving the boundary problem if requesting a box
        boundary_fft = False
        grid.grid = b.reshape(shape)
        del A
    else:
        x = pyamg_solve(A, b, tolerance=tolerance, accel=accel,
                        title="solving electrode boundary conditions")
        grid.grid = x.reshape(shape)

        del A, b

    if boundary_fft:
        # Change boundaries to always use dirichlet
        # This ensures that once we set the boundaries we don't
        # get any side-effects
        periodic = grid.bc[:, 0] == grid.PERIODIC
        bc = np.repeat(np.array([grid.DIRICHLET], np.int32), 6).reshape(3, 2)
        for i in (0, 1, 2):
            if periodic[i]:
                bc[i, :] = grid.PERIODIC
        grid.set_bc(bc)
        A, b = grid.topyamg()

        # Solve only for the boundary fixed
        def sl2idx(grid, sl):
            return grid.pyamg_index(grid.mgrid(sl))

        # Create slices
        sl = [slice(0, g) for g in grid.shape]

        # One boundary at a time
        for i in (0, 1, 2):
            if periodic[i]:
                continue
            new_sl = sl[:]
            new_sl[i] = slice(0, 1)
            idx = sl2idx(grid, new_sl)
            grid.pyamg_fix(A, b, idx, grid.grid[new_sl[0], new_sl[1], new_sl[2]].reshape(-1))
            new_sl[i] = slice(grid.shape[i] - 1, grid.shape[i])
            idx = sl2idx(grid, new_sl)
            grid.pyamg_fix(A, b, idx, grid.grid[new_sl[0], new_sl[1], new_sl[2]].reshape(-1))

        if plot_boundary:
            dat = b.reshape(*grid.shape)
            # now plot every plane
            import matplotlib.pyplot as plt
            slicex3 = np.index_exp[:] * 3
            axs = [
                np.linspace(0, grid.lattice.length[ax], shape, endpoint=False)
                for ax, shape in enumerate(grid.shape)
            ]

            for i in (0, 1, 2):
                idx = list(slicex3)
                j = (i + 1) % 3
                k = (i + 2) % 3
                if i > j:
                    i, j = j, i
                X, Y = np.meshgrid(axs[i], axs[j])

                for v, head in ((0, "bottom"), (-1, "top")):
                    plt.figure()
                    plt.title(f"axis: {'ABC'[k]} ({head})")
                    idx[k] = v
                    plt.contourf(X, Y, dat[tuple(idx)].T)
                    plt.xlabel(f"Distance along {'ABC'[i]} [Ang]")
                    plt.ylabel(f"Distance along {'ABC'[j]} [Ang]")
                    plt.colorbar()

            plt.show()

        grid.grid = _fake()
        x = pyamg_solve(A, b, tolerance=tolerance, accel=accel,
                        title="removing electrode boundaries and solving for edge fixing")

        grid.grid = x.reshape(shape)
        del A, b

    return grid

class DtypeOption(Enum):
    """Data types"""
    d = "d"
    f = "f"
    f64 = "f64"
    f32 = "f32"

class AccelMethod(Enum):
    """Acceleration methods for pyamg"""
    cg = "cg"
    gmres = "gmres"
    fgmres = "fgmres"
    cr = "cr"
    cgnr = "cgnr"
    cgne = "cgne"
    bicgstab = "bicgstab"
    steepest_descent = "steepest_descent"
    minimal_residual = "minimal_residual"

def fftpoisson_fix(
    shape: Annotated[Tuple[int, int, int], CLIOption("-S", "--shape")],
    geometry: Path = Path("siesta.TBT.nc"),
    boundary_condition_a: Annotated[Tuple[str, str], CLIOption("-bc-a", "--boundary-condition-a")] = ("p", "p"),
    boundary_condition_b: Annotated[Tuple[str, str], CLIOption("-bc-b", "--boundary-condition-b")] = ("p", "p"),
    boundary_condition_c: Annotated[Tuple[str, str], CLIOption("-bc-c", "--boundary-condition-c")] = ("p", "p"),
    elec_V: Annotated[Dict[str, float], CLIOption("-V", "--elec-V", )] = {},
    pyamg_shape: Annotated[Tuple[int, int, int], CLIOption("-ps", "--pyamg-shape")] = (-1, -1, -1),
    device: Annotated[Optional[float], CLIOption("-D", "--device")] = None,
    radius: Annotated[float, CLIOption("-R", "--radius")] = 3.,
    dtype: Annotated[DtypeOption, CLIOption("-d", "--dtype")] = DtypeOption.d,
    tolerance: Annotated[float, CLIOption("-T", "--tolerance")] = 1e-7,
    accel: Annotated[AccelMethod, CLIOption("-A", "--acceleration")] = AccelMethod.cg,
    out: Annotated[List[str], CLIOption("-o", "--out", metavar="PATH")] = [],
    box: bool = False,
    boundary_fft: bool = True,
    plot: Optional[int] = None,
    plot_boundary: bool = False,
):
    """FFT Poisson corrections for TranSiesta calculations for arbitrary number of electrodes.
    
    Parameters
    ----------
    geometry:
        siesta.TBT.nc file which contains the geometry and electrode information, 
        currently we cannot read that from fdf-files.
    shape:
        Grid shape, this *has* to be conforming to the TranSiesta calculation, 
        read from output: 'InitMesh: MESH = A x B x C'
    boundary_condition_a:
        Boundary condition along the first lattice vector [periodic/p, neumann/n, dirichlet/d].
        Specify separate BC at the start and end of the lattice vector, respectively.
    boundary_condition_b:
        Boundary condition along the second lattice vector [periodic/p, neumann/n, dirichlet/d]. 
        Specify separate BC at the start and end of the lattice vector, respectively.
    boundary_condition_c:
        Boundary condition along the third lattice vector [periodic/p, neumann/n, dirichlet/d].
        Specify separate BC at the start and end of the lattice vector, respectively.
    elec_V:
        Specify chemical potential on electrode.
    pyamg_shape:
        Grid used to solve the Poisson equation, if shape is different 
        the Grid will be interpolated (order=2) after.
        (-1, -1, -1) means use the same as shape.
    device:
        Fix the value of all device atoms to a value. 
        In some cases this turns out to yield a better box boundary. 
        The default is to *not* fix the potential on the device atoms.
    radius:
        Radius of atoms when figuring out the electrode sizes, 
        this corresponds to the extend of each electrode where boundary conditions are fixed. 
        Should be tuned according to the atomic species [3 Ang]
    dtype:
        Precision of data (d/f64==double, f/f32==single)
    tolerance:
        Precision required for the pyamg solver. 
        NOTE when using single precision arrays this should probably be on the order of 1e-5
    accel:
        Acceleration method for pyamg. 
        May be useful if it fails to converge
    out:
        Output file to store the resulting Poisson solution. 
        It *has* to have TSV.nc file ending to make the file conforming with TranSiesta.
    box:
        Only store the initial box solution (i.e. do not run PyAMG)
    boundary_fft:
        Once the electrode boundary conditions are solved we perform a second solution with boundaries fixed. 
        Using this flag disables this second solution.
    plot:
        Plot grid by averaging over the axis given as argument
    plot_boundary:
        Plot all 6 edges of the box with their fixed values (just before 2nd pyamg solve step)
    
    """

    print(dict(
        geometry=geometry,
        shape=shape,
        boundary_condition_a=boundary_condition_a,
        boundary_condition_b=boundary_condition_b,
        boundary_condition_c=boundary_condition_c,
        elec_V=elec_V,
        pyamg_shape=pyamg_shape,
        device=device,
        radius=radius,
        dtype=dtype,
        tolerance=tolerance,
        accel=accel,
        out=out,
        box=box,
        boundary_fft=boundary_fft,
        plot=plot,
        plot_boundary=plot_boundary,
    ))

    return
    if len(out) == 0:
        print(f">\n>\n>{_script}: No out-files has been specified, work will be carried out but not saved!\n>\n>\n")

    # Read in geometry
    geometry = si.get_sile(geometry).read_geometry()

    # Figure out the electrodes
    elecs_V = {}
    if len(elec_V) == 0:
        print(geometry.names)
        raise ValueError(f"{_script}: Please specify all electrode potentials using --elec-V")

    for name, V in elec_V.items():
        elecs_V[name] = V

    if dtype.value.lower() in ("f", "f32"):
        dtype = np.float32
    elif dtype.value.lower() in ("d", "f64"):
        dtype = np.float64

    # Now we can solve Poisson
    if pyamg_shape[0] == -1:
        shape = shape
    else:
        shape = pyamg_shape

    # Create the boundary conditions
    boundary = []
    boundary.append(boundary_condition_a)
    boundary.append(boundary_condition_b)
    boundary.append(boundary_condition_c)

    V = solve_poisson(geometry, shape, radius=radius, boundary=boundary,
                      dtype=dtype, tolerance=tolerance, box=box,
                      accel=accel.value, boundary_fft=boundary_fft,
                      device_val=device, plot_boundary=plot_boundary,
                      **elecs_V)

    if _DEBUG:
        if not plot is None:
            dat = V.average(plot)
            import matplotlib.pyplot as plt
            axs = [
                np.linspace(0, V.lattice.length[ax], shape, endpoint=False) for ax, shape in enumerate(V.shape)
            ]
            idx = list(range(3))

            # Now plot data
            del axs[plot]
            del idx[plot]

            X, Y = np.meshgrid(*axs)
            plt.contourf(X, Y, np.squeeze(dat.grid).T)
            plt.colorbar()
            plt.title(f"Averaged over {'ABC'[plot]} axis")
            plt.xlabel(f"Distance along {'ABC'[idx[0]]} [Ang]")
            plt.ylabel(f"Distance along {'ABC'[idx[1]]} [Ang]")
            plt.show()

    if np.any(np.array(shape) != np.array(V.shape)):
        print("\nInterpolating the solution...")
        V = V.interp(shape, 2)
        print("Done interpolating!")

    print("")
    # Write solution to the output
    for out_file in out:
        print(f"Writing to file: {out_file}...")
        V.write(out_file)
