# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

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
  --shape-solver [nx ny nz] shape for which the solution is calculated (impacts speed)
  --shape [nx ny nz] final shape of the solution, if shape-solver is not the same the solution will be interpolated (order=2)
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
import argparse as argp
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import sisl as si
from sisl._internal import set_module

__all__ = ["pyamg_solve", "solve_poisson", "fftpoisson_fix_cli", "fftpoisson_fix_run"]


_BC = si.BoundaryCondition

# Base-script name
_script = Path(sys.argv[0]).name

_DEBUG = os.environ.get("SISL_TS_FFT_DEBUG", "False")
# Determine from env-var whether we should use debug mode
_DEBUG = _DEBUG.lower() in ("true", "t", "1", "y", "yes", "on")


def pyamg_solve(A, b, tolerance: float = 1e-12, accel=None, title: str = ""):
    import pyamg

    print(f"\nSetting up pyamg solver... {title}")
    ml = pyamg.aggregation.smoothed_aggregation_solver(A, max_levels=1000)
    del A
    print(ml)
    residuals = []

    def callback(x):
        # residuals calculated in the solve function is a pre-conditioned residual
        # residuals.append(np.linalg.norm(b - A.dot(x)) ** 0.5)
        print(
            "    {:4d}  residual = {:.5e}   x0-residual = {:.5e}".format(
                len(residuals) - 1, residuals[-1], residuals[-1] / residuals[0]
            )
        )

    x = ml.solve(
        b,
        tol=tolerance,
        callback=callback,
        residuals=residuals,
        accel=accel,
        cycle="W",
        maxiter=1e7,
    )
    print("Done solving the Poisson equation!")
    return x


@set_module("sisl_toolbox.transiesta.poisson")
def solve_poisson(
    geometry,
    shape,
    radius: float = 3.0,
    dtype=np.float64,
    tolerance: float = 1e-8,
    accel=None,
    boundary_fft: bool = True,
    device_val: Optional[float] = None,
    plot_boundary: bool = False,
    box: bool = False,
    boundary=None,
    **elecs_V,
):
    """Solve Poisson equation"""
    error = False
    elecs = []
    for name in geometry.names:
        if ("+" in name) or (name in ["Buffer", "Device"]):
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
        raise ValueError(
            f"{_script}: Missing electrode arguments for specifying the bias."
        )

    if boundary is None:
        bc = [[_BC.PERIODIC, _BC.PERIODIC] for _ in range(3)]
    else:
        bc = []

        def bc2bc(s: str) -> str:
            return {
                "periodic": "PERIODIC",
                "p": "PERIODIC",
                _BC.PERIODIC: "PERIODIC",
                "dirichlet": "DIRICHLET",
                "d": "DIRICHLET",
                _BC.DIRICHLET: "DIRICHLET",
                "neumann": "NEUMANN",
                "n": "NEUMANN",
                _BC.NEUMANN: "NEUMANN",
            }.get(s.lower(), s.upper())

        for bottom, top in boundary:
            bc.append([getattr(_BC, bc2bc(bottom)), getattr(_BC, bc2bc(top))])
        if len(bc) != 3:
            raise ValueError(
                f"{_script}: Requires a 3x2 list input for the boundary conditions."
            )

    def _create_shape_tree(xyz, A, B=None):
        """Takes two lists A and B which returns a shape with a binary nesting

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

    # Create grid and specify boundary conditions
    geometry.lattice.set_boundary_condition(bc)
    grid = si.Grid(shape, geometry=geometry, dtype=dtype)

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
    A, b = grid.to.pyamg()

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
        x = pyamg_solve(
            A,
            b,
            tolerance=tolerance,
            accel=accel,
            title="solving electrode boundary conditions",
        )
        grid.grid = x.reshape(shape)

        del A, b

    if boundary_fft:
        # Change boundaries to always use dirichlet
        # This ensures that once we set the boundaries we don't
        # get any side-effects
        BC = si.BoundaryCondition
        periodic = [
            bc == BC.PERIODIC or geometry.nsc[i] > 1
            for i, bc in enumerate(grid.lattice.boundary_condition[:, 0])
        ]
        bc = np.repeat(np.array([BC.DIRICHLET], np.int32), 6).reshape(3, 2)
        for i in (0, 1, 2):
            if periodic[i]:
                bc[i, :] = BC.PERIODIC
        grid.lattice.set_boundary_condition(bc)
        A, b = grid.to.pyamg()

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
            grid.pyamg_fix(
                A, b, idx, grid.grid[new_sl[0], new_sl[1], new_sl[2]].reshape(-1)
            )
            new_sl[i] = slice(grid.shape[i] - 1, grid.shape[i])
            idx = sl2idx(grid, new_sl)
            grid.pyamg_fix(
                A, b, idx, grid.grid[new_sl[0], new_sl[1], new_sl[2]].reshape(-1)
            )

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
        x = pyamg_solve(
            A,
            b,
            tolerance=tolerance,
            accel=accel,
            title="removing electrode boundaries and solving for edge fixing",
        )

        grid.grid = x.reshape(shape)
        del A, b

    return grid


def fftpoisson_fix_cli(subp=None, parser_kwargs={}):
    is_sub = not subp is None

    title = "FFT Poisson corrections for TranSiesta calculations for arbitrary number of electrodes."
    if is_sub:
        global _script
        _script = f"{_script} ts-fft"
        p = subp.add_parser("ts-fft", description=title, help=title, **parser_kwargs)
    else:
        p = argp.ArgumentParser(title, **parser_kwargs)

    tuning = p.add_argument_group(
        "tuning", "Tuning fine details of the Poisson calculation."
    )

    p.add_argument(
        "--geometry",
        "-G",
        default="siesta.TBT.nc",
        metavar="FILE",
        help="siesta.TBT.nc file which contains the geometry and electrode information, currently we cannot read that from fdf-files.",
    )

    p.add_argument(
        "--shape",
        "-s",
        nargs=3,
        type=int,
        required=True,
        metavar=("A", "B", "C"),
        help="Grid shape, this *has* to be conforming to the TranSiesta calculation, read from output: 'InitMesh: MESH = A x B x C'",
    )

    n = {"a": "first", "b": "second", "c": "third"}
    for d in "abc":
        p.add_argument(
            f"--boundary-condition-{d}",
            f"-bc-{d}",
            nargs=2,
            type=str,
            default=["p", "p"],
            metavar=("BOTTOM", "TOP"),
            help=(
                "Boundary condition along the {} lattice vector [periodic/p, neumann/n, dirichlet/d]. "
                "Specify separate BC at the start and end of the lattice vector, respectively.".format(
                    n[d]
                )
            ),
        )

    p.add_argument(
        "--elec-V",
        "-V",
        action="append",
        nargs=2,
        metavar=("NAME", "V"),
        default=[],
        help="Specify chemical potential on electrode",
    )

    p.add_argument(
        "--shape-solver",
        "-ss",
        "--pyamg-shape",
        "-ps",
        nargs=3,
        type=int,
        metavar=("A", "B", "C"),
        default=None,
        help="Grid used to solve the Poisson equation, if shape is different the Grid will be interpolated (order=2) after.",
    )

    p.add_argument(
        "--device",
        "-D",
        type=float,
        default=None,
        metavar="VAL",
        help="Fix the value of all device atoms to a value. In some cases this turns out to yield a better box boundary. The default is to *not* fix the potential on the device atoms.",
    )

    tuning.add_argument(
        "--radius",
        "-R",
        type=float,
        default=3.0,
        metavar="R",
        help=(
            "Radius of atoms when figuring out the electrode sizes, this corresponds to the extend of "
            "each electrode where boundary conditions are fixed. Should be tuned according to the atomic species [3 Ang]"
        ),
    )

    tuning.add_argument(
        "--dtype",
        "-d",
        choices=["d", "f64", "f", "f32"],
        default="d",
        help="Precision of data (d/f64==double, f/f32==single)",
    )

    tuning.add_argument(
        "--tolerance",
        "-T",
        type=float,
        default=1e-7,
        metavar="EPS",
        help="Precision required for the pyamg solver. NOTE when using single precision arrays this should probably be on the order of 1e-5",
    )

    tuning.add_argument(
        "--acceleration",
        "-A",
        dest="accel",
        default="cg",
        metavar="METHOD",
        help="""Acceleration method for pyamg. May be useful if it fails to converge

Try one of: cg, gmres, fgmres, cr, cgnr, cgne, bicgstab, steepest_descent, minimal_residual""",
    )

    test = p.add_argument_group(
        "testing",
        "Options used for testing output. None of these options should be used for production runs!",
    )
    test.add_argument(
        "--box",
        dest="box",
        action="store_true",
        default=False,
        help="Only store the initial box solution (i.e. do not run PyAMG)",
    )

    test.add_argument(
        "--no-boundary-fft",
        action="store_false",
        dest="boundary_fft",
        default=True,
        help="Disable the 2nd solution with the boundaries fixed. Generally, this solver will first solve for the electrode boundary conditions, then for the fixed box boundary conditains (gotten from the first solution).",
    )

    if _DEBUG:
        test.add_argument(
            "--plot",
            dest="plot",
            default=None,
            type=int,
            help="Plot grid by averaging over the axis given as argument",
        )

        test.add_argument(
            "--plot-boundary",
            dest="plot_boundary",
            action="store_true",
            help="Plot all 6 edges of the box with their fixed values (just before 2nd pyamg solve step)",
        )

    p.add_argument(
        "--out",
        "-o",
        action="append",
        default=None,
        help="Output file to store the resulting Poisson solution. It *has* to have TSV.nc file ending to make the file conforming with TranSiesta.",
    )

    if is_sub:
        p.set_defaults(runner=fftpoisson_fix_run)
    else:
        fftpoisson_fix_run(p.parse_args())


def fftpoisson_fix_run(args):
    if args.out is None:
        print(
            f">\n>\n>{_script}: No out-files has been specified, work will be carried out but not saved!\n>\n>\n"
        )

    # Fix the cases where the arguments hasn't been added
    if not _DEBUG:
        args.plot = None
        args.plot_boundary = False

    # Read in geometry
    geometry = si.get_sile(args.geometry).read_geometry()

    # Figure out the electrodes
    elecs_V = {}
    if len(args.elec_V) == 0:
        print(geometry.names)
        raise ValueError(
            f"{_script}: Please specify all electrode potentials using --elec-V"
        )

    for name, V in args.elec_V:
        elecs_V[name] = float(V)

    if args.dtype.lower() in ("f", "f32"):
        dtype = np.float32
    elif args.dtype.lower() in ("d", "f64"):
        dtype = np.float64

    # Now we can solve Poisson
    if args.shape_solver is None:
        shape = args.shape
    else:
        shape = args.shape_solver

    # Create the boundary conditions
    boundary = []
    boundary.append(args.boundary_condition_a)
    boundary.append(args.boundary_condition_b)
    boundary.append(args.boundary_condition_c)

    V = solve_poisson(
        geometry,
        shape,
        radius=args.radius,
        boundary=boundary,
        dtype=dtype,
        tolerance=args.tolerance,
        box=args.box,
        accel=args.accel,
        boundary_fft=args.boundary_fft,
        device_val=args.device,
        plot_boundary=args.plot_boundary,
        **elecs_V,
    )

    if not args.plot is None:
        dat = V.average(args.plot)
        import matplotlib.pyplot as plt

        axs = [
            np.linspace(0, V.lattice.length[ax], shape, endpoint=False)
            for ax, shape in enumerate(V.shape)
        ]
        idx = list(range(3))

        # Now plot data
        del axs[args.plot]
        del idx[args.plot]

        X, Y = np.meshgrid(*axs)
        plt.contourf(X, Y, np.squeeze(dat.grid).T)
        plt.colorbar()
        plt.title(f"Averaged over {'ABC'[args.plot]} axis")
        plt.xlabel(f"Distance along {'ABC'[idx[0]]} [Ang]")
        plt.ylabel(f"Distance along {'ABC'[idx[1]]} [Ang]")
        plt.show()

    if np.any(np.array(args.shape) != np.array(V.shape)):
        print("\nInterpolating the solution...")
        V = V.interp(args.shape, 2)
        print("Done interpolating!")

    print("")
    # Write solution to the output
    if not args.out is None:
        for out in args.out:
            print(f"Writing to file: {out}...")
            V.write(out)


# Import object holding all the CLI
from sisl_toolbox.cli import register_toolbox_cli

register_toolbox_cli(fftpoisson_fix_cli)


if __name__ == "__main__":
    fftpoisson_fix_cli()
