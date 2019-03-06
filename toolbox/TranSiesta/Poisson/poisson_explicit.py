""" Poisson solver for arbitrary electrode positions

Developer: Nick Papior
Contact: nickpapior <at> gmail.com
sisl-version: >=0.9.3

This Poisson solver uses pyamg to calculate an initial guess for the Poisson
solution.

It requires two inputs and has several optional flags.

- The siesta.TBT.nc file which contains the geometry that is to be calculated for
  The reason for using the siesta.TBT.nc file is the ease of use:

    The siesta.TBT.nc contains electrode atoms and device atoms. Hence it
    becomes easy to read in the electrode atomic positions.
    Note that since you'll always do a 0 V calculation this isn't making
    any implications for the requirement of the TBT.nc file.

- The grid size utilized in the grid, this needs to be commensurate with the
  Siesta grid actually used.

This script is a command-line utility with several options (please refer to the
help). There are a few important flags you should know about:

  --tolerance [tol] specify the tolerance of the solution, the tighter the longer solution time
  --pyamg-shape [nx ny nz] shape for which the solution is calculated
  --shape [nx ny nz] final shape of the solution, if pyamg-shape is not the same the solution will be linearly interpolated
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
from __future__ import print_function, division

import sys
import argparse as argp
import numpy as np
import sisl as si
import pyamg

# Retrieve the script-name
_script = sys.argv[0]


def pyamg_solve(A, b, tolerance=1e-12):
    print("\nSetting up pyamg solver...")
    ml = pyamg.aggregation.smoothed_aggregation_solver(A, max_levels=1000)
    print(ml)
    M = ml.aspreconditioner(cycle='W') # pre-conditioner
    residuals = []
    def callback(x):
        # residuals calculated in the solve function is a pre-conditioned residual
        #residuals.append(np.linalg.norm(b - A.dot(x)) ** 0.5)
        print("    {:4d}  residual = {:.5e}   x0-residual = {:.5e}".format(len(residuals) - 1, residuals[-1], residuals[-1] / residuals[0]))
    x = ml.solve(b, tol=tolerance, callback=callback, residuals=residuals, accel="cg", maxiter=1e7)
    print('Done solving the Poisson equation!')
    return x


def solve_poisson(geometry, shape, radius=2.0, dtype=np.float64, tolerance=1e-12, boundary=None, **elecs_V):
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
        raise ValueError("{}: Could not find any electrodes in the geometry.".format(_script))

    error = error or len(elecs) != len(elecs_V)
    if error:
        for name in elecs:
            if not name in elecs_V:
                print(" missing electrode bias: {}".format(name))
        raise ValueError("{}: Missing electrode arguments for specifying the bias.".format(_script))

    if boundary is None:
        bc = [[si.Grid.PERIODIC, si.Grid.PERIODIC] for _ in range(3)]
    else:
        bc = []
        for bottom, top in boundary:
            bottom = bottom.upper()
            top = top.upper()
            bc.append([getattr(si.Grid, bottom), getattr(si.Grid, top)])
        if len(bc) != 3:
            raise ValueError("{}: Requires a 3x2 list input for the boundary conditions.".format(_script))

    def _create_shape_tree(xyz, A, B):
        """ Takes two lists A and B which are forming a tree """
        AA, BB = None, None
        if len(A) == 1:
            #add_A.append(A[0])
            AA = si.Sphere(radius, xyz[A[0]])
            if len(B) == 0:
                return AA

        if len(B) == 1:
            #add_B.append(B[0])
            BB = si.Sphere(radius, xyz[B[0]])
            if len(A) == 0:
                return BB

        # Quick return if these are the final ones
        if AA and BB:
            return AA | BB

        idx_A1, idx_A2 = np.array_split(A, 2)
        idx_B1, idx_B2 = np.array_split(B, 2)

        if not AA:
            AA = _create_shape_tree(xyz, idx_A1, idx_A2)
        if not BB:
            BB = _create_shape_tree(xyz, idx_B1, idx_B2)

        return AA | BB

    #add_A = []
    #add_B = []

    print("Constructing electrode regions for fixing potential")
    elec_shapes = []
    xyz = geometry.xyz
    for elec in elecs:
        print("  - {}".format(elec))
        # Get electrode indices and coordinates
        idx_elec = geometry.names[elec]
        idx_1, idx_2 = np.array_split(idx_elec, 2)
        #add_A.clear()
        #add_B.clear()
        elec_shapes.append(_create_shape_tree(xyz, idx_1, idx_2))
        #assert len(add_A) + len(add_B) == len(idx_elec)

    # Create grid
    grid = si.Grid(shape, geometry=geometry, bc=bc, dtype=dtype)
    class _fake(object):
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

    # Apply electrode constants
    print("\nApplying electrode potentials")
    for i, elec in enumerate(elecs):
        print("  - {}".format(elec))
        idx = grid.index_fold(grid.index(elec_shapes[i]))
        idx = grid.pyamg_index(idx)
        V = elecs_V[elec]
        grid.pyamg_fix(A, b, idx, V)
        del idx
    del elec_shapes

    # Now we have initialized both A and b with correct boundary conditions
    # Lets solve the Poisson equation!
    x = pyamg_solve(A, b, tolerance=tolerance)
    grid.grid = x.reshape(shape)

    del A, b

    return grid


if __name__ == "__main__":

    # Create the argument parser
    p = argp.ArgumentParser("Creation of custom Poisson solutions for TranSiesta calculations with arbitrary number of electrodes.")

    n = {"a": "first", "b": "second", "c": "third"}
    for d in "abc":
        p.add_argument("--boundary-condition-{}".format(d), "-bc-{}".format(d), nargs=2, type=str, default=["periodic", "periodic"],
                       metavar=["BC-bottom", "BC-top"],
                       help=("Boundary condition along the {} lattice vector [periodic, neumann, dirichlet]. "
                             "Provide two to specify separate BC at the start and end of the lattice vector, respectively.".format(n[d])))

    p.add_argument("--radius", "-R", type=float, default=2.,
                   help=("Radius of atoms when figuring out the electrode sizes, this corresponds to the extend of each electrode where boundary"
                         "conditions are fixed."))

    p.add_argument("--dtype", "-d", choices=["d", "f"], default="d",
                   help="Precision of data (d==double, f==single)")

    p.add_argument("--elec-V", "-ebV", nargs=2, action="append", metavar=["NAME", "V"], default=[],
                   help="Specify the potential on the electrode")

    p.add_argument("--tolerance", "-T", type=float, default=1e-10,
                   help="Precision required for the pyamg solver. NOTE when using single precision arrays this should probably be on the order of 1e-5")

    p.add_argument("--shape", "-s", nargs=3, type=int, required=True, metavar=["X", "Y", "Z"],
                   help="Grid shape, this *has* to be conforming to the TranSiesta calculation, read from output: 'InitMesh: MESH = X x Y x Z'")

    p.add_argument("--pyamg-shape", "-ps", nargs=3, type=int, metavar=["X", "Y", "Z"], default=None,
                   help="Grid used to solve the Poisson equation, if shape is different the Grid will be interpolated after.")

    p.add_argument("--geometry", default="siesta.TBT.nc",
                   help="siesta.TBT.nc file which contains the geometry and electrode information, currently we cannot read that from fdf-files.")

    p.add_argument("--out", "-o", action="append", default=None,
                   help="Output file to store the resulting Poisson solution. It *has* to have TSV.nc file ending to make the file conforming ")

    # Parse args
    args = p.parse_args()

    if args.out is None:
        print(">\n>\n>{}: No out-files has been specified, work will be carried out but not saved!\n>\n>\n".format(_script))

    # Read in geometry
    geometry = si.get_sile(args.geometry).read_geometry()

    # Figure out the electrodes
    elecs_V = {}
    if len(args.elec_V) == 0:
        print(geometry.names)
        raise ValueError("{}: Please specify the electrode potentials using --elec-V")

    for name, V in args.elec_V:
        elecs_V[name] = float(V)

    if args.dtype == "f":
        dtype = np.float32
    elif args.dtype == "d":
        dtype = np.float64

    # Now we can solve Poisson
    if not args.pyamg_shape is None:
        shape = args.pyamg_shape
    else:
        shape = args.shape

    # Create the boundary conditions
    boundary = []
    boundary.append(args.boundary_condition_a)
    boundary.append(args.boundary_condition_b)
    boundary.append(args.boundary_condition_c)

    V = solve_poisson(geometry, shape, radius=args.radius, boundary=boundary,
                      dtype=dtype, tolerance=args.tolerance, **elecs_V)

    if not np.all(V.shape == shape):
        print("\nInterpolating the solution...")
        V = V.interp(args.shape)
        print("Done interpolating!")

    print("")
    # Write solution to the output
    for out in args.out:
        print("Writing to file: {}...".format(out))
        V.write(out)
