# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml as yaml_module

import sisl
from sisl._lib._argparse import get_argparse_parser
from sisl.messages import warn
from sisl_toolbox.siesta.minimizer import (
    AtomBasis,
    BADSMinimizeSiesta,
    EnergyMetric,
    FunctionRunner,
    LocalMinimizeSiesta,
    ParticleSwarmsMinimizeSiesta,
    SiestaRunner,
)
from sisl_toolbox.siesta.minimizer._minimize import _log
from sisl_toolbox.siesta.minimizer._yaml_reader import read_yaml


class RhoReuseSiestaRunner(SiestaRunner):

    def run(self):
        pipe = ""
        stdout, stderr = self._get_standard()
        for pre, f in [(">", stdout), ("2>", stderr)]:
            try:
                pipe += f"{pre} {f.name}"
            except Exception:
                pass
        cmd = [str(cmd) for cmd in self.cmd + ["<", self.fdf]]
        _log.debug(f"running Siesta using command[{self.path}]: {' '.join(cmd)} {pipe}")
        # Remove stuff to ensure that we don't read information from prior calculations
        self.clean(
            "*.ion*", "fdf.*.log", "INPUT_TMP.[0-9][0-9][0-9][0-9][0-9]"
        )  # f"{self.systemlabel}.*[!fdf]"

        if (self.path / "Rho.grid.nc").exists():
            if (self.path / "Rho.IN.grid.nc").exists():
                (self.path / "Rho.IN.grid.nc").unlink()

            Path(self.path / "Rho.IN.grid.nc").symlink_to(self.path / "Rho.grid.nc")

            using_rho = False
            with open(self.fdf, "r") as f:
                for line in f.readlines():
                    if "SCF.Read.Charge.NetCDF t" in line:
                        using_rho = True
                        break

            if not using_rho:
                with open(self.fdf, "a") as f:
                    f.write("SCF.Read.Charge.NetCDF t\n")

        return self.hook(
            subprocess.run(
                cmd[:-2],
                cwd=self.path,
                encoding="utf-8",
                stdin=open(self.fdf, "r"),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        )


# Helper functions
def get_row(Z: int) -> Tuple[int, str]:
    """Get the row of the periodic table for a given atomic number.

    If the atomic number is not in the periodic table, a ValueError is raised.

    Parameters
    ----------
    Z : int
        atomic number

    Returns
    -------
    int
        row of the periodic table where the atomic number is located. First
        row is index 0.
    str
        block of the element (s, p, d, f, g).
    """
    Z = abs(Z)

    if 0 < Z <= 2:
        return 0, "s"
    elif Z <= 18:
        rel_Z = Z - 2
        prev_rows = 1

        if rel_Z < 2:
            block = "s"
        else:
            block = "p"

        return ((rel_Z - 1) // 8) + prev_rows, block
    elif Z <= 54:
        rel_Z = Z - 18
        prev_rows = 3

        if rel_Z < 2:
            block = "s"
        elif rel_Z < 12:
            block = "d"
        else:
            block = "p"

        return ((rel_Z - 1) // 18) + prev_rows, block
    elif Z <= 118:
        rel_Z = Z - 54
        prev_rows = 5

        if rel_Z < 2:
            block = "s"
        elif rel_Z < 17:
            block = "f"
        elif rel_Z < 26:
            block = "d"
        else:
            block = "p"

        return ((rel_Z - 1) // 32) + prev_rows, block
    else:
        raise ValueError(f"No such atomic number in the periodic table: {Z}")


def get_valence_orbs_psml(psml: str):
    """Get valence orbitals from psml file.

    Parameters
    ----------
    psml : str
        Path to the psml file.
    """
    import xml.etree.ElementTree as ET

    # Get the root of the xml tree
    tree = ET.parse(psml)
    root = tree.getroot()

    # Iterate recursively over childs until we find the valence-configuration tag,
    # which contains the shells.
    shells = []
    for child in root.iter():
        if child.tag.endswith("valence-configuration"):

            for shell in child:
                shells.append(shell.get("n") + shell.get("l"))
            break

    return shells


def get_valence_orbs_psf(psf: Union[str, Path]):
    """Get valence orbitals from psf file header.

    Parameters
    ----------
    psf : Union[str, Path]
        Path to the psf file.
    """
    with open(psf, "r") as f:

        for _ in range(3):
            line = f.readline()

        shells = [
            shell_spec.split(" ")[0]
            for shell_spec in line.strip().split("/")
            if len(shell_spec) > 0
        ]

    return shells


def generate_basis_config(
    atoms: List[sisl.Atom],
    basis_size: str = "atoms",
    optimize_pol: bool = False,
    pol_optimize_screening: bool = False,
    pol_optimize_width: bool = False,
    initial_atoms_basis: bool = False,
    optimize_atoms_basis: bool = False,
    cutoff_upper_bound: float = 5.5,
    variable_delta: float = 0.05,
):
    """Generates a basis configuration file to be fed to the optimizer.

    Non-polarization orbitals are set to optimize their cutoff radii,
    while polarization orbitals are set to be optimized through the charge
    confinement parameters.

    The function looks at possible pseudopotential files (first psml, then psf) to
    determine which orbitals to include in the basis. Otherwise, it just generates
    the valence orbitals for the usual conventions, without semicores.

    Parameters
    ----------
    atoms : sisl.Atoms
        Atoms object to generate the basis configuration for.
    basis_size : str
        Specification of basis size. "XZ(P)" where X is the number of zetas
        and P is whether to polarize or not.

        X can also be S, D, T, Q for 1, 2, 3, 4 zetas respectively. E.g. "DZ".

        It can also be "atoms", which exactly takes the orbitals present in the atoms object.
    optimize_pol : bool, optional
        If the basis contains polarization orbitals, whether they should be
        explicitly optimized or not. If not, the default polarization orbitals
        will be requested.

        The polarization orbitals are optimized through the charge confinement.
    pol_optimize_screening : bool, optional
        If polarization orbitals are optimized, whether the screening length
        of the charge confinement should be optimized or not.
    pol_optimize_width : bool, optional
        If polarization orbitals are optimized, whether the width of the charge confinement
        should be optimized or not.
    initial_atoms_basis: bool, optional
        If True, the orbitals that are in the Atoms object are taken as the initial values for basis
        optimization.
    optimize_atoms_basis: bool, optional
        If ``initial_atom_basis`` is True, whether to optimize the orbitals read from the basis or not.
    cutoff_upper_bound : float, optional
        Upper bound for the cutoff radii.
    variable_delta : float, optional
        Delta that specifies the step size in changes to the optimized variables.
        It is used differently depending on the optimizer.
    """
    # Parse the basis size
    if basis_size != "atoms":
        try:
            n_zetas = int(basis_size[0])
        except:
            n_zetas = {
                "S": 1,
                "D": 2,
                "T": 3,
                "Q": 4,
            }[basis_size[0]]

        polarization = basis_size[-1] == "P"

        optimize_pol = optimize_pol and polarization
    else:
        n_zetas = None
        polarization = False
        optimize_pol = False

    # Helper function to determine lower bounds for cutoff radii.
    def get_cutoff_lowerbound(orb_name, i_zeta):
        """Returns the lower bound for the cutoff radius for a given orbital and zeta.

        The lower bound depends on which orbital it is and for which zeta the cutoff is.
        """
        if i_zeta > 0:
            if i_zeta == 1:
                return 1
            else:
                return 0
        elif orb_name == "1s":
            return 2
        else:
            return 2.5

    def _orbs_from_row_and_block(table_row, atom_block):
        """To be used if the orbitals to optimize can't be inferred from the pseudopotential."""
        if table_row == 0:
            orbs = ["1s"]
        elif table_row <= 2:
            orbs = [f"{valence_n}s", f"{valence_n}p"]
        elif table_row <= 4:
            orbs = [f"{valence_n - 1}d", f"{valence_n}s"]
            if atom_block != "d":
                orbs.append(f"{valence_n}p")
        else:
            orbs = [f"{valence_n - 2}f", f"{valence_n - 1}d", f"{valence_n}s"]
            if atom_block not in ("d", "f"):
                orbs.append(f"{valence_n}p")
            elif atom_block == "f":
                warn(
                    "The orbitals of f-block atoms might not be correctly determined automatically. Please check that they are correct."
                )

        return orbs

    config_dict = {}
    # Loop through atoms and generate a basis config for each, adding it to config_dict.
    for atom in atoms:
        table_row, atom_block = get_row(atom.Z)

        valence_n = table_row + 1
        pol_orbs = []

        # Find out which orbitals to include for this atom.
        tag = atom.tag
        if basis_size == "atoms":
            orbs = ["dummy"]
        elif Path(f"{tag}.psml").exists():
            # From the psml pseudopotential
            orbs = get_valence_orbs_psml(f"{tag}.psml")
        else:
            # If we didn't find any psml file, we just generate the valence orbitals using
            # the row and block of the atom.
            orbs = _orbs_from_row_and_block(table_row, atom_block)

            # And then look for psf pseudos to see if we need to add semicores.
            # We can't directly infer valence orbitals from psf because the headers
            # also contain the polarization orbitals.
            psf_files = list(Path().glob(f"{tag}.psf")) + list(
                Path().glob(f"{tag}.*.psf")
            )

            if len(psf_files) > 0:
                psf_orbs = get_valence_orbs_psf(psf_files[0])

                # The psf file contains also polarization orbitals,
                # so we need to remove them. What we do actually is to check if there are
                # lower shells than what we created and prepend them to the list of orbitals.
                i = 0
                for i, psf_orb in enumerate(psf_orbs):
                    if psf_orb in orbs:
                        break
                orbs = [*psf_orbs[:i], *orbs]

        orb_to_polarize = orbs[-1]
        # Add polarization orbitals
        if polarization:
            n = int(orb_to_polarize[0])
            l = "spdfg".find(orb_to_polarize[1])

            pol_l = l + 1
            pol_n = max(n, pol_l + 1)

            pol_orbs = [f"{pol_n}{'spdfg'[pol_l]}"]

        # We use the tag of the atom as the key for its basis entry.
        element_config = config_dict[tag] = {
            "element": tag,
        }

        # Read in the initial basis if requested.
        provided_basis = {}
        if initial_atoms_basis or basis_size == "atoms":

            for orbital in atom.orbitals:
                name = orbital.name()
                orb_name, zeta = name[:2], int(name.split("Z")[1][0])

                provided_basis[orb_name, zeta] = orbital.R

                # Add this orbital to the basis if we are requested to set up the
                # basis from the atoms object.
                if basis_size == "atoms":
                    if orb_name not in element_config:
                        element_config[orb_name] = {"basis": {}}

                    basis = element_config[orb_name]["basis"]

                    if optimize_atoms_basis:
                        basis[f"zeta{zeta}"].update(
                            {
                                "initial": orbital.R if orbital.R >= 0 else 3.0,
                                "bounds": [
                                    get_cutoff_lowerbound(orb_name, zeta),
                                    cutoff_upper_bound,
                                ],
                                "delta": variable_delta,
                            }
                        )
                    else:
                        basis[f"zeta{zeta}"] = orbital.R

            if basis_size == "atoms":
                # We have already set up the basis from the atoms.
                continue

        # If we are here, it means that the size of the basis is not determined by the provided
        # atoms object, that is, basis size is something like DZP.

        def get_cutoff_upper_bound(orb_name, i_zeta):
            if (orb_name, i_zeta) in provided_basis:
                return provided_basis[(orb_name, i_zeta)] * 0.95
            else:
                return cutoff_upper_bound

        # Loop through non-polarization orbitals and add them to the basis.
        for orb_name in orbs:
            orb_basis = {}

            if polarization and not optimize_pol:
                if orb_name == orb_to_polarize:
                    orb_basis["pol"] = 1

            for i_zeta in range(n_zetas):
                orb_id = (orb_name, i_zeta + 1)

                orb_basis[f"zeta{i_zeta + 1}"] = {
                    "initial": provided_basis.get(orb_id, 3.0),
                }

                if orb_id not in provided_basis:
                    # Orbital not provided, we need to optimize it from scratch.
                    orb_basis[f"zeta{i_zeta + 1}"].update(
                        {
                            "bounds": [
                                get_cutoff_lowerbound(orb_name, i_zeta),
                                get_cutoff_upper_bound(orb_name, i_zeta),
                            ],
                            "delta": variable_delta,
                        }
                    )
                else:
                    # Orbital provided, we don't need to optimize it.
                    orb_basis[f"zeta{i_zeta + 1}"] = provided_basis[orb_id]

            element_config[orb_name] = {"basis": orb_basis}

        # Loop through polarization orbitals and add them to the basis.
        if polarization:
            for orb_name in pol_orbs:

                # Check if the polarization orbital is already in the provided basis.
                if (orb_name, 1) in provided_basis:
                    # Orbital provided in the basis
                    pol_orb_basis = {}

                    zeta = 1
                    while (orb_name, zeta) in provided_basis:
                        pol_orb_basis[f"zeta{zeta}"] = provided_basis[(orb_name, zeta)]

                        zeta += 1

                elif optimize_pol:
                    # Orbital not provided in the basis, and we need to optimize it.
                    pol_orb_basis = {
                        "polarizes": orb_to_polarize,
                        "charge-confinement": {
                            "charge": {
                                "initial": 3.0,
                                "bounds": [1.0, 10.0],
                                "delta": variable_delta,
                            },
                        },
                    }

                    charge_confinement = pol_orb_basis["charge-confinement"]

                    if pol_optimize_screening:
                        # Screening length (Ang-1)
                        charge_confinement["yukawa"] = {
                            "initial": 0.0,
                            "bounds": [0.0, 3.0],
                            "delta": 0.5,
                        }
                    if pol_optimize_width:
                        # Width for the potential (Ang)
                        charge_confinement["width"] = {
                            "initial": 0.0053,
                            "bounds": [0.0, 0.5],
                            "delta": 0.05,
                        }

                else:
                    continue

                element_config[orb_name] = {"basis": pol_orb_basis}

    return config_dict


def set_minimizer_variables(
    minimizer,
    basis: Dict[str, AtomBasis],
    basis_config: dict,
    zeta: Union[int, None] = None,
    polarization: bool = False,
    optimized_variables: Union[dict, None] = None,
):
    """Sets the variables that the minimizer must optimize.

    A basis will contain all variables, but they must be optimized in a certain order.
    For each step, this function must be called to adapt the basis and set the variables
    on the minimizer.

    Parameters
    ----------
    minimizer : Minimizer
        Minimizer where to add the variables.
    basis : Dict[str, AtomBasis]
        Basis dictionary, which will be modified in-place to create the basis
        for the minimization step.
    basis_config : dict
        basis configuration dictionary, which will also be modified in-place.
    zeta : Union[int, None], optional
        Zeta shell for which we want to optimize cutoff radius. Set it to None
        if we are optimizing polarization orbitals.
    polarization : bool, optional
        Whether we are optimizing polarization orbitals or not.
    optimized_variables : Union[dict, None], optional
        Dictionary containing the already optimized variables. This is required
        unless we are in the first step (optimizing the first zeta shell).
    """
    for atom_key, atom_config in basis_config.items():

        # Loop through all defined orbitals and find those that are polarization orbitals.
        # We are going to perform some modifications on the basis if there are polarization
        # orbitals to optimize.
        for orb_key, orb_config in atom_config.items():

            if not isinstance(orb_config, dict):
                continue

            polarized_orb = orb_config.get("basis", {}).get("polarizes")

            # This orbital is not an orbital that polarizes another one.
            if polarized_orb is None:
                continue

            if polarization:
                # We are optimizing the polarization orbital, get its cutoff radii.
                if optimized_variables is None:
                    raise ValueError(
                        "To optimize the polarizing orbitals we need the optimized zeta cutoffs of the orbitals they polarize."
                    )

                # Remove the specification for a default polarized orbital, since we are going
                # to explicitly include it.
                atom_config[polarized_orb]["basis"].pop("pol")

                # Loop through all the zetas of the orbital that this one polarizes and
                # copy the cutoffs.
                zeta = 0
                while True:
                    zeta += 1

                    # The polarizing orbital has one less Z-shell than the
                    # orbital it polarizes, that's why we stop if there is no next zeta.
                    # SZP is the exception, because we need at least one zeta for the polarizing orbital.
                    next_var_key = f"{atom_key}.{polarized_orb}.z{zeta + 1}"
                    if zeta > 1 and next_var_key not in optimized_variables:
                        break

                    var_key = f"{atom_key}.{polarized_orb}.z{zeta}"
                    orb_config["basis"][f"zeta{zeta}"] = optimized_variables[var_key]

            else:
                # We are not optimizing the polarization orbitals, so we need to tell SIESTA
                # to use its default polarization orbitals, for the polarized orbital.
                atom_config[polarized_orb]["basis"]["pol"] = 1

        # Modify the basis.
        basis[atom_key] = AtomBasis.from_dict(atom_config)

        atom_basis = basis[atom_key]

        # Now that we have the appropiate basis for our next minimization, gather the
        # variable parameters and add them to the optimizer.
        for v in atom_basis.get_variables(basis_config, atom_key):

            var_prefix = ".".join(v.name.split(".")[:-1])

            if v.name.split(".")[-1][0] == "z":
                # This is a cutoff radius for a given Z.

                var_z = int(v.name.split(".")[-1][1:])

                if polarization or (zeta is not None and var_z < zeta + 1):
                    # This is a variable that should have been previously optimized
                    # or a fixed value is provided.
                    if optimized_variables is not None:
                        v.update(optimized_variables[f"{var_prefix}.z{var_z}"])
                elif zeta is not None and var_z == zeta + 1:
                    if (
                        var_z > 1
                        and optimized_variables is not None
                        and f"{var_prefix}.z{var_z - 1}" in optimized_variables
                    ):
                        v.bounds[1] = (
                            optimized_variables[f"{var_prefix}.z{var_z - 1}"] * 0.95
                        )

                    minimizer.add_variable(v)
                elif zeta is not None and var_z > zeta + 1:
                    v.update(0.0)
            else:
                # This is something that is not a cutoff radius. For now, we assume it's a variable
                # for a polarization orbital.
                if polarization:
                    minimizer.add_variable(v)


def write_basis(basis: Dict[str, AtomBasis], fdf: Union[str, Path]):
    """
    Writes a full basis specification to an fdf file.

    Parameters
    ----------
    basis : Dict[str, AtomBasis]
        Dictionary with each value being the basis specification for an atom type.
    fdf : Union[str, Path]
        Path to the fdf file where to write the basis specification.
    """
    full_basis = []
    for atom_key, atom_basis in basis.items():
        full_basis.extend(atom_basis.basis())

    sisl.io.siesta.fdfSileSiesta(fdf, "w").set("PAO.Basis", full_basis)


def write_basis_to_yaml(
    geometry: str,
    size: str = "DZP",
    optimize_pol: bool = False,
    pol_optimize_screening: bool = False,
    pol_optimize_width: bool = False,
    variable_delta: float = 0.05,
    initial_basis: Optional[str] = None,
    optimize_initial_basis: bool = False,
    out: Optional[str] = "basis_config.yaml",
):
    """Generates a basis configuration and writes it to a yaml file.

    This yaml file can then be used to run an optimizer.

    The function looks at possible pseudopotential files (first psml, then psf) to
    determine which orbitals to include in the basis. Otherwise, it just generates
    the valence orbitals for the usual conventions, without semicores.

    Parameters
    ----------
    geometry : str
        Path to the geometry file for which we want to generate the basis configuration.
    size : str, optional
        Specification of basis size. "XZ(P)" where X is the number of zetas
        and P is whether to polarize or not.

        X can also be S, D, T, Q for 1, 2, 3, 4 zetas respectively. E.g. "DZ".

        It can also be "fdf" or "basis" to use the basis size as read from the `initial_basis` argument.
        This forces the `initial_basis` argument to be equal to the `size` argument.
    optimize_pol : bool, optional
        If the basis contains polarization orbitals, whether they should be
        explicitly optimized or not. If not, the default polarization orbitals
        will be requested.
    pol_optimize_screening : bool, optional
        If polarization orbitals are optimized, whether the screening length
        of the charge confinement should be optimized or not.
    pol_optimize_width : bool, optional
        If polarization orbitals are optimized, whether the width of the charge confinement
        should be optimized or not.
    variable_delta : float, optional
        Delta that specifies the step size in changes to the optimized variables.
        It is used differently depending on the optimizer.
    initial_basis: str, optional
        If provided, it should be either "fdf" or "basis", to read the initial values from
        the fdf file or the basis files respectively.

        It is force to the value of `size` if "fdf" or "basis" is provided there.
    optimize_initial_basis: bool, optional
        If an initial basis has been provided, whether to optimize the orbitals read from it.
    out: str, optional
        Path to the yaml file where to write the basis configuration.
    """

    basis_read = initial_basis
    if size in ("basis", "fdf"):
        if initial_basis is not None and initial_basis != size:
            raise ValueError(
                "If `size` is 'basis' or 'fdf', `initial_basis` must be None or equal to `size`."
            )
        basis_read = size
        size = "atoms"

    if basis_read == "fdf":
        atoms = sisl.get_sile(geometry).read_basis(order=["fdf"])
    elif basis_read == "basis":
        atoms = sisl.get_sile(geometry).read_basis()
    else:
        atoms = sisl.get_sile(geometry).read_geometry().atoms.atom

    basis_config = generate_basis_config(
        atoms,
        size,
        optimize_pol=optimize_pol,
        pol_optimize_screening=pol_optimize_screening,
        pol_optimize_width=pol_optimize_width,
        variable_delta=variable_delta,
        initial_atoms_basis=initial_basis is not None,
        optimize_atoms_basis=optimize_initial_basis,
    )

    dumps = yaml_module.dump(basis_config)

    if out is not None:
        with open(out, "w") as f:
            yaml_module.dump({"basis_spec": basis_config}, f)

    return dumps


def optimize_basis(
    geometry: str,
    size: str = "DZP",
    optimize_pol: bool = False,
    variable_delta: float = 0.05,
    basis_spec: dict = {},
    start: str = "z1",  # "z{x}" or pol
    stop: str = "pol",  # "z{x}" or pol
    tol_fun: float = 1e-3,
    tol_stall_iters: int = 4,
    siesta_cmd: str = "siesta",
    out: str = "basisoptim.dat",
    out_fmt: str = "20.17e",
    logging_level: str = "notset",  # Literal["critical", "notset", "info", "debug"]
    optimizer: str = "bads",
    optimizer_kwargs: dict = {},
):
    """Optimizes a basis set for a given geometry.

    The optimization follows the below sequence:

    - First it optimizes the cutoff radii for the first zeta shell.

    - Then it optimizes the cutoff radii for the second zeta shell.

    - ...

    - Then it optimizes the cutoff radii for the last zeta shell.

    - Finally, it optimizes the polarization orbitals (if requested).\n\n

    Appropiate optimization bounds are set for each parameter.\n\n

    At each step, all corresponding parameters are optimized simultaneously.\n\n

    The optimization algorithm can be selected with the `method` argument. Note that
    for all methods it is recommended to run multiple instances of the optimization
    and check that they lead to the same minimum (or pick the best one). Although some
    algorithms might be more reliable than others to avoid local minima.

    If no basis specification is provided, the basis specification is automatically generated
    using the arguments ``size`` and ``optimize_pol``.

    To determine which orbitals to include in the basis, the function looks at possible
    pseudopotential files (first psml, then psf). If they are not there, it just generates
    the valence orbitals for the usual conventions, without semicores.

    This function only provides the simplest specification for basis shapes. If you need more
    customization control, first generate a basis specification and then provide it to this
    function's `basis_spec` argument.

    Parameters
    ----------
    geometry : str
        Path to the geometry file for which we want to generate the basis configuration.
        This fdf file might include other parameters to use when running SIESTA. It can't be
        named RUN.fdf, though.
    size : str, optional
        Specification of basis size. "XZ(P)" where X is the number of zetas
        and P is whether to polarize or not.

        X can also be S, D, T, Q for 1, 2, 3, 4 zetas respectively. E.g. "DZ".

        It can also be "fdf" or "basis" to use the basis size as read from the `initial_basis` argument.
        This forces the `initial_basis` argument to be equal to the `size` argument.
    optimize_pol : bool, optional
        If the basis contains polarization orbitals, whether they should be
        explicitly optimized or not. If not, the default polarization orbitals
        will be requested.
    variable_delta : float, optional
        Delta that specifies the step size in changes to the optimized variables.
        It is used differently depending on the optimizer.
    basis_spec : dict, optional
        An already built basis specification dictionary. If provided, it will be used
        instead of created from the previous arguments.

        You can use ``write_yaml`` to generate a template, and then tailor it to your needs.
    start : str, optional
        At which step to start optimizing. Previous steps will be read from previous runs.
        Can be "z{x}" or "pol" where x is the zeta shell number.
    stop : str, optional
        At which step to stop optimizing. Can be "z{x}" or "pol" where x is the zeta shell number.
        The optimization is stopped AFTER optimizing this step.
    tol_fun : float, optional
        Tolerance for the optimization function value. If the function value changes less than this
        value after some iterations, the optimization is stopped.
    tol_stall_iters : int, optional
        Number of iterations that the function value must not change more than tol_fun to consider
        the optimization ended. Note that one iteration in BADS consists of multiple function evaluations.
    siesta_cmd : str, optional
        Shell command to run siesta.
    out : str, optional
        Path to the file where to write the points evaluated by the optimizer. For each step, a file
        will be generated prefixed with the step name: "{step}_{out}.
    out_fmt : str, optional
        Format string to use when writing the points evaluated by the optimizer.
    logging_level : str, optional
        Logging level to use. Can be "critical", "notset", "info" or "debug".
    optimizer : str, optional
        Which optimizer to use. Can be:
            - "bads": Uses pybads to run the Bayesian Adaptive Direct Search (BADS) algorithm
             for optimization. When benchmarking algorithms, BADS has proven to be the best at finding the
             global minimum, this is why it is the default.
            - "swarms": Uses pyswarms to run a Particle Swarms optimization.
            - "local": Uses the scipy.optimize.minimize function.
    optimizer_kwargs : dict, optional
        Keyword arguments to pass to the optimizer.
    """

    logging.basicConfig(
        format="%(levelname)s:%(message)s",
    )  # level=getattr(logging, logging_level.upper())
    # Set the log level of the sisl logger
    _log.setLevel(getattr(logging, logging_level.upper()))

    optimizer_kwargs = optimizer_kwargs.copy()

    # Generate the basis specification if no basis specification is provided.
    if len(basis_spec) == 0:
        basis_yaml = write_basis_to_yaml(
            geometry=geometry,
            size=size,
            optimize_pol=optimize_pol,
            variable_delta=variable_delta,
            out=None,
        )
        basis_spec = yaml_module.safe_load(basis_yaml)

    # From the basis specification, generate the basis dictionary.
    atom_keys = list(basis_spec.keys())
    basis = {
        atom_key: AtomBasis.from_dict(basis_spec[atom_key]) for atom_key in atom_keys
    }

    # Create a RUN.fdf file that includes the geometry and the basis.
    fdf_path = Path(geometry)
    if fdf_path.exists() and fdf_path.name != "RUN.fdf":
        with open(fdf_path.parent / "RUN.fdf", "w") as f:
            f.write(f"%include BASIS.fdf\n")
            f.write(f"%include {fdf_path}\n")
            f.write(f"SaveRho t\n")

    run_siesta = RhoReuseSiestaRunner(
        ".", cmd=siesta_cmd, stdout="RUN.out", stderr="error"
    )
    run_basis_writer = FunctionRunner(write_basis, basis, run_siesta.path / "BASIS.fdf")
    metric = EnergyMetric(
        run_siesta.absattr("stdout"), energy="basis.enthalpy", failure=0.0
    )

    # Define sequence of running stuff
    runner = run_basis_writer & run_siesta

    # Get the number of zeta shells
    n_zetas = 0
    for atom_config in basis_spec.values():
        for orb_config in atom_config.values():
            if "basis" in orb_config:
                n_zetas = max(
                    n_zetas,
                    len(
                        [
                            key
                            for key in orb_config["basis"].keys()
                            if key.startswith("zeta")
                        ]
                    ),
                )

    optimized_variables = {}

    optim_iters = []
    for zeta in range(n_zetas):
        # We allow the minimizer to go below the cutoff range if it deems it necessary.
        # We also allow it to go above the cutoff range for the first zeta shell.
        if zeta == 0:

            def get_hard_bounds(bounds):
                hard_bounds = np.full_like(bounds, 0.5)

                hard_bounds[:, 1] = np.max(bounds) + 2

                return hard_bounds

        else:

            def get_hard_bounds(bounds):
                hard_bounds = np.full_like(bounds, 0.5)

                hard_bounds[:, 1] = bounds[:, 1]

                return hard_bounds

        optim_iters.append(
            {
                "zeta": zeta,
                "polarization": False,
                "out_prefix": f"z{zeta + 1}",
                "skip": not start.startswith("z") or (int(start[1:]) > zeta + 1),
                "stop": stop.startswith("z") and (int(stop[1:]) < zeta + 1),
                "get_hard_bounds": get_hard_bounds,
                "start_info": "Otimizing zeta shell {zeta + 1}",
            }
        )

    optim_iters.append(
        {
            "zeta": None,
            "polarization": True,
            "out_prefix": "pol",
            "skip": False,
            "stop": True,
            "get_hard_bounds": lambda bounds: bounds,
            "start_info": "Otimizing polarization orbitals",
        }
    )

    # Pick the optimizer
    optimizer = optimizer.lower()
    if optimizer == "bads":
        minimizer_cls = BADSMinimizeSiesta
    elif optimizer == "local":
        minimizer_cls = LocalMinimizeSiesta
    elif optimizer == "swarms":
        minimizer_cls = ParticleSwarmsMinimizeSiesta
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    for optim_iter in optim_iters:
        _log.info(optim_iter["start_info"])

        # Get the minimizer and set the variables that it must optimize
        minimize = minimizer_cls(
            out=f"{optim_iter['out_prefix']}_{out}",
            norm="identity",
            runner=runner,
            metric=metric,
            out_fmt=out_fmt,
        )
        set_minimizer_variables(
            minimize,
            basis,
            basis_spec,
            zeta=optim_iter["zeta"],
            polarization=optim_iter["polarization"],
            optimized_variables=optimized_variables,
        )

        # Check if the user wants to optimize this Z shell
        if not optim_iter.get("skip", False) and len(minimize.variables) > 0:

            if optimizer == "local":
                # Retrieve delta-x for the jacobian for this
                eps = minimize.normalize("delta", with_offset=False)

                result = minimize.run(
                    tol=tol_fun,
                    jac="2-point",
                    options={
                        "disp": True,
                        "ftol": tol_fun,
                        "iprint": 3,
                        "eps": eps,
                        "finite_diff_rel_step": eps,
                        "maxiter": 1000,
                        **optimizer_kwargs.pop("options", {}),
                    },
                    **optimizer_kwargs,
                )
            elif optimizer == "swarms":
                result = minimize.run(**optimizer_kwargs)
            else:
                result = minimize.run(
                    get_hard_bounds=optim_iter["get_hard_bounds"],
                    options={
                        "tol_fun": tol_fun,
                        "tol_stall_iters": tol_stall_iters,
                        **optimizer_kwargs.pop("options", {}),
                    },
                    **optimizer_kwargs,
                )
        else:
            # Just make it read the data
            with minimize:
                pass

        if len(minimize.variables) > 0:
            # Get minimum from all samples
            candidates = minimize.candidates()
            minimize.update(candidates.x_target)

            best_values = {
                var.name: var_best
                for var, var_best in zip(minimize.variables, candidates.x_target)
            }

            optimized_variables.update(best_values)

        # Write the minimum to a file.
        write_basis(basis, f"{optim_iter['out_prefix']}_after_minimize.fdf")

        if optim_iter.get("stop", False):
            break


def basis_cli(subp=None, parser_kwargs={}):
    """Argparse CLI for the basis optimization utilities."""

    # Create main parser
    title = "Basis optimization utilities"
    is_sub = not subp is None
    if is_sub:
        p = subp.add_parser("basis", description=title, help=title, **parser_kwargs)
    else:
        p = argparse.ArgumentParser(title, description=title, **parser_kwargs)

    # If the main parser is executed, just print the help
    p.set_defaults(runner=lambda args: p.print_help())

    # Add the subparsers for the commands
    subp = p.add_subparsers(title="Commands")

    get_argparse_parser(
        optimize_basis, name="optim", subp=subp, parser_kwargs=parser_kwargs
    )
    get_argparse_parser(
        write_basis_to_yaml, name="build", subp=subp, parser_kwargs=parser_kwargs
    )


# Import object holding all the CLI
from sisl_toolbox.cli import register_toolbox_cli

register_toolbox_cli(basis_cli)

# def get_typer_app():
#     """Returns a typer app to use as a CLI for basis optimization."""
#     import typer

#     def yaml_dict(d: str):

#         if isinstance(d, dict):
#             return d

#         return yaml_module.safe_load(d)

#     # Helper function to annotate a function with the help from the docstring.
#     # In this way, typer can display the help on the command line.
#     def _annotate_func(func):
#         import inspect
#         from copy import copy

#         from typing_extensions import Annotated

#         params_help = {}

#         in_parameters = False
#         read_key = None
#         arg_content = ""

#         for line in func.__doc__.split("\n"):
#             if "Parameters" in line:
#                 in_parameters = True
#                 space = line.find("Parameters")
#                 continue

#             if in_parameters:
#                 if len(line) < space + 1:
#                     continue
#                 if len(line) > 1 and line[0] != " ":
#                     break

#                 if line[space] not in (" ", "-"):
#                     if read_key is not None:
#                         params_help[read_key] = arg_content

#                     read_key = line.split(":")[0].strip()
#                     arg_content = ""
#                 else:
#                     if arg_content == "":
#                         arg_content = line.strip().capitalize()
#                     else:
#                         arg_content += " " + line.strip()

#             if line.startswith("------"):
#                 break

#         if read_key is not None:
#             params_help[read_key] = arg_content

#         @functools.wraps(func)
#         def wrapper(*args, config: str = "", **kwargs):
#             return func(*args, **kwargs)

#         def config_callback(ctx, param, param_value):

#             if param_value != "":
#                 with open(param_value, "r") as f:
#                     yaml_config = yaml_module.safe_load(f)

#                 ctx.default_map = ctx.default_map or {}
#                 ctx.default_map.update(yaml_config)

#         sig = inspect.signature(func)

#         new_parameters = []
#         for param in sig.parameters.values():

#             parser = None
#             if param.annotation == dict:
#                 parser = yaml_dict

#             typer_arg_cls = (
#                 typer.Argument
#                 if param.default == inspect.Parameter.empty
#                 else typer.Option
#             )
#             new_parameters.append(
#                 param.replace(
#                     annotation=Annotated[
#                         param.annotation,
#                         typer_arg_cls(help=params_help.get(param.name), parser=parser),
#                     ]
#                 )
#             )

#         new_parameters.append(
#             inspect.Parameter(
#                 "config",
#                 default="",
#                 kind=inspect.Parameter.KEYWORD_ONLY,
#                 annotation=Annotated[
#                     str,
#                     typer.Option(
#                         "--config",
#                         "-c",
#                         help="YAML file to load the configuration from.",
#                         is_eager=True,
#                         callback=config_callback,
#                     ),
#                 ],
#             )
#         )

#         wrapper.__signature__ = sig.replace(parameters=new_parameters)
#         wrapper.__doc__ = func.__doc__[: func.__doc__.find("Parameters\n")]

#         return wrapper

#     # Create the app.
#     app = typer.Typer(name="Basis optimization utilities", rich_markup_mode="markdown")

#     app.command("write-yaml")(_annotate_func(write_basis_to_yaml))
#     app.command("optimize")(_annotate_func(optimize_basis))

#     return app
