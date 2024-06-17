#!/usr/bin/env python
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This example relies on GULP being installed.

# It calculates the dynamical matrix for ZZ-oriented graphene
# and afterward it can write the dynamical matrix to be processed by
# PHtrans for transport properties.

from __future__ import annotations

import sisl

with open("zz.gin", "w") as f:
    f.write(
        """opti conv dist full nosymmetry phon dynamical_matrix nod3
output she
cutd 3.0

cell
17.04 19.67609717 15. 90 90 90

cartesian 128
C core 0.00000 0.00000 0.00000 0 1 0 1 1 1
C core 2.84000 0.00000 0.00000 0 1 0 1 1 1
C core 0.71000 1.22976 0.00000 0 1 0 1 1 1
C core 2.13000 1.22976 0.00000 0 1 0 1 1 1
C core 0.00000 2.45951 0.00000 0 1 0 1 1 1
C core 2.84000 2.45951 0.00000 0 1 0 1 1 1
C core 0.71000 3.68927 0.00000 0 1 0 1 1 1
C core 2.13000 3.68927 0.00000 0 1 0 1 1 1
C core 0.00000 4.91902 0.00000 0 1 0 1 1 1
C core 2.84000 4.91902 0.00000 0 1 0 1 1 1
C core 0.71000 6.14878 0.00000 0 1 0 1 1 1
C core 2.13000 6.14878 0.00000 0 1 0 1 1 1
C core 0.00000 7.37854 0.00000 0 1 0 1 1 1
C core 2.84000 7.37854 0.00000 0 1 0 1 1 1
C core 0.71000 8.60829 0.00000 0 1 0 1 1 1
C core 2.13000 8.60829 0.00000 0 1 0 1 1 1
C core 0.00000 9.83805 0.00000 0 1 0 1 1 1
C core 2.84000 9.83805 0.00000 0 1 0 1 1 1
C core 0.71000 11.06780 0.00000 0 1 0 1 1 1
C core 2.13000 11.06780 0.00000 0 1 0 1 1 1
C core 0.00000 12.29756 0.00000 0 1 0 1 1 1
C core 2.84000 12.29756 0.00000 0 1 0 1 1 1
C core 0.71000 13.52732 0.00000 0 1 0 1 1 1
C core 2.13000 13.52732 0.00000 0 1 0 1 1 1
C core 0.00000 14.75707 0.00000 0 1 0 1 1 1
C core 2.84000 14.75707 0.00000 0 1 0 1 1 1
C core 0.71000 15.98683 0.00000 0 1 0 1 1 1
C core 2.13000 15.98683 0.00000 0 1 0 1 1 1
C core 0.00000 17.21659 0.00000 0 1 0 1 1 1
C core 2.84000 17.21659 0.00000 0 1 0 1 1 1
C core 0.71000 18.44634 0.00000 0 1 0 1 1 1
C core 2.13000 18.44634 0.00000 0 1 0 1 1 1
C core 4.26000 0.00000 0.00000 0 1 0 1 1 1
C core 7.10000 0.00000 0.00000 0 1 0 1 1 1
C core 4.97000 1.22976 0.00000 0 1 0 1 1 1
C core 6.39000 1.22976 0.00000 0 1 0 1 1 1
C core 4.26000 2.45951 0.00000 0 1 0 1 1 1
C core 7.10000 2.45951 0.00000 0 1 0 1 1 1
C core 4.97000 3.68927 0.00000 0 1 0 1 1 1
C core 6.39000 3.68927 0.00000 0 1 0 1 1 1
C core 4.26000 4.91902 0.00000 0 1 0 1 1 1
C core 7.10000 4.91902 0.00000 0 1 0 1 1 1
C core 4.97000 6.14878 0.00000 0 1 0 1 1 1
C core 6.39000 6.14878 0.00000 0 1 0 1 1 1
C core 4.26000 7.37854 0.00000 0 1 0 1 1 1
C core 7.10000 7.37854 0.00000 0 1 0 1 1 1
C core 4.97000 8.60829 0.00000 0 1 0 1 1 1
C core 6.39000 8.60829 0.00000 0 1 0 1 1 1
C core 4.26000 9.83805 0.00000 0 1 0 1 1 1
C core 7.10000 9.83805 0.00000 0 1 0 1 1 1
C core 4.97000 11.06780 0.00000 0 1 0 1 1 1
C core 6.39000 11.06780 0.00000 0 1 0 1 1 1
C core 4.26000 12.29756 0.00000 0 1 0 1 1 1
C core 7.10000 12.29756 0.00000 0 1 0 1 1 1
C core 4.97000 13.52732 0.00000 0 1 0 1 1 1
C core 6.39000 13.52732 0.00000 0 1 0 1 1 1
C core 4.26000 14.75707 0.00000 0 1 0 1 1 1
C core 7.10000 14.75707 0.00000 0 1 0 1 1 1
C core 4.97000 15.98683 0.00000 0 1 0 1 1 1
C core 6.39000 15.98683 0.00000 0 1 0 1 1 1
C core 4.26000 17.21659 0.00000 0 1 0 1 1 1
C core 7.10000 17.21659 0.00000 0 1 0 1 1 1
C core 4.97000 18.44634 0.00000 0 1 0 1 1 1
C core 6.39000 18.44634 0.00000 0 1 0 1 1 1
C core 8.52000 0.00000 0.00000 0 1 0 1 1 1
C core 11.36000 0.00000 0.00000 0 1 0 1 1 1
C core 9.23000 1.22976 0.00000 0 1 0 1 1 1
C core 10.65000 1.22976 0.00000 0 1 0 1 1 1
C core 8.52000 2.45951 0.00000 0 1 0 1 1 1
C core 11.36000 2.45951 0.00000 0 1 0 1 1 1
C core 9.23000 3.68927 0.00000 0 1 0 1 1 1
C core 10.65000 3.68927 0.00000 0 1 0 1 1 1
C core 8.52000 4.91902 0.00000 0 1 0 1 1 1
C core 11.36000 4.91902 0.00000 0 1 0 1 1 1
C core 9.23000 6.14878 0.00000 0 1 0 1 1 1
C core 10.65000 6.14878 0.00000 0 1 0 1 1 1
C core 8.52000 7.37854 0.00000 0 1 0 1 1 1
C core 11.36000 7.37854 0.00000 0 1 0 1 1 1
C core 9.23000 8.60829 0.00000 0 1 0 1 1 1
C core 10.65000 8.60829 0.00000 0 1 0 1 1 1
C core 8.52000 9.83805 0.00000 0 1 0 1 1 1
C core 11.36000 9.83805 0.00000 0 1 0 1 1 1
C core 9.23000 11.06780 0.00000 0 1 0 1 1 1
C core 10.65000 11.06780 0.00000 0 1 0 1 1 1
C core 8.52000 12.29756 0.00000 0 1 0 1 1 1
C core 11.36000 12.29756 0.00000 0 1 0 1 1 1
C core 9.23000 13.52732 0.00000 0 1 0 1 1 1
C core 10.65000 13.52732 0.00000 0 1 0 1 1 1
C core 8.52000 14.75707 0.00000 0 1 0 1 1 1
C core 11.36000 14.75707 0.00000 0 1 0 1 1 1
C core 9.23000 15.98683 0.00000 0 1 0 1 1 1
C core 10.65000 15.98683 0.00000 0 1 0 1 1 1
C core 8.52000 17.21659 0.00000 0 1 0 1 1 1
C core 11.36000 17.21659 0.00000 0 1 0 1 1 1
C core 9.23000 18.44634 0.00000 0 1 0 1 1 1
C core 10.65000 18.44634 0.00000 0 1 0 1 1 1
C core 12.78000 0.00000 0.00000 0 1 0 1 1 1
C core 15.62000 0.00000 0.00000 0 1 0 1 1 1
C core 13.49000 1.22976 0.00000 0 1 0 1 1 1
C core 14.91000 1.22976 0.00000 0 1 0 1 1 1
C core 12.78000 2.45951 0.00000 0 1 0 1 1 1
C core 15.62000 2.45951 0.00000 0 1 0 1 1 1
C core 13.49000 3.68927 0.00000 0 1 0 1 1 1
C core 14.91000 3.68927 0.00000 0 1 0 1 1 1
C core 12.78000 4.91902 0.00000 0 1 0 1 1 1
C core 15.62000 4.91902 0.00000 0 1 0 1 1 1
C core 13.49000 6.14878 0.00000 0 1 0 1 1 1
C core 14.91000 6.14878 0.00000 0 1 0 1 1 1
C core 12.78000 7.37854 0.00000 0 1 0 1 1 1
C core 15.62000 7.37854 0.00000 0 1 0 1 1 1
C core 13.49000 8.60829 0.00000 0 1 0 1 1 1
C core 14.91000 8.60829 0.00000 0 1 0 1 1 1
C core 12.78000 9.83805 0.00000 0 1 0 1 1 1
C core 15.62000 9.83805 0.00000 0 1 0 1 1 1
C core 13.49000 11.06780 0.00000 0 1 0 1 1 1
C core 14.91000 11.06780 0.00000 0 1 0 1 1 1
C core 12.78000 12.29756 0.00000 0 1 0 1 1 1
C core 15.62000 12.29756 0.00000 0 1 0 1 1 1
C core 13.49000 13.52732 0.00000 0 1 0 1 1 1
C core 14.91000 13.52732 0.00000 0 1 0 1 1 1
C core 12.78000 14.75707 0.00000 0 1 0 1 1 1
C core 15.62000 14.75707 0.00000 0 1 0 1 1 1
C core 13.49000 15.98683 0.00000 0 1 0 1 1 1
C core 14.91000 15.98683 0.00000 0 1 0 1 1 1
C core 12.78000 17.21659 0.00000 0 1 0 1 1 1
C core 15.62000 17.21659 0.00000 0 1 0 1 1 1
C core 13.49000 18.44634 0.00000 0 1 0 1 1 1
C core 14.91000 18.44634 0.00000 0 1 0 1 1 1

brenner"""
    )

# Create PHtrans input
with open("ZZ.fdf", "w") as f:
    f.write(
        """SystemLabel ZZ

TBT.DOS.Gf T

TBT.k [51 1 1]

TBT.HS DEVICE_zz.nc

%block TBT.Contour.line
  part line
   from 0. eV to .23 eV
     points 400
      method mid-rule
%endblock TBT.Contour.line

%block TBT.Elec.Left
  HS ELEC_zz.nc
  semi-inf-direction -a2
  electrode-position 1
%endblock
%block TBT.Elec.Right
  HS ELEC_zz.nc
  semi-inf-direction +a2
  electrode-position end -1
%endblock
"""
    )

import os

if not os.path.exists("zz.gout"):
    raise ValueError("zz.gin has not been runned by GULP")

print("Reading output")
gout = sisl.get_sile("zz.gout")
# Correct what to read from the gulp output
gout.set_lattice_key("Cartesian lattice vectors")
# Selectively decide whether you want to read the dynamical
# matrix from the GULP output file or from the
# FORCE_CONSTANTS_2ND file.
order = ["got"]  # GULP output file
# order = ['FC'] # FORCE_CONSTANTS_2ND file

dyn = gout.read_dynamical_matrix(order=order)

# In GULP correcting for Newtons second law is already obeyed
# So this need not be used, however, the precision of output
# may require this anyway.
dyn.apply_newton()

dev = dyn.untile(4, 0)
dev.write("DEVICE_zz.nc")

el = dev.untile(4, 1)
el.write("ELEC_zz.nc")
