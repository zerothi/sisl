! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine write_dm(fname, nspin, no_u, nsc, nnz, ncol, list_col, DM)

  use precision, only: i4, r8
  use sparse_io_m, only: write_sparse, write_data_2d2
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer(i4), intent(in) :: nspin, no_u, nsc(3), nnz
  integer(i4), intent(in) :: ncol(no_u), list_col(nnz)
  real(r8), intent(in) :: DM(nnz,nspin)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin, no_u, nsc, nnz
!f2py intent(in) :: ncol, list_col
!f2py intent(in) :: DM

! Internal variables and arrays
  integer :: iu, is, i, idx, ierr

  call open_file(fname, 'write', 'unknown', 'unformatted', iu)

  ! Also write the supercell.
  write(iu, iostat=ierr) no_u, nspin, nsc
  call iostat_update(ierr)

  ! Sparse pattern
  call write_sparse(iu, no_u, nnz, ncol, list_col)
  call write_data_2d2(iu, no_u, nspin, nnz, ncol, DM)

  call close_file(iu)

end subroutine write_dm
