! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine write_tshs_hs(fname, &
    nspin, na_u, no_u, nnz, &
    nsc1, nsc2, nsc3, &
    cell, xa, lasto, &
    ncol, list_col, H, S, isc)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, na_u, no_u, nnz
  integer, intent(in) :: nsc1, nsc2, nsc3
  real(dp), intent(in) :: cell(3,3), xa(3,na_u)
  integer, intent(in) :: lasto(0:na_u)
  integer, intent(in) :: ncol(no_u), list_col(nnz)
  real(dp), intent(in) :: H(nnz,nspin), S(nnz)
  integer, intent(in) :: isc(3,nsc1*nsc2*nsc3)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin, na_u, no_u, nnz
!f2py intent(in) :: nsc1, nsc2, nsc3
!f2py intent(in) :: cell, xa, lasto
!f2py intent(in) :: ncol, list_col
!f2py intent(in) :: H, S
!f2py intent(in) :: isc

! Internal variables and arrays
  integer :: iu, ierr, is, i, idx

  ! Open file (ensure we start from a clean slate)!
  call open_file(fname, 'write', 'unknown', 'unformatted', iu)

  ! version
  write(iu, iostat=ierr) 1
  call iostat_update(ierr)
  write(iu, iostat=ierr) na_u, no_u, no_u * nsc1 * nsc2 * nsc3, nspin, nnz
  call iostat_update(ierr)

  write(iu, iostat=ierr) nsc1, nsc2, nsc3
  call iostat_update(ierr)
  write(iu, iostat=ierr) cell, xa
  call iostat_update(ierr)
  ! TSGamma, Gamma, onlyS
  write(iu, iostat=ierr) .false., .false., .false.
  call iostat_update(ierr)
  ! kgrid, kdispl
  write(iu, iostat=ierr) (/2, 0, 0, 0, 2, 0, 0, 0, 2/), (/0._dp, 0._dp, 0._dp/)
  call iostat_update(ierr)
  ! Ef, qtot, Temp
  write(iu, iostat=ierr) 0._dp, 1._dp, 0.001_dp
  call iostat_update(ierr)

  ! istep, ia1
  write(iu, iostat=ierr) 0, 0
  call iostat_update(ierr)

  write(iu, iostat=ierr) lasto
  call iostat_update(ierr)

  ! Sparse pattern
  write(iu, iostat=ierr) ncol
  call iostat_update(ierr)
  idx = 0
  do i = 1 , no_u
    write(iu, iostat=ierr) list_col(idx+1:idx+ncol(i))
    call iostat_update(ierr)
    idx = idx + ncol(i)
  end do
  ! Overlap matrix
  idx = 0
  do i = 1 , no_u
    write(iu, iostat=ierr) S(idx+1:idx+ncol(i))
    call iostat_update(ierr)
    idx = idx + ncol(i)
  end do
  ! Hamiltonian matrix
  do is = 1, nspin
    idx = 0
    do i = 1 , no_u
      write(iu, iostat=ierr) H(idx+1:idx+ncol(i),is)
      call iostat_update(ierr)
      idx = idx + ncol(i)
    end do
  end do

  write(iu, iostat=ierr) isc
  call iostat_update(ierr)

  call close_file(iu)

end subroutine write_tshs_hs

