! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.

! All routines read_wfsx_next_* and skip_wfsx_* expect that the file unit is handled externally.
! The file unit should already be open and in the correct position.
! On the other hand, routines with another name manage the opening and closing
! themselves.

! --------------------------------------------------------------
!             Routines to read header information
! --------------------------------------------------------------
! These routines read the information written at the beggining
! of the WFSX file.

subroutine read_wfsx_next_sizes(iu, skip_basis, nspin, no_u, nk, Gamma)
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  integer, intent(in) :: iu
  logical, intent(in) :: skip_basis
  integer, intent(out) :: nspin, no_u, nk
  logical, intent(out) :: Gamma

  ! Internal variables and arrays
  integer :: ierr

  read(iu, iostat=ierr) nk, Gamma
  call iostat_update(ierr)

  read(iu, iostat=ierr) nspin
  call iostat_update(ierr)

  read(iu, iostat=ierr) no_u
  call iostat_update(ierr)

  if (skip_basis) then
    read(iu, iostat=ierr) ! This contains basis information
    call iostat_update(ierr)
  endif 

end subroutine read_wfsx_next_sizes

subroutine read_wfsx_sizes(fname, nspin, no_u, nk, Gamma)
  use io_m, only: open_file, close_file

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: nspin, no_u, nk
  logical, intent(out) :: Gamma

  ! Internal variables and arrays
  integer :: iu

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  call read_wfsx_next_sizes(iu, .false., nspin, no_u, nk, Gamma)

  call close_file(iu)

end subroutine read_wfsx_sizes

subroutine read_wfsx_next_basis(iu, no_u, atom_indices, atom_labels, &
   orb_index_atom, orb_n, orb_simmetry)  
    use io_m, only: iostat_update

    implicit none

    ! Input parameters
    integer, intent(in) :: iu, no_u
    integer, intent(out) :: atom_indices(no_u), orb_index_atom(no_u)
    character, intent(out) :: atom_labels(no_u, 20), orb_simmetry(no_u, 20)
    integer, intent(out) :: orb_n(no_u)

  ! Internal variables and arrays
    integer :: j, ierr

    read(iu, iostat=ierr) (atom_indices(j), atom_labels(j, :), orb_index_atom(j), &
                 orb_n(j), orb_simmetry(j, :), j=1,no_u)

    call iostat_update(ierr)

end subroutine read_wfsx_next_basis

! --------------------------------------------------------------
!             Routines to read k point information
! --------------------------------------------------------------
! These routines read the information for a given k point in the
! WFSX file.

subroutine read_wfsx_next_info(iu, ispin, ik, k, kw, nwf)
    use io_m, only: iostat_update

    implicit none

    integer, parameter :: dp = selected_real_kind(p=15)

    ! Input parameters
    integer, intent(in) :: iu
    integer, intent(out) :: ispin, ik
    real(dp), intent(out) :: k(3), kw
    integer, intent(out) :: nwf

  ! Internal variables and arrays
    integer :: ierr

    ! read information here
    read(iu, iostat=ierr) ik, k, kw
    call iostat_update(ierr)
    read(iu, iostat=ierr) ispin
    call iostat_update(ierr)
    read(iu, iostat=ierr) nwf
    call iostat_update(ierr)

end subroutine read_wfsx_next_info

subroutine read_wfsx_index_info(fname, ispin, ik, k, kw, nwf)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispin, ik
  real(dp), intent(out) :: k(3), kw
  integer, intent(out) :: nwf

  integer :: file_ispin, file_ik
  ! Internal variables and arrays
  integer :: iu

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  call skip_wfsx_index(iu, ispin, ik)

  call read_wfsx_next_info(iu, file_ispin, file_ik, k, kw, nwf)

  call close_file(iu)

end subroutine read_wfsx_index_info

subroutine read_wfsx_next_index_check(iu, ispin, ik, nwf, fail)
  use io_m, only: iostat_update, iostat_query

  implicit none

  integer, intent(in) :: iu, ispin, ik, nwf
  integer, intent(inout) :: fail

  integer :: ierr
  integer :: file_ispin, file_ik, file_nwf

  ! read information here
  read(iu, iostat=ierr) file_ik ! k, kw
  call iostat_update(ierr)
  read(iu, iostat=ierr) file_ispin
  call iostat_update(ierr)
  read(iu, iostat=ierr) file_nwf
  call iostat_update(ierr)

  call iostat_query(fail)
  if ( file_ik /= ik ) then
    fail = -1
  end if
  if ( file_ispin /= ispin ) then
    fail = -2
  end if
  if ( file_nwf /= nwf ) then
    fail = -3
  end if

end subroutine read_wfsx_next_index_check

! --------------------------------------------------------------
!               Routines that skip over things
! --------------------------------------------------------------
! Useful if we want to ignore contents of the file. This routines
! ALWAYS depend on the file unit being handled externally.

subroutine skip_wfsx_next_eigenstate(iu)
  use io_m, only: iostat_update

  integer, intent(in) :: iu

  integer :: nwf

  read(iu, iostat=ierr) ! ik, k, kw
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! ispin
  call iostat_update(ierr)
  read(iu, iostat=ierr) nwf
  call iostat_update(ierr)

  call skip_wfsx_next_vals(iu, nwf)

end subroutine skip_wfsx_next_eigenstate

subroutine skip_wfsx_next_vals(iu, nwf)
  use io_m, only: iostat_update

  integer, intent(in) :: iu, nwf

  integer :: iwf

  do iwf = 1, nwf
    read(iu, iostat=ierr) ! indwf
    call iostat_update(ierr)
    read(iu, iostat=ierr) ! eig [eV]
    call iostat_update(ierr)
    read(iu, iostat=ierr) ! state
    call iostat_update(ierr)
  end do

end subroutine skip_wfsx_next_vals

subroutine skip_wfsx_index(iu, ispin, ik)
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: ispin, ik

  ! Internal variables
  integer :: no_u
  integer :: nk, nspin, i, j
  logical :: Gamma

  call read_wfsx_next_sizes(iu, .true., nspin, no_u, nk, Gamma)

  ! Check that the requested indices make sense
  if ( ik > nk ) call iostat_update(-1)
  if ( ispin > nspin ) call iostat_update(-2)

  ! Skip until the k index that we need
  do i = 1 , ik - 1
    do j = 1 , nspin
      call skip_wfsx_next_eigenstate(iu)
    end do
  end do

  ! Skip until the desired spin index
  do i = 1 , ispin - 1
    call skip_wfsx_next_eigenstate(iu)
  end do

end subroutine skip_wfsx_index

! --------------------------------------------------------------
!             Routines that read the next values
! --------------------------------------------------------------
! There are three different cases:
! - 1: When only the gamma point is in the WFSX file. The state
!   is in a real array.
! - 2: All other cases where spin is not non-colinear. The state 
!   is in a complex array.
! - 4: For non-colinear spins. The state's first dimension is two
!   times larger than the number of actual orbitals in the system.

subroutine read_wfsx_next_1(iu, no_u, nwf, idx, eig, state)
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: no_u, nwf
  integer, intent(out) :: idx(nwf)
  real(dp), intent(out) :: eig(nwf)
  real(sp), intent(out) :: state(no_u, nwf)

  integer :: iwf
  ! Internal variables and arrays
  integer :: ierr

  do iwf = 1, nwf
    read(iu, iostat=ierr) idx(iwf)
    call iostat_update(ierr)
    read(iu, iostat=ierr) eig(iwf)
    call iostat_update(ierr)
    read(iu, iostat=ierr) state(:,iwf)
    call iostat_update(ierr)
  end do

end subroutine read_wfsx_next_1

subroutine read_wfsx_next_2(iu, no_u, nwf, idx, eig, state)
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: no_u, nwf
  integer, intent(out) :: idx(nwf)
  real(dp), intent(out) :: eig(nwf)
  complex(sp), intent(out) :: state(no_u, nwf)

  integer :: iwf
  ! Internal variables and arrays
  integer :: ierr

  do iwf = 1, nwf
    read(iu, iostat=ierr) idx(iwf)
    call iostat_update(ierr)
    read(iu, iostat=ierr) eig(iwf)
    call iostat_update(ierr)
    read(iu, iostat=ierr) state(:,iwf)
    call iostat_update(ierr)
  end do

end subroutine read_wfsx_next_2

subroutine read_wfsx_next_4(iu, no_u, nwf, idx, eig, state)
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: no_u, nwf
  integer, intent(out) :: idx(nwf)
  real(dp), intent(out) :: eig(nwf)
  complex(sp), intent(out) :: state(2*no_u, nwf)

  integer :: iwf
  ! Internal variables and arrays
  integer :: ierr

  do iwf = 1, nwf
    read(iu, iostat=ierr) idx(iwf)
    call iostat_update(ierr)
    read(iu, iostat=ierr) eig(iwf)
    call iostat_update(ierr)
    read(iu, iostat=ierr) state(:,iwf)
    call iostat_update(ierr)
  end do

end subroutine read_wfsx_next_4

subroutine read_wfsx_index_1(fname, ispin, ik, no_u, nwf, idx, eig, state)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispin, ik, no_u, nwf
  integer, intent(out) :: idx(nwf)
  real(dp), intent(out) :: eig(nwf)
  real(sp), intent(out) :: state(no_u, nwf)

  ! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  call skip_wfsx_index(iu, ispin, ik)

  call read_wfsx_next_index_check(iu, ispin, ik, nwf, ierr)
  if ( ierr /= 0 ) then
    call iostat_update(ierr)
    call close_file(iu)
    return
  end if

  call read_wfsx_next_1(iu, no_u, nwf, idx, eig, state)

  call close_file(iu)

end subroutine read_wfsx_index_1

subroutine read_wfsx_index_2(fname, ispin, ik, no_u, nwf, idx, eig, state)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispin, ik, no_u, nwf
  integer, intent(out) :: idx(nwf)
  real(dp), intent(out) :: eig(nwf)
  complex(sp), intent(out) :: state(no_u, nwf)

  ! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  call skip_wfsx_index(iu, ispin, ik)

  call read_wfsx_next_index_check(iu, ispin, ik, nwf, ierr)
  if ( ierr /= 0 ) then
    call iostat_update(ierr)
    call close_file(iu)
    return
  end if

  call read_wfsx_next_2(iu, no_u, nwf, idx, eig, state)

  call close_file(iu)

end subroutine read_wfsx_index_2

subroutine read_wfsx_index_4(fname, ispin, ik, no_u, nwf, idx, eig, state)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispin, ik, no_u, nwf
  integer, intent(out) :: idx(nwf)
  real(dp), intent(out) :: eig(nwf)
  complex(sp), intent(out) :: state(2*no_u, nwf)

  ! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  call skip_wfsx_index(iu, ispin, ik)

  call read_wfsx_next_index_check(iu, ispin, ik, nwf, ierr)
  if ( ierr /= 0 ) then
    call iostat_update(ierr)
    call close_file(iu)
    return
  end if

  call read_wfsx_next_4(iu, no_u, nwf, idx, eig, state)

  call close_file(iu)

end subroutine read_wfsx_index_4

! --------------------------------------------------------------
!   Routines that read the information of all k points
! --------------------------------------------------------------
! They skip the actual values of the eigenstates and might be
! useful to check the contents of the file.

subroutine read_wfsx_next_all_info(iu, nspin, nk, ks, kw, nwf)
  ! This function should be called after reading sizes.
  ! It reads the k value, weight and number of wavefunctions
  ! of all the k points in the file.
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: nspin, nk
  real(dp), intent(out) :: ks(nk, 3), kw(nk)
  integer, intent(out) :: nwf(nspin, nk)

  ! Internal variables and arrays
  integer :: ik, ispin, file_ik, file_ispin
  integer :: l_nspin

  l_nspin = size(nwf, 1)

  do ik = 1 , nk
    do ispin = 1, l_nspin
      ! Notice that if there is more than one spin, we write more than once
      ! to the same position. It doesn't matter, since the k value and the k weight
      ! is the same for all spin indices.
      call read_wfsx_next_info(iu, file_ispin, file_ik, ks(ik, :), kw(ik), nwf(ispin, ik))
      call skip_wfsx_next_vals(iu, nwf(ispin, ik))
    end do
  end do

end subroutine read_wfsx_next_all_info
