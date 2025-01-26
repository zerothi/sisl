! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine read_tshs_version(fname, version)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: version

  ! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: version

  integer :: iu, ierr
  integer :: tmp(5)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Set a default version in case it won't be found
  version = -1

  read(iu, iostat=ierr) tmp
  if ( ierr == 0 ) then
    ! Valid original version
    version = 0

  else ! ierr /= 0

    ! we have a version
    rewind(iu)
    read(iu, iostat=ierr) version

    if ( version /= 1 ) then
      ! Signal we do not know about this file
      call iostat_update(-3)
    end if

  end if
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_tshs_version

subroutine read_tshs_sizes(fname, nspin, na_u, no_u, n_s, nnz)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: nspin, na_u, no_u, n_s, nnz

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: nspin, na_u, no_u, n_s, nnz

! Internal variables and arrays
  integer :: iu, ierr
  integer :: version, tmp(5)

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then

    nspin = 0
    na_u = 0
    no_u = 0
    n_s = 0
    nnz = 0

    call iostat_update(-1)
    return

  end if

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) ! version
  call iostat_update(ierr)
  ! Read the sizes
  !na_u, no_u, no_s, nspin, n_nzsg
  read(iu, iostat=ierr) tmp
  call iostat_update(ierr)

  ! Copy the readed variables
  nspin = tmp(4)
  na_u = tmp(1)
  no_u = tmp(2)
  n_s = tmp(3) / tmp(2)
  nnz = tmp(5)

  call close_file(iu)

end subroutine read_tshs_sizes

subroutine read_tshs_ef(fname, Ef)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  real(dp), intent(out) :: Ef

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Ef

! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) ! version
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! na_u, no_u, no_s, nspin, n_nzsg
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! nsc
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! cell, xa
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! Gamma, TSGamma, onlyS
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! kscell, kdispl
  call iostat_update(ierr)
  read(iu, iostat=ierr) Ef ! Qtot, Temp
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_tshs_ef

subroutine read_tshs_k(fname, kcell, kdispl)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  character(len=*), intent(in) :: fname
  integer, intent(out) :: kcell(3,3)
  real(dp), intent(out) :: kdispl(3)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: kcell, kdispl

  integer :: iu, ierr, version

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) ! version
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! na_u, no_u, no_s, nspin, n_nzsg
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! nsc
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! cell, xa
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! Gamma, TSGamma, onlyS
  call iostat_update(ierr)
  read(iu, iostat=ierr) kcell, kdispl
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_tshs_k

subroutine read_tshs_cell(fname, n_s, nsc, cell, isc)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: n_s
  integer, intent(out) :: nsc(3)
  real(dp), intent(out) :: cell(3,3)
  integer, intent(out) :: isc(3,n_s)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: n_s
!f2py intent(out) :: nsc, cell, isc

! Internal variables and arrays
  integer :: iu, ierr, i, is
  integer :: version, tmp(5)
  logical :: Gamma, TSGamma, onlyS

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then

    nsc = 0
    cell = 0._dp
    isc = 0

    call iostat_update(-1)
    return

  end if

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) ! version
  call iostat_update(ierr)
  ! Now we may read the sizes
  read(iu, iostat=ierr) tmp
  call iostat_update(ierr)

  ! Read the stuff...
  read(iu, iostat=ierr) nsc
  call iostat_update(ierr)
  read(iu, iostat=ierr) cell ! xa
  call iostat_update(ierr)
  read(iu, iostat=ierr) Gamma, TSGamma, onlyS
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! kscell, kdispl
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! Ef, Qtot, Temp
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! istep, ia1
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! lasto
  call iostat_update(ierr)

  ! Sparse pattern
  read(iu, iostat=ierr) ! ncol
  call iostat_update(ierr)
  do i = 1 , tmp(2)
    read(iu, iostat=ierr) ! list_col
    call iostat_update(ierr)
  end do
  ! Overlap matrix
  do i = 1 , tmp(2)
    read(iu, iostat=ierr) ! S
    call iostat_update(ierr)
  end do
  if ( .not. onlyS ) then
    ! Hamiltonian matrix
    do is = 1, tmp(4)
      do i = 1 , tmp(2)
        read(iu, iostat=ierr) ! H
        call iostat_update(ierr)
      end do
    end do
  end if
  if ( .not. Gamma ) then
    read(iu, iostat=ierr) isc
    call iostat_update(ierr)
  end if

  call close_file(iu)

end subroutine read_tshs_cell

subroutine read_tshs_geom(fname, na_u, xa, lasto)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: na_u
  real(dp), intent(out) :: xa(3,na_u)
  integer, intent(out) :: lasto(0:na_u)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: na_u
!f2py intent(out) :: xa, lasto

! Internal variables and arrays
  integer :: iu, ierr
  integer :: version, tmp(5)
  real(dp) :: cell(3,3)

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then

    xa = 0._dp
    cell = 0._dp

    call iostat_update(-1)
    return

  end if

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) ! version
  call iostat_update(ierr)
  ! Now we may read the sizes
  read(iu, iostat=ierr) tmp
  call iostat_update(ierr)

  ! Read the stuff...
  read(iu, iostat=ierr) ! nsc
  call iostat_update(ierr)
  read(iu, iostat=ierr) cell, xa
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! Gamma, TSGamma, onlyS
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! kscell, kdispl
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! Ef, Qtot, Temp
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! istep, ia1
  call iostat_update(ierr)
  read(iu, iostat=ierr) lasto(0:na_u)
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_tshs_geom

subroutine read_tshs_hs(fname, nspin, no_u, nnz, ncol, list_col, H, S)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, no_u, nnz
  integer, intent(out) :: ncol(no_u), list_col(nnz)
  real(dp), intent(out) :: H(nnz, nspin), S(nnz)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: nspin, no_u, nnz
!f2py intent(out) :: ncol, list_col
!f2py intent(out) :: H, S

! Internal variables and arrays
  integer :: iu, ierr, i, is, idx
  integer :: version, tmp(5)
  real(dp) :: Ef
  logical :: Gamma, TSGamma, onlyS

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then

    ncol = -1
    list_col = -1
    H = 0._dp
    S = 0._dp

    call iostat_update(-1)
    return

  end if

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) ! version
  call iostat_update(ierr)
  ! Now we may read the sizes
  read(iu, iostat=ierr) tmp
  call iostat_update(ierr)

  ! Read the stuff...
  read(iu, iostat=ierr) ! nsc
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! cell, xa
  call iostat_update(ierr)
  read(iu, iostat=ierr) Gamma, TSGamma, onlyS
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! kscell, kdispl
  call iostat_update(ierr)
  read(iu, iostat=ierr) Ef ! Qtot, Temp
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! istep, ia1
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! lasto
  call iostat_update(ierr)

  ! Sparse pattern
  read(iu, iostat=ierr) ncol
  call iostat_update(ierr)
  idx = 0
  do i = 1 , tmp(2)
    read(iu, iostat=ierr) list_col(idx+1:idx+ncol(i))
    call iostat_update(ierr)
    idx = idx + ncol(i)
  end do
  ! Overlap matrix
  idx = 0
  do i = 1 , tmp(2)
    read(iu, iostat=ierr) S(idx+1:idx+ncol(i))
    call iostat_update(ierr)
    idx = idx + ncol(i)
  end do
  ! Hamiltonian matrix
  if ( onlyS ) then
    H(:,:) = 0._dp
  else
    do is = 1, tmp(4)
      idx = 0
      do i = 1 , tmp(2)
        read(iu, iostat=ierr) H(idx+1:idx+ncol(i),is)
        call iostat_update(ierr)
        idx = idx + ncol(i)
      end do
      ! Move to Ef = 0
      if ( is <= 2 ) then
        H(:,is) = H(:,is) - Ef * S(:)
      end if
    end do
  end if

  call close_file(iu)

end subroutine read_tshs_hs

subroutine read_tshs_s(fname, no_u, nnz, ncol, list_col, S)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: no_u, nnz
  integer, intent(out) :: ncol(no_u), list_col(nnz)
  real(dp), intent(out) :: S(nnz)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: no_u, nnz
!f2py intent(out) :: ncol, list_col
!f2py intent(out) :: S

! Internal variables and arrays
  integer :: iu, ierr, i, idx
  integer :: version, tmp(5)

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then

    ncol = -1
    list_col = -1
    S = 0._dp

    call iostat_update(-1)
    return

  end if

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) ! version
  call iostat_update(ierr)
  ! Now we may read the sizes
  read(iu, iostat=ierr) tmp
  call iostat_update(ierr)

  ! Read the stuff...
  read(iu, iostat=ierr) ! nsc
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! cell, xa
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! Gamma, TSGamma, onlyS
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! kscell, kdispl
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! Ef, Qtot, Temp
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! istep, ia1
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! lasto
  call iostat_update(ierr)

  ! Sparse pattern
  read(iu, iostat=ierr) ncol
  call iostat_update(ierr)
  idx = 0
  do i = 1 , tmp(2)
    read(iu, iostat=ierr) list_col(idx+1:idx+ncol(i))
    call iostat_update(ierr)
    idx = idx + ncol(i)
  end do
  ! Overlap matrix
  idx = 0
  do i = 1 , tmp(2)
    read(iu, iostat=ierr) S(idx+1:idx+ncol(i))
    call iostat_update(ierr)
    idx = idx + ncol(i)
  end do

  call close_file(iu)

end subroutine read_tshs_s
