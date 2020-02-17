subroutine read_wfsx_sizes(fname, nspin, no_u, nk, Gamma)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: nspin, no_u, nk
  logical, intent(out) :: Gamma

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: nspin
!f2py intent(out) :: no_u
!f2py intent(out) :: nk
!f2py intent(out) :: Gamma

! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) nk, Gamma
  call iostat_update(ierr)

  read(iu, iostat=ierr) nspin
  call iostat_update(ierr)

  read(iu, iostat=ierr) no_u
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_wfsx_sizes

subroutine read_wfsx_index_info(fname, ispin, ik, k, kw, nw)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispin, ik
  real(dp), intent(out) :: k(3), kw
  integer, intent(out) :: nw

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: ispin
!f2py intent(in) :: ik
!f2py intent(out) :: k
!f2py intent(out) :: kw
!f2py intent(out) :: nw

  integer :: i
! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  call skip_wfsx_index(iu, ispin, ik)

  ! read information here
  read(iu) i, k, kw
  call iostat_update(ierr)
  read(iu) ! ispin
  call iostat_update(ierr)
  read(iu) nw
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_wfsx_index_info

subroutine skip_wfsx_index(iu, ispin, ik)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: ispin, ik

! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: ispin, ik

  integer :: nk, spin, i, j, k, nwf
! Internal variables and arrays
  integer :: ierr

  read(iu, iostat=ierr) nk !, is_gamma
  call iostat_update(ierr)

  if ( ik > nk ) then
    call iostat_update(-1)
    call close_file(iu)
    return
  end if

  read(iu, iostat=ierr) spin
  call iostat_update(ierr)
  if ( ispin > spin ) then
    call iostat_update(-1)
    call close_file(iu)
    return
  end if

  read(iu, iostat=ierr) ! no_u
  call iostat_update(ierr)

  ! Read pass the spin
  do j = 1, ispin - 1
    do i = 1 , nk
      read(iu, iostat=ierr) ! ik, k, kw
      call iostat_update(ierr)
      read(iu, iostat=ierr) ! ispin
      call iostat_update(ierr)
      read(iu, iostat=ierr) nwf
      call iostat_update(ierr)
      do k = 1, nwf
        read(iu, iostat=ierr) ! indwf
        call iostat_update(ierr)
        read(iu, iostat=ierr) ! eig [eV]
        call iostat_update(ierr)
        read(iu, iostat=ierr) ! state
        call iostat_update(ierr)
      end do
    end do
  end do

  ! skip the indices not requested
  do i = 1 , ik - 1
    read(iu, iostat=ierr) ! ik, k, kw
    call iostat_update(ierr)
    read(iu, iostat=ierr) ! ispin
    call iostat_update(ierr)
    read(iu, iostat=ierr) nwf
    call iostat_update(ierr)
    do k = 1, nwf
      read(iu, iostat=ierr) ! indwf
      call iostat_update(ierr)
      read(iu, iostat=ierr) ! eig [eV]
      call iostat_update(ierr)
      read(iu, iostat=ierr) ! state
      call iostat_update(ierr)
    end do
  end do

end subroutine skip_wfsx_index


subroutine read_wfsx_index_1(fname, ispin, ik, no_u, nw, index, eig, state)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispin, ik, no_u, nw
  integer, intent(out) :: index(nw)
  real(dp), intent(out) :: eig(nw)
  real(sp), intent(out) :: state(no_u, nw)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: ispin, ik, no_u, nw
!f2py intent(out) :: index, eig, state

  integer :: k, nwf
! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  call skip_wfsx_index(iu, ispin, ik)

  ! read information here
  read(iu, iostat=ierr) ! ik, k, kw
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! ispin
  call iostat_update(ierr)
  read(iu, iostat=ierr) nwf
  call iostat_update(ierr)

  if ( nw /= nwf ) then
    call iostat_update(-1)
    call close_file(iu)
    return
  end if

  do k = 1, nw
    read(iu, iostat=ierr) index(k)
    call iostat_update(ierr)
    read(iu, iostat=ierr) eig(k)
    call iostat_update(ierr)
    read(iu, iostat=ierr) state(:,k)
    call iostat_update(ierr)
  end do

  call close_file(iu)

end subroutine read_wfsx_index_1

subroutine read_wfsx_index_2(fname, ispin, ik, no_u, nw, index, eig, state)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispin, ik, no_u, nw
  integer, intent(out) :: index(nw)
  real(dp), intent(out) :: eig(nw)
  complex(sp), intent(out) :: state(no_u, nw)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: ispin, ik, no_u, nw
!f2py intent(out) :: index, eig, state

  integer :: k, nwf
! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  call skip_wfsx_index(iu, ispin, ik)

  ! read information here
  read(iu, iostat=ierr) ! ik, k, kw
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! ispin
  call iostat_update(ierr)
  read(iu, iostat=ierr) nwf
  call iostat_update(ierr)

  if ( nw /= nwf ) then
    call iostat_update(-1)
    close(iu)
    return
  end if

  do k = 1, nw
    read(iu, iostat=ierr) index(k)
    call iostat_update(ierr)
    read(iu, iostat=ierr) eig(k)
    call iostat_update(ierr)
    read(iu, iostat=ierr) state(:,k)
    call iostat_update(ierr)
  end do

  call close_file(iu)

end subroutine read_wfsx_index_2

subroutine read_wfsx_index_4(fname, ispin, ik, no_u, nw, index, eig, state)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispin, ik, no_u, nw
  integer, intent(out) :: index(nw)
  real(dp), intent(out) :: eig(nw)
  complex(sp), intent(out) :: state(2*no_u, nw)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: ispin, ik, no_u, nw
!f2py intent(out) :: index, eig, state

  integer :: k, nwf
! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  call skip_wfsx_index(iu, ispin, ik)

  ! read information here
  read(iu, iostat=ierr) ! ik, k, kw
  call iostat_update(ierr)
  read(iu, iostat=ierr) ! ispin
  call iostat_update(ierr)
  read(iu, iostat=ierr) nwf
  call iostat_update(ierr)

  if ( nw /= nwf ) then
    call iostat_update(-1)
    call close_file(iu)
    return
  end if

  do k = 1, nw
    read(iu, iostat=ierr) index(k)
    call iostat_update(ierr)
    read(iu, iostat=ierr) eig(k)
    call iostat_update(ierr)
    read(iu, iostat=ierr) state(:,k)
    call iostat_update(ierr)
  end do

  call close_file(iu)

end subroutine read_wfsx_index_4
