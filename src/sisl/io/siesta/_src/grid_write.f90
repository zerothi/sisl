! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.

subroutine write_grid_header(iu, nspin, mesh1, mesh2, mesh3, cell)
  use io_m, only: iostat_update

  implicit none

  ! Precision
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: nspin, mesh1, mesh2, mesh3
  real(dp), intent(in) :: cell(3,3)

! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: nspin, mesh1, mesh2, mesh3
!f2py intent(in) :: cell

! Internal variables and arrays
  integer :: ierr

  write(iu, iostat=ierr) cell(:,:)
  call iostat_update(ierr)

  write(iu, iostat=ierr) mesh1, mesh2, mesh3, nspin
  call iostat_update(ierr)

end subroutine write_grid_header

subroutine write_grid_sp(iu, grid, mesh1, mesh2, mesh3)
  use io_m, only: iostat_update

  implicit none

  ! Precision
  integer, parameter :: sp = selected_real_kind(p=6)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: mesh1, mesh2, mesh3
  real(sp), intent(in) :: grid(mesh1, mesh2, mesh3)

! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: mesh1, mesh2, mesh3
!f2py intent(in) :: grid

! Internal variables and arrays
  integer :: ierr
  integer :: iz, iy

  do iz = 1, mesh3
    do iy = 1, mesh2
      write(iu, iostat=ierr) grid(:,iy,iz)
      call iostat_update(ierr)
    end do
  end do

end subroutine write_grid_sp

subroutine write_grid_dp(iu, grid, mesh1, mesh2, mesh3)
  use io_m, only: iostat_update

  implicit none

  ! Precision
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: mesh1, mesh2, mesh3
  real(dp), intent(in) :: grid(mesh1, mesh2, mesh3)

! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: mesh1, mesh2, mesh3
!f2py intent(in) :: grid

! Internal variables and arrays
  integer :: ierr
  integer :: iz, iy

  do iz = 1, mesh3
    do iy = 1, mesh2
      write(iu, iostat=ierr) real(grid(:,iy,iz),sp)
      call iostat_update(ierr)
    end do
  end do

end subroutine write_grid_dp
