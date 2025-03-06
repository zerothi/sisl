! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine read_grid_sizes(fname, nspin, mesh)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: nspin
  integer, intent(out) :: mesh(3)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: nspin
!f2py intent(out) :: mesh

! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) ! cell(:,:)
  call iostat_update(ierr)

  read(iu, iostat=ierr) mesh, nspin
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_grid_sizes

subroutine read_grid_cell(fname, cell)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in):: fname
  real(dp), intent(out) :: cell(3,3)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: cell

! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) cell(:,:)
  call iostat_update(ierr)
  cell(:,:) = cell(:,:)

  call close_file(iu)

end subroutine read_grid_cell

subroutine read_grid(fname, nspin, mesh1, mesh2, mesh3, grid)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, mesh1, mesh2, mesh3
  real(sp), intent(out) :: grid(mesh1,mesh2,mesh3,nspin)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin, mesh1, mesh2, mesh3
!f2py intent(out) :: cell, grid

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, iz, iy

  ! Local readables
  integer :: lnspin, lmesh(3)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) ! cell
  call iostat_update(ierr)

  read(iu, iostat=ierr) lmesh, lnspin
  call iostat_update(ierr)
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmesh(1) /= mesh1 ) stop 'Error in reading data, not allocated, mesh'
  if ( lmesh(2) /= mesh2 ) stop 'Error in reading data, not allocated, mesh'
  if ( lmesh(3) /= mesh3 ) stop 'Error in reading data, not allocated, mesh'

  do is = 1, nspin

    do iz = 1, mesh3
      do iy = 1, mesh2
        read(iu, iostat=ierr) grid(:,iy,iz,is)
        call iostat_update(ierr)
      end do
    end do

  end do

  call close_file(iu)

end subroutine read_grid
