subroutine write_grid(fname, nspin, mesh1, mesh2, mesh3, cell, grid)
  use io_m, only: open_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, mesh1, mesh2, mesh3
  real(dp), intent(in) :: cell(3,3)
  real(sp), intent(in) :: grid(mesh1, mesh2, mesh3)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin, mesh1, mesh2, mesh3
!f2py intent(in) :: cell, grid

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, iz, iy

  ! Open file (ensure we start from a clean slate)!
  call open_file(fname, 'write', 'unknown', 'unformatted', iu)

  write(iu, iostat=ierr) cell(:,:) / Ang
  call iostat_update(ierr)

  write(iu, iostat=ierr) mesh1, mesh2, mesh3, nspin
  call iostat_update(ierr)

  do is = 1, nspin

    do iz = 1, mesh3
      do iy = 1, mesh2
        write(iu, iostat=ierr) grid(:,iy,iz)
        call iostat_update(ierr)
      end do
    end do

  end do

  close(iu)

end subroutine write_grid


