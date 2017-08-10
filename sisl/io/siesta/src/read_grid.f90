subroutine read_grid(fname, nspin, mesh1, mesh2, mesh3, cell, grid)

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, mesh1, mesh2, mesh3
  real(dp), intent(out) :: cell(3,3)
  real(sp), intent(out) :: grid(mesh1,mesh2,mesh3)
  
! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin
!f2py intent(in) :: mesh1
!f2py intent(in) :: mesh2
!f2py intent(in) :: mesh3
!f2py intent(out) :: cell
!f2py intent(out) :: grid

! Internal variables and arrays
  integer :: iu
  integer :: is, iz, iy

  ! Local readables
  integer :: lnspin, lmesh(3)

  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) cell(:,:)
  cell = cell * Ang

  read(iu) lmesh, lnspin
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmesh(1) /= mesh1 ) stop 'Error in reading data, not allocated, mesh'
  if ( lmesh(2) /= mesh2 ) stop 'Error in reading data, not allocated, mesh'
  if ( lmesh(3) /= mesh3 ) stop 'Error in reading data, not allocated, mesh'

  do is = 1, nspin

     do iz = 1, mesh3
        do iy = 1, mesh2
           read(iu) grid(:,iy,iz)
        end do
     end do
     
  end do

  close(iu)

end subroutine read_grid
