subroutine read_grid(fname, nspin, mesh1, mesh2, mesh3, cell, grid)

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*) :: fname
  integer :: nspin, mesh1, mesh2, mesh3
  real(dp) :: cell(3,3)
  real(sp) :: grid(mesh1,mesh2,mesh3)
! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin, mesh(3)
!f2py real*8, intent(out), dimension(3,3) :: cell
!f2py real*4, intent(out), dimension(mesh1,mesh2,mesh3) :: grid

! Internal variables and arrays
  integer :: iu
  integer :: is, iz, iy

  ! Local readables
  integer :: lnspin, lmesh(3)

  iu = 1804
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) cell(:,1)
  read(iu) cell(:,2)
  read(iu) cell(:,3)

  read(iu) lmesh, lnspin
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmesh(1) /= mesh1 ) stop 'Error in reading data, not allocated, mesh'
  if ( lmesh(2) /= mesh2 ) stop 'Error in reading data, not allocated, mesh'
  if ( lmesh(3) /= mesh3 ) stop 'Error in reading data, not allocated, mesh'

  do is = 1, nspin

     do iz = 1, mesh(3)
        do iy = 1, mesh(2)
           read(iu) grid(:,iy,iz)
        end do
     end do
     
  end do

  close(iu)

end subroutine read_grid
