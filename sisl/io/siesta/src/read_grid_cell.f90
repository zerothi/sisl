subroutine read_grid_cell(fname, cell)

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*) :: fname
  real(dp) :: cell(3,3)
! Define f2py intents
!f2py intent(in) :: fname
!f2py real*8, intent(out), dimension(3,3) :: cell

! Internal variables and arrays
  integer :: iu

  iu = 1804
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) cell(:,1)
  read(iu) cell(:,2)
  read(iu) cell(:,3)
  cell = cell * Ang

  close(iu)

end subroutine read_grid_cell
