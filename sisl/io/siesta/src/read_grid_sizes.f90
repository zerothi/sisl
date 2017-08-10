subroutine read_grid_sizes(fname, nspin, mesh)
  
  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*) :: fname
  integer :: nspin
  integer :: mesh(3)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: nspin
!f2py intent(out) :: mesh

! Internal variables and arrays
  integer :: iu

  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) ! cell(:,:)

  read(iu) mesh, nspin

  close(iu)

end subroutine read_grid_sizes
