subroutine read_hsx_header(fname,Gamma,nspin,no_u,no_s,maxnh)
  
  implicit none

  ! Input parameters
  character(len=*) :: fname
  logical :: Gamma
  integer :: no_u, no_s, nspin, maxnh
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, no_u, no_s, nspin, maxnh

! Internal variables and arrays
  integer :: iu

! Open file
  iu = 1850
  open( iu, file=fname, form='unformatted', status='unknown' )      
  
! Read overall data
  read(iu) no_u, no_s, nspin, maxnh

! Read logical
  read(iu) Gamma

  close(iu)
  
end subroutine read_hsx_header
