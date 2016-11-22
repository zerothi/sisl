subroutine read_hs_header(fname,Gamma,nspin,no_u,no_s,maxnh)

  ! Input parameters
  character(len=*) :: fname
  logical :: Gamma
  integer :: no_u, no_s, nspin, maxnh
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, no_u, no_s, nspin, maxnh

! Internal variables and arrays
  integer :: iu

  iu = 1804
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) no_u, no_s, nspin, maxnh

  read(iu) Gamma

  close(iu)

end subroutine read_hs_header
