subroutine read_tshs_header(fname,Gamma,nspin,no_u,no_s,maxnh)

  ! Input parameters
  character(len=*) :: fname
  logical :: Gamma
  integer :: no_u, no_s, nspin, maxnh
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, no_u, no_s, nspin, maxnh

! Internal variables and arrays
  integer :: iu, tmp

  iu = 1804
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) tmp, no_u, no_s, nspin, maxnh

  ! wrap reads
  read(iu) !xa
  read(iu) !iza
  read(iu) !ucell

  read(iu) Gamma

  close(iu)

end subroutine read_tshs_header
