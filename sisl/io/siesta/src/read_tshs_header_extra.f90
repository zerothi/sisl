subroutine read_tshs_header_extra(fname,na_u,ucell,Ef,Qtot,Temp)

  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*) :: fname
  integer :: na_u
  real(dp) :: ucell(3,3), Ef, Qtot, Temp
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: na_u
!f2py real*8, intent(out), dimension(3,3) :: ucell
!f2py intent(out) :: Ef, Qtot, Temp

! Internal variables and arrays
  integer :: iu, tmp(4)
  logical :: Gamma
  
  iu = 1804
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) na_u, tmp(1), tmp(2), tmp(3), tmp(4)

  ! wrap reads
  read(iu) !xa
  read(iu) !iza
  read(iu) ucell
  ucell = ucell * Ang

  read(iu) Gamma
  read(iu) !onlyS
  read(iu) !TSGamma
  read(iu) !kscell
  read(iu) !kdispl
  read(iu) !istep,ia1
  read(iu) !lasto

  if ( .not. Gamma ) then
     read(iu) ! indxuo
  end if

  read(iu) ! numh

  read(iu) Qtot, Temp 
  read(iu) Ef
  Ef = Ef * eV

  close(iu)

end subroutine read_tshs_header_extra
