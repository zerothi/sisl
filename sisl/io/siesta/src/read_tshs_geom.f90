subroutine read_tshs_geom(fname, na_u, xa, lasto)

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*) :: fname
  integer :: na_u
  real(dp) :: xa(3,na_u)
  integer :: lasto(0:na_u)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: na_u
!f2py intent(out) :: xa
!f2py intent(out) :: lasto

! Internal variables and arrays
  integer :: iu
  integer :: version, tmp(5)
  real(dp) :: cell(3,3)

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then
     
     xa = 0._dp
     cell = 0._dp
     
     return
     
  end if

  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')
  read(iu) ! version
  ! Now we may read the sizes
  read(iu) tmp

  ! Read the stuff...
  read(iu) ! nsc
  read(iu) cell, xa
  xa = xa * Ang
  read(iu) ! Gamma, TSGamma, onlyS
  read(iu) ! kscell, kdispl
  read(iu) ! Ef, Qtot, Temp
  read(iu) ! istep, ia1
  read(iu) lasto(0:na_u)

  close(iu)

end subroutine read_tshs_geom
