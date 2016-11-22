subroutine read_tshs_extra(fname,na_u, &
     lasto,xa)

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*) :: fname
  integer :: na_u
  integer :: lasto(na_u) ! Notice that we do not allow the "first" index
  real(dp) :: xa(3,na_u)
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in) :: na_u
!f2py integer, intent(out), dimension(na_u)  :: lasto
!f2py real*8, intent(out), dimension(3,na_u) :: xa

! Internal variables and arrays
  integer :: iu

  ! Local readables
  integer :: lna_u, lno_s, lno_u, lnspin, lmaxnh

  iu = 1805
  open(iu,file=trim(fname),status='old',form='unformatted')
  read(iu) lna_u, lno_u, lno_s, lnspin, lmaxnh
  if ( lna_u /= na_u ) stop 'Error in reading data, not allocated, na_u'

  ! wrap reads
  read(iu) xa
  xa = xa * Ang
  read(iu) !iza
  read(iu) !ucell

  read(iu) !lGamma

  read(iu) !onlyS
  read(iu) !TSGamma
  read(iu) !kscell
  read(iu) !kdispl
  read(iu) !istep,ia1
  read(iu) is,lasto

  close(iu)

end subroutine read_tshs_extra
