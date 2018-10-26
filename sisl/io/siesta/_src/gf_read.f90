subroutine read_open_gf( fname, iu )

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: iu

  ! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: iu
  
  ! Open file
  call free_unit(iu)
  open( iu, file=trim(fname), form='unformatted', status='old', action='read' )
  
end subroutine read_open_gf

subroutine read_gf_sizes( iu, nspin, no_u, nkpt, NE)

  implicit none

  ! Precision 
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(out) :: nspin, no_u, nkpt, NE

  ! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(out) :: nspin
!f2py intent(out) :: no_u
!f2py intent(out) :: nkpt
!f2py intent(out) :: NE

  ! Local variables
  integer :: na_used

  read(iu) nspin !cell
  read(iu) !na_u, no_u
  read(iu) na_used, no_u
  read(iu) !xa_used, lasto_used
  read(iu) !.false., Bloch, pre_expand
  read(iu) !mu

  ! k-points
  read(iu) nkpt
  read(iu) !
  
  read(iu) NE

end subroutine read_gf_sizes

subroutine read_gf_header( iu, nkpt, kpt, NE, E )
  
  implicit none

  ! Precision 
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: nkpt, NE
  real(dp), intent(out) :: kpt(3,nkpt)
  complex(dp), intent(out) :: E(NE)

  ! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: nkpt
!f2py intent(in) :: NE
!f2py intent(out) :: kpt
!f2py intent(out) :: E

  read(iu) !nspin, cell
  read(iu) !na_u, no_u
  read(iu) !na_used, no_used
  read(iu) !xa_used, lasto_used
  read(iu) !.false., Bloch, pre_expand
  read(iu) !mu

  ! k-points
  read(iu) !nkpt
  read(iu) kpt
  
  read(iu) !NE
  read(iu) E

end subroutine read_gf_header

subroutine read_gf_hs( iu, no_u, H, S)
  
  implicit none

  ! Precision 
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: no_u

  ! Variables for the size
  complex(dp), intent(out) :: H(no_u,no_u), S(no_u,no_u)

  ! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: no_u
!f2py intent(out) :: H
!f2py intent(out) :: S

  read(iu) !ik, iE, E
  read(iu) H
  read(iu) S

end subroutine read_gf_hs

subroutine read_gf_se( iu, no_u, iE, SE )
  
  implicit none

  ! Precision 
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: no_u
  integer, intent(in) :: iE
  complex(dp), intent(out) :: SE(no_u,no_u)

  ! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: no_u
!f2py intent(in) :: iE
!f2py intent(out) :: SE

  if ( iE > 1 ) then
    read(iu) !ik, iE, E
  end if
  read(iu) SE

end subroutine read_gf_se
