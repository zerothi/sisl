subroutine write_open_gf( fname, iu )

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: iu

  ! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: iu, test
  
  ! Open file
  call free_unit(iu)
  open( iu, file=trim(fname), form='unformatted', status='unknown', action='write' )
  
end subroutine write_open_gf

subroutine write_gf_header( iu, nspin, cell, na_u, no_u, na_used, no_used, &
    xa_used, lasto_used, Bloch, pre_expand, mu, nkpt, kpt, kw, NE, E )
  
  implicit none

  ! Precision 
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  ! Variables for the size
  integer, intent(in) :: nspin
  real(dp), intent(in) :: cell(3,3)
  integer, intent(in) :: na_u, no_u, na_used, no_used
  real(dp), intent(in) :: xa_used(3,na_used)
  integer, intent(in) :: lasto_used(0:na_used)
  integer, intent(in) :: Bloch(3)
  integer, intent(in) :: pre_expand
  real(dp), intent(in) :: mu
  integer, intent(in) :: nkpt
  real(dp), intent(in) :: kpt(3,nkpt), kw(nkpt)
  integer, intent(in) :: NE
  complex(dp), intent(in) :: E(NE)

! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: nspin, cell, na_u, no_u, na_used, no_used
!f2py intent(in) :: xa_used, lasto_used, Bloch, pre_expand
!f2py intent(in) :: mu
!f2py intent(in) :: nkpt, kpt, w, NE, E

  write(iu) nspin, cell
  write(iu) na_u, no_u
  write(iu) na_used, no_used
  write(iu) xa_used, lasto_used
  write(iu) .false., Bloch, pre_expand
  write(iu) mu

  ! k-points
  write(iu) nkpt
  write(iu) kpt, kw
  
  write(iu) NE
  write(iu) E

end subroutine write_gf_header

subroutine write_gf_hs( iu, ik, iE, E, no_u, H, S)
  
  implicit none

  ! Precision 
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: ik, iE
  complex(dp), intent(in) :: E
  ! Variables for the size
  integer, intent(in) :: no_u
  complex(dp), intent(in) :: H(no_u,no_u), S(no_u,no_u)

! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: ik, iE, E
!f2py intent(in) :: no_u
!f2py intent(in) :: H, S
  
  write(iu) ik, iE, E
  write(iu) H
  write(iu) S

end subroutine write_gf_hs

subroutine write_gf_se( iu, ik, iE, E, no_u, SE )
  
  implicit none

  ! Precision 
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: ik, iE
  complex(dp), intent(in) :: E
  ! Variables for the size
  integer, intent(in) :: no_u
  complex(dp), intent(in) :: SE(no_u,no_u)

! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: ik, iE, E
!f2py intent(in) :: no_u
!f2py intent(in) :: SE

  if ( iE > 1 ) then
    write(iu) ik, iE, E
  end if
  write(iu) SE

end subroutine write_gf_se

subroutine close_gf( iu )

  implicit none

  ! Input parameters
  integer, intent(in) :: iu

  ! Define f2py intents
!f2py intent(in) :: iu
  
  ! Open file
  close( iu )
  
end subroutine close_gf
