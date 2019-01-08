subroutine read_open_gf(fname, iu )
  use io_m, only: open_file

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: iu

  ! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: iu

  ! Open file
  call open_file(fname, 'read', 'old', 'unformatted', iu)

end subroutine read_open_gf

subroutine read_gf_sizes(iu, nspin, no_u, nkpt, NE)
  use io_m, only: iostat_update

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
  integer :: na_used, ierr

  read(iu, iostat=ierr) nspin !cell
  call iostat_update(ierr)
  read(iu, iostat=ierr) !na_u, no_u
  call iostat_update(ierr)
  read(iu, iostat=ierr) na_used, no_u
  call iostat_update(ierr)
  read(iu, iostat=ierr) !xa_used, lasto_used
  call iostat_update(ierr)
  read(iu, iostat=ierr) !.false., Bloch, pre_expand
  call iostat_update(ierr)
  read(iu, iostat=ierr) !mu
  call iostat_update(ierr)

  ! k-points
  read(iu, iostat=ierr) nkpt
  call iostat_update(ierr)
  read(iu, iostat=ierr) !
  call iostat_update(ierr)

  read(iu, iostat=ierr) NE
  call iostat_update(ierr)

end subroutine read_gf_sizes

subroutine read_gf_header(iu, nkpt, kpt, NE, E)
  use io_m, only: iostat_update

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

  integer :: ierr

  read(iu, iostat=ierr) !nspin, cell
  call iostat_update(ierr)
  read(iu, iostat=ierr) !na_u, no_u
  call iostat_update(ierr)
  read(iu, iostat=ierr) !na_used, no_used
  call iostat_update(ierr)
  read(iu, iostat=ierr) !xa_used, lasto_used
  call iostat_update(ierr)
  read(iu, iostat=ierr) !.false., Bloch, pre_expand
  call iostat_update(ierr)
  read(iu, iostat=ierr) !mu
  call iostat_update(ierr)

  ! k-points
  read(iu, iostat=ierr) !nkpt
  call iostat_update(ierr)
  read(iu, iostat=ierr) kpt
  call iostat_update(ierr)

  read(iu, iostat=ierr) !NE
  call iostat_update(ierr)
  read(iu, iostat=ierr) E
  call iostat_update(ierr)

end subroutine read_gf_header

subroutine read_gf_hs(iu, no_u, H, S)
  use io_m, only: iostat_update

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

  integer :: ierr

  read(iu, iostat=ierr) !ik, iE, E
  call iostat_update(ierr)
  read(iu, iostat=ierr) H
  call iostat_update(ierr)
  read(iu, iostat=ierr) S
  call iostat_update(ierr)

end subroutine read_gf_hs

subroutine read_gf_se( iu, no_u, iE, SE )
  use io_m, only: iostat_update

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

  integer :: ierr

  if ( iE > 1 ) then
    read(iu, iostat=ierr) !ik, iE, E
    call iostat_update(ierr)
  end if
  read(iu, iostat=ierr) SE
  call iostat_update(ierr)

end subroutine read_gf_se
