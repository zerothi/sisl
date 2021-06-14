! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine write_gf_header( iu, nspin, cell, na_u, no_u, na_used, no_used, &
    xa_used, lasto_used, Bloch, pre_expand, mu, nkpt, kpt, kw, NE, E)
  use io_m, only: iostat_update

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

  integer :: ierr

  write(iu, iostat=ierr) nspin, cell
  call iostat_update(ierr)
  write(iu, iostat=ierr) na_u, no_u
  call iostat_update(ierr)
  write(iu, iostat=ierr) na_used, no_used
  call iostat_update(ierr)
  write(iu, iostat=ierr) xa_used, lasto_used
  call iostat_update(ierr)
  write(iu, iostat=ierr) .false., Bloch, pre_expand
  call iostat_update(ierr)
  write(iu, iostat=ierr) mu
  call iostat_update(ierr)

  ! k-points
  write(iu, iostat=ierr) nkpt
  call iostat_update(ierr)
  write(iu, iostat=ierr) kpt, kw
  call iostat_update(ierr)

  write(iu, iostat=ierr) NE
  call iostat_update(ierr)
  write(iu, iostat=ierr) E
  call iostat_update(ierr)

end subroutine write_gf_header

subroutine write_gf_hs(iu, ik, E, no_u, H, S)
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: ik
  complex(dp), intent(in) :: E
  ! Variables for the size
  integer, intent(in) :: no_u
  complex(dp), intent(in) :: H(no_u,no_u), S(no_u,no_u)

! Define f2py intents
!f2py intent(in) :: iu
!f2py intent(in) :: ik, E
!f2py intent(in) :: no_u
!f2py intent(in) :: H, S

  integer :: ierr

  ! ik and iE are Python indices
  write(iu, iostat=ierr) ik + 1, 1, E
  call iostat_update(ierr)
  write(iu, iostat=ierr) H
  call iostat_update(ierr)
  write(iu, iostat=ierr) S
  call iostat_update(ierr)

end subroutine write_gf_hs

subroutine write_gf_se(iu, ik, iE, E, no_u, SE)
  use io_m, only: iostat_update

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

  integer :: ierr

  ! ik and iE are Python indices
  if ( iE > 0 ) then
    write(iu, iostat=ierr) ik + 1, iE + 1, E
    call iostat_update(ierr)
  end if
  write(iu, iostat=ierr) SE
  call iostat_update(ierr)

end subroutine write_gf_se
