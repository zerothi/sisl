! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
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

subroutine read_gf_find(iu, nspin, nkpt, NE, &
    cstate, cspin, ckpt, cE, cis_read, istate, ispin, ikpt, iE)
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  integer, intent(in) :: iu
  integer, intent(in) :: nspin, nkpt, NE
  integer, intent(in) :: cstate, cspin, ckpt, cE, cis_read
  integer, intent(in) :: istate, ispin, ikpt, iE

  ! Define f2py intents
!f2py intent(in) :: iu, nspin, nkpt, NE
!f2py intent(in) :: cstate, cspin, ckpt, cE, cis_read
!f2py intent(in) :: istate, ispin, ikpt, iE

  integer :: ierr, i

  ! We calculate the current record position
  ! Then we calculate the resulting record position
  ! Finally we backspace or read to the record position
  integer :: crec, irec

  if ( istate == -1 ) then
    ! Requested header read! REWIND!
    rewind(iu, iostat=ierr)
    call iostat_update(ierr)
    return
  end if

  ! ikpt and iE are Python indices
  ! Find linear record index
  crec = linear_rec(cstate, cspin, ckpt, cE, cis_read)
  irec = linear_rec(istate, ispin, ikpt, iE, 0) ! stop just in front

  if ( crec < irec ) then
    do i = crec, irec - 1
      read(iu, iostat=ierr) ! record
      call iostat_update(ierr)
    end do
  else if ( crec > irec ) then
    do i = irec, crec - 1
      backspace(iu, iostat=ierr) ! record
      call iostat_update(ierr)
    end do
  end if

contains

  function linear_rec(state, ispin, ikpt, iE, is_read) result(irec)
    ! Note that these indices are 0-based
    integer, intent(in) :: state, ispin, ikpt, iE, is_read
    integer :: irec

    integer :: nHS, nSE

    if ( state == -1 .and. is_read == 0 ) then
      ! We are at the start of the file
      irec = 0
    else
      ! records in header
      irec = 10
    end if
    ! Return if the record requested is the header
    if ( state == -1 ) return

    ! Skip to the spin
    nHS = ispin * nkpt
    nSE = ispin * nkpt * NE
    ! per H and S we also have ik, iE, E
    ! per SE we also have ik, iE, E (except for the first energy-point where we don't have it)
    irec = irec + nHS * (3 - 1) + nSE * 2

    ! Skip to the k-point
    nHS = ikpt
    nSE = ikpt * NE
    irec = irec + nHS * (3 - 1) + nSE * 2

    ! If the state is 0, it means that we should read beyond H and S for the given k-point
    if ( state == 0 .and. is_read == 0 ) return
    ! Skip the HS and line
    irec = irec + 3
    if ( state == 0 ) return

    ! Skip to the energy-point
    if ( iE > 0 ) then
      irec = irec + iE * 2 - 1 ! correct the iE == 1 ik, iE, E line
    end if

    if ( is_read == 1 ) then
      ! Means that we already have read past this entry
      if ( iE > 0 ) then
        irec = irec + 2
      else
        irec = irec + 1
      end if
    end if

  end function linear_rec

end subroutine read_gf_find

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
  integer :: f_ik, f_iE

  integer :: ierr

  read(iu, iostat=ierr) f_ik, f_iE ! E
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

  integer :: f_ik, f_iE
  integer :: ierr

  if ( iE > 0 ) then
    read(iu, iostat=ierr) f_ik, f_iE ! E
    call iostat_update(ierr)
    if ( iE + 1 /= f_iE ) then
      ! Signal something is wrong!
      call iostat_update(999)
    end if
  end if
  read(iu, iostat=ierr) SE
  call iostat_update(ierr)

end subroutine read_gf_se
