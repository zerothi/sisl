! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
!> Module for handling free units and checking whether stuff succeeded
module io_m

  implicit none
  private

  public :: open_file
  public :: open_file_read
  public :: open_file_write
  public :: rewind_file
  public :: iostat_reset
  public :: iostat_update
  public :: iostat_query
  public :: close_file

  integer :: io_stat = 0

contains

  !< Open the file `file` using the given `action`, `status` and `form` specifications.
  subroutine open_file(file, action, status, form, unit)
    character(len=*), intent(in) :: file
    character(len=*), intent(in) :: action, status, form
    integer, intent(out) :: unit

    logical :: opened
    integer :: ierr

    ! Check out whether the file is already opened and
    ! if we can reuse it...
    unit = -1
    inquire(file=file, number=unit, opened=opened, iostat=ierr)
    call iostat_update(ierr)

    if ( unit > 0 ) then

      ! The file is already open
      ! Depending on the action, we need to rewind or close it
      select case ( action )
      case ( 'r', 'R', 'read', 'READ' )

        ! It is already opened, simply rewind and return...
        rewind(unit)
        return

      case ( 'w', 'W', 'write', 'WRITE' )

        call close_file(unit)

      end select

    end if

    ! Always reset
    call iostat_reset()

    ! We need to open it a-new
    unit = 999
    opened = .true.
    do while ( opened )

      unit = unit + 1
      inquire(unit, opened=opened)

    end do

    open(unit, file=trim(file), status=status, form=form, action=action, iostat=ierr)
    call iostat_update(ierr)

  end subroutine open_file

  !< Close file from a unit
  subroutine close_file(iu)
    integer, intent(in) :: iu
    integer :: ierr

    ! Define f2py intents
!f2py intent(in) :: iu

    ! Open file
    close(iu, iostat=ierr)

    call iostat_update(ierr)

  end subroutine close_file

  !< Open file for read-mode
  subroutine open_file_read(fname, iu)
    ! Input parameters
    character(len=*), intent(in) :: fname
    integer, intent(out) :: iu

    ! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: iu

    call open_file(fname, 'read', 'old', 'unformatted', iu)

  end subroutine open_file_read

  !< Open file for write-mode
  subroutine open_file_write(fname, iu)
    ! Input parameters
    character(len=*), intent(in) :: fname
    integer, intent(out) :: iu

    ! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: iu

    call open_file(fname, 'write', 'unknown', 'unformatted', iu)

  end subroutine open_file_write

  !< Rewind file to read from beginning
  subroutine rewind_file( iu )

    ! Input parameters
    integer, intent(in) :: iu

  ! Define f2py intents
!f2py intent(in) :: iu

    ! Open file
    rewind(iu)

  end subroutine rewind_file

  !< Initialize global io stat
  subroutine iostat_reset()
    io_stat = 0
  end subroutine iostat_reset

  !< Update global status, only overwrite if not used
  subroutine iostat_update(iostat)
    integer, intent(in) :: iostat

    ! Define f2py intents
!f2py intent(out)  :: iostat

    if ( io_stat == 0 ) io_stat = iostat

  end subroutine iostat_update

  !< Query the status of io
  subroutine iostat_query(iostat)
    integer, intent(out) :: iostat

    ! Define f2py intents
!f2py intent(out)  :: iostat

    iostat = io_stat

  end subroutine iostat_query

end module io_m
