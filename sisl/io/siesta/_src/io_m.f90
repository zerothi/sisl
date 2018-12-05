!> Module for handling free units and checking whether stuff succeeded
module io_m

  implicit none

  public :: free_unit
  public :: iostat_reset
  public :: iostat_update
  public :: iostat_query

  integer, private :: io_stat = 0

contains

  !< Get the next free unit
  subroutine free_unit(u, stat_reset)
    integer, intent(out) :: u
    logical, intent(in), optional :: stat_reset

    ! Define f2py intents
!f2py intent(out) :: u
!f2py intent(in), optional :: u

    logical :: opened

    u = 999
    opened = .true.
    do while ( opened )

      u = u + 1
      inquire(u, opened=opened)

    end do

    ! Default to reset the global iostat
    opened = .true.
    if ( present(stat_reset) ) opened = stat_reset
    if ( opened ) then
      call iostat_reset()
    end if

  end subroutine free_unit

  !< Initialize global io stat
  subroutine iostat_reset()
    io_stat = 0
  end subroutine iostat_reset

  !< Update global status, only overwrite if not used
  subroutine iostat_update(iostat)
    integer, intent(in) :: iostat
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
