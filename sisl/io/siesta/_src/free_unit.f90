subroutine free_unit(u)

  implicit none

  integer :: u
  
  ! Define f2py intents
!f2py intent(out)  :: u

  logical :: opened

  u = 999
  opened = .true.
  do while ( opened )

     u = u + 1
     inquire(u, opened=opened)

  end do

end subroutine free_unit
