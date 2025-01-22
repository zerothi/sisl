module precision

  implicit none

  integer, parameter :: r4 = selected_real_kind(6)
  integer, parameter :: r8 = selected_real_kind(15)
  integer, parameter :: i4 = selected_int_kind(6)
  integer, parameter :: i8 = selected_int_kind(18)

end module precision
