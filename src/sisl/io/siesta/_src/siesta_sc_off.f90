! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine siesta_sc_off(nsa, nsb, nsc, isc)
  integer, intent(in) :: nsa, nsb, nsc
  ! Sadly product requires -lgfortran
  ! and f2py assumes nsc(1) to be a function
  integer, intent(out) :: isc(3,nsa*nsb*nsc)

  integer :: x, y, z, i
  integer :: nx, ny, nz

  i = 0
  do z = 0, nsc - 1
    nz = linear2pm(z, nsc)
    do y = 0, nsb - 1
      ny = linear2pm(y, nsb)
      do x = 0, nsa - 1
        nx = linear2pm(x, nsa)
        i = i + 1
        isc(1,i) = nx
        isc(2,i) = ny
        isc(3,i) = nz
      end do
    end do
  end do

contains

  pure function linear2pm(i,n) result(j)
    integer, intent(in) :: i, n
    integer :: j
    if ( i > n / 2 ) then
      j = -n + i
    else
      j = i
    end if
  end function linear2pm

end subroutine siesta_sc_off
