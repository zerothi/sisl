! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine write_hsx1(fname, &
    label, Z, no, n, l, zeta, &
    xa, isa, lasto, &
    ucell, nsc, isc_off, &
    ncol, row, &
    H, S, is_dp, &
    Ef, Qtot, temp, &
    nspecies, na_u, no_max, n_s, nspin, no_u, nnz)

  use precision, only: r4, r8
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspecies
  integer, intent(in) :: na_u, isa(na_u), lasto(na_u)
  real(r8), intent(in) :: xa(3,na_u)
  real(r8), intent(in) :: ucell(3,3)
  character(len=20), intent(in) :: label(nspecies)
  real(r8), intent(in) :: Z(nspecies)
  integer, intent(in) :: no(nspecies)
  integer, intent(in) :: no_max
  integer, intent(in), dimension(no_max, nspecies) :: n, l, zeta
  integer, intent(in) :: n_s, nsc(3), isc_off(3, n_s)
  integer, intent(in) :: nspin, no_u, nnz
  integer, intent(in) :: ncol(no_u), row(nnz)
  real(r8), intent(in) :: H(nnz, nspin), S(nnz)
  logical, intent(in) :: is_dp
  real(r8), intent(in) :: Ef, Qtot, temp

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: na_u, xa, isa, lasto
!f2py intent(in) :: nspecies, label, Z, no
!f2py intent(in) :: no_max, n, l, zeta
!f2py intent(in) :: ucell, n_s, nsc, isc_off
!f2py intent(in) :: no_u, nnz, nspin
!f2py intent(in) :: ncol, row
!f2py intent(in) :: H, S, is_dp, Ef, Qtot, temp

! Internal variables and arrays
  integer :: iu, ierr
  integer :: i, is, idx

  ! Open file (ensure we start from a clean slate)!
  call open_file(fname, 'write', 'unknown', 'unformatted', iu)

  write(iu, iostat=ierr) 1
  call iostat_update(ierr)

! Write header information
  write(iu, iostat=ierr) is_dp
  call iostat_update(ierr)

! Write overall data
  write(iu, iostat=ierr) na_u, no_u, nspin, nspecies, nsc
  call iostat_update(ierr)
  write(iu, iostat=ierr) ucell, Ef, qtot, temp
  call iostat_update(ierr)
  write(iu, iostat=ierr) isc_off, xa, isa, lasto
  call iostat_update(ierr)

  write(iu, iostat=ierr) (label(is), Z(is), no(is), is=1,nspecies)
  call iostat_update(ierr)
  do is = 1, nspecies
    write(iu, iostat=ierr) (n(i,is), l(i,is), zeta(i,is), i=1,no(is))
    call iostat_update(ierr)
  end do

  write(iu, iostat=ierr) ncol
  call iostat_update(ierr)

  idx = 1
  do i = 1, no_u
    write(iu, iostat=ierr) row(idx:idx+ncol(i)-1)
    call iostat_update(ierr)
    idx = idx + ncol(i)
  end do

  if ( is_dp ) then

    do is = 1 , nspin
      idx = 1
      do i = 1 , no_u
        write(iu, iostat=ierr) H(idx:idx+ncol(i)-1,is)
        call iostat_update(ierr)
        idx = idx + ncol(i)
      end do
    end do

    idx = 1
    do i = 1 , no_u
      write(iu, iostat=ierr) S(idx:idx+ncol(i)-1)
      call iostat_update(ierr)
      idx = idx + ncol(i)
    end do

  else

    do is = 1 , nspin
      idx = 1
      do i = 1 , no_u
        write(iu, iostat=ierr) real(H(idx:idx+ncol(i)-1,is), r4)
        call iostat_update(ierr)
        idx = idx + ncol(i)
      end do
    end do

    idx = 1
    do i = 1 , no_u
      write(iu, iostat=ierr) real(S(idx:idx+ncol(i)-1), r4)
      call iostat_update(ierr)
      idx = idx + ncol(i)
    end do

  end if

  call close_file(iu)

end subroutine write_hsx1

subroutine write_hsx2(fname, &
    label, Z, no, n, l, zeta, &
    xa, isa, lasto, &
    ucell, nsc, isc_off, &
    ncol, row, &
    H, S, is_dp, &
    Ef, Qtot, temp, &
    k_cell, k_displ, &
    ! Sizes in the end
    nspecies, no_max, na_u, n_s, nspin, no_u, nnz)

  use precision, only: r4, r8
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspecies
  character(len=20), intent(in) :: label(nspecies)
  real(r8), intent(in) :: Z(nspecies)
  integer, intent(in) :: no(nspecies)
  integer, intent(in) :: no_max
  integer, intent(in), dimension(no_max, nspecies) :: n, l, zeta
  integer, intent(in) :: na_u, isa(na_u), lasto(na_u)
  real(r8), intent(in) :: xa(3,na_u), ucell(3,3)
  integer, intent(in) :: n_s, nsc(3), isc_off(3, n_s)
  integer, intent(in) :: nspin, no_u, nnz
  integer, intent(in) :: ncol(no_u), row(nnz)
  real(r8), intent(in) :: H(nnz, nspin), S(nnz)
  logical, intent(in) :: is_dp
  real(r8), intent(in) :: Ef, Qtot, temp
  integer, intent(in) :: k_cell(3,3)
  real(r8), intent(in) :: k_displ(3)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: na_u, xa, isa, lasto
!f2py intent(in) :: nspecies, label, Z, no
!f2py intent(in) :: no_max, n, l, zeta
!f2py intent(in) :: ucell, n_s, nsc, isc_off
!f2py intent(in) :: no_u, nnz, nspin
!f2py intent(in) :: ncol, row
!f2py intent(in) :: H, S, is_dp, Ef, Qtot, temp
!f2py intent(in) :: k_cell, k_displ

! Internal variables and arrays
  integer :: iu, ierr
  integer :: i, is, idx

  ! Open file (ensure we start from a clean slate)!
  call open_file(fname, 'write', 'unknown', 'unformatted', iu)

  write(iu, iostat=ierr) 2
  call iostat_update(ierr)

! Write header information
  write(iu, iostat=ierr) is_dp
  call iostat_update(ierr)

! Write overall data
  write(iu, iostat=ierr) na_u, no_u, nspin, nspecies, nsc
  call iostat_update(ierr)
  write(iu, iostat=ierr) ucell, Ef, qtot, temp
  call iostat_update(ierr)
  write(iu, iostat=ierr) isc_off, xa, isa, lasto
  call iostat_update(ierr)

  write(iu, iostat=ierr) (label(is), Z(is), no(is), is=1,nspecies)
  call iostat_update(ierr)
  do is = 1, nspecies
    write(iu, iostat=ierr) (n(i,is), l(i,is), zeta(i,is), i=1,no(is))
    call iostat_update(ierr)
  end do

  write(iu, iostat=ierr) k_cell, k_displ
  call iostat_update(ierr)

  write(iu, iostat=ierr) ncol
  call iostat_update(ierr)

  idx = 1
  do i = 1, no_u
    write(iu, iostat=ierr) row(idx:idx+ncol(i)-1)
    call iostat_update(ierr)
    idx = idx + ncol(i)
  end do

  if ( is_dp ) then

    do is = 1 , nspin
      idx = 1
      do i = 1 , no_u
        write(iu, iostat=ierr) H(idx:idx+ncol(i)-1,is)
        call iostat_update(ierr)
        idx = idx + ncol(i)
      end do
    end do

    idx = 1
    do i = 1 , no_u
      write(iu, iostat=ierr) S(idx:idx+ncol(i)-1)
      call iostat_update(ierr)
      idx = idx + ncol(i)
    end do

  else

    do is = 1 , nspin
      idx = 1
      do i = 1 , no_u
        write(iu, iostat=ierr) real(H(idx:idx+ncol(i)-1,is), r4)
        call iostat_update(ierr)
        idx = idx + ncol(i)
      end do
    end do

    idx = 1
    do i = 1 , no_u
      write(iu, iostat=ierr) real(S(idx:idx+ncol(i)-1), r4)
      call iostat_update(ierr)
      idx = idx + ncol(i)
    end do

  end if

  call close_file(iu)

end subroutine write_hsx2
