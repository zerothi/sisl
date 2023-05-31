! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine write_hsx0(fname, Gamma, no_u, no_s, nspin, maxnh, &
    numh, listhptr, listh, H, S, xij, Qtot, temp)

  use precision, only: r4, r8
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*) :: fname
  logical :: Gamma
  integer :: no_u, no_s, nspin, maxnh
  integer :: listh(maxnh), numh(no_u), listhptr(no_u)
  real(r8) :: H(maxnh,nspin), S(maxnh), xij(3,maxnh), Qtot, temp

! Define f2py intents
!f2py intent(in) :: fname, Gamma, no_u, no_s, nspin, maxnh
!f2py intent(in) :: numh, listhptr, listh, H, S, xij, Qtot, temp

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, im, k
  integer :: indxuo(no_s)

  ! Open file (ensure we start from a clean slate)!
  call open_file(fname, 'write', 'unknown', 'unformatted', iu)

! Write overall data
  write(iu, iostat=ierr) no_u, no_s, nspin, maxnh
  call iostat_update(ierr)

! Write logical
  write(iu, iostat=ierr) Gamma
  call iostat_update(ierr)

! Write out indxuo
  if (.not. Gamma) then
    do ih = 1 , no_s
      im = mod(ih,no_u)
      if ( im == 0 ) im = no_u
      indxuo(ih) = im
    end do
    write(iu, iostat=ierr) (indxuo(ih),ih=1,no_s)
    call iostat_update(ierr)
  end if

  write(iu, iostat=ierr) (numh(ih),ih=1,no_u)
  call iostat_update(ierr)

! Write listh
  do ih = 1 , no_u
    write(iu, iostat=ierr) (listh(listhptr(ih)+im),im = 1,numh(ih))
    call iostat_update(ierr)
  end do

! Write Hamiltonian
  do is = 1 , nspin
    do ih = 1 , no_u
      write(iu, iostat=ierr) (real(H(listhptr(ih)+im,is),kind=r4),im=1,numh(ih))
      call iostat_update(ierr)
    end do
  end do

! Write Overlap matrix
  do ih = 1,no_u
    write(iu, iostat=ierr) (real(S(listhptr(ih)+im),kind=r4),im = 1,numh(ih))
    call iostat_update(ierr)
  end do

  write(iu, iostat=ierr) Qtot,temp
  call iostat_update(ierr)

  do ih = 1 , no_u
    write(iu, iostat=ierr) ((real(xij(k,listhptr(ih)+im),kind=r4), k=1,3),im =1,numh(ih))
    call iostat_update(ierr)
  end do

  call close_file(iu)

end subroutine write_hsx0

subroutine write_hsx1(fname, &
    nspecies, &
    na_u, xa, isa, lasto, &
    label, Z, no_max, no, n, l, zeta, &
    ucell, n_s, nsc, isc_off, &
    nspin, &
    no_u, no_s, nnz, &
    ncol, row, &
    H, S, is_dp, &
    Ef, Qtot, temp)

  use precision, only: r4, r8
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*) :: fname
  integer :: nspecies
  integer :: na_u, isa(na_u), lasto(na_u)
  real(r8) :: xa(3,na_u)
  real(r8) :: ucell(3,3)
  character(len=*) :: label(nspecies)
  integer :: Z(nspecies), no_max, no(nspecies)
  integer, dimension(nspecies, no_max) :: n, l, zeta
  integer :: n_s, nsc(3), isc_off(3, n_s)
  integer :: nspin, no_u, no_s, nnz
  integer :: ncol(no_u), row(nnz)
  real(r8) :: H(nnz, nspin), S(nnz)
  logical :: is_dp
  real(r8) :: Ef, Qtot, temp

! Define f2py intents
!f2py intent(in) :: fname, nspecies, na_u, xa, isa, lasto
!f2py intent(in) :: label, Z, no_max, no, n, l, zeta
!f2py intent(in) :: ucell, n_s, nsc, isc_off, nspin
!f2py intent(in) :: no_u, no_s, nnz, ncol, row
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
    write(iu, iostat=ierr) (n(is,i), l(is,i), zeta(is,i), i=1,no(is))
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

