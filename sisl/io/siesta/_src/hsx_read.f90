! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine read_hsx_version(fname, version)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: version

  ! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: version

  integer :: iu, ierr
  integer :: tmp(4)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Set a default version in case it won't be found
  version = -1

  read(iu, iostat=ierr) tmp
  if ( ierr == 0 ) then
    ! Valid original version
    version = 0

  else ! ierr /= 0

    ! we have a version
    rewind(iu)
    read(iu, iostat=ierr) version

    if ( version /= 1 ) then
      ! Signal we do not know about this file
      call iostat_update(-3)
    end if

  end if
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_version


subroutine read_hsx_sizes(fname, Gamma, nspin, na_u, no_u, no_s, maxnh)

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(out) :: Gamma
  integer, intent(out) :: nspin, na_u, no_u, no_s, maxnh

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, nspin, na_u, no_u, no_s, maxnh

  ! Internal variables and arrays
  integer :: version

  call read_hsx_version(fname, version)

  if ( version == 0 ) then ! old

    call read_hsx_sizes0(fname, Gamma, nspin, na_u, no_u, no_s, maxnh)

  else if ( version == 1 ) then

    call read_hsx_sizes1(fname, Gamma, nspin, na_u, no_u, no_s, maxnh)

  end if

end subroutine read_hsx_sizes

subroutine read_hsx_sizes0(fname, Gamma, nspin, na_u, no_u, no_s, maxnh)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(out) :: Gamma
  integer, intent(out) :: nspin, na_u, no_u, no_s, maxnh

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, nspin, na_u, no_u, no_s, maxnh

  ! Internal variables and arrays
  integer :: iu, ierr
  integer :: io, is, nspecies
  character(len=20), allocatable :: label(:)
  real(dp), allocatable :: zval(:)
  integer, allocatable :: no(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) no_u, no_s, nspin, maxnh
  call iostat_update(ierr)
  read(iu, iostat=ierr) gamma
  call iostat_update(ierr)
  if ( .not. gamma ) then
    read(iu, iostat=ierr) !indxuo
    call iostat_update(ierr)
  end if
  read(iu, iostat=ierr) !numh
  call iostat_update(ierr)
  do io = 1, no_u * (nspin + 1)
    read(iu, iostat=ierr) !H and S
    call iostat_update(ierr)
  end do
  read(iu, iostat=ierr) !Qtot, temp
  call iostat_update(ierr)
  do io = 1, no_u
    read(iu, iostat=ierr) !xij
    call iostat_update(ierr)
  end do

  ! Now read in the geometry information
  read(iu, iostat=ierr) nspecies
  call iostat_update(ierr)
  allocate(label(nspecies), zval(nspecies), no(nspecies))
  read(iu, iostat=ierr) (label(is), zval(is),no(is),is=1,nspecies)

  do is = 1, nspecies
    do io = 1 , no(is)
      read(iu, iostat=ierr) !n(is,io), l(is,io), zeta(is,io)
      call iostat_update(ierr)
    end do
  end do
  deallocate(label, zval, no)

  read(iu, iostat=ierr) na_u
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_sizes0

subroutine read_hsx_sizes1(fname, Gamma, nspin, na_u, no_u, no_s, maxnh)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(out) :: Gamma
  integer, intent(out) :: nspin, na_u, no_u, no_s, maxnh

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, nspin, na_u, no_u, no_s, maxnh

  ! Internal variables and arrays
  integer :: version, nspecies, nsc(3)
  integer, allocatable :: numh(:)
  integer :: is
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) version
  call iostat_update(ierr)
  if ( version /= 1 ) then
    call iostat_update(-3)
    return
  end if
  read(iu, iostat=ierr) !is_dp
  call iostat_update(ierr)
  read(iu, iostat=ierr) na_u, no_u, nspin, nspecies, nsc
  call iostat_update(ierr)
  no_s = product(nsc) * no_u
  Gamma = no_s == no_u

  read(iu, iostat=ierr) !ucell, Ef, qtot, temp
  call iostat_update(ierr)
  read(iu, iostat=ierr) !isc_off, xa, isa, lasto(1:na_u)
  call iostat_update(ierr)
  read(iu, iostat=ierr) !(label(is), zval(is), no(is), is=1,nspecies)
  call iostat_update(ierr)

  do is = 1, nspecies
    read(iu, iostat=ierr) !(nquant(is,io), lquant(is,io), zeta(is,io), io=1,no(is))
    call iostat_update(ierr)
  end do

  allocate(numh(no_u))
  read(iu, iostat=ierr) numh
  call iostat_update(ierr)
  maxnh = sum(numh)
  deallocate(numh)

  call close_file(iu)

end subroutine read_hsx_sizes1

subroutine read_hsx_ef(fname, Ef)

  implicit none

  integer, parameter :: dp = selected_real_kind(p=14)

  ! Input parameters
  character(len=*), intent(in) :: fname
  real(dp), intent(out) :: Ef

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Ef

  ! Internal variables and arrays
  integer :: version

  call read_hsx_version(fname, version)

  if ( version == 0 ) then ! old

    call read_hsx_ef0(fname, Ef)

  else if ( version == 1 ) then

    call read_hsx_ef1(fname, Ef)

  end if

end subroutine read_hsx_ef


subroutine read_hsx_ef0(fname, Ef)
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=14)

  ! Input parameters
  character(len=*), intent(in) :: fname
  real(dp), intent(out) :: Ef

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Ef

  Ef = huge(1._dp)
  call iostat_update(-3)

end subroutine read_hsx_ef0

subroutine read_hsx_ef1(fname, Ef)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=14)

  ! Input parameters
  character(len=*), intent(in) :: fname
  real(dp), intent(out) :: Ef

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Ef

  ! Internal variables and arrays
  integer :: version
  real(dp) :: ucell(3,3)
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) version
  call iostat_update(ierr)
  if ( version /= 1 ) then
    call iostat_update(-3)
    return
  end if
  read(iu, iostat=ierr) !is_dp
  call iostat_update(ierr)
  read(iu, iostat=ierr) !na_u, no_u, nspin, nspecies, nsc
  call iostat_update(ierr)

  read(iu, iostat=ierr) ucell, Ef! , qtot, temp
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_ef1


!< Internal method for skipping species information in version 1 and later
!< The unit *must* be located just after
!<     read(iu) isc_off, xa, isa, lasto(1:na_u)

subroutine internal_read_hsx_skip_specie1(iu, nspecies)
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  integer, intent(in) :: iu, nspecies

  ! Internal variables and arrays
  integer :: ierr, is

  read(iu, iostat=ierr) !(label(is), zval(is), no(is), is=1,nspecies)
  call iostat_update(ierr)

  do is = 1, nspecies
    read(iu, iostat=ierr) !(nquant(is,io), lquant(is,io), zeta(is,io), io=1, no(is))
    call iostat_update(ierr)
  end do

end subroutine internal_read_hsx_skip_specie1


subroutine read_hsx_hsx0(fname, Gamma, nspin, no_u, no_s, maxnh, &
    numh, listh, H, S, xij)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(in) :: Gamma
  integer, intent(in) :: nspin, no_u, no_s, maxnh
  integer, intent(out) :: numh(no_u), listh(maxnh)
  real(sp), intent(out) :: H(maxnh,nspin), S(maxnh), xij(3,maxnh)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: Gamma, nspin, no_u, no_s, maxnh
!f2py intent(out) :: numh, listh
!f2py intent(out) :: H, S, xij

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, im
  integer, allocatable :: listhptr(:)

  ! Local readables
  logical :: lGamma
  integer :: lno_s, lno_u, lnspin, lmaxnh

  call open_file(fname, 'read', 'old', 'unformatted', iu)

! Read overall data
  read(iu, iostat=ierr) lno_u, lno_s, lnspin, lmaxnh
  call iostat_update(ierr)
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lno_s /= no_s ) stop 'Error in reading data, not allocated, no_s'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmaxnh /= maxnh ) stop 'Error in reading data, not allocated, maxnh'

! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)
  if ( lGamma .neqv. Gamma ) stop 'Error in reading data, not allocated'

! Read out indxuo
  if (.not. Gamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) numh
  call iostat_update(ierr)

  ! Create listhptr
  allocate(listhptr(no_u))
  listhptr(1) = 0
  do ih = 2 , no_u
    listhptr(ih) = listhptr(ih-1) + numh(ih-1)
  end do

! Read listh
  do ih = 1 , no_u
    im = numh(ih)
    read(iu, iostat=ierr) listh(listhptr(ih)+1:listhptr(ih)+im)
    call iostat_update(ierr)
  end do

! Read Hamiltonian
  do is = 1 , nspin
    do ih = 1 , no_u
      im = numh(ih)
      read(iu, iostat=ierr) H(listhptr(ih)+1:listhptr(ih)+im,is)
      call iostat_update(ierr)
    end do
  end do

! Read Overlap matrix
  do ih = 1,no_u
    im = numh(ih)
    read(iu, iostat=ierr) S(listhptr(ih)+1:listhptr(ih)+im)
    call iostat_update(ierr)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

  if ( Gamma ) then
    xij = 0._sp
  else
    do ih = 1 , no_u
      im = numh(ih)
      read(iu, iostat=ierr) xij(1:3,listhptr(ih)+1:listhptr(ih)+im)
      call iostat_update(ierr)
    end do
  end if

  deallocate(listhptr)

  call close_file(iu)

end subroutine read_hsx_hsx0

subroutine read_hsx_hsx1(fname, nspin, no_u, no_s, maxnh, &
    numh, listh, H, S, isc)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=14)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, no_u, no_s, maxnh
  integer, intent(out) :: numh(no_u), listh(maxnh), isc(3,no_s/no_u)
  real(dp), intent(out) :: H(maxnh,nspin), S(maxnh)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin, no_u, no_s, maxnh
!f2py intent(out) :: numh, listh
!f2py intent(out) :: H, S, isc

! Internal variables and arrays
  integer :: iu, ierr, version
  integer :: is, ih, im
  integer, allocatable :: listhptr(:)

  ! Local readables
  logical :: is_dp
  integer :: lna_u, lno_u, lnspin
  integer :: nspecies, nsc(3)

  real(sp), allocatable :: sbuf(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) version
  call iostat_update(ierr)
  if ( version /= 1 ) then
    call iostat_update(-3)
    return
  end if

  read(iu, iostat=ierr) is_dp
  call iostat_update(ierr)

  read(iu, iostat=ierr) lna_u, lno_u, lnspin, nspecies, nsc
  call iostat_update(ierr)
  if ( lno_u /= no_u ) call iostat_update(-6)
  if ( lno_u * product(nsc) /= no_s ) call iostat_update(-6)
  if ( lnspin /= nspin ) call iostat_update(-6)

  read(iu, iostat=ierr) !ucell, Ef, qtot, temp
  call iostat_update(ierr)

  read(iu, iostat=ierr) isc!, xa, isa, lasto
  call iostat_update(ierr)

  call internal_read_hsx_skip_specie1(iu, nspecies)

  read(iu, iostat=ierr) numh
  call iostat_update(ierr)

  ! Create listhptr
  allocate(listhptr(no_u))
  listhptr(1) = 0
  do ih = 2 , no_u
    listhptr(ih) = listhptr(ih-1) + numh(ih-1)
  end do

  ! Read listh
  do ih = 1 , no_u
    im = numh(ih)
    read(iu, iostat=ierr) listh(listhptr(ih)+1:listhptr(ih)+im)
    call iostat_update(ierr)
  end do

  if ( is_dp ) then

    ! Read Hamiltonian
    do is = 1 , nspin
      do ih = 1 , no_u
        im = numh(ih)
        read(iu, iostat=ierr) H(listhptr(ih)+1:listhptr(ih)+im,is)
        call iostat_update(ierr)
      end do
    end do

    ! Read Overlap matrix
    do ih = 1,no_u
      im = numh(ih)
      read(iu, iostat=ierr) S(listhptr(ih)+1:listhptr(ih)+im)
      call iostat_update(ierr)
    end do

  else

    allocate(sbuf(maxval(numh)))

    ! Read Hamiltonian
    do is = 1, nspin
      do ih = 1, no_u
        im = numh(ih)
        read(iu, iostat=ierr) sbuf(1:im)
        call iostat_update(ierr)
        H(listhptr(ih)+1:listhptr(ih)+im,is) = sbuf(1:im)
      end do
    end do

    ! Read Overlap matrix
    do ih = 1, no_u
      im = numh(ih)
      read(iu, iostat=ierr) sbuf(1:im)
      call iostat_update(ierr)
      S(listhptr(ih)+1:listhptr(ih)+im) = sbuf(1:im)
    end do

    deallocate(sbuf)

  end if

  deallocate(listhptr)

  call close_file(iu)

end subroutine read_hsx_hsx1


subroutine read_hsx_sx0(fname, Gamma, nspin, no_u, no_s, maxnh, &
    numh, listh, S, xij)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(in) :: Gamma
  integer, intent(in) :: nspin, no_u, no_s, maxnh
  integer, intent(out) :: numh(no_u), listh(maxnh)
  real(sp), intent(out) :: S(maxnh), xij(3,maxnh)

  ! Define f2py intents
  !f2py intent(in) :: fname
  !f2py intent(in) :: Gamma, nspin, no_u, no_s, maxnh
  !f2py intent(out) :: numh, listh
  !f2py intent(out) :: S, xij

  ! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, im
  integer, allocatable :: listhptr(:)

  ! Local readables
  logical :: lGamma
  integer :: lno_s, lno_u, lnspin, lmaxnh

  real(sp), allocatable :: buf(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) lno_u, lno_s, lnspin, lmaxnh
  call iostat_update(ierr)
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lno_s /= no_s ) stop 'Error in reading data, not allocated, no_s'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmaxnh /= maxnh ) stop 'Error in reading data, not allocated, maxnh'

  ! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)
  if ( lGamma .neqv. Gamma ) stop 'Error in reading data, not allocated'

  ! Read out indxuo
  if (.not. Gamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) numh
  call iostat_update(ierr)

  ! Create listhptr
  allocate(listhptr(no_u))
  listhptr(1) = 0
  do ih = 2 , no_u
    listhptr(ih) = listhptr(ih-1) + numh(ih-1)
  end do

  allocate(buf(maxval(numh)*3))

  ! Read listh
  do ih = 1 , no_u
    im = numh(ih)
    read(iu, iostat=ierr) listh(listhptr(ih)+1:listhptr(ih)+im)
    call iostat_update(ierr)
  end do

  ! Read Hamiltonian
  do is = 1 , nspin
    do ih = 1 , no_u
      im = numh(ih)
      read(iu, iostat=ierr)
      call iostat_update(ierr)
    end do
  end do

  ! Read Overlap matrix
  do ih = 1,no_u
    im = numh(ih)
    read(iu, iostat=ierr) buf(1:im)
    call iostat_update(ierr)
    S(listhptr(ih)+1:listhptr(ih)+im) = buf(1:im)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

  if ( Gamma ) then
    xij = 0._sp
  else
    do ih = 1 , no_u
      im = numh(ih)
      read(iu, iostat=ierr) buf(1:im*3)
      call iostat_update(ierr)
      xij(1:3,listhptr(ih)+1:listhptr(ih)+im) = reshape(buf(1:im*3),(/3,im/))
    end do
  end if

  deallocate(buf, listhptr)

  call close_file(iu)

end subroutine read_hsx_sx0

subroutine read_hsx_sx1(fname, nspin, no_u, no_s, maxnh, &
    numh, listh, S, isc)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=14)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, no_u, no_s, maxnh
  integer, intent(out) :: numh(no_u), listh(maxnh), isc(3,no_s/no_u)
  real(dp), intent(out) :: S(maxnh)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin, no_u, no_s, maxnh
!f2py intent(out) :: numh, listh
!f2py intent(out) :: S, isc

! Internal variables and arrays
  integer :: iu, ierr, version
  integer :: is, ih, im
  integer, allocatable :: listhptr(:)

  ! Local readables
  logical :: is_dp
  integer :: lna_u, lno_u, lnspin
  integer :: nspecies, nsc(3)

  real(sp), allocatable :: sbuf(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) version
  call iostat_update(ierr)
  if ( version /= 1 ) then
    call iostat_update(-3)
    return
  end if

  read(iu, iostat=ierr) is_dp
  call iostat_update(ierr)

  read(iu, iostat=ierr) lna_u, lno_u, lnspin, nspecies, nsc
  call iostat_update(ierr)
  if ( lno_u /= no_u ) call iostat_update(-6)
  if ( lno_u * product(nsc) /= no_s ) call iostat_update(-6)
  if ( lnspin /= nspin ) call iostat_update(-6)

  read(iu, iostat=ierr) !ucell, Ef, qtot, temp
  call iostat_update(ierr)

  read(iu, iostat=ierr) isc!, xa, isa, lasto
  call iostat_update(ierr)

  call internal_read_hsx_skip_specie1(iu, nspecies)

  read(iu, iostat=ierr) numh
  call iostat_update(ierr)

  ! Create listhptr
  allocate(listhptr(no_u))
  listhptr(1) = 0
  do ih = 2 , no_u
    listhptr(ih) = listhptr(ih-1) + numh(ih-1)
  end do

  ! Read listh
  do ih = 1 , no_u
    im = numh(ih)
    read(iu, iostat=ierr) listh(listhptr(ih)+1:listhptr(ih)+im)
    call iostat_update(ierr)
  end do

  ! Read Hamiltonian
  do is = 1 , nspin * no_u
    read(iu, iostat=ierr) !
    call iostat_update(ierr)
  end do

  if ( is_dp ) then

    ! Read Overlap matrix
    do ih = 1,no_u
      im = numh(ih)
      read(iu, iostat=ierr) S(listhptr(ih)+1:listhptr(ih)+im)
      call iostat_update(ierr)
    end do

  else

    allocate(sbuf(maxval(numh)))

    ! Read Overlap matrix
    do ih = 1, no_u
      im = numh(ih)
      read(iu, iostat=ierr) sbuf(1:im)
      call iostat_update(ierr)
      S(listhptr(ih)+1:listhptr(ih)+im) = sbuf(1:im)
    end do

    deallocate(sbuf)

  end if

  deallocate(listhptr)

  call close_file(iu)

end subroutine read_hsx_sx1

subroutine read_hsx_geom1(fname, na_u, cell, nsc, xa, lasto)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: na_u
  integer, intent(out) :: lasto(na_u), nsc(3)
  real(dp), intent(out) :: cell(3,3), xa(3,na_u)

  ! Define f2py intents
!f2py intent(in) :: fname, na_u
!f2py intent(out) :: cell, nsc, isa, xa, lasto

  integer :: iu, ierr, version
  integer :: lna_u, no_u, nspin, nspecies
  integer, allocatable :: isc(:,:), isa(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) version
  call iostat_update(ierr)
  if ( version /= 1 ) then
    call iostat_update(-3)
    return
  end if

  read(iu, iostat=ierr) ! is_dp
  call iostat_update(ierr)

  read(iu, iostat=ierr) lna_u, no_u, nspin, nspecies, nsc
  call iostat_update(ierr)
  if ( lna_u /= na_u ) call iostat_update(-6)

  read(iu, iostat=ierr) cell !, Ef, qtot, temp
  call iostat_update(ierr)

  allocate(isc(3, product(nsc)), isa(na_u))
  read(iu, iostat=ierr) isc, xa, isa, lasto
  call iostat_update(ierr)
  deallocate(isc, isa)

  call close_file(iu)

end subroutine read_hsx_geom1

subroutine read_hsx_specie_sizes(fname, no_u, na_u, nspecies)

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: no_u, na_u, nspecies

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: no_u, na_u, nspecies

! Internal variables and arrays
  integer :: version

  call read_hsx_version(fname, version)

  if ( version == 0 ) then
    call read_hsx_specie_sizes0(fname, no_u, na_u, nspecies)
  else if ( version == 1 ) then
    call read_hsx_specie_sizes1(fname, no_u, na_u, nspecies)
  end if

end subroutine read_hsx_specie_sizes

subroutine read_hsx_specie_sizes0(fname, no_u, na_u, nspecies)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: no_u, na_u, nspecies

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: no_u, na_u, nspecies

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, io

  ! Local readables
  integer :: lno_s, lnspin, lmaxnh
  logical :: lGamma
  character(len=20), allocatable :: label(:)
  real(dp), allocatable :: zval(:)
  integer, allocatable :: no(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) no_u, lno_s, lnspin, lmaxnh
  call iostat_update(ierr)

  ! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)

  ! Read out indxuo
  if (.not. lGamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) ! numh
  call iostat_update(ierr)

  ! Read listh
  do ih = 1 , no_u
    read(iu, iostat=ierr) !listh
    call iostat_update(ierr)
  end do

  ! Read Hamiltonian
  do is = 1, no_u * (lnspin + 1)
    read(iu, iostat=ierr) ! H and S
    call iostat_update(ierr)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

  do ih = 1 , no_u
    read(iu, iostat=ierr) ! xij
    call iostat_update(ierr)
  end do

  ! Now read in the geometry information
  read(iu, iostat=ierr) nspecies
  call iostat_update(ierr)
  allocate(label(nspecies), zval(nspecies), no(nspecies))
  read(iu, iostat=ierr) (label(is), zval(is),no(is),is=1,nspecies)

  do is = 1, nspecies
    do io = 1 , no(is)
      read(iu, iostat=ierr) !n(is,io), l(is,io), zeta(is,io)
      call iostat_update(ierr)
    end do
  end do
  deallocate(label, zval, no)

  read(iu, iostat=ierr) na_u
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_specie_sizes0

subroutine read_hsx_specie_sizes1(fname, no_u, na_u, nspecies)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: no_u, na_u, nspecies

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: no_u, na_u, nspecies

! Internal variables and arrays
  integer :: iu, ierr, version
  integer :: lnspin, nsc(3)

  ! Local readables

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) version
  call iostat_update(ierr)
  if ( version /= 1 ) then
    call iostat_update(-3)
    return
  end if

  read(iu, iostat=ierr) !is_dp
  call iostat_update(ierr)

  ! Read overall data
  read(iu, iostat=ierr) na_u, no_u, lnspin, nspecies, nsc
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_specie_sizes1

subroutine read_hsx_species(fname, nspecies, no_u, na_u, label, zval, no, isa)

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspecies, no_u, na_u
  character(len=1), intent(out) :: label(20,nspecies)
  real(dp), intent(out) :: zval(nspecies)
  integer, intent(out) :: no(nspecies)
  integer, intent(out) :: isa(na_u)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspecies, no_u, na_u
!f2py intent(out) :: label, zval, no, isa

  integer :: version

  call read_hsx_version(fname, version)

  if ( version == 0 ) then ! old

    call read_hsx_species0(fname, nspecies, no_u, na_u, label, zval, no, isa)

  else if ( version == 1 ) then

    call read_hsx_species1(fname, nspecies, no_u, na_u, label, zval, no, isa)

  end if

end subroutine read_hsx_species

subroutine read_hsx_species0(fname, nspecies, no_u, na_u, label, zval, no, isa)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=9)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspecies, no_u, na_u
  character(len=1), intent(out) :: label(20,nspecies)
  real(dp), intent(out) :: zval(nspecies)
  integer, intent(out) :: no(nspecies)
  integer, intent(out) :: isa(na_u)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspecies, no_u, na_u
!f2py intent(out) :: label, zval, no, isa

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, io

  ! Local readables
  integer :: lno_u, lnspecies, lno_s, lna_u, lnspin, lmaxnh
  logical :: lGamma

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) lno_u , lno_s, lnspin, lmaxnh
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  call iostat_update(ierr)

! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)

! Read out indxuo
  if (.not. lGamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) ! numh
  call iostat_update(ierr)

! Read listh
  do ih = 1 , no_u
    read(iu, iostat=ierr) !listh
    call iostat_update(ierr)
  end do

! Read Hamiltonian
  do is = 1 , lnspin
    do ih = 1 , no_u
      read(iu, iostat=ierr) ! H
      call iostat_update(ierr)
    end do
  end do

! Read Overlap matrix
  do ih = 1,no_u
    read(iu, iostat=ierr) ! S
    call iostat_update(ierr)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

! Read xij
  do ih = 1 , no_u
    read(iu, iostat=ierr) ! xij
    call iostat_update(ierr)
  end do

  ! Now read in the geometry information
  read(iu, iostat=ierr) lnspecies
  if ( lnspecies /= nspecies ) stop 'Error in reading data, not allocated, nspecies'
  call iostat_update(ierr)

  read(iu, iostat=ierr) (label(1:20,is), zval(is),no(is),is=1,nspecies)
  call iostat_update(ierr)
  do is = 1, nspecies
    do io = 1 , no(is)
      read(iu, iostat=ierr) !n(is,io), l(is,io), zeta(is,io)
      call iostat_update(ierr)
    end do
  end do

  read(iu, iostat=ierr) lna_u
  if ( lna_u /= na_u ) stop 'Error in reading data, not allocated, na_u'
  call iostat_update(ierr)
  read(iu, iostat=ierr) isa
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_species0

subroutine read_hsx_species1(fname, nspecies, no_u, na_u, label, zval, no, isa)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspecies, no_u, na_u
  character(len=1), intent(out) :: label(20,nspecies)
  real(dp), intent(out) :: zval(nspecies)
  integer, intent(out) :: no(nspecies)
  integer, intent(out) :: isa(na_u)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspecies, no_u, na_u
!f2py intent(out) :: label, zval, no, isa

! Internal variables and arrays
  integer :: iu, ierr, version
  integer :: is

  ! Local readables
  integer :: lno_u, lnspecies, lna_u, lnspin, nsc(3)

  integer, allocatable :: isc(:,:)
  real(dp), allocatable :: xa(:,:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) version
  call iostat_update(ierr)
  if ( version /= 1 ) then
    call iostat_update(-3)
    return
  end if

  read(iu, iostat=ierr) !is_dp
  call iostat_update(ierr)

  read(iu, iostat=ierr) lna_u, lno_u, lnspin, lnspecies, nsc
  call iostat_update(ierr)
  if ( lnspecies /= nspecies ) call iostat_update(-6)
  if ( lno_u /= no_u ) call iostat_update(-6)

  read(iu, iostat=ierr) !ucell, Ef, qtot, temp
  call iostat_update(ierr)

  allocate(isc(3,product(nsc)), xa(3, lna_u))

  read(iu, iostat=ierr) isc, xa, isa!, lasto
  call iostat_update(ierr)
  deallocate(isc, xa)

  read(iu, iostat=ierr) (label(1:20,is), zval(is),no(is),is=1,nspecies)
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_species1

subroutine read_hsx_specie(fname, ispecie, no_specie, n_specie, l_specie, zeta_specie)

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispecie, no_specie
  integer, intent(out) :: n_specie(no_specie), l_specie(no_specie), zeta_specie(no_specie)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: ispecie, no_specie
!f2py intent(out) :: n_specie, l_specie, zeta_specie

! Internal variables and arrays
  integer :: version

  call read_hsx_version(fname, version)

  if ( version == 0 ) then ! old

    call read_hsx_specie0(fname, ispecie, no_specie, n_specie, l_specie, zeta_specie)

  else if ( version == 1 ) then

    call read_hsx_specie1(fname, ispecie, no_specie, n_specie, l_specie, zeta_specie)

  end if

end subroutine read_hsx_specie

subroutine read_hsx_specie0(fname, ispecie, no_specie, n_specie, l_specie, zeta_specie)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispecie, no_specie
  integer, intent(out) :: n_specie(no_specie), l_specie(no_specie), zeta_specie(no_specie)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: ispecie, no_specie
!f2py intent(out) :: n_specie, l_specie, zeta_specie

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, io

  ! Local readables
  logical :: lGamma
  integer :: lnspecies, lno_u, lno_s, lnspin
  character(len=20), allocatable :: label(:)
  real(dp), allocatable :: zval(:)
  integer, allocatable :: no(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) lno_u, lno_s, lnspin !, lmaxnh
  call iostat_update(ierr)

! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)

! Read out indxuo
  if (.not. lGamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) ! numh
  call iostat_update(ierr)

! Read listh
  do ih = 1 , lno_u
    read(iu, iostat=ierr) !listh
    call iostat_update(ierr)
  end do

! Read Hamiltonian
  do is = 1 , lnspin
    do ih = 1 , lno_u
      read(iu, iostat=ierr) ! H
      call iostat_update(ierr)
    end do
  end do

! Read Overlap matrix
  do ih = 1, lno_u
    read(iu, iostat=ierr) ! S
    call iostat_update(ierr)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

! Read xij
  do ih = 1 , lno_u
    read(iu, iostat=ierr) ! xij
    call iostat_update(ierr)
  end do

  ! Now read in the geometry information
  read(iu, iostat=ierr) lnspecies
  if ( ispecie < 1 ) call iostat_update(-6)
  if ( lnspecies < ispecie ) call iostat_update(-6)
  call iostat_update(ierr)
  allocate(label(lnspecies))
  allocate(zval(lnspecies))
  allocate(no(lnspecies))

  read(iu, iostat=ierr) (label(is), zval(is),no(is),is=1,lnspecies)
  call iostat_update(ierr)
  do is = 1, lnspecies
    if ( is == ispecie ) then
      do io = 1 , no(is)
        read(iu, iostat=ierr) n_specie(io), l_specie(io), zeta_specie(io)
        call iostat_update(ierr)
      end do
    else
      do io = 1 , no(is)
        read(iu, iostat=ierr) !n(is,io), l(is,io), zeta(is,io)
        call iostat_update(ierr)
      end do
    end if
  end do

  deallocate(label, zval, no)

  call close_file(iu)

end subroutine read_hsx_specie0

subroutine read_hsx_specie1(fname, ispecie, no_specie, n_specie, l_specie, zeta_specie)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispecie, no_specie
  integer, intent(out) :: n_specie(no_specie), l_specie(no_specie), zeta_specie(no_specie)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: ispecie, no_specie
!f2py intent(out) :: n_specie, l_specie, zeta_specie

! Internal variables and arrays
  integer :: iu, ierr, version
  integer :: is, io

  ! Local readables
  integer :: nspecies, lna_u, lno_u, lnspin, nsc(3)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) version
  call iostat_update(ierr)
  if ( version /= 1 ) then
    call iostat_update(-3)
    return
  end if

  read(iu, iostat=ierr) !is_dp
  call iostat_update(ierr)

  ! Read overall data
  read(iu, iostat=ierr) lna_u, lno_u, lnspin, nspecies, nsc
  if ( ispecie < 1 ) call iostat_update(-6)
  if ( nspecies < ispecie ) call iostat_update(-6)

  call iostat_update(ierr)

  read(iu, iostat=ierr) !ucell, Ef, qtot, temp
  call iostat_update(ierr)

  read(iu, iostat=ierr) !isc, xa, isa, lasto
  call iostat_update(ierr)

  read(iu, iostat=ierr) !(label(is), zval(is),no(is),is=1,lnspecies)
  call iostat_update(ierr)

  do is = 1, nspecies
    if ( is == ispecie ) then
      read(iu, iostat=ierr) (n_specie(io), l_specie(io), zeta_specie(io), io=1,no_specie)
      call iostat_update(ierr)
    else
      read(iu, iostat=ierr) !n(is,io), l(is,io), zeta(is,io)
      call iostat_update(ierr)
    end if
  end do

  call close_file(iu)

end subroutine read_hsx_specie1
