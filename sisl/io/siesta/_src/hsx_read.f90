subroutine read_hsx_sizes(fname, Gamma, nspin, no_u, no_s, maxnh)
  use io_m, only: open_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(out) :: Gamma
  integer, intent(out) :: nspin, no_u, no_s, maxnh

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, nspin, no_u, no_s, maxnh

! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

! Read overall data
  read(iu, iostat=ierr) no_u, no_s, nspin, maxnh
  call iostat_update(ierr)

! Read logical
  read(iu, iostat=ierr) Gamma
  call iostat_update(ierr)

  close(iu)

end subroutine read_hsx_sizes

subroutine read_hsx_hsx(fname, Gamma, nspin, no_u, no_s, maxnh, &
    numh, listh, H, S, xij)
  use io_m, only: open_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  real(sp), parameter :: eV = 13.60580_sp
  real(sp), parameter :: Ang = 0.529177_sp

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
  integer :: listhptr(maxnh)

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
      read(iu, iostat=ierr) buf(1:im)
      call iostat_update(ierr)
      H(listhptr(ih)+1:listhptr(ih)+im,is) = buf(1:im) * eV
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
      xij(1:3,listhptr(ih)+1:listhptr(ih)+im) = reshape(buf(1:im*3),(/3,im/)) * Ang
    end do
  end if

  deallocate(buf)

  close(iu)

end subroutine read_hsx_hsx

subroutine read_hsx_s(fname, Gamma, nspin, no_u, no_s, maxnh, &
    numh, listh, S)
  use io_m, only: open_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(in) :: Gamma
  integer, intent(in) :: nspin, no_u, no_s, maxnh
  integer, intent(out) :: numh(no_u), listh(maxnh)
  real(sp), intent(out) :: S(maxnh)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: Gamma, nspin, no_u, no_s, maxnh
!f2py intent(out) :: numh, listh
!f2py intent(out) :: S

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, im
  integer :: listhptr(maxnh)

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
  listhptr(1) = 0
  do ih = 2 , no_u
    listhptr(ih) = listhptr(ih-1) + numh(ih-1)
  end do

  allocate(buf(maxval(numh)))

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

  deallocate(buf)

  close(iu)

end subroutine read_hsx_s
