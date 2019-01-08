subroutine read_hs_header(fname, Gamma, nspin, no_u, no_s, maxnh)
  use io_m, only: open_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(out) :: Gamma
  integer, intent(out) :: no_u, no_s, nspin, maxnh

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, no_u, no_s, nspin, maxnh

! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) no_u, no_s, nspin, maxnh
  call iostat_update(ierr)

  read(iu, iostat=ierr) Gamma
  call iostat_update(ierr)

  close(iu)

end subroutine read_hs_header

subroutine read_hs(fname, Gamma, nspin, no_u,no_s,maxnh, &
    numh,listhptr,listh,H,S,xij)
  use io_m, only: open_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(in) :: Gamma
  integer, intent(in) :: no_u, no_s, nspin, maxnh
  integer, intent(out) :: listh(maxnh), numh(no_u), listhptr(no_u)
  real(dp), intent(out) :: H(maxnh,nspin), S(maxnh), xij(3,maxnh)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in) :: Gamma, no_u, no_s, nspin, maxnh
!f2py intent(out) :: numh, listhptr, listh
!f2py intent(out) :: H, S, xij

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, im, ip

  ! Local readables
  logical :: lGamma
  integer :: lno_s, lno_u, lnspin, lmaxnh

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  read(iu, iostat=ierr) lno_u, lno_s, lnspin, lmaxnh
  call iostat_update(ierr)
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lno_s /= no_s ) stop 'Error in reading data, not allocated, no_s'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmaxnh /= maxnh ) stop 'Error in reading data, not allocated, maxnh'

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

! Read listh
  do ih = 1 , no_u
    ip = listhptr(ih)
    do im = 1 , numh(ih)
      read(iu, iostat=ierr) listh(ip+im)
      call iostat_update(ierr)
    end do
  end do

! Read Hamiltonian
  do is = 1 , nspin
    do ih = 1 , no_u
      ip = listhptr(ih)
      do im = 1 , numh(ih)
        read(iu, iostat=ierr) H(ip+im,is)
        call iostat_update(ierr)
        H(ip+im,is) = H(ip+im,is) * eV
      end do
    end do
  end do

! Read Hamiltonian
  do ih = 1 , no_u
    ip = listhptr(ih)
    do im = 1 , numh(ih)
      read(iu, iostat=ierr) S(ip+im)
      call iostat_update(ierr)
    end do
  end do

  read(iu, iostat=ierr) !qtot, temp 
  call iostat_update(ierr)

! Read xij
  if ( .not. Gamma ) then
    do ih = 1 , no_u
      ip = listhptr(ih)
      do im = 1 , numh(ih)
        read(iu, iostat=ierr) xij(1:3,ip + im)
        call iostat_update(ierr)
        xij(1:3,ip + im) = xij(1:3,ip + im) * Ang
      end do
    end do
  end if

  close(iu)

end subroutine read_hs
