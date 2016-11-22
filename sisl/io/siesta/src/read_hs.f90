subroutine read_hs(fname, Gamma, nspin, no_u,no_s,maxnh, &
     numh,listhptr,listh,H,S,xij)

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*) :: fname
  logical :: Gamma
  integer :: no_u, no_s, nspin, maxnh
  integer :: listh(maxnh), numh(no_u), listhptr(no_u)
  real(dp) :: H(maxnh,nspin), S(maxnh), xij(3,maxnh)
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in) :: Gamma, no_u, no_s, nspin, maxnh
!f2py integer, intent(out), dimension(no_u)  :: numh, listhptr
!f2py integer, intent(out), dimension(maxnh) :: listh
!f2py real*8, intent(out), dimension(maxnh) :: S
!f2py real*8, intent(out), dimension(maxnh,nspin) :: H
!f2py real*8, intent(out), dimension(3,maxnh) :: xij

! Internal variables and arrays
  integer :: iu
  integer :: is, ih, im, ip

  ! Local readables
  logical :: lGamma
  integer :: lno_s, lno_u, lnspin, lmaxnh

  iu = 1804
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) lno_u, lno_s, lnspin, lmaxnh
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lno_s /= no_s ) stop 'Error in reading data, not allocated, no_s'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmaxnh /= maxnh ) stop 'Error in reading data, not allocated, maxnh'

  read(iu) lGamma
  if ( lGamma .neqv. Gamma ) stop 'Error in reading data, not allocated'

! Read out indxuo
  if (.not. Gamma) then
     read(iu) ! indxuo
  end if

  read(iu) numh

! Create listhptr
  listhptr(1) = 0
  do ih = 2 , no_u
     listhptr(ih) = listhptr(ih-1) + numh(ih-1)
  end do

! Read listh
  do ih = 1 , no_u
     ip = listhptr(ih)
     do im = 1 , numh(ih)
        read(iu) listh(ip+im)
     end do
  end do

! Read Hamiltonian
  do is = 1 , nspin
     do ih = 1 , no_u
        ip = listhptr(ih)
        do im = 1 , numh(ih)
           read(iu) H(ip+im,is)
           H(ip+im,is) = H(ip+im,is) * eV
        end do
     end do
  end do

! Read Hamiltonian
  do ih = 1 , no_u
     ip = listhptr(ih)
     do im = 1 , numh(ih)
        read(iu) S(ip+im)
     end do
  end do

  read(iu) !qtot, temp 

! Read xij
  if ( .not. Gamma ) then
     do ih = 1 , no_u
        ip = listhptr(ih)
        do im = 1 , numh(ih)
           read(iu) xij(1:3,ip + im)
           xij(1:3,ip + im) = xij(1:3,ip + im) * Ang
        end do
     end do
  end if

  close(iu)

end subroutine read_hs
