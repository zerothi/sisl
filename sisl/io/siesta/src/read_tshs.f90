subroutine read_TSHS(fname,Gamma, no_u,no_s,nspin,maxnh, &
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
  logical :: lGamma, onlyS
  integer :: lno_s, lno_u, lnspin, lmaxnh
  real(dp), allocatable :: buf(:)

  iu = 1804
  open(iu,file=trim(fname),status='old',form='unformatted')
  read(iu) is, lno_u, lno_s, lnspin, lmaxnh
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lno_s /= no_s ) stop 'Error in reading data, not allocated, no_s'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmaxnh /= maxnh ) stop 'Error in reading data, not allocated, maxnh'

  ! wrap reads
  read(iu) !xa
  read(iu) !iza
  read(iu) !ucell

  read(iu) lGamma
  if ( lGamma .neqv. Gamma ) stop 'Error in reading data, not allocated'

  read(iu) onlyS
  read(iu) !TSGamma
  read(iu) !kscell
  read(iu) !kdispl
  read(iu) !istep,ia1
  read(iu) !lasto

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

  read(iu) !qtot, temp 
  read(iu) !Ef

! Read listh
  do ih = 1 , no_u
     ip = listhptr(ih)
     im = numh(ih)
     read(iu) listh(ip+1:ip+im)
  end do

! Read overlap
  do ih = 1 , no_u
     ip = listhptr(ih)
     im = numh(ih)
     read(iu) S(ip+1:ip+im)
  end do

! Read Hamiltonian
  if ( .not. onlyS ) then
     do is = 1 , nspin
        do ih = 1 , no_u
           ip = listhptr(ih)
           im = numh(ih)
           read(iu) H(ip+1:ip+im,is)
           H(ip+1:ip+im,is) = H(ip+1:ip+im,is) * eV
        end do
     end do
  end if

! Read xij
  if ( .not. Gamma ) then
     allocate(buf(maxval(numh)*3))
     do ih = 1 , no_u
        ip = listhptr(ih)
        im = numh(ih)
        read(iu) buf(1:im),buf(im+1:im*2),buf(im*2+1:im*3)
        xij(1,ip+1:ip+im) = buf(1:im) * Ang
        xij(2,ip+1:ip+im) = buf(im+1:im*2) * Ang
        xij(3,ip+1:ip+im) = buf(im*2+1:im*3) * Ang
     end do
     deallocate(buf)
  end if

  close(iu)

end subroutine read_tshs
