subroutine read_hsx( fname, Gamma, no_u, no_s, nspin, maxnh, &
     numh, listhptr, listh, H, S, xij)
  
  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(sp), parameter :: eV = 13.60580_sp
  real(sp), parameter :: Ang = 0.529177_sp

  ! Input parameters
  character(len=*) :: fname
  logical :: Gamma
  integer :: no_u, no_s, nspin, maxnh
  integer :: listh(maxnh), numh(no_u), listhptr(no_u)
  real(sp) :: H(maxnh,nspin), S(maxnh), xij(3,maxnh)
! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: Gamma, no_u, no_s, nspin, maxnh
!f2py integer, intent(out), dimension(no_u) :: numh, listhptr
!f2py integer, intent(out), dimension(maxnh) :: listh
!f2py real*4, intent(out), dimension(maxnh) :: S
!f2py real*4, intent(out), dimension(maxnh,nspin) :: H
!f2py real*4, intent(out), dimension(3,maxnh) :: xij

! Internal variables and arrays
  integer :: iu
  integer :: is, ih, im

  ! Local readables
  logical :: lGamma
  integer :: lno_s, lno_u, lnspin, lmaxnh

  real(sp), allocatable :: buf(:)

! Open file
  iu = 1850
  open( iu, file=fname, form='unformatted', status='unknown' )      
  
! Read overall data
  read(iu) lno_u, lno_s, lnspin, lmaxnh
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lno_s /= no_s ) stop 'Error in reading data, not allocated, no_s'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmaxnh /= maxnh ) stop 'Error in reading data, not allocated, maxnh'

! Read logical
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

  allocate(buf(maxval(numh)*3))

! Read listh
  do ih = 1 , no_u
     im = numh(ih)
     read(iu) listh(listhptr(ih)+1:listhptr(ih)+im)
  end do

! Read Hamiltonian
  do is = 1 , nspin
     do ih = 1 , no_u
        im = numh(ih)
        read(iu) buf(1:im)
        H(listhptr(ih)+1:listhptr(ih)+im,is) = buf(1:im) * eV
     end do
  end do

! Read Overlap matrix
  do ih = 1,no_u
     im = numh(ih)
     read(iu) buf(1:im)
     S(listhptr(ih)+1:listhptr(ih)+im) = buf(1:im)
  end do

  read(iu) !Qtot,temp

  if ( .not. Gamma ) then
     do ih = 1 , no_u
        im = numh(ih)
        read(iu) buf(1:im*3)
        xij(1:3,listhptr(ih)+1:listhptr(ih)+im) = reshape(buf(1:im*3),(/3,im/)) * Ang
     end do
  end if

  deallocate(buf)

  close(iu)

end subroutine read_hsx
