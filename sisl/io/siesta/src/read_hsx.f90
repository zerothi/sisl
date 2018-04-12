subroutine read_hsx_sizes(fname,Gamma,nspin,no_u,no_s,maxnh)
  
  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(out) :: Gamma
  integer, intent(out) :: nspin, no_u, no_s, maxnh
  
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, nspin, no_u, no_s, maxnh

! Internal variables and arrays
  integer :: iu

  ! Open file
  call free_unit(iu)
  open( iu, file=trim(fname), form='unformatted', status='unknown' )
  
! Read overall data
  read(iu) no_u, no_s, nspin, maxnh

! Read logical
  read(iu) Gamma

  close(iu)
  
end subroutine read_hsx_sizes

subroutine read_hsx_hsx( fname, Gamma, nspin, no_u, no_s, maxnh, &
     numh, listh, H, S, xij)
  
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
  integer :: iu
  integer :: is, ih, im
  integer :: listhptr(maxnh)

  ! Local readables
  logical :: lGamma
  integer :: lno_s, lno_u, lnspin, lmaxnh

  real(sp), allocatable :: buf(:)

! Open file
  call free_unit(iu)
  open( iu, file=trim(fname), form='unformatted', status='unknown' )
  
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

  if ( Gamma ) then
     xij = 0._sp
  else
     do ih = 1 , no_u
        im = numh(ih)
        read(iu) buf(1:im*3)
        xij(1:3,listhptr(ih)+1:listhptr(ih)+im) = reshape(buf(1:im*3),(/3,im/)) * Ang
     end do
  end if

  deallocate(buf)

  close(iu)

end subroutine read_hsx_hsx

subroutine read_hsx_s( fname, Gamma, nspin, no_u, no_s, maxnh, &
     numh, listh, S)

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
  integer :: iu
  integer :: is, ih, im
  integer :: listhptr(maxnh)

  ! Local readables
  logical :: lGamma
  integer :: lno_s, lno_u, lnspin, lmaxnh

  real(sp), allocatable :: buf(:)

  ! Open file
  call free_unit(iu)
  open( iu, file=trim(fname), form='unformatted', status='unknown' )

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

  allocate(buf(maxval(numh)))

! Read listh
  do ih = 1 , no_u
     im = numh(ih)
     read(iu) listh(listhptr(ih)+1:listhptr(ih)+im)
  end do

! Read Hamiltonian
  do is = 1 , nspin
     do ih = 1 , no_u
        im = numh(ih)
        read(iu)
     end do
  end do

! Read Overlap matrix
  do ih = 1,no_u
     im = numh(ih)
     read(iu) buf(1:im)
     S(listhptr(ih)+1:listhptr(ih)+im) = buf(1:im)
  end do

  deallocate(buf)

  close(iu)

end subroutine read_hsx_s
