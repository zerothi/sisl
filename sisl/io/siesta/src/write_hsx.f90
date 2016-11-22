subroutine write_hsx( fname, Gamma, no_u, no_s, nspin, maxnh, &
     numh, listhptr, listh, H, S, xij, Qtot, temp)
  
  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*) :: fname
  logical :: Gamma
  integer :: no_u, no_s, nspin, maxnh
  integer :: listh(maxnh), numh(no_u), listhptr(no_u)
  real(dp) :: H(maxnh,nspin), S(maxnh), xij(3,maxnh), Qtot, temp
! Define f2py intents
!f2py intent(in) :: fname, Gamma, no_u, no_s, nspin, maxnh
!f2py intent(in) :: numh, listhptr, listh, H, S, xij, Qtot, temp

! Internal variables and arrays
  integer :: iu
  integer :: is, ih, im, k
  integer :: indxuo(no_s)

! Open file
  iu = 1850 ! Constant
  open( iu, file=fname, form='unformatted', status='unknown' )      
  
! Write overall data
  write(iu) no_u, no_s, nspin, maxnh

! Write logical
  write(iu) Gamma

! Write out indxuo
  if (.not. Gamma) then
     do ih = 1 , no_s
        im = mod(ih,no_u)
        if ( im == 0 ) im = no_u
        indxuo(ih) = im
     end do
     write(iu) (indxuo(ih),ih=1,no_s)
  end if

  write(iu) (numh(ih),ih=1,no_u)

! Write listh
  do ih = 1 , no_u
     write(iu) (listh(listhptr(ih)+im),im = 1,numh(ih))
  end do

! Write Hamiltonian
  do is = 1 , nspin
     do ih = 1 , no_u
        write(iu) (real(H(listhptr(ih)+im,is),kind=sp),im=1,numh(ih))
     end do
  end do

! Write Overlap matrix
  do ih = 1,no_u
     write(iu) (real(S(listhptr(ih)+im),kind=sp),im = 1,numh(ih))
  end do

  write(iu) Qtot,temp

  do ih = 1 , no_u
     write(iu) ((real(xij(k,listhptr(ih)+im),kind=sp), k=1,3),im =1,numh(ih))
  enddo

  close(iu)
end subroutine write_hsx

