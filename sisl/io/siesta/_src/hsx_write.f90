subroutine write_hsx(fname, Gamma, no_u, no_s, nspin, maxnh, &
    numh, listhptr, listh, H, S, xij, Qtot, temp)
  use io_m, only: open_file
  use io_m, only: iostat_update

  implicit none

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
  real(dp) :: H(maxnh,nspin), S(maxnh), xij(3,maxnh), Qtot, temp

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
      write(iu, iostat=ierr) (real(H(listhptr(ih)+im,is)/eV,kind=sp),im=1,numh(ih))
      call iostat_update(ierr)
    end do
  end do

! Write Overlap matrix
  do ih = 1,no_u
    write(iu, iostat=ierr) (real(S(listhptr(ih)+im),kind=sp),im = 1,numh(ih))
    call iostat_update(ierr)
  end do

  write(iu, iostat=ierr) Qtot,temp
  call iostat_update(ierr)

  do ih = 1 , no_u
    write(iu, iostat=ierr) ((real(xij(k,listhptr(ih)+im)/Ang,kind=sp), k=1,3),im =1,numh(ih))
    call iostat_update(ierr)
  end do

  close(iu)

end subroutine write_hsx

