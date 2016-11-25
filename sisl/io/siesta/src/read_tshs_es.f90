subroutine read_tshs_es(fname, nspin, no_u, nnz, ncol, list_col, H, S)

  implicit none
  
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*) :: fname
  integer :: nspin, no_u, nnz
  integer :: ncol(no_u), list_col(nnz)
  real(dp) :: H(nnz, nspin), S(nnz)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: nspin
!f2py intent(in)  :: no_u
!f2py intent(in)  :: nnz
!f2py intent(out) :: ncol
!f2py intent(out) :: list_col
!f2py intent(out) :: H
!f2py intent(out) :: S

! Internal variables and arrays
  integer :: iu, i, is, idx
  integer :: version, tmp(5)
  real(dp) :: Ef
  logical :: Gamma

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then
     
     ncol = -1
     list_col = -1
     H = 0._dp
     S = 0._dp
     
     return
     
  end if

  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')
  read(iu) ! version
  ! Now we may read the sizes
  read(iu) tmp

  ! Read the stuff...
  read(iu) ! nsc
  read(iu) ! cell, xa
  read(iu) Gamma ! TSGamma, onlyS
  read(iu) ! kscell, kdispl
  read(iu) Ef ! Qtot, Temp
  read(iu) ! istep, ia1
  read(iu) ! lasto

  ! Sparse pattern
  read(iu) ncol
  idx = 0
  do i = 1 , tmp(2)
     read(iu) list_col(idx+1:idx+ncol(i))
     idx = idx + ncol(i)
  end do
  ! Overlap matrix
  idx = 0
  do i = 1 , tmp(2)
     read(iu) S(idx+1:idx+ncol(i))
     idx = idx + ncol(i)
  end do
  ! Hamiltonian matrix
  do is = 1, tmp(4)
     idx = 0
     do i = 1 , tmp(2)
        read(iu) H(idx+1:idx+ncol(i),is)
        idx = idx + ncol(i)
     end do
     ! Move to Ef = 0
     H(:,is) = H(:,is) - Ef * S(:)
     ! Change to eV
     H(:,is) = H(:,is) * eV
  end do

  close(iu)

end subroutine read_tshs_es
