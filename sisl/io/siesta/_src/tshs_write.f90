subroutine write_tshs_hs(fname, &
     nspin, na_u, no_u, nnz, &
     nsc1, nsc2, nsc3, &
     cell, xa, lasto, &
     ncol, list_col, H, S, isc)
  
  implicit none
  
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, na_u, no_u, nnz
  integer, intent(in) :: nsc1, nsc2, nsc3
  real(dp), intent(in) :: cell(3,3), xa(3,na_u)
  integer, intent(in) :: lasto(0:na_u)
  integer, intent(in) :: ncol(no_u), list_col(nnz)
  real(dp), intent(in) :: H(nnz,nspin), S(nnz)
  integer, intent(in) :: isc(3,nsc1*nsc2*nsc3)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin, na_u, no_u, nnz
!f2py intent(in) :: nsc1, nsc2, nsc3
!f2py intent(in) :: cell, xa, lasto
!f2py intent(in) :: ncol, list_col
!f2py intent(in) :: H, S
!f2py intent(in) :: isc

! Internal variables and arrays
  integer :: iu, is, i, idx

  call free_unit(iu)
  open( iu, file=trim(fname), status='unknown', form='unformatted', action='write')
  ! version
  write(iu) 1
  write(iu) na_u, no_u, no_u * nsc1 * nsc2 * nsc3, nspin, nnz

  write(iu) nsc1, nsc2, nsc3
  write(iu) cell, xa
  ! TSGamma, Gamma, onlyS
  write(iu) .false., .false., .false.
  ! kgrid, kdispl
  write(iu) (/2, 0, 0, 0, 2, 0, 0, 0, 2/), (/0._dp, 0._dp, 0._dp/)
  ! Ef, qtot, Temp
  write(iu) 0._dp, 1._dp, 0.001_dp

  ! istep, ia1
  write(iu) 0, 0

  write(iu) lasto

  ! Sparse pattern
  write(iu) ncol
  idx = 0
  do i = 1 , no_u
     write(iu) list_col(idx+1:idx+ncol(i))
     idx = idx + ncol(i)
  end do
  ! Overlap matrix
  idx = 0
  do i = 1 , no_u
     write(iu) S(idx+1:idx+ncol(i))
     idx = idx + ncol(i)
  end do
  ! Hamiltonian matrix
  do is = 1, nspin
     idx = 0
     do i = 1 , no_u
        write(iu) H(idx+1:idx+ncol(i),is)
        idx = idx + ncol(i)
     end do
  end do

  write(iu) isc

  close(iu)

end subroutine write_tshs_hs

