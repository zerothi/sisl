subroutine read_tshs_cell(fname, n_s, nsc, cell, isc)

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*) :: fname
  integer :: n_s
  integer :: nsc(3)
  real(dp) :: cell(3,3)
  integer :: isc(3,n_s)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: n_s
!f2py intent(out) :: nsc
!f2py intent(out) :: cell
!f2py intent(out) :: isc

! Internal variables and arrays
  integer :: iu, i, is
  integer :: version, tmp(5)
  logical :: Gamma
  
  call read_tshs_version(fname, version)

  if ( version /= 1 ) then

     nsc = 0
     cell = 0._dp
     isc = 0
     
     return
     
  end if

  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')
  read(iu) ! version
  ! Now we may read the sizes
  read(iu) tmp

  ! Read the stuff...
  read(iu) nsc
  read(iu) cell ! xa
  cell = cell * Ang
  read(iu) Gamma ! TSGamma, onlyS
  read(iu) ! kscell, kdispl
  read(iu) ! Ef, Qtot, Temp
  read(iu) ! istep, ia1
  read(iu) ! lasto

  ! Sparse pattern
  read(iu) ! ncol
  do i = 1 , tmp(2)
     read(iu) ! list_col
  end do
  ! Overlap matrix
  do i = 1 , tmp(2)
     read(iu) ! S
  end do
  ! Hamiltonian matrix
  do is = 1, tmp(4)
     do i = 1 , tmp(2)
        read(iu) ! H
     end do
  end do
  if ( .not. Gamma ) then
     read(iu) isc
  end if
  
  close(iu)

end subroutine read_tshs_cell
