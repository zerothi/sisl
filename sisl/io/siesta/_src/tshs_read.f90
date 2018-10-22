subroutine read_tshs_version(fname, version)

  implicit none
  
  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: version

  ! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: version

  integer :: iu, err
  integer :: tmp(5)

  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')
  read(iu, iostat=err) tmp
  if ( err /= 0 ) then
     ! we have a version
     rewind(iu)
     read(iu) version
  else
     version = 0
  end if

  close(iu)
  
end subroutine read_tshs_version

subroutine read_tshs_sizes(fname, nspin, na_u, no_u, n_s, nnz)

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: nspin, na_u, no_u, n_s, nnz

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: nspin, na_u, no_u, n_s, nnz

! Internal variables and arrays
  integer :: iu
  integer :: version, tmp(5)

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then
     
     nspin = 0
     na_u = 0
     no_u = 0
     n_s = 0
     nnz = 0
     
     return
     
  end if
  
  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')
  read(iu) ! version
  ! Read the sizes
  !na_u, no_u, no_s, nspin, n_nzsg
  read(iu) tmp

  ! Copy the readed variables
  nspin = tmp(4)
  na_u = tmp(1)
  no_u = tmp(2)
  n_s = tmp(3) / tmp(2)
  nnz = tmp(5)
  
  close(iu)

end subroutine read_tshs_sizes

subroutine read_tshs_cell(fname, n_s, nsc, cell, isc)

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: n_s
  integer, intent(out) :: nsc(3)
  real(dp), intent(out) :: cell(3,3)
  integer, intent(out) :: isc(3,n_s)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: n_s
!f2py intent(out) :: nsc, cell, isc

! Internal variables and arrays
  integer :: iu, i, is
  integer :: version, tmp(5)
  logical :: Gamma, TSGamma, onlyS
  
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
  read(iu) Gamma, TSGamma, onlyS
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
  if ( .not. onlyS ) then
    ! Hamiltonian matrix
    do is = 1, tmp(4)
      do i = 1 , tmp(2)
        read(iu) ! H
      end do
    end do
  end if
  if ( .not. Gamma ) then
    read(iu) isc
  end if
  
  close(iu)
  
end subroutine read_tshs_cell

subroutine read_tshs_geom(fname, na_u, xa, lasto)

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: na_u
  real(dp), intent(out) :: xa(3,na_u)
  integer, intent(out) :: lasto(0:na_u)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: na_u
!f2py intent(out) :: xa, lasto

! Internal variables and arrays
  integer :: iu
  integer :: version, tmp(5)
  real(dp) :: cell(3,3)

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then
     
     xa = 0._dp
     cell = 0._dp
     
     return
     
  end if

  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')
  read(iu) ! version
  ! Now we may read the sizes
  read(iu) tmp

  ! Read the stuff...
  read(iu) ! nsc
  read(iu) cell, xa
  xa = xa * Ang
  read(iu) ! Gamma, TSGamma, onlyS
  read(iu) ! kscell, kdispl
  read(iu) ! Ef, Qtot, Temp
  read(iu) ! istep, ia1
  read(iu) lasto(0:na_u)

  close(iu)

end subroutine read_tshs_geom

subroutine read_tshs_hs(fname, nspin, no_u, nnz, ncol, list_col, H, S)

  implicit none
  
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, no_u, nnz
  integer, intent(out) :: ncol(no_u), list_col(nnz)
  real(dp), intent(out) :: H(nnz, nspin), S(nnz)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: nspin, no_u, nnz
!f2py intent(out) :: ncol, list_col
!f2py intent(out) :: H, S

! Internal variables and arrays
  integer :: iu, i, is, idx
  integer :: version, tmp(5)
  real(dp) :: Ef
  logical :: Gamma, TSGamma, onlyS

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
  read(iu) Gamma, TSGamma, onlyS
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
  if ( onlyS ) then
    H(:,:) = 0._dp
  else
    do is = 1, tmp(4)
      idx = 0
      do i = 1 , tmp(2)
        read(iu) H(idx+1:idx+ncol(i),is)
        idx = idx + ncol(i)
      end do
      ! Move to Ef = 0
      if ( is <= 2 ) then
        H(:,is) = H(:,is) - Ef * S(:)
      end if
      ! Change to eV
      H(:,is) = H(:,is) * eV
    end do
  end if

  close(iu)

end subroutine read_tshs_hs

subroutine read_tshs_s(fname, no_u, nnz, ncol, list_col, S)

  implicit none

  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp
  real(dp), parameter :: Ang = 0.529177_dp

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: no_u, nnz
  integer, intent(out) :: ncol(no_u), list_col(nnz)
  real(dp), intent(out) :: S(nnz)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in)  :: no_u, nnz
!f2py intent(out) :: ncol, list_col
!f2py intent(out) :: S

! Internal variables and arrays
  integer :: iu, i, idx
  integer :: version, tmp(5)

  call read_tshs_version(fname, version)

  if ( version /= 1 ) then

     ncol = -1
     list_col = -1
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
  read(iu) ! Gamma, TSGamma, onlyS
  read(iu) ! kscell, kdispl
  read(iu) ! Ef, Qtot, Temp
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

  close(iu)

end subroutine read_tshs_s

