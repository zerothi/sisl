! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine read_tsde_sizes(fname, nspin, no_u, nsc, nnz)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: no_u, nspin, nsc(3), nnz

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: no_u, nspin, nsc, nnz

! Internal variables and arrays
  integer :: iu, ierr
  integer, allocatable :: num_col(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! First try and see if nsc is present
  read(iu, iostat=ierr) no_u, nspin, nsc
  if ( ierr /= 0 ) then
    rewind(iu)
    read(iu, iostat=ierr) no_u, nspin
    nsc(:) = 0
  end if
  call iostat_update(ierr)

  allocate(num_col(no_u))
  read(iu, iostat=ierr) num_col
  call iostat_update(ierr)
  nnz = sum(num_col)
  deallocate(num_col)

  call close_file(iu)

end subroutine read_tsde_sizes

subroutine read_tsde_dm(fname, nspin, no_u, nsc, nnz, &
    ncol, list_col, DM)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: no_u, nspin, nsc(3), nnz
  integer, intent(out) :: ncol(no_u), list_col(nnz)
  real(dp), intent(out) :: DM(nnz,nspin)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in) :: no_u, nspin, nsc, nnz
!f2py intent(out) :: ncol, list_col
!f2py intent(out) :: DM

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, io, n

  ! Local readables
  integer :: lno_u, lnspin, lnsc(3)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! First try and see if nsc is present
  read(iu,iostat=ierr) lno_u, lnspin, lnsc
  if ( ierr /= 0 ) then
    rewind(iu)
    read(iu, iostat=ierr) lno_u, lnspin
    lnsc(:) = 0
  end if
  call iostat_update(ierr)
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( any(lnsc /= nsc) ) stop 'Error in reading data, not allocated, nsc'

  read(iu, iostat=ierr) ncol
  call iostat_update(ierr)
  if ( nnz /= sum(ncol) ) stop 'Error in reading data, not allocated, nnz'

  ! Read list_col
  n = 0
  do io = 1 , no_u
    read(iu, iostat=ierr) list_col(n+1:n+ncol(io))
    call iostat_update(ierr)
    n = n + ncol(io)
  end do

  ! Read Density matrix
  do is = 1 , nspin
    n = 0
    do io = 1 , no_u
      read(iu, iostat=ierr) DM(n+1:n+ncol(io), is)
      call iostat_update(ierr)
      n = n + ncol(io)
    end do
  end do

  call close_file(iu)

end subroutine read_tsde_dm

subroutine read_tsde_ef(fname, Ef)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  real(dp), intent(out) :: Ef

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Ef

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, io

  ! Local readables
  integer :: lno_u, lnspin

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! First try and see if nsc is present
  read(iu,iostat=ierr) lno_u, lnspin
  call iostat_update(ierr)

  ! Skip ncol
  read(iu, iostat=ierr) ! ncol
  call iostat_update(ierr)

  ! Skip list_col
  do io = 1 , lno_u
    read(iu, iostat=ierr) ! list_col(n+1:n+ncol(io))
    call iostat_update(ierr)
  end do

  ! Skip density matrix and energy density matrix
  do is = 1 , lnspin * 2
    do io = 1 , lno_u
      read(iu, iostat=ierr) ! DM(n+1:n+ncol(io), is)
      call iostat_update(ierr)
    end do
  end do

  read(iu, iostat=ierr) Ef
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_tsde_ef

subroutine read_tsde_edm(fname, nspin, no_u, nsc, nnz, &
    ncol, list_col, EDM)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: no_u, nspin, nsc(3), nnz
  integer, intent(out) :: ncol(no_u), list_col(nnz)
  real(dp), intent(out) :: EDM(nnz,nspin)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in) :: no_u, nspin, nsc, nnz
!f2py intent(out) :: ncol, list_col
!f2py intent(out) :: EDM

  ! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, io, n

  ! Local readables
  integer :: lno_u, lnspin, lnsc(3)

  real(dp) :: Ef
  real(dp), allocatable :: DM(:,:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! First try and see if nsc is present
  read(iu, iostat=ierr) lno_u, lnspin, lnsc
  if ( ierr /= 0 ) then
    rewind(iu)
    read(iu, iostat=ierr) lno_u, lnspin
    lnsc(:) = 0
  end if
  call iostat_update(ierr)
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( any(lnsc /= nsc) ) stop 'Error in reading data, not allocated, nsc'

  read(iu) ncol
  call iostat_update(ierr)
  if ( nnz /= sum(ncol) ) stop 'Error in reading data, not allocated, nnz'

  allocate(DM(nnz, nspin))

  ! Read list_col
  n = 0
  do io = 1 , no_u
    read(iu, iostat=ierr) list_col(n+1:n+ncol(io))
    call iostat_update(ierr)
    n = n + ncol(io)
  end do

  ! Skip density matrix
  do is = 1 , nspin
    n = 0
    do io = 1 , no_u
      read(iu, iostat=ierr) DM(n+1:n+ncol(io), is)
      call iostat_update(ierr)
      n = n + ncol(io)
    end do
  end do

  ! Read energy density matrix
  do is = 1 , nspin
    n = 0
    do io = 1 , no_u
      read(iu, iostat=ierr) EDM(n+1:n+ncol(io), is)
      call iostat_update(ierr)
      n = n + ncol(io)
    end do
  end do

  ! Read Fermi energy
  read(iu, iostat=ierr) Ef
  call iostat_update(ierr)

  do is = 1 , nspin
    do io = 1 , nnz
      EDM(io,is) = EDM(io,is) - Ef * DM(io, is)
    end do
  end do

  ! Clean-up
  deallocate(DM)

  call close_file(iu)

end subroutine read_tsde_edm
