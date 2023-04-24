! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
subroutine read_dm_sizes(fname, nspin, no_u, nsc, nnz)

  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: nspin, no_u, nsc(3), nnz

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

end subroutine read_dm_sizes

subroutine read_dm(fname, nspin, no_u, nsc, nnz, ncol, list_col, DM)

  use precision, only: i4, r8
  use sparse_io_m, only: read_sparse, read_data_2d2
  use io_m, only: open_file, close_file
  use io_m, only: iostat_reset, iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer(i4), intent(in) :: no_u, nspin, nsc(3), nnz
  integer(i4), intent(out) :: ncol(no_u), list_col(nnz)
  real(r8), intent(out) :: DM(nnz,nspin)

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in) :: no_u, nspin, nsc, nnz
!f2py intent(out) :: ncol, list_col
!f2py intent(out) :: DM

! Internal variables and arrays
  integer(i4) :: iu, ierr

  ! Local readables
  integer(i4) :: lno_u, lnspin, lnsc(3)

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

  call read_sparse(iu, no_u, nnz, ncol, list_col)
  call read_data_2d2(iu, no_u, nspin, nnz, ncol, DM)

  call close_file(iu)

end subroutine read_dm
