subroutine read_dm_sizes(fname, nspin, no_u, nsc, nnz)
  use io_m, only: open_file
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

  close(iu)

end subroutine read_dm_sizes

subroutine read_dm(fname, nspin, no_u, nsc, nnz, ncol, list_col, DM)
  use io_m, only: open_file
  use io_m, only: iostat_reset, iostat_update

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

  close(iu)

end subroutine read_dm
