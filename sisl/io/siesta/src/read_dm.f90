subroutine read_dm_sizes(fname,nspin,no_u, nnz)

  implicit none
  
  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: no_u, nspin, nnz
  
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: no_u, nspin, nnz

! Internal variables and arrays
  integer :: iu
  integer, allocatable :: num_col(:)

  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) no_u, nspin
  allocate(num_col(no_u))
  read(iu) num_col
  nnz = sum(num_col)
  deallocate(num_col)

  close(iu)

end subroutine read_dm_sizes

subroutine read_dm(fname, nspin, no_u, nnz, &
     ncol,list_col,DM)

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: no_u, nspin, nnz
  integer, intent(out) :: ncol(no_u), list_col(nnz)
  real(dp), intent(out) :: DM(nnz,nspin)
  
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in) :: no_u, nspin, nnz
!f2py intent(out) :: ncol, list_col
!f2py intent(out) :: DM

! Internal variables and arrays
  integer :: iu
  integer :: is, io, n

  ! Local readables
  integer :: lno_u, lnspin

  call free_unit(iu)
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) lno_u, lnspin
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'

  read(iu) ncol
  if ( nnz /= sum(ncol) ) stop 'Error in reading data, not allocated, nnz'

  ! Read list_col
  n = 0
  do io = 1 , no_u
     read(iu) list_col(n+1:n+ncol(io))
     n = n + ncol(io)
  end do

! Read Density matrix
  do is = 1 , nspin
     n = 0
     do io = 1 , no_u
        read(iu) DM(n+1:n+ncol(io), is)
        n = n + ncol(io)
     end do
  end do

  close(iu)

end subroutine read_dm
