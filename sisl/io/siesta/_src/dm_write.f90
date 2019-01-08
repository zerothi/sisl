subroutine write_dm(fname, nspin, no_u, nsc, nnz, ncol, list_col, DM)
  use io_m, only: open_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspin, no_u, nsc(3), nnz
  integer, intent(in) :: ncol(no_u), list_col(nnz)
  real(dp), intent(in) :: DM(nnz,nspin)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspin, no_u, nsc, nnz
!f2py intent(in) :: ncol, list_col
!f2py intent(in) :: DM

! Internal variables and arrays
  integer :: iu, is, i, idx, ierr

  call open_file(fname, 'write', 'unknown', 'unformatted', iu)

  ! Also write the supercell.
  write(iu, iostat=ierr) no_u, nspin, nsc
  call iostat_update(ierr)

  ! Sparse pattern
  write(iu, iostat=ierr) ncol
  call iostat_update(ierr)
  idx = 0
  do i = 1 , no_u
    write(iu, iostat=ierr) list_col(idx+1:idx+ncol(i))
    call iostat_update(ierr)
    idx = idx + ncol(i)
  end do
  ! Density matrix
  do is = 1, nspin
    idx = 0
    do i = 1 , no_u
      write(iu, iostat=ierr) DM(idx+1:idx+ncol(i),is)
      call iostat_update(ierr)
      idx = idx + ncol(i)
    end do
  end do

  close(iu)

end subroutine write_dm

