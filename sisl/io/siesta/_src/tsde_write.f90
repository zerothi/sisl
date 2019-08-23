subroutine write_tsde_dm_edm(fname, nspin, no_u, nsc, nnz, &
    ncol, list_col, DM, EDM, Ef)
  use io_m, only: open_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)
  integer, parameter :: dp = selected_real_kind(p=15)
  real(dp), parameter :: eV = 13.60580_dp

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: no_u, nspin, nsc(3), nnz
  integer, intent(in) :: ncol(no_u), list_col(nnz)
  real(dp), intent(in) :: DM(nnz,nspin), EDM(nnz,nspin)
  real(dp), intent(in) :: Ef

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in) :: no_u, nspin, nsc, nnz
!f2py intent(in) :: ncol, list_col
!f2py intent(in) :: DM, EDM, Ef

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, io, n

  call open_file(fname, 'write', 'unknown', 'unformatted', iu)

  ! First try and see if nsc is present
  write(iu,iostat=ierr) no_u, nspin, nsc
  call iostat_update(ierr)

  write(iu, iostat=ierr) ncol
  call iostat_update(ierr)

  ! Write list_col
  n = 0
  do io = 1 , no_u
    write(iu, iostat=ierr) list_col(n+1:n+ncol(io))
    call iostat_update(ierr)
    n = n + ncol(io)
  end do

  ! Write Density matrix
  do is = 1 , nspin
    n = 0
    do io = 1 , no_u
      write(iu, iostat=ierr) DM(n+1:n+ncol(io), is)
      call iostat_update(ierr)
      n = n + ncol(io)
    end do
  end do

  ! Write energy density matrix
  do is = 1 , nspin
    n = 0
    do io = 1 , no_u
      write(iu, iostat=ierr) EDM(n+1:n+ncol(io), is) / eV
      call iostat_update(ierr)
      n = n + ncol(io)
    end do
  end do

  ! Write Fermi-level
  write(iu, iostat=ierr) Ef / eV
  call iostat_update(ierr)

  close(iu)

end subroutine write_tsde_dm_edm
