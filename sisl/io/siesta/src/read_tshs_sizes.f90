subroutine read_tshs_sizes(fname, nspin, na_u, no_u, n_s, nnz)

  implicit none

  ! Input parameters
  character(len=*) :: fname
  integer :: nspin
  integer :: na_u
  integer :: no_u
  integer :: n_s
  integer :: nnz

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: nspin
!f2py intent(out) :: na_u
!f2py intent(out) :: no_u
!f2py intent(out) :: n_s
!f2py intent(out) :: nnz

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
