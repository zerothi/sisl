subroutine read_tshs_version(fname, version)

  implicit none
  
  ! Input parameters
  character(len=*) :: fname
  integer :: version

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
