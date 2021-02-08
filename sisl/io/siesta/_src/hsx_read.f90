subroutine read_hsx_sizes(fname, Gamma, nspin, no_u, no_s, maxnh)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(out) :: Gamma
  integer, intent(out) :: nspin, no_u, no_s, maxnh

! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: Gamma, nspin, no_u, no_s, maxnh

! Internal variables and arrays
  integer :: iu, ierr

  call open_file(fname, 'read', 'old', 'unformatted', iu)

! Read overall data
  read(iu, iostat=ierr) no_u, no_s, nspin, maxnh
  call iostat_update(ierr)

! Read logical
  read(iu, iostat=ierr) Gamma
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_sizes

subroutine read_hsx_hsx(fname, Gamma, nspin, no_u, no_s, maxnh, &
    numh, listh, H, S, xij)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(in) :: Gamma
  integer, intent(in) :: nspin, no_u, no_s, maxnh
  integer, intent(out) :: numh(no_u), listh(maxnh)
  real(sp), intent(out) :: H(maxnh,nspin), S(maxnh), xij(3,maxnh)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: Gamma, nspin, no_u, no_s, maxnh
!f2py intent(out) :: numh, listh
!f2py intent(out) :: H, S, xij

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, im
  integer, allocatable :: listhptr(:)

  ! Local readables
  logical :: lGamma
  integer :: lno_s, lno_u, lnspin, lmaxnh

  real(sp), allocatable :: buf(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

! Read overall data
  read(iu, iostat=ierr) lno_u, lno_s, lnspin, lmaxnh
  call iostat_update(ierr)
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lno_s /= no_s ) stop 'Error in reading data, not allocated, no_s'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmaxnh /= maxnh ) stop 'Error in reading data, not allocated, maxnh'

! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)
  if ( lGamma .neqv. Gamma ) stop 'Error in reading data, not allocated'

! Read out indxuo
  if (.not. Gamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) numh
  call iostat_update(ierr)

  ! Create listhptr
  allocate(listhptr(no_u))
  listhptr(1) = 0
  do ih = 2 , no_u
    listhptr(ih) = listhptr(ih-1) + numh(ih-1)
  end do

  allocate(buf(maxval(numh)*3))

! Read listh
  do ih = 1 , no_u
    im = numh(ih)
    read(iu, iostat=ierr) listh(listhptr(ih)+1:listhptr(ih)+im)
    call iostat_update(ierr)
  end do

! Read Hamiltonian
  do is = 1 , nspin
    do ih = 1 , no_u
      im = numh(ih)
      read(iu, iostat=ierr) buf(1:im)
      call iostat_update(ierr)
      H(listhptr(ih)+1:listhptr(ih)+im,is) = buf(1:im)
    end do
  end do

! Read Overlap matrix
  do ih = 1,no_u
    im = numh(ih)
    read(iu, iostat=ierr) buf(1:im)
    call iostat_update(ierr)
    S(listhptr(ih)+1:listhptr(ih)+im) = buf(1:im)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

  if ( Gamma ) then
    xij = 0._sp
  else
    do ih = 1 , no_u
      im = numh(ih)
      read(iu, iostat=ierr) buf(1:im*3)
      call iostat_update(ierr)
      xij(1:3,listhptr(ih)+1:listhptr(ih)+im) = reshape(buf(1:im*3),(/3,im/))
    end do
  end if

  deallocate(buf, listhptr)

  call close_file(iu)

end subroutine read_hsx_hsx

subroutine read_hsx_sx(fname, Gamma, nspin, no_u, no_s, maxnh, &
    numh, listh, S, xij)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  ! Precision 
  integer, parameter :: sp = selected_real_kind(p=6)

  ! Input parameters
  character(len=*), intent(in) :: fname
  logical, intent(in) :: Gamma
  integer, intent(in) :: nspin, no_u, no_s, maxnh
  integer, intent(out) :: numh(no_u), listh(maxnh)
  real(sp), intent(out) :: S(maxnh), xij(3,maxnh)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: Gamma, nspin, no_u, no_s, maxnh
!f2py intent(out) :: numh, listh
!f2py intent(out) :: S, xij

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, im
  integer, allocatable :: listhptr(:)

  ! Local readables
  logical :: lGamma
  integer :: lno_s, lno_u, lnspin, lmaxnh

  real(sp), allocatable :: buf(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) lno_u, lno_s, lnspin, lmaxnh
  call iostat_update(ierr)
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lno_s /= no_s ) stop 'Error in reading data, not allocated, no_s'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'
  if ( lmaxnh /= maxnh ) stop 'Error in reading data, not allocated, maxnh'

! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)
  if ( lGamma .neqv. Gamma ) stop 'Error in reading data, not allocated'

! Read out indxuo
  if (.not. Gamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) numh
  call iostat_update(ierr)

! Create listhptr
  allocate(listhptr(no_u))
  listhptr(1) = 0
  do ih = 2 , no_u
    listhptr(ih) = listhptr(ih-1) + numh(ih-1)
  end do

  allocate(buf(maxval(numh)*3))

! Read listh
  do ih = 1 , no_u
    im = numh(ih)
    read(iu, iostat=ierr) listh(listhptr(ih)+1:listhptr(ih)+im)
    call iostat_update(ierr)
  end do

! Read Hamiltonian
  do is = 1 , nspin
    do ih = 1 , no_u
      im = numh(ih)
      read(iu, iostat=ierr)
      call iostat_update(ierr)
    end do
  end do

! Read Overlap matrix
  do ih = 1,no_u
    im = numh(ih)
    read(iu, iostat=ierr) buf(1:im)
    call iostat_update(ierr)
    S(listhptr(ih)+1:listhptr(ih)+im) = buf(1:im)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

  if ( Gamma ) then
    xij = 0._sp
  else
    do ih = 1 , no_u
      im = numh(ih)
      read(iu, iostat=ierr) buf(1:im*3)
      call iostat_update(ierr)
      xij(1:3,listhptr(ih)+1:listhptr(ih)+im) = reshape(buf(1:im*3),(/3,im/))
    end do
  end if

  deallocate(buf, listhptr)

  call close_file(iu)

end subroutine read_hsx_sx

subroutine read_hsx_specie_sizes(fname, no_u, na_u, nspecies)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(out) :: no_u, na_u, nspecies

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(out) :: no_u, na_u, nspecies

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, io

  ! Local readables
  integer :: lno_s, lnspin, lmaxnh
  logical :: lGamma
  character(len=20), allocatable :: labelfis(:)
  real(dp), allocatable :: zvalfis(:)
  integer, allocatable :: nofis(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) no_u, lno_s, lnspin, lmaxnh
  call iostat_update(ierr)

! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)

! Read out indxuo
  if (.not. lGamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) ! numh
  call iostat_update(ierr)

! Read listh
  do ih = 1 , no_u
    read(iu, iostat=ierr) !listh
    call iostat_update(ierr)
  end do

! Read Hamiltonian
  do is = 1 , lnspin
    do ih = 1 , no_u
      read(iu, iostat=ierr) ! H
      call iostat_update(ierr)
    end do
  end do

! Read Overlap matrix
  do ih = 1,no_u
    read(iu, iostat=ierr) ! S
    call iostat_update(ierr)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

  do ih = 1 , no_u
    read(iu, iostat=ierr) ! xij
    call iostat_update(ierr)
  end do

  ! Now read in the geometry information
  read(iu, iostat=ierr) nspecies
  call iostat_update(ierr)
  allocate(labelfis(nspecies), zvalfis(nspecies), nofis(nspecies))
  read(iu, iostat=ierr) (labelfis(is), zvalfis(is),nofis(is),is=1,nspecies)

  do is = 1, nspecies
    do io = 1 , nofis(is)
      read(iu, iostat=ierr) !nfio(is,io), lfio(is,io), zetafio(is,io)
      call iostat_update(ierr)
    end do
  end do

  read(iu, iostat=ierr) na_u
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_specie_sizes

subroutine read_hsx_species(fname, nspecies, no_u, na_u, labelfis, zvalfis, nofis, isa)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: sp = selected_real_kind(p=9)
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: nspecies, no_u, na_u
  character(len=1), intent(out) :: labelfis(20,nspecies)
  real(dp), intent(out) :: zvalfis(nspecies)
  integer, intent(out) :: nofis(nspecies)
  integer, intent(out) :: isa(na_u)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: nspecies, no_u, na_u
!f2py intent(out) :: labelfis, zvalfis, nofis, isa

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, io

  ! Local readables
  integer :: lno_u, lnspecies, lno_s, lna_u, lnspin, lmaxnh
  logical :: lGamma

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) lno_u , lno_s, lnspin, lmaxnh
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  call iostat_update(ierr)

! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)

! Read out indxuo
  if (.not. lGamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) ! numh
  call iostat_update(ierr)

! Read listh
  do ih = 1 , no_u
    read(iu, iostat=ierr) !listh
    call iostat_update(ierr)
  end do

! Read Hamiltonian
  do is = 1 , lnspin
    do ih = 1 , no_u
      read(iu, iostat=ierr) ! H
      call iostat_update(ierr)
    end do
  end do

! Read Overlap matrix
  do ih = 1,no_u
    read(iu, iostat=ierr) ! S
    call iostat_update(ierr)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

! Read xij
  do ih = 1 , no_u
    read(iu, iostat=ierr) ! xij
    call iostat_update(ierr)
  end do

  ! Now read in the geometry information
  read(iu, iostat=ierr) lnspecies
  if ( lnspecies /= nspecies ) stop 'Error in reading data, not allocated, nspecies'
  call iostat_update(ierr)

  read(iu, iostat=ierr) (labelfis(1:20,is), zvalfis(is),nofis(is),is=1,nspecies)
  call iostat_update(ierr)
  do is = 1, nspecies
    do io = 1 , nofis(is)
      read(iu, iostat=ierr) !nfio(is,io), lfio(is,io), zetafio(is,io)
      call iostat_update(ierr)
    end do
  end do

  read(iu, iostat=ierr) lna_u
  if ( lna_u /= na_u ) stop 'Error in reading data, not allocated, na_u'
  call iostat_update(ierr)
  read(iu, iostat=ierr) isa
  call iostat_update(ierr)

  call close_file(iu)

end subroutine read_hsx_species

subroutine read_hsx_specie(fname, ispecie, no_specie, n_specie, l_specie, zeta_specie)
  use io_m, only: open_file, close_file
  use io_m, only: iostat_update

  implicit none

  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*), intent(in) :: fname
  integer, intent(in) :: ispecie, no_specie
  integer, intent(out) :: n_specie(no_specie), l_specie(no_specie), zeta_specie(no_specie)

! Define f2py intents
!f2py intent(in) :: fname
!f2py intent(in) :: ispecie, no_specie
!f2py intent(out) :: n_specie, l_specie, zeta_specie

! Internal variables and arrays
  integer :: iu, ierr
  integer :: is, ih, io

  ! Local readables
  logical :: lGamma
  integer :: lnspecies, lno_u, lno_s, lnspin
  character(len=20), allocatable :: labelfis(:)
  real(dp), allocatable :: zvalfis(:)
  integer, allocatable :: nofis(:)

  call open_file(fname, 'read', 'old', 'unformatted', iu)

  ! Read overall data
  read(iu, iostat=ierr) lno_u, lno_s, lnspin !, lmaxnh
  call iostat_update(ierr)

! Read logical
  read(iu, iostat=ierr) lGamma
  call iostat_update(ierr)

! Read out indxuo
  if (.not. lGamma) then
    read(iu, iostat=ierr) ! indxuo
    call iostat_update(ierr)
  end if

  read(iu, iostat=ierr) ! numh
  call iostat_update(ierr)

! Read listh
  do ih = 1 , lno_u
    read(iu, iostat=ierr) !listh
    call iostat_update(ierr)
  end do

! Read Hamiltonian
  do is = 1 , lnspin
    do ih = 1 , lno_u
      read(iu, iostat=ierr) ! H
      call iostat_update(ierr)
    end do
  end do

! Read Overlap matrix
  do ih = 1, lno_u
    read(iu, iostat=ierr) ! S
    call iostat_update(ierr)
  end do

  read(iu, iostat=ierr) !Qtot,temp
  call iostat_update(ierr)

! Read xij
  do ih = 1 , lno_u
    read(iu, iostat=ierr) ! xij
    call iostat_update(ierr)
  end do

  ! Now read in the geometry information
  read(iu, iostat=ierr) lnspecies
  if ( lnspecies < ispecie ) stop 'Error in reading data, not allocated, nspecies<ispecie'
  call iostat_update(ierr)
  allocate(labelfis(lnspecies))
  allocate(zvalfis(lnspecies))
  allocate(nofis(lnspecies))

  read(iu, iostat=ierr) (labelfis(is), zvalfis(is),nofis(is),is=1,lnspecies)
  call iostat_update(ierr)
  do is = 1, lnspecies
    if ( is == ispecie ) then
      do io = 1 , nofis(is)
        read(iu, iostat=ierr) n_specie(io), l_specie(io), zeta_specie(io)
        call iostat_update(ierr)
      end do
    else
      do io = 1 , nofis(is)
        read(iu, iostat=ierr) !nfio(is,io), lfio(is,io), zetafio(is,io)
        call iostat_update(ierr)
      end do
    end if
  end do

  deallocate(labelfis, zvalfis, nofis)

  call close_file(iu)

end subroutine read_hsx_specie
