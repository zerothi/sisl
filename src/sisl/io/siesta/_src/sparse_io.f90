module sparse_io_m

  use precision
  use io_m, only: iostat_update

  implicit none
  private

  public :: write_sparse
  public :: write_data_1d, write_data_2d1, write_data_2d2
  public :: read_sparse
  public :: read_data_1d, read_data_2d1, read_data_2d2

contains

  subroutine read_skip(iu, n)
     
    ! Input parameters
    integer(i4), intent(in) :: iu, n

    integer(i4) :: i
    integer(i4) :: ierr

    do i = 1, n
      read(iu, iostat=ierr) ! nothing
      call iostat_update(ierr)
    end do

  end subroutine read_skip

  subroutine read_sparse(iu, no_u, nnz, ncol, list_col)
    
    ! Input parameters
    integer(i4), intent(in) :: iu
    integer(i4), intent(in) :: no_u, nnz
    integer(i4), intent(inout) :: ncol(no_u), list_col(nnz)

    integer(i4) :: io, idx
    integer(i4) :: ierr

    read(iu, iostat=ierr) ncol
    call iostat_update(ierr)
    if ( nnz /= sum(ncol) ) then
      call iostat_update(10)
    end if

    ! Read list_col
    idx = 0
    do io = 1 , no_u
      read(iu, iostat=ierr) list_col(idx+1:idx+ncol(io))
      call iostat_update(ierr)
      idx = idx + ncol(io)
    end do

  end subroutine read_sparse

  subroutine read_data_1d(iu, no_u, nnz, ncol, M)
    
     ! Input parameters
    integer(i4), intent(in) :: iu
    integer(i4), intent(in) :: no_u, nnz
    integer(i4), intent(in) :: ncol(no_u)
    real(r8), intent(inout) :: M(nnz)

    integer(i4) :: io, idx
    integer(i4) :: ierr

    idx = 0
    do io = 1 , no_u
      read(iu, iostat=ierr) M(idx+1:idx+ncol(io))
      call iostat_update(ierr)
      idx = idx + ncol(io)
    end do

  end subroutine read_data_1d
  
  subroutine read_data_2d1(iu, no_u, dim2, nnz, ncol, M)
    
     ! Input parameters
    integer(i4), intent(in) :: iu
    integer(i4), intent(in) :: no_u, dim2, nnz
    integer(i4), intent(in) :: ncol(no_u)
    real(r8), intent(inout) :: M(dim2,nnz)

    integer(i4) :: io, idx
    integer(i4) :: ierr

    idx = 0
    do io = 1 , no_u
      read(iu, iostat=ierr) M(:, idx+1:idx+ncol(io))
      call iostat_update(ierr)
      idx = idx + ncol(io)
    end do

  end subroutine read_data_2d1
  
  subroutine read_data_2d2(iu, no_u, dim2, nnz, ncol, M)
    
     ! Input parameters
    integer(i4), intent(in) :: iu
    integer(i4), intent(in) :: no_u, dim2, nnz
    integer(i4), intent(in) :: ncol(no_u)
    real(r8), intent(inout) :: M(nnz, dim2)

    integer(i4) :: io, d2, idx
    integer(i4) :: ierr

    do d2 = 1, dim2
      idx = 0
      do io = 1 , no_u
        read(iu, iostat=ierr) M(idx+1:idx+ncol(io),d2)
        call iostat_update(ierr)
        idx = idx + ncol(io)
      end do
    end do

  end subroutine read_data_2d2

  subroutine write_sparse(iu, no_u, nnz, ncol, list_col)
    
    ! Input parameters
    integer(i4), intent(in) :: iu
    integer(i4), intent(in) :: no_u, nnz
    integer(i4), intent(in) :: ncol(no_u), list_col(nnz)

    integer(i4) :: io, idx
    integer(i4) :: ierr

    write(iu, iostat=ierr) ncol
    call iostat_update(ierr)
    
    idx = 0
    do io = 1 , no_u
      write(iu, iostat=ierr) list_col(idx+1:idx+ncol(io))
      call iostat_update(ierr)
      idx = idx + ncol(io)
    end do

  end subroutine write_sparse

  subroutine write_data_1d(iu, no_u, nnz, ncol, M)
    
    ! Input parameters
    integer(i4), intent(in) :: iu
    integer(i4), intent(in) :: no_u, nnz
    integer(i4), intent(in) :: ncol(no_u)
    real(r8), intent(in) :: M(nnz)

    integer(i4) :: io, idx
    integer(i4) :: ierr

    idx = 0
    do io = 1 , no_u
      write(iu, iostat=ierr) M(idx+1:idx+ncol(io))
      call iostat_update(ierr)
      idx = idx + ncol(io)
    end do

  end subroutine write_data_1d

  subroutine write_data_2d1(iu, no_u, dim2, nnz, ncol, M)
    
    ! Input parameters
    integer(i4), intent(in) :: iu
    integer(i4), intent(in) :: no_u, dim2, nnz
    integer(i4), intent(in) :: ncol(no_u)
    real(r8), intent(in) :: M(dim2,nnz)

    integer(i4) :: io, idx
    integer(i4) :: ierr

    idx = 0
    do io = 1 , no_u
      write(iu, iostat=ierr) M(:,idx+1:idx+ncol(io))
      call iostat_update(ierr)
      idx = idx + ncol(io)
    end do

  end subroutine write_data_2d1

  subroutine write_data_2d2(iu, no_u, dim2, nnz, ncol, M)
    
    ! Input parameters
    integer(i4), intent(in) :: iu
    integer(i4), intent(in) :: no_u, dim2, nnz
    integer(i4), intent(in) :: ncol(no_u)
    real(r8), intent(in) :: M(nnz,dim2)

    integer(i4) :: io, d2, idx
    integer :: ierr

    do d2 = 1, dim2
      idx = 0
      do io = 1 , no_u
        write(iu, iostat=ierr) M(idx+1:idx+ncol(io), d2)
        call iostat_update(ierr)
        idx = idx + ncol(io)
      end do
    end do

  end subroutine write_data_2d2

end module sparse_io_m
  
