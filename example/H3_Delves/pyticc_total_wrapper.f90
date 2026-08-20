subroutine pyticc_total_grid(bonds, V, n_grid)
    implicit none

    integer, intent(in) :: n_grid
    real(8), intent(in) :: bonds(3, n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i
    real(8) :: derivatives(3)

    do i = 1, n_grid
        call bkmp2(bonds(:, i), V(i), derivatives, -1)
    end do
end subroutine pyticc_total_grid
