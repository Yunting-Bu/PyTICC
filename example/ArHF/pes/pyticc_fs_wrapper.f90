subroutine pyticc_lambda_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none

    integer, intent(in) :: n_coordinate, n_grid
    real(8), intent(in) :: RR, coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid, 2)
    real(8), parameter :: pi = acos(-1.0d0)
    integer :: i

    do i = 1, n_grid
        call interaction_PES(coordinates(1, i), RR, coordinates(2, i) * 180.0d0 / pi, V(i, 1))
        V(i, 2) = 0.0d0
    end do
end subroutine pyticc_lambda_grid
