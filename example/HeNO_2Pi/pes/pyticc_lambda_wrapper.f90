subroutine pyticc_lambda_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none
    integer, intent(in) :: n_coordinate, n_grid
    real(8), intent(in) :: RR, coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid, 2)
    real(8), external :: VSUM, VDIF
    real(8), parameter :: cm_to_hartree = 1.0d0 / 219474.6313705d0
    integer :: i

    do i = 1, n_grid
        V(i, 1) = VSUM(RR, cos(coordinates(2, i))) * cm_to_hartree
        V(i, 2) = VDIF(RR, cos(coordinates(2, i))) * cm_to_hartree
    end do
end subroutine pyticc_lambda_grid

subroutine pyticc_monomer_y_grid(r, V, n_grid)
    implicit none
    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)
    real(8), parameter :: re = 2.1746401324355977d0
    real(8), parameter :: De = 0.2934488409742969d0
    real(8), parameter :: morse_a = 1.321245963231734d0
    integer :: i

    do i = 1, n_grid
        V(i) = De * (1.0d0 - exp(-morse_a * (r(i) - re)))**2
    end do
end subroutine pyticc_monomer_y_grid
