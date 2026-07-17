subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none

    integer, intent(in) :: n_coordinate
    integer, intent(in) :: n_grid
    real(8), intent(in) :: RR
    real(8), intent(in) :: coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i
    real(8), parameter :: pi = acos(-1.0d0)

    do i = 1, n_grid
        call interaction_PES( &
            coordinates(1, i), &
            RR, &
            coordinates(2, i) * 180.0d0 / pi, &
            V(i) &
        )
    end do
end subroutine pyticc_interaction_grid


subroutine pyticc_monomer_y_grid(r, V, n_grid)
    implicit none

    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i
    real(8), parameter :: autoang = 0.5292177249d0
    real(8), parameter :: autocm = 219474.63137d0
    real(8), parameter :: re = 0.9168d0 / autoang
    real(8), parameter :: De = 49361.6d0 / autocm
    real(8), parameter :: a1 = 2.23729d0
    real(8), parameter :: a2 = 1.12367d0
    real(8), parameter :: a3 = 0.568735d0
    real(8), parameter :: a4 = 0.00918165d0
    real(8), parameter :: a5 = 0.0080782d0
    real(8), parameter :: a6 = -0.0351117d0
    real(8) :: p

    do i = 1, n_grid
        p = r(i) - re
        V(i) = -De * ( &
            1.0d0 + a1 * p + a2 * p**2 + a3 * p**3 + &
            a4 * p**4 + a5 * p**5 + a6 * p**6 &
        ) * exp(-a1 * p)
    end do
end subroutine pyticc_monomer_y_grid
