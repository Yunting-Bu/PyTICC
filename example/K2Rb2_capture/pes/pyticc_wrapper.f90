subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none

    integer, intent(in) :: n_coordinate
    integer, intent(in) :: n_grid
    real(8), intent(in) :: RR
    real(8), intent(in) :: coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i
    real(8) :: geometry(6)
    real(8), parameter :: pi = acos(-1.0d0)

    if (n_coordinate /= 5) then
        error stop "KRb+KRb PES requires r1, r2, theta1, theta2, and phi"
    end if

    do i = 1, n_grid
        geometry = [ &
            RR, &
            coordinates(1, i), &
            coordinates(2, i), &
            coordinates(3, i) * 180.0d0 / pi, &
            coordinates(4, i) * 180.0d0 / pi, &
            coordinates(5, i) * 180.0d0 / pi &
        ]
        call interaction_potential_ABCD(geometry, V(i))
    end do
end subroutine pyticc_interaction_grid


subroutine pyticc_monomer_x_grid(r, V, n_grid)
    implicit none

    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i

    do i = 1, n_grid
        call potential_AB(r(i), V(i))
    end do
end subroutine pyticc_monomer_x_grid


subroutine pyticc_monomer_y_grid(r, V, n_grid)
    implicit none

    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i

    do i = 1, n_grid
        call potential_CD(r(i), V(i))
    end do
end subroutine pyticc_monomer_y_grid
