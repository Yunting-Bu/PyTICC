subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none

    integer, intent(in) :: n_coordinate
    integer, intent(in) :: n_grid
    real(8), intent(in) :: RR
    real(8), intent(in) :: coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i
    real(8) :: R_h2
    real(8) :: theta_h2

    do i = 1, n_grid
        call arhd_to_arh2( &
            RR, &
            coordinates(1, i), &
            coordinates(2, i), &
            R_h2, &
            theta_h2 &
        )
        call PESH2_Ar(R_h2, coordinates(1, i), theta_h2, V(i), 0)
    end do
end subroutine pyticc_interaction_grid


subroutine arhd_to_arh2(RR, r_hd, theta_hd, R_h2, theta_h2)
    implicit none

    real(8), intent(in) :: RR
    real(8), intent(in) :: r_hd
    real(8), intent(in) :: theta_hd
    real(8), intent(out) :: R_h2
    real(8), intent(out) :: theta_h2

    real(8), parameter :: mass_h = 1.00782503223d0
    real(8), parameter :: mass_d = 2.01410177812d0
    real(8), parameter :: pi = acos(-1.0d0)
    real(8) :: alpha_cos
    real(8) :: distance_to_end
    real(8) :: half_bond
    real(8) :: theta_cos
    real(8) :: distance_com_to_end

    ! Convert the Ar--HD center-of-mass Jacobi vector to the Ar--H2
    ! geometric-center convention used by the isotope-independent PES.
    distance_com_to_end = mass_h * r_hd / (mass_h + mass_d)
    distance_to_end = sqrt( &
        RR**2 + distance_com_to_end**2 &
        - 2.0d0 * RR * distance_com_to_end * cos(theta_hd) &
    )
    alpha_cos = ( &
        distance_to_end**2 + distance_com_to_end**2 - RR**2 &
    ) / (2.0d0 * distance_to_end * distance_com_to_end)
    alpha_cos = min(1.0d0, max(-1.0d0, alpha_cos))

    half_bond = 0.5d0 * r_hd
    R_h2 = sqrt( &
        distance_to_end**2 + half_bond**2 &
        - 2.0d0 * distance_to_end * half_bond * alpha_cos &
    )
    theta_cos = ( &
        R_h2**2 + half_bond**2 - distance_to_end**2 &
    ) / (2.0d0 * R_h2 * half_bond)
    theta_cos = min(1.0d0, max(-1.0d0, theta_cos))
    theta_h2 = acos(theta_cos) * 180.0d0 / pi
end subroutine arhd_to_arh2


subroutine pyticc_monomer_y_grid(r, V, n_grid)
    implicit none

    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i
    real(8), parameter :: bohr_to_angstrom = 0.529177249d0
    real(8), parameter :: hartree_to_ev = 27.21138d0
    real(8), parameter :: r_e = 0.7414d0
    real(8), parameter :: D_e = 4.747d0 / hartree_to_ev
    real(8), parameter :: a1 = 3.961d0
    real(8), parameter :: a2 = 4.064d0
    real(8), parameter :: a3 = 3.574d0
    real(8) :: displacement

    do i = 1, n_grid
        displacement = r(i) * bohr_to_angstrom - r_e
        V(i) = -D_e * ( &
            1.0d0 + a1 * displacement + a2 * displacement**2 &
            + a3 * displacement**3 &
        ) * exp(-a1 * displacement)
    end do
end subroutine pyticc_monomer_y_grid
