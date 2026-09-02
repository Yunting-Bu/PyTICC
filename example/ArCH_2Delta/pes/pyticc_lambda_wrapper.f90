subroutine pyticc_lambda_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none
    integer, intent(in) :: n_coordinate, n_grid
    real(8), intent(in) :: RR, coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid, 2)
    real(8) :: va, vd, r_ang, gamma
    real(8), parameter :: bohr_to_angstrom = 0.529177210903d0
    real(8), parameter :: cm_to_hartree = 1.0d0 / 219474.6313705d0
    integer, parameter :: isurf = 1
    real(8), parameter :: dre = -0.25d0
    external :: chad_pes
    integer :: i

    do i = 1, n_grid
        r_ang = RR * bohr_to_angstrom
        gamma = coordinates(2, i)
        call chad_pes(r_ang, gamma, isurf, dre, va, vd)
        V(i, 1) = va * cm_to_hartree
        V(i, 2) = -vd * cm_to_hartree
    end do
end subroutine pyticc_lambda_grid

subroutine pyticc_monomer_y_grid(r, V, n_grid)
    implicit none
    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)
    real(8), parameter :: re = 2.08266d0
    real(8), parameter :: De = 0.117055d0
    real(8), parameter :: morse_a = 1.1362d0
    integer :: i

    do i = 1, n_grid
        V(i) = De * (1.0d0 - exp(-morse_a * (r(i) - re)))**2
    end do
end subroutine pyticc_monomer_y_grid