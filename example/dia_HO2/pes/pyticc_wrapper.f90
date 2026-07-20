module pyticc_ho2_adapter_state
    implicit none

    logical, save :: initialized = .false.

contains

    subroutine ensure_ho2_pes_initialized
        implicit none

        if (.not. initialized) then
            call pes_init
            initialized = .true.
        end if
    end subroutine ensure_ho2_pes_initialized

end module pyticc_ho2_adapter_state


subroutine pyticc_diabatic_interaction_grid(RR, coordinates, V, n_coordinate, n_grid, n_state)
    use pyticc_ho2_adapter_state, only: ensure_ho2_pes_initialized
    implicit none

    integer, intent(in) :: n_coordinate
    integer, intent(in) :: n_grid
    integer, intent(in) :: n_state
    real(8), intent(in) :: RR
    real(8), intent(in) :: coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid, n_state, n_state)

    integer :: electronic_state
    integer :: i
    real(8), parameter :: au_per_ev = 27.21138d0
    real(8), parameter :: cm_per_ev = 8065.5d0
    real(8), parameter :: pi = acos(-1.0d0)
    real(8) :: bonds(3)
    real(8) :: cos_theta
    real(8) :: dpem(2, 2)
    real(8) :: monomer
    real(8) :: r_oh_1
    real(8) :: r_oh_2
    real(8) :: r_oo
    real(8) :: theta_degree
    real(8) :: vlr
    real(8), external :: switchR

    if (n_coordinate /= 2) stop "HO2 DPEM requires r_OO and Jacobi theta"
    if (n_state /= 2) stop "HO2 DPEM has exactly two electronic states"
    call ensure_ho2_pes_initialized

    do i = 1, n_grid
        r_oo = coordinates(1, i)
        cos_theta = cos(coordinates(2, i))
        theta_degree = coordinates(2, i) * 180.0d0 / pi
        r_oh_1 = sqrt(max(0.0d0, RR**2 + 0.25d0 * r_oo**2 - RR * r_oo * cos_theta))
        r_oh_2 = sqrt(max(0.0d0, RR**2 + 0.25d0 * r_oo**2 + RR * r_oo * cos_theta))
        bonds = [r_oh_1, r_oo, r_oh_2]

        call pot0(2, dpem, bonds)
        do electronic_state = 1, 2
            call potential_AB(r_oo, monomer, electronic_state)
            dpem(electronic_state, electronic_state) = dpem(electronic_state, electronic_state) - monomer
        end do

        call get_long(theta_degree, RR, r_oo, vlr)
        if (r_oo < 3.0d0 .and. r_oo > 1.5d0) then
            dpem(1, 1) = vlr / au_per_ev / cm_per_ev * switchR(RR) + (1.0d0 - switchR(RR)) * dpem(1, 1)
        end if

        if (RR >= 7.0d0) then
            dpem(1, 2) = 0.0d0
            dpem(2, 1) = 0.0d0
            dpem(2, 2) = 0.0d0
        end if

        V(i, :, :) = dpem
    end do
end subroutine pyticc_diabatic_interaction_grid


subroutine pyticc_diabatic_monomer_grid(r, V, n_grid, n_state)
    use pyticc_ho2_adapter_state, only: ensure_ho2_pes_initialized
    implicit none

    integer, intent(in) :: n_grid
    integer, intent(in) :: n_state
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid, n_state)

    integer :: electronic_state
    integer :: i

    if (n_state /= 2) stop "HO2 DPEM has exactly two electronic states"
    call ensure_ho2_pes_initialized

    do electronic_state = 1, n_state
        do i = 1, n_grid
            call potential_AB(r(i), V(i, electronic_state), electronic_state)
        end do
    end do
end subroutine pyticc_diabatic_monomer_grid
