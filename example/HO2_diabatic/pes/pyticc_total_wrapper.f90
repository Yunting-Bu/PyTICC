module pyticc_ho2_total_state
    implicit none

    logical, save :: initialized = .false.

contains

    subroutine ensure_ho2_total_initialized
        implicit none

        if (.not. initialized) then
            call pes_init
            initialized = .true.
        end if
    end subroutine ensure_ho2_total_initialized

end module pyticc_ho2_total_state


subroutine pyticc_total_grid(bonds, V, n_grid)
    use pyticc_ho2_total_state, only: ensure_ho2_total_initialized
    implicit none

    integer, intent(in) :: n_grid
    real(8), intent(in) :: bonds(3, n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i
    real(8) :: ho2_bonds(3)
    real(8) :: lower_surface(1, 1)

    call ensure_ho2_total_initialized
    do i = 1, n_grid
        ! PyTICC order for A=O, B=O, C=H is (OO,OH,OH).
        ! This pot0 version expects (OH1,OO,OH2).
        ho2_bonds = [bonds(2, i), bonds(1, i), bonds(3, i)]
        call pot0(1, lower_surface, ho2_bonds)
        V(i) = lower_surface(1, 1)
    end do
end subroutine pyticc_total_grid
