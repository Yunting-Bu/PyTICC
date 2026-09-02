subroutine pyticc_total_grid(bonds, V, n_grid)
    implicit none

    integer, intent(in) :: n_grid
    real(8), intent(in) :: bonds(3, n_grid)
    real(8), intent(out) :: V(n_grid)

    logical, save :: initialized = .false.
    integer :: i
    real(8) :: r_oh1, r_oh2, r_hh, cosine

    if (.not. initialized) then
        call potreadadi
        initialized = .true.
    end if

    do i = 1, n_grid
        r_oh1 = bonds(1, i)
        r_oh2 = bonds(2, i)
        r_hh = bonds(3, i)
        cosine = (r_oh1**2 + r_oh2**2 - r_hh**2) / (2.d0 * r_oh1 * r_oh2)
        cosine = max(-1.d0, min(1.d0, cosine))
        call h2oadipes(r_oh1, r_oh2, cosine, V(i), 1)
    end do
end subroutine pyticc_total_grid
