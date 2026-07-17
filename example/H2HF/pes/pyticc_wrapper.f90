subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none

    integer, intent(in) :: n_coordinate
    integer, intent(in) :: n_grid
    real(8), intent(in) :: RR
    real(8), intent(in) :: coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i
    real(8) :: V_total
    real(8), parameter :: pi = acos(-1.0d0)

    if (n_coordinate /= 5) then
        stop "H2HF interaction PES requires r_H2, r_HF, theta_H2, theta_HF, and phi"
    end if

    do i = 1, n_grid
        call pesh3f( &
            coordinates(1, i), &
            coordinates(2, i), &
            RR, &
            coordinates(3, i) * 180.0d0 / pi, &
            coordinates(4, i) * 180.0d0 / pi, &
            coordinates(5, i) * 180.0d0 / pi, &
            V(i), &
            V_total &
        )
    end do
end subroutine pyticc_interaction_grid


subroutine pyticc_monomer_x_grid(r, V, n_grid)
    implicit none

    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i

    do i = 1, n_grid
        call vh2(r(i), V(i))
    end do
end subroutine pyticc_monomer_x_grid


subroutine pyticc_monomer_y_grid(r, V, n_grid)
    implicit none

    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)

    integer :: i

    do i = 1, n_grid
        call vhf(r(i), V(i))
    end do
end subroutine pyticc_monomer_y_grid


real(8) function spgndr(l, mm, x)
    implicit none

    integer, intent(in) :: l
    integer, intent(in) :: mm
    integer :: i
    integer :: m
    real(8), intent(in) :: x
    real(8) :: factor
    real(8), external :: plgndr

    m = abs(mm)
    factor = (2 * l + 1) / 2.0d0
    do i = l - m + 1, l + m
        factor = factor / i
    end do
    factor = sqrt(factor)
    if (mm < 0 .and. mod(m, 2) /= 0) factor = -factor
    spgndr = factor * plgndr(l, m, x)
end function spgndr


real(8) function plgndr(l, m, x)
    implicit none

    integer, intent(in) :: l
    integer, intent(in) :: m
    integer :: i
    integer :: ll
    real(8), intent(in) :: x
    real(8) :: factor
    real(8) :: pll
    real(8) :: pmm
    real(8) :: pmmp1
    real(8) :: somx2

    if (m < 0 .or. m > l .or. abs(x) > 1.0d0) then
        stop "Invalid associated Legendre arguments"
    end if

    pmm = 1.0d0
    if (m > 0) then
        somx2 = sqrt((1.0d0 - x) * (1.0d0 + x))
        factor = 1.0d0
        do i = 1, m
            pmm = -pmm * factor * somx2
            factor = factor + 2.0d0
        end do
    end if

    if (l == m) then
        plgndr = pmm
        return
    end if

    pmmp1 = x * (2 * m + 1) * pmm
    if (l == m + 1) then
        plgndr = pmmp1
        return
    end if

    do ll = m + 2, l
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    end do
    plgndr = pll
end function plgndr
