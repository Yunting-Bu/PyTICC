subroutine pyticc_lambda_grid(RR, coordinates, V, n_coordinate, n_grid)
    use arnopes, only: surf_tk_jac
    implicit none
    integer, intent(in) :: n_coordinate, n_grid
    real(8), intent(in) :: RR, coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid, 2)
    real(8) :: jc(3), val(2), vno, vnop
    real(8), parameter :: cm_to_hartree = 1.0d0 / 219474.6313705d0
    real(8), parameter :: de = 53434.0d0
    real(8), parameter :: re = 2.174644592d0
    real(8), parameter :: be = 1.451769543d0
    integer :: i

    do i = 1, n_grid
        jc(1) = coordinates(1, i)
        jc(2) = RR
        jc(3) = coordinates(2, i) * 180.0d0 / 3.141592653589793d0
        call surf_tk_jac(jc, val)
        vnop = de * (exp(-be * (jc(1) - re)) - 1.0d0)**2
        V(i, 1) = (val(1) - vnop) * cm_to_hartree
        V(i, 2) = (-val(2)) * cm_to_hartree
    end do
end subroutine pyticc_lambda_grid

subroutine pyticc_monomer_y_grid(r, V, n_grid)
    implicit none
    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)
    real(8), parameter :: cm_to_hartree = 1.0d0 / 219474.6313705d0
    real(8), parameter :: de = 53434.0d0
    real(8), parameter :: re = 2.174644592d0
    real(8), parameter :: be = 1.451769543d0
    integer :: i

    do i = 1, n_grid
        V(i) = de * (exp(-be * (r(i) - re)) - 1.0d0)**2 * cm_to_hartree
    end do
end subroutine pyticc_monomer_y_grid