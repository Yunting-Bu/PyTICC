subroutine pes_init
    implicit none
end subroutine pes_init

subroutine potsub(bonds, energy_ev)
    implicit none
    real(8), intent(in) :: bonds(3)
    real(8), intent(out) :: energy_ev
    real(8) :: energy_hartree(1, 1)

    call pot0(1, energy_hartree, bonds)
    energy_ev = energy_hartree(1, 1) * 27.2114d0
end subroutine potsub
