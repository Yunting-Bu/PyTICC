subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none

    integer, intent(in) :: n_coordinate, n_grid
    real(8), intent(in) :: RR
    real(8), intent(in) :: coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid)

    real(8), parameter :: bohr_to_angstrom = 0.529177210903d0
    real(8), parameter :: ev_per_hartree = 27.211386245988d0
    real(8), parameter :: mass_h = 1.00782503223d0
    real(8), parameter :: mass_o = 15.99491461957d0
    logical, save :: initialized = .false.
    integer :: i
    real(8) :: atoms(3, 4), monomer_atoms(3, 4)
    real(8) :: total_energy, monomer_energy

    if (.not. initialized) then
        call pes_init
        initialized = .true.
    end if

    do i = 1, n_grid
        call corrected_radau_cartesian( &
            RR, coordinates(:, i), mass_h, mass_o, atoms)
        atoms = atoms * bohr_to_angstrom
        monomer_atoms = atoms
        monomer_atoms(:, 3) = [0.d0, 0.d0, 99.d0]

        call HOHCl_pes_interface(atoms, total_energy)
        call HOHCl_pes_interface(monomer_atoms, monomer_energy)
        V(i) = (total_energy - monomer_energy) / ev_per_hartree
    end do
end subroutine pyticc_interaction_grid


subroutine corrected_radau_cartesian(RR, coordinates, mass_h, mass_o, atoms)
    implicit none

    real(8), intent(in) :: RR, coordinates(5), mass_h, mass_o
    real(8), intent(out) :: atoms(3, 4)

    real(8) :: atom_h1(3), atom_o(3), atom_h2(3), center_hh(3)
    real(8) :: center_mass(3), rotated(3)
    real(8) :: r1, r2, theta1, theta2, phi
    real(8) :: half_theta, total_mass, cos_phi, sin_phi
    real(8) :: cos_theta2, sin_theta2

    r1 = coordinates(1)
    r2 = coordinates(2)
    theta1 = coordinates(3)
    theta2 = coordinates(4)
    phi = coordinates(5)
    half_theta = 0.5d0 * theta1
    total_mass = 2.d0 * mass_h + mass_o

    atom_h1 = [r1 * sin(half_theta), 0.d0, -r1 * cos(half_theta)]
    atom_h2 = [-r2 * sin(half_theta), 0.d0, -r2 * cos(half_theta)]
    center_hh = 0.5d0 * (atom_h1 + atom_h2)
    atom_o = (1.d0 - sqrt(total_mass / mass_o)) * center_hh
    center_mass = (mass_h * atom_h1 + mass_o * atom_o + mass_h * atom_h2) / total_mass
    atom_h1 = atom_h1 - center_mass
    atom_o = atom_o - center_mass
    atom_h2 = atom_h2 - center_mass

    cos_phi = cos(phi)
    sin_phi = sin(phi)
    cos_theta2 = cos(theta2)
    sin_theta2 = sin(theta2)

    call rotate_monomer_atom(atom_h1, cos_phi, sin_phi, cos_theta2, sin_theta2, rotated)
    atoms(:, 1) = rotated
    call rotate_monomer_atom(atom_h2, cos_phi, sin_phi, cos_theta2, sin_theta2, rotated)
    atoms(:, 2) = rotated
    atoms(:, 3) = [0.d0, 0.d0, RR]
    call rotate_monomer_atom(atom_o, cos_phi, sin_phi, cos_theta2, sin_theta2, rotated)
    atoms(:, 4) = rotated
end subroutine corrected_radau_cartesian


subroutine rotate_monomer_atom(point, cos_phi, sin_phi, cos_theta, sin_theta, rotated)
    implicit none

    real(8), intent(in) :: point(3), cos_phi, sin_phi, cos_theta, sin_theta
    real(8), intent(out) :: rotated(3)
    real(8) :: x_phi, y_phi

    x_phi = cos_phi * point(1) - sin_phi * point(2)
    y_phi = sin_phi * point(1) + cos_phi * point(2)
    rotated(1) = cos_theta * x_phi + sin_theta * point(3)
    rotated(2) = y_phi
    rotated(3) = -sin_theta * x_phi + cos_theta * point(3)
end subroutine rotate_monomer_atom
