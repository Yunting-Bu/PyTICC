from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.angle import gauss_legendre_dvr, lambda_minus, lambda_plus, norm_YjK
from pyticc.basis.podvr import VibPODVR

TriatomPES = Callable[[NDArray[np.float64]], NDArray[np.float64]]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TriatomHamiltonianData:
    """
    PES and quadrature data shared by all triatomic Hamiltonian blocks.

    Members:
        radial_1: VibPODVR - first Radau-coordinate vibrational basis
        radial_2: VibPODVR - second Radau-coordinate vibrational basis
        cos_theta: NDArray[np.float64] - Gauss-Legendre grids in cos(theta1), shape
            (n_theta,)
        theta_weights: NDArray[np.float64] - Gauss-Legendre weights, shape
            (n_theta,)
        V_res: NDArray[np.float64] - residual monomer potential on (r1, r2, theta1),
            shape (n_r1, n_r2, n_theta)
        B: NDArray[np.float64] - first inverse-square radial matrix, shape
            (n_v1, n_v1)
        C: NDArray[np.float64] - second inverse-square radial matrix, shape
            (n_v2, n_v2)
        angular: dict[int, NDArray[np.float64]] - normalized associated Legendre
            functions; each value has shape (n_theta, j1max + 1)
    """

    radial_1: VibPODVR
    radial_2: VibPODVR
    cos_theta: NDArray[np.float64]
    theta_weights: NDArray[np.float64]
    V_res: NDArray[np.float64]
    B: NDArray[np.float64]
    C: NDArray[np.float64]
    angular: dict[int, NDArray[np.float64]]


# ----------------------------------------------------------------------------------------
def prepare_triatom_hamiltonian(
    potential: TriatomPES,
    radial_1: VibPODVR,
    radial_2: VibPODVR,
    masses: tuple[float, float, float],
    equilibrium: tuple[float, float, float],
    n_theta: int,
    j1max: int,
) -> TriatomHamiltonianData:
    r"""
    Prepare the coordinate-independent data for the Radau triatomic Hamiltonian.

    Formula:
        V_res(r1,r2,theta1) = V_ABC(r1,r2,theta1) - V1(r1) - V2(r2)
        B_v'v = <v'|1/(2 m_A r1^2)|v>
        C_v'v = <v'|1/(2 m_C r2^2)|v>

    Inputs:
        potential: TriatomPES - vectorized monomer PES mapping coordinates with
            shape (3, n_point) to values with shape (n_point,)
        radial_1: VibPODVR - first Radau-coordinate vibrational basis
        radial_2: VibPODVR - second Radau-coordinate vibrational basis
        masses: tuple[float, float, float] - masses of atoms A, B, and C in atomic units
        equilibrium: tuple[float, float, float] - equilibrium (r1, r2, theta1) in atomic units and radians
        n_theta: int - number of Gauss-Legendre grids in cos(theta1)
        j1max: int - maximum bending angular momentum

    Returns:
        data: TriatomHamiltonianData - reusable data with a residual-potential grid
            of shape (n_r1, n_r2, n_theta), radial matrices of shapes
            (n_v1, n_v1) and (n_v2, n_v2), and angular arrays with leading shape
            (n_theta, ...)
    """
    cos_theta, theta_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta)
    theta = np.arccos(cos_theta)
    r1_eq, r2_eq, theta_eq = equilibrium

    r1_grid, r2_grid, theta_grid = np.meshgrid(radial_1.grids, radial_2.grids, theta, indexing="ij")
    coordinates = np.stack((r1_grid.reshape(-1), r2_grid.reshape(-1), theta_grid.reshape(-1)))
    V_ABC = np.asarray(potential(coordinates), dtype=np.float64).reshape(r1_grid.shape)

    coordinates_1 = np.stack(
        (
            radial_1.grids,
            np.full(radial_1.grids.size, r2_eq),
            np.full(radial_1.grids.size, theta_eq),
        )
    )
    coordinates_2 = np.stack(
        (
            np.full(radial_2.grids.size, r1_eq),
            radial_2.grids,
            np.full(radial_2.grids.size, theta_eq),
        )
    )
    V_1 = np.asarray(potential(coordinates_1), dtype=np.float64)
    V_2 = np.asarray(potential(coordinates_2), dtype=np.float64)
    V_res = V_ABC - V_1[:, None, None] - V_2[None, :, None]

    mass_A, _, mass_C = masses
    P = radial_1.wavefunctions
    Q = radial_2.wavefunctions
    B = np.einsum("pa,p,pb->ab", P, 1.0 / (2.0 * mass_A * radial_1.grids**2), P, optimize=True)
    C = np.einsum("pa,p,pb->ab", Q, 1.0 / (2.0 * mass_C * radial_2.grids**2), Q, optimize=True)
    angular = {
        omega: np.column_stack(
            [np.asarray(norm_YjK(j1, omega, cos_theta), dtype=np.float64) if j1 >= abs(omega) else np.zeros(n_theta) for j1 in range(j1max + 1)]
        )
        for omega in range(-j1max, j1max + 1)
    }

    return TriatomHamiltonianData(
        radial_1=radial_1,
        radial_2=radial_2,
        cos_theta=cos_theta,
        theta_weights=theta_weights,
        V_res=V_res,
        B=B,
        C=C,
        angular=angular,
    )


# ----------------------------------------------------------------------------------------
def get_hmat_triatom_unsym(
    data: TriatomHamiltonianData,
    j2: int,
    qn: NDArray[np.int64],
) -> NDArray[np.float64]:
    r"""
    Construct one unsymmetrized triatomic monomer Hamiltonian block.

    Formula:
        h_ABC = h1 + h2 + T_vr + V_res

    The rovibrational kinetic matrix uses Eq. (20) and the residual potential uses
    Eq. (23) of the ABC+D reference. Only Delta Omega = 0, +/-1, +/-2 kinetic
    couplings are generated.

    Inputs:
        data: TriatomHamiltonianData - shared radial, angular, and PES data
        j2: int - total rotational angular momentum of the triatomic monomer
        qn: NDArray[np.int64] - unsymmetrized primitive quantum numbers
            (j1, omega, v1, v2), shape (n_primitive, 4)

    Returns:
        H: NDArray[np.float64] - real symmetric Hamiltonian matrix, shape
            (n_primitive, n_primitive)
    """
    n = qn.shape[0]
    H = np.zeros((n, n), dtype=np.float64)
    lookup = {tuple(int(value) for value in state): index for index, state in enumerate(qn)}
    vmax_1 = data.radial_1.energies.size - 1
    vmax_2 = data.radial_2.energies.size - 1
    j1max = max(int(np.max(qn[:, 0])), 0)
    angular_matrices: dict[tuple[str, int], NDArray[np.float64]] = {}

    def angular_matrix(kind: str, omega: int) -> NDArray[np.float64]:
        """Return and cache one j1-space angular coupling matrix, shape (n_j1, n_j1)."""
        key = (kind, omega)
        if key in angular_matrices:
            return angular_matrices[key]

        x = data.cos_theta
        sin_theta = np.sqrt(1.0 - x**2)
        if kind == "E":
            omega_prime = omega
            factor = 1.0 / (1.0 + x)
        elif kind == "Dp":
            omega_prime = omega + 1
            factor = x / sin_theta
        elif kind == "Dm":
            omega_prime = omega - 1
            factor = x / sin_theta
        elif kind == "Fp":
            omega_prime = omega + 2
            factor = 1.0 / (1.0 + x)
        elif kind == "Fm":
            omega_prime = omega - 2
            factor = 1.0 / (1.0 + x)
        elif kind == "Gp":
            omega_prime = omega + 1
            factor = 1.0 / sin_theta
        elif kind == "Gm":
            omega_prime = omega - 1
            factor = 1.0 / sin_theta
        elif kind == "Hp":
            omega_prime = omega + 2
            factor = np.ones_like(x)
        else:
            omega_prime = omega - 2
            factor = np.ones_like(x)

        if abs(omega_prime) > j1max:
            result = np.zeros((j1max + 1, j1max + 1))
        else:
            bra = data.angular[omega_prime]
            ket = data.angular[omega]
            result = bra.T @ ((data.theta_weights * factor)[:, None] * ket)
        angular_matrices[key] = result
        return result

    for index, (_, _, v1, v2) in enumerate(qn):
        H[index, index] += data.radial_1.energies[v1] + data.radial_2.energies[v2]

    sqrt_weight = np.sqrt(data.theta_weights)
    V_flat = data.V_res.reshape(-1)
    for omega in sorted({int(value) for value in qn[:, 1]}):
        indices = np.flatnonzero(qn[:, 1] == omega)
        grid_basis = np.empty((indices.size, V_flat.size), dtype=np.float64)
        for local, index in enumerate(indices):
            j1, _, v1, v2 = qn[index]
            values = (
                data.radial_1.wavefunctions[:, v1, None, None]
                * data.radial_2.wavefunctions[None, :, v2, None]
                * (sqrt_weight * data.angular[omega][:, j1])[None, None, :]
            )
            grid_basis[local] = values.reshape(-1)
        H[np.ix_(indices, indices)] += (grid_basis * V_flat[None, :]) @ grid_basis.T

    for ket, (j1, omega, v1, v2) in enumerate(qn):
        for delta in (-2, -1, 0, 1, 2):
            omega_prime = int(omega + delta)
            if abs(omega_prime) > j2 or abs(omega_prime) > j1max:
                continue
            for j1_prime in range(abs(omega_prime), j1max + 1):
                vibrational_pairs = {(v1_prime, int(v2)) for v1_prime in range(vmax_1 + 1)}
                vibrational_pairs.update((int(v1), v2_prime) for v2_prime in range(vmax_2 + 1))
                for v1_prime, v2_prime in vibrational_pairs:
                    bra = lookup.get((j1_prime, omega_prime, v1_prime, v2_prime))
                    if bra is None:
                        continue
                    ct_1 = data.B[v1_prime, v1] if v2_prime == v2 else 0.0
                    ct_2 = data.C[v2_prime, v2] if v1_prime == v1 else 0.0
                    same_j = j1_prime == j1

                    if delta == 0:
                        coefficient = (j2 * (j2 + 1) - omega**2) / 8.0 + j1 * (j1 + 1) if same_j else 0.0
                        coefficient += (j2 * (j2 + 1) - 3 * omega**2) * angular_matrix("E", int(omega))[j1_prime, j1] / 4.0
                        value = (ct_1 + ct_2) * coefficient
                    elif delta == 1:
                        coefficient = (2 * omega + 1) * (
                            angular_matrix("Gp", int(omega))[j1_prime, j1] - angular_matrix("Dp", int(omega))[j1_prime, j1]
                        )
                        if same_j:
                            coefficient -= 2.0 * lambda_plus(int(j1), int(omega))
                        value = (ct_1 - ct_2) * lambda_plus(j2, int(omega)) * coefficient / 4.0
                    elif delta == -1:
                        coefficient = (2 * omega - 1) * (
                            angular_matrix("Gm", int(omega))[j1_prime, j1] - angular_matrix("Dm", int(omega))[j1_prime, j1]
                        )
                        if same_j:
                            coefficient -= 2.0 * lambda_minus(int(j1), int(omega))
                        value = (ct_1 - ct_2) * lambda_minus(j2, int(omega)) * coefficient / 4.0
                    elif delta == 2:
                        angular = 2.0 * angular_matrix("Fp", int(omega))[j1_prime, j1] - angular_matrix("Hp", int(omega))[j1_prime, j1]
                        value = (ct_1 + ct_2) * lambda_plus(j2, int(omega)) * lambda_plus(j2, int(omega + 1)) * angular / 16.0
                    else:
                        angular = 2.0 * angular_matrix("Fm", int(omega))[j1_prime, j1] - angular_matrix("Hm", int(omega))[j1_prime, j1]
                        value = (ct_1 + ct_2) * lambda_minus(j2, int(omega)) * lambda_minus(j2, int(omega - 1)) * angular / 16.0
                    H[bra, ket] += value

    return np.asarray(0.5 * (H + H.T), dtype=np.float64)
