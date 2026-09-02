from collections.abc import Sequence
from dataclasses import dataclass
from math import prod, sqrt

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc._typing import JaxDevice
from pyticc.basis.angle import clebsch_gordan_half, norm_reduced_wigner_d_half
from pyticc.fine_structure.diatom_diatom import FSDiatomDiatomBasis
from pyticc.matrix.interaction.fs_diatom_diatom import _packed_positions, _primitive_amplitudes
from pyticc.pes.spin_resolved_diatom_diatom import OrbitalState


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpinResolvedFSDiatomDiatomVBasis:
    """Packed kernels for ``sum_S P_S tensor W_orb^(S)``.

    ``kernel[g,e,p]`` contracts geometric point ``g`` and electronic PES
    component ``e=(S,alpha_bra,alpha_ket)`` into lower-triangle channel pair
    ``p``. The electronic axes are ordered by ``two_total_spins`` and
    ``orbital_states``.
    """

    n_channel: int
    grid_shape: tuple[int, int, int, int, int]
    two_total_spins: tuple[int, ...]
    orbital_states: tuple[OrbitalState, ...]
    pair_rows: NDArray[np.int64]
    pair_columns: NDArray[np.int64]
    kernel: NDArray[np.complex128]

    @property
    def electronic_shape(self) -> tuple[int, int, int]:
        """Return ``(n_spin,n_orbital,n_orbital)``."""
        n_orbital = len(self.orbital_states)
        return len(self.two_total_spins), n_orbital, n_orbital


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpinResolvedFSDiatomDiatomVBasisDevice:
    """Device-resident total-spin-resolved contraction kernel."""

    kernel: jax.Array


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _raw_wigner_d(
    two_S: int,
    two_mu: int,
    two_sigma: int,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the unnormalized Wigner ``d^S_(mu,sigma)``."""
    normalized = norm_reduced_wigner_d_half(two_S, two_mu, two_sigma, theta)
    return np.asarray(normalized, dtype=np.float64) / sqrt((two_S + 1.0) / 2.0)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _spin_projector_kernel(
    two_S_X: int,
    two_S_Y: int,
    two_total_spin: int,
    two_sigma_X_bra: int,
    two_sigma_Y_bra: int,
    two_sigma_X_ket: int,
    two_sigma_Y_ket: int,
    theta_X: NDArray[np.float64],
    theta_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r"""Return ``<Sigma'_X Sigma'_Y|P_S|Sigma_X Sigma_Y>`` on the angle grid.

    The BF spin basis is coupled along the intermolecular axis. Consistently
    with the rotor phase used by the scalar 5D kernel, molecule X has Euler
    angles ``(phi,theta_X,0)`` and molecule Y has ``(0,theta_Y,0)``.
    """
    result = np.zeros((theta_X.size, theta_Y.size, phi.size), dtype=np.complex128)
    mu_X_values = range(-two_S_X, two_S_X + 1, 2)
    mu_Y_values = range(-two_S_Y, two_S_Y + 1, 2)
    d_X_bra = {two_mu: _raw_wigner_d(two_S_X, two_mu, two_sigma_X_bra, theta_X) for two_mu in mu_X_values}
    d_X_ket = {two_mu: _raw_wigner_d(two_S_X, two_mu, two_sigma_X_ket, theta_X) for two_mu in mu_X_values}
    d_Y_bra = {two_mu: _raw_wigner_d(two_S_Y, two_mu, two_sigma_Y_bra, theta_Y) for two_mu in mu_Y_values}
    d_Y_ket = {two_mu: _raw_wigner_d(two_S_Y, two_mu, two_sigma_Y_ket, theta_Y) for two_mu in mu_Y_values}
    for two_mu_X_bra in mu_X_values:
        for two_mu_Y_bra in mu_Y_values:
            two_M_bra = two_mu_X_bra + two_mu_Y_bra
            coefficient_bra = clebsch_gordan_half(
                two_S_X,
                two_mu_X_bra,
                two_S_Y,
                two_mu_Y_bra,
                two_total_spin,
            )
            if coefficient_bra == 0.0:
                continue
            for two_mu_X_ket in mu_X_values:
                two_mu_Y_ket = two_M_bra - two_mu_X_ket
                if abs(two_mu_Y_ket) > two_S_Y or (two_S_Y - two_mu_Y_ket) % 2:
                    continue
                coefficient_ket = clebsch_gordan_half(
                    two_S_X,
                    two_mu_X_ket,
                    two_S_Y,
                    two_mu_Y_ket,
                    two_total_spin,
                )
                if coefficient_ket == 0.0:
                    continue
                phase = np.exp(0.5j * (two_mu_X_bra - two_mu_X_ket) * phi)
                result += (
                    coefficient_bra
                    * coefficient_ket
                    * d_X_bra[two_mu_X_bra][:, None, None]
                    * d_X_ket[two_mu_X_ket][:, None, None]
                    * d_Y_bra[two_mu_Y_bra][None, :, None]
                    * d_Y_ket[two_mu_Y_ket][None, :, None]
                    * phase[None, None, :]
                )
    return result


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _ladder_factor(two_S: int, two_mu: int, direction: int) -> float:
    """Return the spin raising/lowering factor for doubled quantum numbers."""
    S = two_S / 2.0
    mu = two_mu / 2.0
    return sqrt(S * (S + 1.0) - mu * (mu + direction))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _spin_dipole_kernel(
    two_S_X: int,
    two_S_Y: int,
    two_sigma_X_bra: int,
    two_sigma_Y_bra: int,
    two_sigma_X_ket: int,
    two_sigma_Y_ket: int,
    theta_X: NDArray[np.float64],
    theta_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r"""Return the dimensionless direct spin-dipole kernel in the molecular frames.

    Formula:
        With the intermolecular axis as BF z,

        T_dd = -2 S_Xz S_Yz + (S_X+ S_Y- + S_X- S_Y+)/2.

        Molecular-frame spin functions are rotated to BF projections by
        ``|S Sigma> = sum_mu D^S_(mu,Sigma)(omega)|S mu>``. Molecule X uses
        ``omega_X=(phi,theta_X,0)`` and molecule Y uses
        ``omega_Y=(0,theta_Y,0)``. The returned array is
        ``<Sigma'_X Sigma'_Y|T_dd|Sigma_X Sigma_Y>``.

    Inputs:
        two_S_X: int - twice spin of monomer X
        two_S_Y: int - twice spin of monomer Y
        two_sigma_X_bra: int - twice molecular-frame bra projection on X
        two_sigma_Y_bra: int - twice molecular-frame bra projection on Y
        two_sigma_X_ket: int - twice molecular-frame ket projection on X
        two_sigma_Y_ket: int - twice molecular-frame ket projection on Y
        theta_X: NDArray[np.float64] - X polar angles in radians
        theta_Y: NDArray[np.float64] - Y polar angles in radians
        phi: NDArray[np.float64] - torsional angles in radians

    Returns:
        kernel: NDArray[np.complex128] - shape
            ``(n_theta_X,n_theta_Y,n_phi)``
    """
    result = np.zeros((theta_X.size, theta_Y.size, phi.size), dtype=np.complex128)
    mu_X_values = range(-two_S_X, two_S_X + 1, 2)
    mu_Y_values = range(-two_S_Y, two_S_Y + 1, 2)
    d_X_bra = {two_mu: _raw_wigner_d(two_S_X, two_mu, two_sigma_X_bra, theta_X) for two_mu in mu_X_values}
    d_X_ket = {two_mu: _raw_wigner_d(two_S_X, two_mu, two_sigma_X_ket, theta_X) for two_mu in mu_X_values}
    d_Y_bra = {two_mu: _raw_wigner_d(two_S_Y, two_mu, two_sigma_Y_bra, theta_Y) for two_mu in mu_Y_values}
    d_Y_ket = {two_mu: _raw_wigner_d(two_S_Y, two_mu, two_sigma_Y_ket, theta_Y) for two_mu in mu_Y_values}
    for two_mu_X_ket in mu_X_values:
        for two_mu_Y_ket in mu_Y_values:
            transitions = (
                (two_mu_X_ket, two_mu_Y_ket, -2.0 * (two_mu_X_ket / 2.0) * (two_mu_Y_ket / 2.0)),
                (
                    two_mu_X_ket + 2,
                    two_mu_Y_ket - 2,
                    0.5 * _ladder_factor(two_S_X, two_mu_X_ket, 1) * _ladder_factor(two_S_Y, two_mu_Y_ket, -1),
                ),
                (
                    two_mu_X_ket - 2,
                    two_mu_Y_ket + 2,
                    0.5 * _ladder_factor(two_S_X, two_mu_X_ket, -1) * _ladder_factor(two_S_Y, two_mu_Y_ket, 1),
                ),
            )
            for two_mu_X_bra, two_mu_Y_bra, coefficient in transitions:
                if coefficient == 0.0 or two_mu_X_bra not in d_X_bra or two_mu_Y_bra not in d_Y_bra:
                    continue
                phase = np.exp(0.5j * (two_mu_X_bra - two_mu_X_ket) * phi)
                result += (
                    coefficient
                    * d_X_bra[two_mu_X_bra][:, None, None]
                    * d_X_ket[two_mu_X_ket][:, None, None]
                    * d_Y_bra[two_mu_Y_bra][None, :, None]
                    * d_Y_ket[two_mu_Y_ket][None, :, None]
                    * phase[None, None, :]
                )
    return result


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def magnetic_dipole_matrix(
    basis: FSDiatomDiatomBasis,
    theta_X: NDArray[np.float64],
    theta_weights_X: NDArray[np.float64],
    theta_Y: NDArray[np.float64],
    theta_weights_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Build the dimensionless direct electron-spin dipole matrix.

    Formula:
        For channel amplitudes ``A_(eta,e)(g)`` used by the orbital PES kernel,

        M_dd(eta',eta) = delta_(K'K)/pi sum_g sum_(Lambda_X,Lambda_Y)
          Re[A*_(eta',e')(g) T_dd(Sigma',Sigma;g) A_(eta,e)(g)],

        where the signed orbital projections are conserved and ``T_dd`` is
        defined by :func:`_spin_dipole_kernel`. The physical interaction added
        by the scattering Hamiltonian is ``C_dd M_dd/R^3``.

    Inputs:
        basis: FSDiatomDiatomBasis - parity-adapted two-diatom channels
        theta_X: NDArray[np.float64] - X polar nodes in radians
        theta_weights_X: NDArray[np.float64] - weights over cos(theta_X)
        theta_Y: NDArray[np.float64] - Y polar nodes in radians
        theta_weights_Y: NDArray[np.float64] - weights over cos(theta_Y)
        phi: NDArray[np.float64] - torsional nodes on [0,pi]
        phi_weights: NDArray[np.float64] - torsional weights

    Returns:
        matrix: NDArray[np.float64] - real symmetric dimensionless matrix with
            shape ``(n_channel,n_channel)``
    """
    grid_shape, amplitudes = _primitive_amplitudes(
        basis,
        theta_X,
        theta_weights_X,
        theta_Y,
        theta_weights_Y,
        phi,
        phi_weights,
    )
    n_grid = prod(grid_shape)
    matrix = np.zeros((basis.n_channel, basis.n_channel), dtype=np.float64)
    kernel_cache: dict[tuple[int, int, int, int], NDArray[np.complex128]] = {}
    for row in range(basis.n_channel):
        for column in range(row + 1):
            if basis[row].two_K != basis[column].two_K:
                continue
            value = 0.0
            for bra_key, bra_amplitude in amplitudes[row].items():
                for ket_key, ket_amplitude in amplitudes[column].items():
                    if bra_key[0] != ket_key[0] or bra_key[2] != ket_key[2]:
                        continue
                    spin_key = (bra_key[1], bra_key[3], ket_key[1], ket_key[3])
                    if spin_key not in kernel_cache:
                        kernel_cache[spin_key] = np.broadcast_to(
                            _spin_dipole_kernel(
                                basis.monomer_X.two_S,
                                basis.monomer_Y.two_S,
                                *spin_key,
                                theta_X,
                                theta_Y,
                                phi,
                            )[None, None],
                            grid_shape,
                        ).reshape(n_grid)
                    value += float(np.sum(np.real(np.conjugate(bra_amplitude) * kernel_cache[spin_key] * ket_amplitude)))
            matrix[row, column] = value / np.pi
            matrix[column, row] = matrix[row, column]
    return matrix


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare(
    basis: FSDiatomDiatomBasis,
    two_total_spins: Sequence[int],
    orbital_states: Sequence[OrbitalState],
    theta_X: NDArray[np.float64],
    theta_weights_X: NDArray[np.float64],
    theta_Y: NDArray[np.float64],
    theta_weights_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> SpinResolvedFSDiatomDiatomVBasis:
    r"""Prepare the total-spin-resolved two-diatom interaction kernel.

    For primitive electronic labels
    ``e=(Lambda_X,Sigma_X,Lambda_Y,Sigma_Y)``, this evaluates

    ``G[g,S,alpha',alpha,eta',eta] = A*_(eta',e') Q^S_(Sigma',Sigma) A_(eta,e)/pi``

    subject to ``alpha'=(Lambda'_X,Lambda'_Y)`` and
    ``alpha=(Lambda_X,Lambda_Y)``. Contracting it with dense Hermitian orbital
    matrices ``W^(S)_(alpha',alpha)`` gives the channel interaction. Retaining
    the complex kernel is essential when ``W`` itself has imaginary couplings.
    """
    theta_X = np.asarray(theta_X, dtype=np.float64)
    theta_Y = np.asarray(theta_Y, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    spins = tuple(int(value) for value in two_total_spins)
    orbitals = tuple(orbital_states)
    if not spins or len(set(spins)) != len(spins):
        raise ValueError("two_total_spins must contain unique values")
    if not orbitals or len(set(orbitals)) != len(orbitals):
        raise ValueError("orbital_states must contain unique values")

    grid_shape, amplitudes = _primitive_amplitudes(
        basis,
        theta_X,
        theta_weights_X,
        theta_Y,
        theta_weights_Y,
        phi,
        phi_weights,
    )
    n_grid = prod(grid_shape)
    n_electronic = len(spins) * len(orbitals) ** 2
    pair_rows, pair_columns = np.tril_indices(basis.n_channel)
    kernel = np.zeros((n_grid, n_electronic, pair_rows.size), dtype=np.complex128)
    orbital_indices = {(state.two_lambda_X, state.two_lambda_Y): index for index, state in enumerate(orbitals)}
    projector_cache: dict[tuple[int, int, int, int, int], NDArray[np.complex128]] = {}

    for pair_index, (row, column) in enumerate(zip(pair_rows, pair_columns, strict=True)):
        row_index = int(row)
        column_index = int(column)
        if basis[row_index].two_K != basis[column_index].two_K:
            continue
        for bra_key, bra_amplitude in amplitudes[row_index].items():
            bra_orbital = orbital_indices.get((bra_key[0], bra_key[2]))
            if bra_orbital is None:
                continue
            for ket_key, ket_amplitude in amplitudes[column_index].items():
                ket_orbital = orbital_indices.get((ket_key[0], ket_key[2]))
                if ket_orbital is None:
                    continue
                for spin_index, two_total_spin in enumerate(spins):
                    projector_key = (two_total_spin, bra_key[1], bra_key[3], ket_key[1], ket_key[3])
                    if projector_key not in projector_cache:
                        projector_cache[projector_key] = _spin_projector_kernel(
                            basis.monomer_X.two_S,
                            basis.monomer_Y.two_S,
                            *projector_key,
                            theta_X,
                            theta_Y,
                            phi,
                        )
                    projector = np.broadcast_to(
                        projector_cache[projector_key][None, None],
                        grid_shape,
                    ).reshape(n_grid)
                    electronic_index = spin_index * len(orbitals) ** 2 + bra_orbital * len(orbitals) + ket_orbital
                    kernel[:, electronic_index, pair_index] += np.conjugate(bra_amplitude) * projector * ket_amplitude
    kernel *= 1.0 / np.pi
    return SpinResolvedFSDiatomDiatomVBasis(
        n_channel=basis.n_channel,
        grid_shape=grid_shape,
        two_total_spins=spins,
        orbital_states=orbitals,
        pair_rows=np.asarray(pair_rows, dtype=np.int64),
        pair_columns=np.asarray(pair_columns, dtype=np.int64),
        kernel=np.ascontiguousarray(kernel),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _potential_batches(
    V_basis: SpinResolvedFSDiatomDiatomVBasis,
    potential: NDArray[np.float64] | NDArray[np.complex128] | jax.Array,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128] | jax.Array, bool]:
    """Validate and flatten spin/orbital PES grids."""
    values = potential if isinstance(potential, jax.Array) else np.asarray(potential)
    expected = (*V_basis.grid_shape, *V_basis.electronic_shape)
    n_grid = prod(V_basis.grid_shape)
    n_electronic = prod(V_basis.electronic_shape)
    if values.shape == expected:
        return values.reshape(1, n_grid, n_electronic), False
    if values.ndim == len(expected) + 1 and values.shape[1:] == expected:
        return values.reshape(values.shape[0], n_grid, n_electronic), True
    if values.ndim == 3 and values.shape[1:] == (n_grid, n_electronic):
        return values, True
    message = f"Spin-resolved FS diatom-diatom PES grid has shape {values.shape}, expected {expected} with optional leading R axis"
    logger.error(message)
    raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def contract(
    V_basis: SpinResolvedFSDiatomDiatomVBasis,
    potential: NDArray[np.float64] | NDArray[np.complex128],
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Contract spin-resolved orbital PES grids into Hermitian matrices."""
    batches, batched = _potential_batches(V_basis, potential)
    host_batches = np.asarray(batches)
    indices = tuple(range(V_basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    pair_rows, pair_columns, packed = _packed_positions(V_basis.n_channel, indices)
    contracted = np.einsum("bge,gep->bp", host_batches, V_basis.kernel[:, :, packed], optimize=True)
    matrix = np.zeros((host_batches.shape[0], len(indices), len(indices)), dtype=np.result_type(host_batches, V_basis.kernel))
    matrix[:, pair_rows, pair_columns] = contracted
    off_diagonal = pair_rows != pair_columns
    matrix[:, pair_columns[off_diagonal], pair_rows[off_diagonal]] = np.conjugate(contracted[:, off_diagonal])
    diagonal = ~off_diagonal
    matrix[:, pair_rows[diagonal], pair_columns[diagonal]] = np.real(contracted[:, diagonal])
    return matrix if batched else matrix[0]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def device_basis(
    V_basis: SpinResolvedFSDiatomDiatomVBasis,
    device: JaxDevice,
) -> SpinResolvedFSDiatomDiatomVBasisDevice:
    """Copy the complete spin-resolved kernel to one JAX device."""
    return SpinResolvedFSDiatomDiatomVBasisDevice(jax.device_put(V_basis.kernel, device))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@jax.jit
def _contract_device(potential: jax.Array, kernel: jax.Array) -> jax.Array:
    return jnp.einsum("bge,gep->bp", potential, kernel, optimize=True)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def contract_device(
    V_basis: SpinResolvedFSDiatomDiatomVBasis,
    basis_device: SpinResolvedFSDiatomDiatomVBasisDevice,
    potential: NDArray[np.float64] | NDArray[np.complex128] | jax.Array,
    device: JaxDevice,
    channel_indices: Sequence[int] | None = None,
) -> jax.Array:
    """Contract selected exact channel blocks on a JAX device."""
    batches, batched = _potential_batches(V_basis, potential)
    indices = tuple(range(V_basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    pair_rows, pair_columns, packed = _packed_positions(V_basis.n_channel, indices)
    potential_device = jax.device_put(batches, device)
    contracted = _contract_device(potential_device, basis_device.kernel[:, :, packed])
    matrix = jnp.zeros(
        (potential_device.shape[0], len(indices), len(indices)),
        dtype=jnp.result_type(potential_device, basis_device.kernel),
        device=device,
    )
    matrix = matrix.at[:, pair_rows, pair_columns].set(contracted)
    off_diagonal = pair_rows != pair_columns
    matrix = matrix.at[:, pair_columns[off_diagonal], pair_rows[off_diagonal]].set(jnp.conjugate(contracted[:, off_diagonal]))
    diagonal = ~off_diagonal
    matrix = matrix.at[:, pair_rows[diagonal], pair_columns[diagonal]].set(jnp.real(contracted[:, diagonal]))
    return matrix if batched else matrix[0]


# ----------------------------------------------------------------------------------------
