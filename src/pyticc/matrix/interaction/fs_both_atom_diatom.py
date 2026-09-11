from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc._typing import JaxDevice
from pyticc.basis.angle import clebsch_gordan_half, norm_reduced_wigner_d_half
from pyticc.fine_structure.atom_diatom import FSAtomDiatomBasis
from pyticc.matrix.interaction.fs_diatom_diatom import _packed_positions
from pyticc.matrix.interaction.fs_diatom_diatom_spin import _raw_wigner_d
from pyticc.pes.spin_resolved_atom_diatom import AtomDiatomOrbitalState

ElectronicKey = tuple[int, int, int, int]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpinResolvedFSBothAtomDiatomVBasis:
    """Packed kernels for ``sum_S P_S tensor W_orb^(S)``.

    ``kernel[g,e,p]`` contracts geometric point ``g`` and electronic PES
    component ``e=(S,alpha_bra,alpha_ket)`` into lower-triangle channel pair
    ``p``. The electronic axes are ordered by ``two_total_spins`` and
    ``orbital_states``.
    """

    n_channel: int
    grid_shape: tuple[int, int]
    two_total_spins: tuple[int, ...]
    orbital_states: tuple[AtomDiatomOrbitalState, ...]
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
class SpinResolvedFSBothAtomDiatomVBasisDevice:
    """Device-resident total-spin-resolved contraction kernel."""

    kernel: jax.Array


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _primitive_amplitudes(
    basis: FSAtomDiatomBasis,
    theta: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
    *,
    helicity_sign: int = 1,
) -> tuple[tuple[int, int], list[dict[ElectronicKey, NDArray[np.complex128]]]]:
    r"""Build weighted channel amplitudes resolved by signed primitive state.

    Formula:
        For channel ``eta=(level_X,block_Y,tau_Y,j_12,K)`` and signed primitive
        electronic labels ``e=(lambda,mu,Lambda,Sigma)``,

        A_(eta,e)(r,theta) = sum_(kappa_X,kappa_Y)
          <j_X kappa_X, j_Y kappa_Y|j_12 K>
          <L lambda, S mu|j_X kappa_X>
          X^(Y)_((Lambda,Sigma),tau_Y) chi_v(r)
          sqrt(w) dtilde^(j_Y)_(kappa_Y,Omega)(theta),

        with ``kappa_X+kappa_Y=K``, ``kappa_X=lambda+mu``, and
        ``dtilde=sqrt((2j+1)/2)d`` normalized for integration over cos(theta).
        The atomic projections are already body-fixed, so the atom contributes
        no rotor function. ``Omega=Lambda+Sigma``. The quadrature weight is
        absorbed into A; the PODVR radial contraction is discrete.
    """
    theta = np.asarray(theta, dtype=np.float64)
    theta_weights = np.asarray(theta_weights, dtype=np.float64)
    vib_Y = basis.monomer_Y.vib
    grid_shape = (vib_Y.grids.size, theta.size)
    n_grid = prod(grid_shape)
    sqrt_weight = np.sqrt(theta_weights)
    atom = basis.atom
    atomic_cache: dict[tuple[int, int, int], float] = {}
    angular_cache: dict[tuple[int, int, int], NDArray[np.float64]] = {}
    amplitudes: list[dict[ElectronicKey, NDArray[np.complex128]]] = []
    for channel in basis:
        block_Y = basis.monomer_Y.blocks[channel.block_Y]
        two_j_X = atom.two_j_levels[channel.level_X]
        two_j_Y = block_Y.two_j
        signed_coefficients = block_Y.transform @ block_Y.coefficients[:, channel.tau_Y]
        components = [
            (state, float(coefficient))
            for state, coefficient in zip(block_Y.primitive_states, signed_coefficients, strict=True)
            if abs(coefficient) >= 1.0e-15
        ]
        two_K_signed = helicity_sign * channel.two_K
        channel_amplitudes: dict[ElectronicKey, NDArray[np.complex128]] = {}
        for two_lambda in range(-atom.two_L, atom.two_L + 1, 2):
            for two_mu in range(-atom.two_S, atom.two_S + 1, 2):
                two_kappa_X = two_lambda + two_mu
                if abs(two_kappa_X) > two_j_X or (two_j_X - two_kappa_X) % 2:
                    continue
                atomic_key = (two_lambda, two_mu, two_j_X)
                if atomic_key not in atomic_cache:
                    atomic_cache[atomic_key] = clebsch_gordan_half(atom.two_L, two_lambda, atom.two_S, two_mu, two_j_X)
                cg_atom = atomic_cache[atomic_key]
                if cg_atom == 0.0:
                    continue
                two_kappa_Y = two_K_signed - two_kappa_X
                if abs(two_kappa_Y) > two_j_Y or (two_j_Y - two_kappa_Y) % 2:
                    continue
                cg_couple = clebsch_gordan_half(two_j_X, two_kappa_X, two_j_Y, two_kappa_Y, channel.two_j12)
                if cg_couple == 0.0:
                    continue
                factor = cg_atom * cg_couple
                for state, coefficient in components:
                    angular_key = (state.two_j, two_kappa_Y, state.two_omega)
                    if angular_key not in angular_cache:
                        angular_cache[angular_key] = sqrt_weight * np.asarray(
                            norm_reduced_wigner_d_half(state.two_j, two_kappa_Y, state.two_omega, theta),
                            dtype=np.float64,
                        )
                    amplitude = factor * coefficient * np.multiply.outer(vib_Y.wavefunctions[:, state.v], angular_cache[angular_key]).reshape(n_grid)
                    electronic_key = (two_lambda, two_mu, state.two_lambda, state.two_sigma)
                    if electronic_key in channel_amplitudes:
                        channel_amplitudes[electronic_key] += amplitude
                    else:
                        channel_amplitudes[electronic_key] = amplitude
        amplitudes.append(channel_amplitudes)
    return grid_shape, amplitudes


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _spin_projector_kernel(
    two_S_X: int,
    two_S_Y: int,
    two_total_spin: int,
    two_mu_X_bra: int,
    two_sigma_Y_bra: int,
    two_mu_X_ket: int,
    two_sigma_Y_ket: int,
    theta: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r"""Return ``<mu'_X Sigma'_Y|P_S|mu_X Sigma_Y>`` on the theta grid.

    The atomic spin projections ``mu`` are already body-fixed, because the
    atom has no molecular axis. The molecular spin projections ``Sigma`` live
    on the molecular axis and are rotated to the body-fixed z axis with the
    Euler angles ``omega_Y=(0,theta,0)``:

    Q^S_(mu' Sigma',mu Sigma) = sum_(mu'_Y mu_Y)
      d^{S_Y*}_{mu'_Y Sigma'}(theta) P^S_(mu'_X mu'_Y, mu_X mu_Y)
      d^{S_Y}_{mu_Y Sigma}(theta).
    """
    result = np.zeros(theta.size, dtype=np.complex128)
    mu_Y_values = range(-two_S_Y, two_S_Y + 1, 2)
    d_Y_bra = {two_mu_Y: _raw_wigner_d(two_S_Y, two_mu_Y, two_sigma_Y_bra, theta) for two_mu_Y in mu_Y_values}
    d_Y_ket = {two_mu_Y: _raw_wigner_d(two_S_Y, two_mu_Y, two_sigma_Y_ket, theta) for two_mu_Y in mu_Y_values}
    for two_mu_Y_bra in mu_Y_values:
        two_M_S = two_mu_X_bra + two_mu_Y_bra
        coefficient_bra = clebsch_gordan_half(two_S_X, two_mu_X_bra, two_S_Y, two_mu_Y_bra, two_total_spin)
        if coefficient_bra == 0.0:
            continue
        two_mu_Y_ket = two_M_S - two_mu_X_ket
        if abs(two_mu_Y_ket) > two_S_Y or (two_S_Y - two_mu_Y_ket) % 2:
            continue
        coefficient_ket = clebsch_gordan_half(two_S_X, two_mu_X_ket, two_S_Y, two_mu_Y_ket, two_total_spin)
        if coefficient_ket == 0.0:
            continue
        result += coefficient_bra * coefficient_ket * np.conj(d_Y_bra[two_mu_Y_bra]) * d_Y_ket[two_mu_Y_ket]
    return result


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare(
    basis: FSAtomDiatomBasis,
    two_total_spins: Sequence[int],
    orbital_states: Sequence[AtomDiatomOrbitalState],
    theta: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
) -> SpinResolvedFSBothAtomDiatomVBasis:
    r"""Prepare the total-spin-resolved atom-diatom interaction kernel.

    For primitive electronic labels
    ``e=(lambda,mu,Lambda,Sigma)``, this evaluates

    ``G[g,S,alpha',alpha,eta',eta] = A*_(eta',e') Q^S_(mu'Sigma',mu Sigma) A_(eta,e)``

    subject to ``alpha'=(lambda',Lambda')`` and
    ``alpha=(lambda,Lambda)``. Contracting it with dense Hermitian orbital
    matrices ``W^(S)_(alpha',alpha)`` gives the channel interaction. Retaining
    the complex kernel is essential when ``W`` itself has imaginary couplings.

    The basis stores one representative with nonnegative ``K``. For a
    parity-conserving scalar interaction, the two equal diagonal-helicity
    contributions of a normalized ``K>0`` parity pair reduce exactly to the
    positive-``K`` primitive matrix element. Consequently no separate
    ``-K`` amplitude or additional factor of one half enters this kernel.
    """
    theta = np.asarray(theta, dtype=np.float64)
    spins = tuple(int(value) for value in two_total_spins)
    orbitals = tuple(orbital_states)
    if not spins or len(set(spins)) != len(spins):
        raise ValueError("two_total_spins must contain unique values")
    if not orbitals or len(set(orbitals)) != len(orbitals):
        raise ValueError("orbital_states must contain unique values")

    grid_shape, amplitudes = _primitive_amplitudes(basis, theta, theta_weights)
    n_grid = prod(grid_shape)
    n_electronic = len(spins) * len(orbitals) ** 2
    pair_rows, pair_columns = np.tril_indices(basis.n_channel)
    kernel = np.zeros((n_grid, n_electronic, pair_rows.size), dtype=np.complex128)
    orbital_indices = {(state.two_lambda_atom, state.two_lambda_Y): index for index, state in enumerate(orbitals)}
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
                            basis.atom.two_S,
                            basis.monomer_Y.two_S,
                            *projector_key,
                            theta,
                        )
                    projector = np.broadcast_to(projector_cache[projector_key][None, :], grid_shape).reshape(n_grid)
                    electronic_index = spin_index * len(orbitals) ** 2 + bra_orbital * len(orbitals) + ket_orbital
                    kernel[:, electronic_index, pair_index] += projector * np.conjugate(bra_amplitude) * ket_amplitude
    return SpinResolvedFSBothAtomDiatomVBasis(
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
    V_basis: SpinResolvedFSBothAtomDiatomVBasis,
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
    message = f"Spin-resolved FS A+BC PES grid has shape {values.shape}, expected {expected} with optional leading R axis"
    logger.error(message)
    raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def contract(
    V_basis: SpinResolvedFSBothAtomDiatomVBasis,
    potential: NDArray[np.float64] | NDArray[np.complex128],
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Contract spin-resolved orbital PES grids into Hermitian matrices."""
    batches, batched = _potential_batches(V_basis, potential)
    host_batches = np.asarray(batches)
    indices = tuple(range(V_basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    pair_rows, pair_columns, packed = _packed_positions(V_basis.n_channel, indices)
    kernel = V_basis.kernel if channel_indices is None else V_basis.kernel[:, :, packed]
    contracted = host_batches.reshape(host_batches.shape[0], -1) @ kernel.reshape(kernel.shape[0] * kernel.shape[1], packed.size)
    selected = np.asarray(indices, dtype=np.int64)
    reversed_pairs = selected[pair_rows] < selected[pair_columns]
    contracted[:, reversed_pairs] = np.conjugate(contracted[:, reversed_pairs])
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
    V_basis: SpinResolvedFSBothAtomDiatomVBasis,
    device: JaxDevice,
) -> SpinResolvedFSBothAtomDiatomVBasisDevice:
    """Copy the complete spin-resolved kernel to one JAX device."""
    return SpinResolvedFSBothAtomDiatomVBasisDevice(jax.device_put(V_basis.kernel, device))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@jax.jit
def _contract_device(potential: jax.Array, kernel: jax.Array) -> jax.Array:
    return potential.reshape(potential.shape[0], -1) @ kernel.reshape(kernel.shape[0] * kernel.shape[1], kernel.shape[-1])


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def contract_device(
    V_basis: SpinResolvedFSBothAtomDiatomVBasis,
    basis_device: SpinResolvedFSBothAtomDiatomVBasisDevice,
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
    selected = np.asarray(indices, dtype=np.int64)
    reversed_pairs = selected[pair_rows] < selected[pair_columns]
    contracted = jnp.where(reversed_pairs[None, :], jnp.conjugate(contracted), contracted)
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
