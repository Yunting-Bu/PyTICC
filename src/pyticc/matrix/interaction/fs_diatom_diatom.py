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
from pyticc.fine_structure.diatom_diatom import FSDiatomDiatomBasis

ElectronicKey = tuple[int, int, int, int]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSDiatomDiatomVBasis:
    """
    Packed scalar-PES quadrature kernels for two fine-structure diatoms.

    Members:
        n_channel: int - complete fine-structure channel count
        grid_shape: tuple[int,int,int,int,int] - grid sizes ordered as
            ``(r_X,r_Y,theta_X,theta_Y,phi)``
        pair_rows: NDArray[np.int64] - lower-triangle row indices, shape
            ``(n_pair,)``
        pair_columns: NDArray[np.int64] - lower-triangle column indices, shape
            ``(n_pair,)``
        kernel: NDArray[np.float64] - scalar-PES contraction operator with
            shape ``(n_grid,n_pair)``
    """

    n_channel: int
    grid_shape: tuple[int, int, int, int, int]
    pair_rows: NDArray[np.int64]
    pair_columns: NDArray[np.int64]
    kernel: NDArray[np.float64]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSDiatomDiatomVBasisDevice:
    """Device-resident scalar-PES contraction kernel for two FS diatoms.

    Members:
        kernel: jax.Array - complete packed kernel, shape ``(n_grid,n_pair)``
    """

    kernel: jax.Array


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _angular_amplitude(
    two_j_X: int,
    two_omega_X: int,
    two_j_Y: int,
    two_omega_Y: int,
    two_j12: int,
    two_K: int,
    theta_X: NDArray[np.float64],
    theta_weights_X: NDArray[np.float64],
    theta_Y: NDArray[np.float64],
    theta_weights_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r"""
    Build one weighted coupled-rotor amplitude for fixed primitive Omegas.

    Formula:
        With doubled quantum numbers supplied to the function, the physical
        amplitude on the angular grid is

        A(theta_X,theta_Y,phi) = sqrt(w_X w_Y w_phi)
          sum_(m_X+m_Y=K)
          <j_X m_X,j_Y m_Y|j_12 K>
          dtilde^(j_X)_(m_X,Omega_X)(theta_X)
          dtilde^(j_Y)_(m_Y,Omega_Y)(theta_Y)
          exp(i m_X phi).

        ``dtilde=sqrt((2j+1)/2)d`` is normalized for integration over
        ``cos(theta)``. The torsional quadrature spans ``[0,pi]``; the matching
        ``1/pi`` symmetry normalization is applied by :func:`prepare`.

    Inputs:
        two_j_X: int - twice monomer-X angular momentum
        two_omega_X: int - twice monomer-X molecular-axis projection
        two_j_Y: int - twice monomer-Y angular momentum
        two_omega_Y: int - twice monomer-Y molecular-axis projection
        two_j12: int - twice coupled monomer angular momentum
        two_K: int - twice BF helicity
        theta_X: NDArray[np.float64] - monomer-X polar nodes in radians
        theta_weights_X: NDArray[np.float64] - quadrature weights over
            ``cos(theta_X)``
        theta_Y: NDArray[np.float64] - monomer-Y polar nodes in radians
        theta_weights_Y: NDArray[np.float64] - quadrature weights over
            ``cos(theta_Y)``
        phi: NDArray[np.float64] - torsional nodes in radians
        phi_weights: NDArray[np.float64] - quadrature weights over ``[0,pi]``

    Returns:
        amplitude: NDArray[np.complex128] - weighted angular amplitude with
            shape ``(n_theta_X,n_theta_Y,n_phi)``
    """
    amplitude = np.zeros((theta_X.size, theta_Y.size, phi.size), dtype=np.complex128)
    sqrt_weight_X = np.sqrt(theta_weights_X)
    sqrt_weight_Y = np.sqrt(theta_weights_Y)
    sqrt_weight_phi = np.sqrt(phi_weights)
    for two_m_X in range(-two_j_X, two_j_X + 1, 2):
        two_m_Y = two_K - two_m_X
        if abs(two_m_Y) > two_j_Y or (two_j_Y - two_m_Y) % 2:
            continue
        coefficient = clebsch_gordan_half(two_j_X, two_m_X, two_j_Y, two_m_Y, two_j12)
        if coefficient == 0.0:
            continue
        d_X = sqrt_weight_X * np.asarray(
            norm_reduced_wigner_d_half(two_j_X, two_m_X, two_omega_X, theta_X),
            dtype=np.float64,
        )
        d_Y = sqrt_weight_Y * np.asarray(
            norm_reduced_wigner_d_half(two_j_Y, two_m_Y, two_omega_Y, theta_Y),
            dtype=np.float64,
        )
        phase = sqrt_weight_phi * np.exp(0.5j * two_m_X * phi)
        amplitude += coefficient * d_X[:, None, None] * d_Y[None, :, None] * phase[None, None, :]
    return amplitude


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _primitive_amplitudes(
    basis: FSDiatomDiatomBasis,
    theta_X: NDArray[np.float64],
    theta_weights_X: NDArray[np.float64],
    theta_Y: NDArray[np.float64],
    theta_weights_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
    *,
    helicity_sign: int = 1,
) -> tuple[tuple[int, int, int, int, int], list[dict[ElectronicKey, NDArray[np.complex128]]]]:
    """Build weighted channel amplitudes resolved by signed primitive state."""
    theta_X = np.asarray(theta_X, dtype=np.float64)
    theta_Y = np.asarray(theta_Y, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    theta_weights_X = np.asarray(theta_weights_X, dtype=np.float64)
    theta_weights_Y = np.asarray(theta_weights_Y, dtype=np.float64)
    phi_weights = np.asarray(phi_weights, dtype=np.float64)
    vib_X = basis.monomer_X.vib
    vib_Y = basis.monomer_Y.vib
    grid_shape = (vib_X.grids.size, vib_Y.grids.size, theta_X.size, theta_Y.size, phi.size)
    n_grid = prod(grid_shape)
    angular_cache: dict[tuple[int, int, int, int, int, int], NDArray[np.complex128]] = {}
    amplitudes: list[dict[ElectronicKey, NDArray[np.complex128]]] = []

    source = tuple(basis) if basis.exchange is None else basis.exchange.source_channels
    for channel in source:
        block_X = basis.monomer_X.blocks[channel.block_X]
        block_Y = basis.monomer_Y.blocks[channel.block_Y]
        coefficients_X = block_X.transform @ block_X.coefficients[:, channel.tau_X]
        coefficients_Y = block_Y.transform @ block_Y.coefficients[:, channel.tau_Y]
        channel_amplitudes: dict[ElectronicKey, NDArray[np.complex128]] = {}
        for state_X, coefficient_X in zip(block_X.primitive_states, coefficients_X, strict=True):
            if abs(coefficient_X) < 1.0e-15:
                continue
            radial_X = vib_X.wavefunctions[:, state_X.v]
            for state_Y, coefficient_Y in zip(block_Y.primitive_states, coefficients_Y, strict=True):
                coefficient = coefficient_X * coefficient_Y
                if abs(coefficient) < 1.0e-15:
                    continue
                angular_key = (
                    state_X.two_j,
                    state_X.two_omega,
                    state_Y.two_j,
                    state_Y.two_omega,
                    channel.two_j12,
                    helicity_sign * channel.two_K,
                )
                if angular_key not in angular_cache:
                    angular_cache[angular_key] = _angular_amplitude(
                        *angular_key,
                        theta_X,
                        theta_weights_X,
                        theta_Y,
                        theta_weights_Y,
                        phi,
                        phi_weights,
                    )
                radial = np.multiply.outer(radial_X, vib_Y.wavefunctions[:, state_Y.v])
                amplitude = coefficient * (radial[:, :, None, None, None] * angular_cache[angular_key][None, None]).reshape(n_grid)
                electronic_key = (state_X.two_lambda, state_X.two_sigma, state_Y.two_lambda, state_Y.two_sigma)
                if electronic_key in channel_amplitudes:
                    channel_amplitudes[electronic_key] += amplitude
                else:
                    channel_amplitudes[electronic_key] = amplitude
        amplitudes.append(channel_amplitudes)
    if basis.exchange is not None:
        adapted: list[dict[ElectronicKey, NDArray[np.complex128]]] = []
        for indices, weights in zip(basis.exchange.source_indices, basis.exchange.coefficients, strict=True):
            combined: dict[ElectronicKey, NDArray[np.complex128]] = {}
            for index, weight in zip(indices, weights, strict=True):
                if weight == 0.0:
                    continue
                for key, amplitude in amplitudes[index].items():
                    if key in combined:
                        combined[key] += weight * amplitude
                    else:
                        combined[key] = weight * amplitude
            adapted.append(combined)
        amplitudes = adapted
    return grid_shape, amplitudes


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare(
    basis: FSDiatomDiatomBasis,
    theta_X: NDArray[np.float64],
    theta_weights_X: NDArray[np.float64],
    theta_Y: NDArray[np.float64],
    theta_weights_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> FSDiatomDiatomVBasis:
    r"""
    Prepare a spin-independent scalar-PES kernel in the two-FS-diatom basis.

    Formula:
        For channel ``eta=(n_X,n_Y,j_12,K)`` and signed primitive electronic
        pair ``e=(Lambda_X,Sigma_X,Lambda_Y,Sigma_Y)``, define

        A_(eta,e)(xi) = chi_(v_X)(r_X) chi_(v_Y)(r_Y)
          X^(X)_(e_X,tau_X) X^(Y)_(e_Y,tau_Y)
          sum_(m_X+m_Y=K) CG dtilde_X dtilde_Y exp(i m_X phi).

        The spin-independent scalar interaction is the identity in both signed
        primitive electronic spaces,

        <e'|V|e> = V(xi) delta_(e'e),

        so the real field-free BF matrix is

        V_(eta'eta)(R) = delta_(K'K) / pi
          sum_g V(R,g) sum_e Re[A*_(eta',e)(g) A_(eta,e)(g)].

        Grid index ``g`` is ordered as
        ``(r_X,r_Y,theta_X,theta_Y,phi)``. Polar and torsional Gaussian weights
        are absorbed into A; PODVR radial contraction is discrete. Monomer
        signed coefficients are ``X=U C`` and are therefore already normalized.
        For exchange-adapted channels replace A by A T_eta, using the real
        expansion in basis.exchange; this gives T_eta.T V T_eta without
        applying any additional normalization to the resulting matrix.

    Inputs:
        basis: FSDiatomDiatomBasis - parity-adapted two-diatom channels
        theta_X: NDArray[np.float64] - monomer-X polar nodes in radians
        theta_weights_X: NDArray[np.float64] - weights over ``cos(theta_X)``
        theta_Y: NDArray[np.float64] - monomer-Y polar nodes in radians
        theta_weights_Y: NDArray[np.float64] - weights over ``cos(theta_Y)``
        phi: NDArray[np.float64] - torsional nodes on ``[0,pi]`` in radians
        phi_weights: NDArray[np.float64] - torsional quadrature weights

    Returns:
        V_basis: FSDiatomDiatomVBasis - reusable packed scalar-PES kernel
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

    pair_rows, pair_columns = np.tril_indices(basis.n_channel)
    kernel = np.zeros((n_grid, pair_rows.size), dtype=np.float64)
    for pair_index, (row, column) in enumerate(zip(pair_rows, pair_columns, strict=True)):
        row_index = int(row)
        column_index = int(column)
        if basis[row_index].two_K != basis[column_index].two_K:
            continue
        common_electronic_states = amplitudes[row_index].keys() & amplitudes[column_index].keys()
        for electronic_key in common_electronic_states:
            kernel[:, pair_index] += np.real(np.conjugate(amplitudes[row_index][electronic_key]) * amplitudes[column_index][electronic_key])
    kernel *= 1.0 / np.pi
    return FSDiatomDiatomVBasis(
        n_channel=basis.n_channel,
        grid_shape=grid_shape,
        pair_rows=np.asarray(pair_rows, dtype=np.int64),
        pair_columns=np.asarray(pair_columns, dtype=np.int64),
        kernel=np.ascontiguousarray(kernel),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _potential_batches(
    V_basis: FSDiatomDiatomVBasis,
    potential: NDArray[np.float64] | jax.Array,
) -> tuple[NDArray[np.float64] | jax.Array, bool]:
    """Validate and flatten a scalar PES grid, retaining its host or device type."""
    values = potential if isinstance(potential, jax.Array) else np.asarray(potential, dtype=np.float64)
    n_grid = prod(V_basis.grid_shape)
    if values.shape == V_basis.grid_shape:
        return values.reshape(1, n_grid), False
    if values.ndim == len(V_basis.grid_shape) + 1 and values.shape[1:] == V_basis.grid_shape:
        return values.reshape(values.shape[0], n_grid), True
    if values.ndim == 2 and values.shape[1] == n_grid:
        return values, True
    message = f"Scalar FS diatom-diatom PES grid has shape {values.shape}, expected {V_basis.grid_shape} with optional leading R axis"
    logger.error(message)
    raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _packed_positions(n_channel: int, indices: tuple[int, ...]) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return selected lower-triangle output indices and complete packed positions."""
    if len(set(indices)) != len(indices) or any(index < 0 or index >= n_channel for index in indices):
        message = "channel_indices must be unique complete-basis positions"
        logger.error(message)
        raise ValueError(message)
    pair_rows, pair_columns = np.tril_indices(len(indices))
    selected = np.asarray(indices, dtype=np.int64)
    global_rows = selected[pair_rows]
    global_columns = selected[pair_columns]
    packed_rows = np.maximum(global_rows, global_columns)
    packed_columns = np.minimum(global_rows, global_columns)
    packed = packed_rows * (packed_rows + 1) // 2 + packed_columns
    return np.asarray(pair_rows, dtype=np.int64), np.asarray(pair_columns, dtype=np.int64), np.asarray(packed, dtype=np.int64)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def contract(
    V_basis: FSDiatomDiatomVBasis,
    potential: NDArray[np.float64],
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    r"""
    Contract a scalar PES into two-fine-structure-diatom channel matrices.

    Formula:
        For every selected lower-triangle pair ``p=(eta',eta)``,

        V_p(R) = sum_g V(R,g) G_(g,p),

        where G is the spin-independent kernel prepared by :func:`prepare`.
        The upper triangle is filled by ``V_(eta,eta')=V_(eta',eta)``.

    Inputs:
        V_basis: FSDiatomDiatomVBasis - prepared host contraction kernel
        potential: NDArray[np.float64] - scalar values with shape
            ``grid_shape`` or ``(n_R,*grid_shape)``
        channel_indices: Sequence[int] | None - optional unique complete-basis
            positions in requested output order

    Returns:
        matrix: NDArray[np.float64] - symmetric matrix with shape
            ``(n_selected,n_selected)`` or ``(n_R,n_selected,n_selected)``
    """
    batches, batched = _potential_batches(V_basis, potential)
    host_batches = np.asarray(batches, dtype=np.float64)
    indices = tuple(range(V_basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    pair_rows, pair_columns, packed = _packed_positions(V_basis.n_channel, indices)
    contracted = host_batches @ V_basis.kernel[:, packed]
    matrix = np.zeros((host_batches.shape[0], len(indices), len(indices)), dtype=np.float64)
    matrix[:, pair_rows, pair_columns] = contracted
    matrix[:, pair_columns, pair_rows] = contracted
    return matrix if batched else matrix[0]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def device_basis(V_basis: FSDiatomDiatomVBasis, device: JaxDevice) -> FSDiatomDiatomVBasisDevice:
    """Copy a packed scalar two-FS-diatom kernel to one JAX device."""
    return FSDiatomDiatomVBasisDevice(kernel=jax.device_put(V_basis.kernel, device))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@jax.jit
def _contract_device(potential: jax.Array, kernel: jax.Array) -> jax.Array:
    """Contract flattened scalar PES batches with selected packed kernels."""
    return potential @ kernel


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def contract_device(
    V_basis: FSDiatomDiatomVBasis,
    basis_device: FSDiatomDiatomVBasisDevice,
    potential: NDArray[np.float64] | jax.Array,
    device: JaxDevice,
    channel_indices: Sequence[int] | None = None,
) -> jax.Array:
    """Contract a scalar two-FS-diatom PES into channel matrices on a JAX device.

    Inputs:
        V_basis: FSDiatomDiatomVBasis - host basis metadata
        basis_device: FSDiatomDiatomVBasisDevice - device-resident kernel
        potential: NDArray[np.float64] | jax.Array - scalar PES grid with an
            optional leading radial batch
        device: JaxDevice - contraction device
        channel_indices: Sequence[int] | None - optional complete-basis positions

    Returns:
        matrix: jax.Array - symmetric selected channel matrix, optionally
            preceded by the radial batch axis
    """
    batches, batched = _potential_batches(V_basis, potential)
    indices = tuple(range(V_basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    pair_rows, pair_columns, packed = _packed_positions(V_basis.n_channel, indices)
    potential_device = jax.device_put(batches, device)
    contracted = _contract_device(potential_device, basis_device.kernel[:, packed])
    matrix = jnp.zeros((potential_device.shape[0], len(indices), len(indices)), dtype=jnp.float64, device=device)
    matrix = matrix.at[:, pair_rows, pair_columns].set(contracted)
    matrix = matrix.at[:, pair_columns, pair_rows].set(contracted)
    return matrix if batched else matrix[0]


# ----------------------------------------------------------------------------------------
