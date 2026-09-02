import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from loguru import logger
from numpy.typing import NDArray

from pyticc._typing import JaxDevice

jax.config.update("jax_enable_x64", True)

LogDInput = jax.Array | NDArray[np.float64] | NDArray[np.complex128]


# ----------------------------------------------------------------------------------------
def _device_array(value: LogDInput | float, device: JaxDevice | None, dtype: DTypeLike | None = None) -> jax.Array:
    """Place one input directly on the requested device and optionally cast it there."""
    array = jax.device_put(value, device)
    return jnp.asarray(array, dtype=dtype)


# ----------------------------------------------------------------------------------------
def _validate_square(name: str, value: LogDInput) -> tuple[int, ...]:
    """Validate trailing square matrix axes and return the complete array shape."""
    shape = tuple(value.shape)
    if len(shape) < 2 or shape[-2] != shape[-1]:
        message = f"{name} must end with square matrix axes, but got shape={shape}"
        logger.error(message)
        raise ValueError(message)
    return shape


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _diagonal_matrix(diagonal: jax.Array) -> jax.Array:
    """Embed vectors with shape (..., n) as diagonal matrices with shape (..., n, n)."""
    n_channel = diagonal.shape[-1]
    identity = jnp.eye(n_channel, dtype=diagonal.dtype)
    return diagonal[..., :, None] * identity


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def initialize_logD_inelastic(Wmat: LogDInput) -> jax.Array:
    r"""
    Initialize a real log-derivative matrix for inelastic scattering.

    Formula:
        Y_ij(R0) = delta_ij sqrt(abs(W_ii(R0))).

    Inputs:
        Wmat: LogDInput - radial equation matrix or batch of matrices, shape
            (..., n_channel, n_channel)

    Returns:
        Ymat: jax.Array - real initial log-derivative matrices with the same shape
            (..., n_channel, n_channel) as Wmat
    """
    _validate_square("Wmat", Wmat)
    diagonal = jnp.real(jnp.diagonal(jnp.asarray(Wmat), axis1=-2, axis2=-1))
    return _diagonal_matrix(jnp.sqrt(jnp.abs(diagonal)))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def initialize_logD_capture(Wmat: LogDInput) -> jax.Array:
    r"""
    Initialize a complex incoming-wave log-derivative matrix for capture.

    Formula:
        Y_ii(R0) = sqrt(W_ii),                W_ii >= 0,
        Y_ii(R0) = -i sqrt(-W_ii),            W_ii < 0.

    Inputs:
        Wmat: LogDInput - real radial equation matrix or batch of matrices, shape
            (..., n_channel, n_channel)

    Returns:
        Ymat: jax.Array - complex initial log-derivative matrices with the same shape
            (..., n_channel, n_channel) as Wmat
    """
    _validate_square("Wmat", Wmat)
    diagonal = jnp.real(jnp.diagonal(jnp.asarray(Wmat), axis1=-2, axis2=-1))
    values = jnp.where(diagonal >= 0.0, jnp.sqrt(jnp.maximum(diagonal, 0.0)), -1.0j * jnp.sqrt(jnp.maximum(-diagonal, 0.0)))
    return _diagonal_matrix(values.astype(jnp.complex128))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _reference_values(radial_half_step: jax.Array, reference_values: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""
    Build diagonal LDMD reference values from the sector-midpoint potential.

    Formula:
        p_j^2 >= 0 :
            y1_{ij} = \delta_{ij} p_j coth(p_j h)
            y2_{ij} = \delta_{ij} p_j scsh(p_j h)
        p_j^2 < 0 :
            y1_{ij} = \delta_{ij} |p_j| cot(|p_j| h)
            y2_{ij} = \delta_{ij} |p_j| scs(|p_j| h)

    Inputs:
        radial_half_step: jax.Array - scalar half-sector step, shape ()
        reference_values: jax.Array - midpoint diagonal, shape (n_channel,)

    Returns:
        y1_values: jax.Array - same-end reference diagonal, shape (n_channel,)
        y2_values: jax.Array - cross-end reference diagonal, shape (n_channel,)
    """
    magnitude = jnp.sqrt(jnp.abs(reference_values))
    argument = magnitude * radial_half_step
    small = jnp.abs(argument) < 1.0e-5
    safe_argument = jnp.where(small, 1.0, argument)

    y1_positive = magnitude / jnp.tanh(safe_argument)
    y1_negative = magnitude / jnp.tan(safe_argument)
    y2_positive = magnitude / jnp.sinh(safe_argument)
    y2_negative = magnitude / jnp.sin(safe_argument)
    y1_series = 1.0 / radial_half_step + reference_values * radial_half_step / 3.0 - reference_values**2 * radial_half_step**3 / 45.0
    y2_series = 1.0 / radial_half_step - reference_values * radial_half_step / 6.0 + 7.0 * reference_values**2 * radial_half_step**3 / 360.0

    y1_values = jnp.where(small, y1_series, jnp.where(reference_values >= 0.0, y1_positive, y1_negative))
    y2_values = jnp.where(small, y2_series, jnp.where(reference_values >= 0.0, y2_positive, y2_negative))
    return y1_values, y2_values


# ----------------------------------------------------------------------------------------
def _add_diagonal(matrix: jax.Array, diagonal: jax.Array) -> jax.Array:
    """Add one vector to the trailing matrix diagonal without materializing a diagonal matrix."""
    indices = jnp.diag_indices(matrix.shape[-1])
    return matrix.at[..., indices[0], indices[1]].add(diagonal)


# ----------------------------------------------------------------------------------------
def _correction_matrices(
    radial_half_step: jax.Array,
    W_start: jax.Array,
    W_mid: jax.Array,
    W_end: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build energy-independent LDMD correction matrices for one sector."""
    dtype = jnp.result_type(W_start.dtype, W_mid.dtype, W_end.dtype)
    W_start = W_start.astype(dtype)
    W_mid = W_mid.astype(dtype)
    W_end = W_end.astype(dtype)
    n_channel = W_mid.shape[-1]
    identity = jnp.eye(n_channel, dtype=dtype)
    reference_values = jnp.real(jnp.diag(W_mid))
    reference = jnp.diag(reference_values.astype(dtype))
    T_start = W_start - reference
    T_mid = W_mid - reference
    T_end = W_end - reference
    Q_start = radial_half_step * T_start / 3.0
    Q_mid = 4.0 / radial_half_step * (jnp.linalg.solve(identity - radial_half_step**2 * T_mid / 6.0, identity) - identity)
    Q_end = radial_half_step * T_end / 3.0
    return reference_values, Q_start, Q_mid, Q_end


# ----------------------------------------------------------------------------------------
def _propagate_with_corrections(
    Ymat: jax.Array,
    radial_half_step: jax.Array,
    reference_values: jax.Array,
    Q_start: jax.Array,
    Q_mid: jax.Array,
    Q_end: jax.Array,
) -> jax.Array:
    """Propagate one energy using precomputed energy-independent sector corrections."""
    dtype = jnp.result_type(Ymat.dtype, Q_start.dtype, Q_mid.dtype, Q_end.dtype)
    Ymat = Ymat.astype(dtype)
    Q_start = Q_start.astype(dtype)
    Q_mid = Q_mid.astype(dtype)
    Q_end = Q_end.astype(dtype)
    y1_values, y2_values = _reference_values(radial_half_step, reference_values)
    y1_values = y1_values.astype(dtype)
    y2_values = y2_values.astype(dtype)
    y2 = _diagonal_matrix(y2_values)

    start_matrix = _add_diagonal(Ymat + Q_start, y1_values)
    start_solution = jnp.linalg.solve(start_matrix, y2)
    Y_mid = _add_diagonal(Q_mid, y1_values) - y2_values[:, None] * start_solution

    mid_matrix = _add_diagonal(Y_mid + Q_mid, y1_values)
    mid_solution = jnp.linalg.solve(mid_matrix, y2)
    Y_end = _add_diagonal(Q_end, y1_values) - y2_values[:, None] * mid_solution
    return 0.5 * (Y_end + Y_end.T)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _propagate_logD_sector_jax(
    Ymat: jax.Array,
    radial_half_step: jax.Array,
    W_start: jax.Array,
    W_mid: jax.Array,
    W_end: jax.Array,
) -> jax.Array:
    r"""
    Execute one JAX-compatible two-half-step LDMD sector update.

    Every matrix input and the returned log derivative has shape
    (n_channel, n_channel); ``radial_half_step`` has scalar shape ().

    Formula:
        p_j^2 = W_mid_{jj}
        T_{ij} = W_{ij} - \delta_{ij} p_j^2
        Q_start = h T_start / 3
        Q_mid = 4 / h [I - (I - h^2 T_mid / 6)^{-1}]
        Q_end = h T_end / 3
    """
    reference_values, Q_start, Q_mid, Q_end = _correction_matrices(radial_half_step, W_start, W_mid, W_end)
    return _propagate_with_corrections(Ymat, radial_half_step, reference_values, Q_start, Q_mid, Q_end)


# ----------------------------------------------------------------------------------------


_propagate_logD_sector_compiled = jax.jit(_propagate_logD_sector_jax)


# ----------------------------------------------------------------------------------------
def propagate_logD_sector(
    Ymat: LogDInput,
    radial_half_step: float,
    W_start: LogDInput,
    W_mid: LogDInput,
    W_end: LogDInput,
    *,
    device: JaxDevice | None = None,
) -> jax.Array:
    r"""
    Propagate a log-derivative matrix through one Manolopoulos LDMD sector.

    The sector contains two half-steps, ``radial_start -> radial_mid -> radial_end``,
    each of length ``radial_half_step``. The diagonal of ``W_mid`` is the
    reference potential.

    Inputs:
        Ymat: LogDInput - log-derivative matrix at the sector start, shape
            (n_channel, n_channel)
        radial_half_step: float - half-sector radial step in atomic units
        W_start: LogDInput - radial equation matrix at the sector start, shape
            (n_channel, n_channel)
        W_mid: LogDInput - radial equation matrix at the sector midpoint, shape
            (n_channel, n_channel)
        W_end: LogDInput - radial equation matrix at the sector end, shape
            (n_channel, n_channel)
        device: JaxDevice | None - optional explicit JAX execution device

    Returns:
        Y_end: jax.Array - log-derivative matrix at the sector end, shape
            (n_channel, n_channel)
    """
    shape = _validate_square("Ymat", Ymat)
    if radial_half_step <= 0.0:
        message = f"radial_half_step must be positive, but got radial_half_step={radial_half_step}"
        logger.error(message)
        raise ValueError(message)
    for name, value in (("W_start", W_start), ("W_mid", W_mid), ("W_end", W_end)):
        if _validate_square(name, value) != shape:
            message = f"Ymat and W matrices must have the same shape, but got Ymat.shape={shape}, {name}.shape={value.shape}"
            logger.error(message)
            raise ValueError(message)

    return _propagate_logD_sector_compiled(
        _device_array(Ymat, device),
        _device_array(radial_half_step, device, jnp.float64),
        _device_array(W_start, device),
        _device_array(W_mid, device),
        _device_array(W_end, device),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _propagate_logD_jax(
    Y_initial: jax.Array,
    total_energies: jax.Array,
    reduced_mass: jax.Array,
    radial_half_steps: jax.Array,
    W_base_start: jax.Array,
    W_base_mid: jax.Array,
    W_base_end: jax.Array,
) -> jax.Array:
    """
    Propagate an energy batch over all sectors inside one compiled JAX kernel.

    ``Y_initial`` has shape (n_energy, n_channel, n_channel), energies have shape
    (n_energy,), half-steps have shape (n_sector,), and each W_base array has shape
    (n_sector, n_channel, n_channel). The return shape matches ``Y_initial``.
    """
    energy_shift = 2.0 * reduced_mass * total_energies[:, None]
    reference_values, Q_start, Q_mid, Q_end = jax.vmap(_correction_matrices)(
        radial_half_steps,
        W_base_start,
        W_base_mid,
        W_base_end,
    )

    def scan_sector(
        Ymat: jax.Array,
        sector: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        """Advance a Y batch with shape (n_energy, n_channel, n_channel) by one sector."""
        radial_half_step, sector_reference, sector_Q_start, sector_Q_mid, sector_Q_end = sector
        energy_reference_values = sector_reference[None, :] - energy_shift
        Y_end = jax.vmap(_propagate_with_corrections, in_axes=(0, None, 0, None, None, None))(
            Ymat,
            radial_half_step,
            energy_reference_values,
            sector_Q_start,
            sector_Q_mid,
            sector_Q_end,
        )
        return Y_end, None

    Y_final, _ = jax.lax.scan(scan_sector, Y_initial, (radial_half_steps, reference_values, Q_start, Q_mid, Q_end))
    return Y_final


# ----------------------------------------------------------------------------------------


_propagate_logD_compiled = jax.jit(_propagate_logD_jax)


# ----------------------------------------------------------------------------------------
def propagate_logD(
    Y_initial: LogDInput,
    total_energies: LogDInput,
    reduced_mass: float,
    radial_half_steps: LogDInput,
    W_base_start: LogDInput,
    W_base_mid: LogDInput,
    W_base_end: LogDInput,
    *,
    device: JaxDevice | None = None,
) -> jax.Array:
    r"""
    Propagate all total energies through a sequence of LDMD sectors with JAX.

    ``W_base`` is independent of total energy,

        W_base(R) = U / R**2 + 2 * reduced_mass * [V(R) + diag(E_int)],

    and the energy-dependent matrix is

        W(R; Etot) = W_base(R) - 2 * reduced_mass * Etot * I.

    ``jax.vmap`` propagates total energies and ``jax.lax.scan`` propagates sectors.

    Inputs:
        Y_initial: LogDInput - initial matrices with shape (n_energy, n_channel, n_channel)
        total_energies: LogDInput - total energies with shape (n_energy,)
        reduced_mass: float - collision reduced mass in atomic units
        radial_half_steps: LogDInput - radial half-step for each sector, shape
            (n_sector,)
        W_base_start: LogDInput - energy-independent matrices at sector starts,
            shape (n_sector, n_channel, n_channel)
        W_base_mid: LogDInput - energy-independent matrices at sector midpoints,
            shape (n_sector, n_channel, n_channel)
        W_base_end: LogDInput - energy-independent matrices at sector ends, shape
            (n_sector, n_channel, n_channel)
        device: JaxDevice | None - optional explicit JAX execution device

    Returns:
        Y_final: jax.Array - final log-derivative matrices, shape
            (n_energy, n_channel, n_channel)
    """
    step_shape = tuple(radial_half_steps.shape)

    if len(step_shape) != 1:
        message = f"radial_half_steps must be one-dimensional, but got shape={step_shape}"
        logger.error(message)
        raise ValueError(message)

    matrix_dtype = jnp.result_type(Y_initial, W_base_start, W_base_mid, W_base_end)
    return _propagate_logD_compiled(
        _device_array(Y_initial, device, matrix_dtype),
        _device_array(total_energies, device, jnp.float64),
        _device_array(reduced_mass, device, jnp.float64),
        _device_array(radial_half_steps, device, jnp.float64),
        _device_array(W_base_start, device),
        _device_array(W_base_mid, device),
        _device_array(W_base_end, device),
    )


# ----------------------------------------------------------------------------------------
