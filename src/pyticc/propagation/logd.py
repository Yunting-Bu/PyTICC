import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpy.typing import NDArray

jax.config.update("jax_enable_x64", True)

LogDInput = jax.Array | NDArray[np.float64] | NDArray[np.complex128]


def _validate_square(name: str, value: LogDInput) -> tuple[int, ...]:
    shape = tuple(value.shape)
    if len(shape) < 2 or shape[-2] != shape[-1]:
        message = f"{name} must end with square matrix axes, but got shape={shape}"
        logger.error(message)
        raise ValueError(message)
    return shape


def _diagonal_matrix(diagonal: jax.Array) -> jax.Array:
    n_channel = diagonal.shape[-1]
    identity = jnp.eye(n_channel, dtype=diagonal.dtype)
    return diagonal[..., :, None] * identity


# ----------------------------------------------------------------------------------------
def initialize_logD_inelastic(Wmat: LogDInput) -> jax.Array:
    r"""
    Initialize a real log-derivative matrix for inelastic scattering.

    Formula:
        Y_ij(R0) = delta_ij sqrt(abs(W_ii(R0))).

    Inputs:
        Wmat: LogDInput - radial equation matrix or batch of matrices

    Returns:
        Ymat: jax.Array - real initial log-derivative matrix
    """
    _validate_square("Wmat", Wmat)
    diagonal = jnp.diagonal(jnp.asarray(Wmat, dtype=jnp.float64), axis1=-2, axis2=-1)
    return _diagonal_matrix(jnp.sqrt(jnp.abs(diagonal)))


# ----------------------------------------------------------------------------------------
def initialize_logD_capture(Wmat: LogDInput) -> jax.Array:
    r"""
    Initialize a complex incoming-wave log-derivative matrix for capture.

    Formula:
        Y_ii(R0) = sqrt(W_ii),                W_ii >= 0,
        Y_ii(R0) = -i sqrt(-W_ii),            W_ii < 0.

    Inputs:
        Wmat: LogDInput - real radial equation matrix or batch of matrices

    Returns:
        Ymat: jax.Array - complex initial log-derivative matrix
    """
    _validate_square("Wmat", Wmat)
    diagonal = jnp.diagonal(jnp.asarray(Wmat, dtype=jnp.float64), axis1=-2, axis2=-1)
    values = jnp.where(diagonal >= 0.0, jnp.sqrt(jnp.maximum(diagonal, 0.0)), -1.0j * jnp.sqrt(jnp.maximum(-diagonal, 0.0)))
    return _diagonal_matrix(values.astype(jnp.complex128))


def _reference_matrices(radial_half_step: jax.Array, W_mid: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    reference_values = jnp.real(jnp.diag(W_mid))
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
    reference = jnp.diag(reference_values.astype(W_mid.dtype))
    return jnp.diag(y1_values.astype(W_mid.dtype)), jnp.diag(y2_values.astype(W_mid.dtype)), reference


def _propagate_logD_sector_jax(
    Ymat: jax.Array,
    radial_half_step: jax.Array,
    W_start: jax.Array,
    W_mid: jax.Array,
    W_end: jax.Array,
) -> jax.Array:
    dtype = jnp.result_type(Ymat.dtype, W_start.dtype, W_mid.dtype, W_end.dtype)
    Ymat = Ymat.astype(dtype)
    W_start = W_start.astype(dtype)
    W_mid = W_mid.astype(dtype)
    W_end = W_end.astype(dtype)
    n_channel = Ymat.shape[-1]
    identity = jnp.eye(n_channel, dtype=dtype)
    y1, y2, reference = _reference_matrices(radial_half_step, W_mid)

    T_start = W_start - reference
    T_mid = W_mid - reference
    T_end = W_end - reference
    Q_start = radial_half_step * T_start / 3.0
    Q_mid = 4.0 / radial_half_step * (jnp.linalg.solve(identity - radial_half_step**2 * T_mid / 6.0, identity) - identity)
    Q_end = radial_half_step * T_end / 3.0

    Y_mid = y1 + Q_mid - y2 @ jnp.linalg.solve(Ymat + y1 + Q_start, y2)
    Y_end = y1 + Q_end - y2 @ jnp.linalg.solve(Y_mid + y1 + Q_mid, y2)
    return 0.5 * (Y_end + Y_end.T)


_propagate_logD_sector_compiled = jax.jit(_propagate_logD_sector_jax)


# ----------------------------------------------------------------------------------------
def propagate_logD_sector(
    Ymat: LogDInput,
    radial_half_step: float,
    W_start: LogDInput,
    W_mid: LogDInput,
    W_end: LogDInput,
) -> jax.Array:
    r"""
    Propagate a log-derivative matrix through one Manolopoulos LDMD sector.

    The sector contains two half-steps, ``radial_start -> radial_mid -> radial_end``,
    each of length ``radial_half_step``. The diagonal of ``W_mid`` is the
    reference potential.

    Inputs:
        Ymat: LogDInput - log-derivative matrix at the sector start
        radial_half_step: float - half-sector radial step in atomic units
        W_start: LogDInput - radial equation matrix at the sector start
        W_mid: LogDInput - radial equation matrix at the sector midpoint
        W_end: LogDInput - radial equation matrix at the sector end

    Returns:
        Y_end: jax.Array - log-derivative matrix at the sector end
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
        jnp.asarray(Ymat),
        jnp.asarray(radial_half_step, dtype=jnp.float64),
        jnp.asarray(W_start),
        jnp.asarray(W_mid),
        jnp.asarray(W_end),
    )


def _propagate_logD_jax(
    Y_initial: jax.Array,
    total_energies: jax.Array,
    reduced_mass: jax.Array,
    radial_half_steps: jax.Array,
    W_base_start: jax.Array,
    W_base_mid: jax.Array,
    W_base_end: jax.Array,
) -> jax.Array:
    n_channel = Y_initial.shape[-1]
    identity = jnp.eye(n_channel, dtype=W_base_start.dtype)
    energy_shift = 2.0 * reduced_mass * total_energies[:, None, None] * identity

    def scan_sector(Ymat: jax.Array, sector: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, None]:
        radial_half_step, base_start, base_mid, base_end = sector
        W_start = base_start[None, :, :] - energy_shift
        W_mid = base_mid[None, :, :] - energy_shift
        W_end = base_end[None, :, :] - energy_shift
        Y_end = jax.vmap(_propagate_logD_sector_jax, in_axes=(0, None, 0, 0, 0))(Ymat, radial_half_step, W_start, W_mid, W_end)
        return Y_end, None

    Y_final, _ = jax.lax.scan(scan_sector, Y_initial, (radial_half_steps, W_base_start, W_base_mid, W_base_end))
    return Y_final


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
        radial_half_steps: LogDInput - radial half-step for each sector
        W_base_start: LogDInput - energy-independent matrices at sector starts
        W_base_mid: LogDInput - energy-independent matrices at sector midpoints
        W_base_end: LogDInput - energy-independent matrices at sector ends

    Returns:
        Y_final: jax.Array - final log-derivative matrices for all energies
    """
    Y_shape = _validate_square("Y_initial", Y_initial)
    energy_shape = tuple(total_energies.shape)
    step_shape = tuple(radial_half_steps.shape)
    if len(Y_shape) != 3 or energy_shape != (Y_shape[0],):
        message = f"Y_initial must have shape (n_energy, n_channel, n_channel) matching total_energies, but got {Y_shape} and {energy_shape}"
        logger.error(message)
        raise ValueError(message)
    if len(step_shape) != 1:
        message = f"radial_half_steps must be one-dimensional, but got shape={step_shape}"
        logger.error(message)
        raise ValueError(message)
    if reduced_mass <= 0.0:
        message = f"reduced_mass must be positive, but got reduced_mass={reduced_mass}"
        logger.error(message)
        raise ValueError(message)

    expected_W_shape = (step_shape[0], Y_shape[1], Y_shape[2])
    for name, value in (("W_base_start", W_base_start), ("W_base_mid", W_base_mid), ("W_base_end", W_base_end)):
        if tuple(value.shape) != expected_W_shape:
            message = f"{name} must have shape {expected_W_shape}, but got {value.shape}"
            logger.error(message)
            raise ValueError(message)

    return _propagate_logD_compiled(
        jnp.asarray(Y_initial),
        jnp.asarray(total_energies, dtype=jnp.float64),
        jnp.asarray(reduced_mass, dtype=jnp.float64),
        jnp.asarray(radial_half_steps, dtype=jnp.float64),
        jnp.asarray(W_base_start, dtype=jnp.float64),
        jnp.asarray(W_base_mid, dtype=jnp.float64),
        jnp.asarray(W_base_end, dtype=jnp.float64),
    )


# ----------------------------------------------------------------------------------------
