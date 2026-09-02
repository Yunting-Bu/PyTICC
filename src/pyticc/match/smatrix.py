import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.bessel import modified_bessel_K_logD, riccati_bessel_jy


# ----------------------------------------------------------------------------------------
def _reference_functions(
    energy: float,
    Rmatch: float,
    reduced_mass: float,
    E_int: NDArray[np.float64],
    L: NDArray[np.float64],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Evaluate open-channel reference functions and closed-channel decay log derivatives.

    Closed-channel growing reference functions are not evaluated because the physical
    scattering boundary condition contains only the decaying solution.

    Formula:
        J_o = k_o^{-1/2} \hat j_L(k_o R),
        N_o = k_o^{-1/2} \hat n_L(k_o R),
        Gamma_c = d/dR ln[sqrt(kappa_c R) K_{L+1/2}(kappa_c R)].

    Inputs:
        E_int: NDArray[np.float64] - channel thresholds, shape (n_channel,)
        L: NDArray[np.float64] - orbital angular momenta, shape (n_channel,)

    Returns:
        open_indices: NDArray[np.int64] - global open-channel indices, shape (n_open,)
        closed_indices: NDArray[np.int64] - global closed-channel indices, shape (n_closed,)
        J_open: NDArray[np.float64] - regular open reference values, shape (n_open,)
        N_open: NDArray[np.float64] - irregular open reference values, shape (n_open,)
        J_prime_open: NDArray[np.float64] - regular open reference derivatives, shape (n_open,)
        N_prime_open: NDArray[np.float64] - irregular open reference derivatives, shape (n_open,)
        Gamma_closed: NDArray[np.float64] - decaying closed-reference log derivatives,
            shape (n_closed,)
    """
    energy_difference = energy - E_int
    if np.any(energy_difference == 0.0):
        message = f"Asymptotic matching is undefined exactly at a channel threshold Etot={energy}"
        logger.error(message)
        raise ValueError(message)

    open_indices = np.asarray(np.flatnonzero(energy_difference > 0.0), dtype=np.int64)
    closed_indices = np.asarray(np.flatnonzero(energy_difference < 0.0), dtype=np.int64)
    J_open = np.empty(open_indices.size, dtype=np.float64)
    N_open = np.empty(open_indices.size, dtype=np.float64)
    J_prime_open = np.empty(open_indices.size, dtype=np.float64)
    N_prime_open = np.empty(open_indices.size, dtype=np.float64)
    Gamma_closed = np.empty(closed_indices.size, dtype=np.float64)

    for open_position, index in enumerate(open_indices):
        wave_number = float(np.sqrt(2.0 * reduced_mass * abs(energy_difference[index])))
        argument = wave_number * Rmatch
        j_value, n_value, j_derivative, n_derivative = riccati_bessel_jy(float(L[index]), argument)
        J_open[open_position] = j_value / np.sqrt(wave_number)
        N_open[open_position] = n_value / np.sqrt(wave_number)
        J_prime_open[open_position] = np.sqrt(wave_number) * j_derivative
        N_prime_open[open_position] = np.sqrt(wave_number) * n_derivative

    for closed_position, index in enumerate(closed_indices):
        wave_number = float(np.sqrt(2.0 * reduced_mass * abs(energy_difference[index])))
        argument = wave_number * Rmatch
        K_logD = modified_bessel_K_logD(float(L[index] + 0.5), argument)
        Gamma_closed[closed_position] = 0.5 / Rmatch + wave_number * K_logD

    return open_indices, closed_indices, J_open, N_open, J_prime_open, N_prime_open, Gamma_closed


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _open_reaction_matrix(
    Y: NDArray[np.float64] | NDArray[np.complex128],
    open_indices: NDArray[np.int64],
    closed_indices: NDArray[np.int64],
    J_open: NDArray[np.float64],
    N_open: NDArray[np.float64],
    J_prime_open: NDArray[np.float64],
    N_prime_open: NDArray[np.float64],
    Gamma_closed: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r"""
    Eliminate closed channels and construct the physical open-open reaction matrix.

    Formula:
        Y_eff = Y_oo - Y_oc (Y_cc - Gamma_c)^(-1) Y_co,

        K_oo = -(Y_eff N_o - N'_o)^(-1) (Y_eff J_o - J'_o).

    Here ``Gamma_c = N'_c N_c^(-1)`` is the logarithmic derivative of the
    exponentially decaying closed-channel solution. Thus all propagated closed-channel
    couplings remain in ``Y_eff`` while no growing closed-channel reference function is
    constructed. Open reference functions are energy normalized.

    Inputs:
        Y: NDArray - asymptotic SF log-derivative matrix, shape (n_channel,n_channel)
        open_indices: NDArray[np.int64] - global open-channel indices, shape (n_open,)
        closed_indices: NDArray[np.int64] - global closed-channel indices, shape (n_closed,)
        J_open: NDArray[np.float64] - regular open reference values, shape (n_open,)
        N_open: NDArray[np.float64] - irregular open reference values, shape (n_open,)
        J_prime_open: NDArray[np.float64] - regular open reference derivatives, shape (n_open,)
        N_prime_open: NDArray[np.float64] - irregular open reference derivatives, shape (n_open,)
        Gamma_closed: NDArray[np.float64] - decaying closed-reference log derivatives,
            shape (n_closed,)

    Returns:
        reaction_open: NDArray - physical reaction matrix, shape (n_open,n_open)
    """
    Y_open = Y[np.ix_(open_indices, open_indices)]
    if closed_indices.size:
        Y_open_closed = Y[np.ix_(open_indices, closed_indices)]
        Y_closed_open = Y[np.ix_(closed_indices, open_indices)]
        closed_operator = Y[np.ix_(closed_indices, closed_indices)] - np.diag(Gamma_closed)
        Y_open = Y_open - Y_open_closed @ np.linalg.solve(closed_operator, Y_closed_open)

    left = Y_open * N_open[np.newaxis, :] - np.diag(N_prime_open)
    right = Y_open * J_open[np.newaxis, :] - np.diag(J_prime_open)
    return np.asarray(-np.linalg.solve(left, right), dtype=np.complex128)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Smat(
    Ymat: ArrayLike,
    Rmatch: float,
    Etot: EnergyInput,
    reduced_mass: float,
    E_int: ArrayLike,
    L: ArrayLike,
) -> tuple[NDArray[np.complex128], ...]:
    r"""
    Match SF log-derivative matrices to asymptotic scattering matrices.

    Open channels use energy-normalized Riccati-Bessel reference functions. Closed
    channels are eliminated through their decaying-reference logarithmic derivatives,
    leaving the physical open-open reaction matrix without evaluating growing closed
    solutions. The returned matrices follow the order of
    ``flatnonzero(E_int < Etot[i])``.

    Formula:
        Y_eff = Y_oo - Y_oc @ (Y_cc - Gamma_c)**(-1) @ Y_co,
        K_oo = -(Y_eff @ N_o - N'_o)**(-1) @ (Y_eff @ J_o - J'_o),
        S = (I + i * K_oo)**(-1) @ (I - i * K_oo).

    Inputs:
        Ymat: ArrayLike - SF log-derivative matrices with shape (n_energy, n_channel, n_channel)
        Rmatch: float - asymptotic matching distance in atomic units
        Etot: EnergyInput - total-energy array with shape (n_energy,), or a
            one-column text file in atomic units
        reduced_mass: float - collision reduced mass in atomic units
        E_int: ArrayLike - asymptotic channel internal energies in atomic units,
            shape (n_channel,)
        L: ArrayLike - SF orbital angular momenta, including non-integral values,
            shape (n_channel,)

    Returns:
        Smat: tuple[NDArray[np.complex128], ...] - one matrix per energy; element i
            has shape (n_open[i], n_open[i])
    """
    energies = get_Etot(Etot)
    Ymat_array = np.asarray(Ymat)
    E_int_array = np.asarray(E_int, dtype=np.float64)
    L_array = np.asarray(L, dtype=np.float64)
    n_channel = E_int_array.size

    expected_shape = (energies.size, n_channel, n_channel)
    if Ymat_array.shape != expected_shape:
        message = f"Ymat must have shape {expected_shape}, but got {Ymat_array.shape}"
        logger.error(message)
        raise ValueError(message)
    if E_int_array.ndim != 1 or L_array.shape != E_int_array.shape:
        message = f"E_int and L must be one-dimensional arrays with the same shape, but got {E_int_array.shape} and {L_array.shape}"
        logger.error(message)
        raise ValueError(message)
    if Rmatch <= 0.0 or reduced_mass <= 0.0:
        message = f"Rmatch and reduced_mass must be positive, but got Rmatch={Rmatch}, reduced_mass={reduced_mass}"
        logger.error(message)
        raise ValueError(message)
    if np.any(L_array < 0.0):
        message = "Orbital angular momenta L must be non-negative"
        logger.error(message)
        raise ValueError(message)

    scattering_matrices: list[NDArray[np.complex128]] = []
    for energy_index, energy in enumerate(energies):
        references = _reference_functions(float(energy), Rmatch, reduced_mass, E_int_array, L_array)
        open_indices, closed_indices, J_open, N_open, J_prime_open, N_prime_open, Gamma_closed = references
        if open_indices.size == 0:
            scattering_matrices.append(np.empty((0, 0), dtype=np.complex128))
            continue

        Y = Ymat_array[energy_index]
        reaction_open = _open_reaction_matrix(
            Y,
            open_indices,
            closed_indices,
            J_open,
            N_open,
            J_prime_open,
            N_prime_open,
            Gamma_closed,
        )
        identity = np.eye(open_indices.size, dtype=np.complex128)
        Smat = np.linalg.solve(identity + 1.0j * reaction_open, identity - 1.0j * reaction_open)
        scattering_matrices.append(np.asarray(Smat, dtype=np.complex128))

    return tuple(scattering_matrices)


# ----------------------------------------------------------------------------------------
