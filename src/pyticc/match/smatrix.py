import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.bessel import modified_bessel_IK_logD, riccati_bessel_jy


def _reference_matrices(
    energy: float,
    Rmatch: float,
    reduced_mass: float,
    E_int: NDArray[np.float64],
    L: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    energy_difference = energy - E_int
    if np.any(energy_difference == 0.0):
        message = f"Asymptotic matching is undefined exactly at a channel threshold Etot={energy}"
        logger.error(message)
        raise ValueError(message)

    open_mask = energy_difference > 0.0
    n_channel = E_int.size
    J = np.ones(n_channel, dtype=np.float64)
    N = np.ones(n_channel, dtype=np.float64)
    J_prime = np.empty(n_channel, dtype=np.float64)
    N_prime = np.empty(n_channel, dtype=np.float64)

    for index in range(n_channel):
        wave_number = np.sqrt(2.0 * reduced_mass * abs(energy_difference[index]))
        argument = wave_number * Rmatch
        if open_mask[index]:
            j_value, n_value, j_derivative, n_derivative = riccati_bessel_jy(float(L[index]), argument)
            J[index] = j_value / np.sqrt(wave_number)
            N[index] = n_value / np.sqrt(wave_number)
            J_prime[index] = np.sqrt(wave_number) * j_derivative
            N_prime[index] = np.sqrt(wave_number) * n_derivative
        else:
            I_logD, K_logD = modified_bessel_IK_logD(float(L[index] + 0.5), argument)
            J_prime[index] = 0.5 / Rmatch + wave_number * I_logD
            N_prime[index] = 0.5 / Rmatch + wave_number * K_logD

    return np.diag(J), np.diag(N), np.diag(J_prime), np.diag(N_prime), open_mask


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
    channels use logarithmic derivatives of modified Bessel functions and remain in
    the full reaction matrix until its open-open block is extracted. The returned
    matrices follow the order of ``flatnonzero(E_int < Etot[i])``.

    Formula:
        K = -(Y @ N - N')**(-1) @ (Y @ J - J'),
        S = (I + i * K_oo)**(-1) @ (I - i * K_oo).

    Inputs:
        Ymat: ArrayLike - SF log-derivative matrices with shape (n_energy, n_channel, n_channel)
        Rmatch: float - asymptotic matching distance in atomic units
        Etot: EnergyInput - total-energy array or one-column text file in atomic units
        reduced_mass: float - collision reduced mass in atomic units
        E_int: ArrayLike - asymptotic channel internal energies in atomic units
        L: ArrayLike - SF orbital angular momenta, including non-integral values

    Returns:
        Smat: tuple[NDArray[np.complex128], ...] - open-channel scattering matrix at each energy
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
        J, N, J_prime, N_prime, open_mask = _reference_matrices(float(energy), Rmatch, reduced_mass, E_int_array, L_array)
        open_indices = np.flatnonzero(open_mask)
        if open_indices.size == 0:
            scattering_matrices.append(np.empty((0, 0), dtype=np.complex128))
            continue

        Y = Ymat_array[energy_index]
        reaction_matrix = -np.linalg.solve(Y @ N - N_prime, Y @ J - J_prime)
        reaction_open = reaction_matrix[np.ix_(open_indices, open_indices)]
        identity = np.eye(open_indices.size, dtype=np.complex128)
        Smat = np.linalg.solve(identity + 1.0j * reaction_open, identity - 1.0j * reaction_open)
        scattering_matrices.append(np.asarray(Smat, dtype=np.complex128))

    return tuple(scattering_matrices)


# ----------------------------------------------------------------------------------------
