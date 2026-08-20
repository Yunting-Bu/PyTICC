import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from scipy import special

from pyticc.basis.delves import DelvesBasis, midpoint_quad, sine_basis, theta_max
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.bessel import riccati_bessel_jy
from pyticc.match.delves import DelvesAsymptoticBasis, build_delves_asymptotic_basis, transform_logD_to_delves_channels
from pyticc.matrix.delves import mass_scale
from pyticc.pes.total import TotalPES
from pyticc.propagation.delves import DelvesPropagationResult


# ----------------------------------------------------------------------------------------
def match_delves(
    result: DelvesPropagationResult,
    Etot: EnergyInput,
    basis: DelvesBasis,
    total_pes: TotalPES,
) -> tuple[DelvesAsymptoticBasis, tuple[NDArray[np.complex128], ...]]:
    r"""
    Match one completed Delves propagation without exposing intermediate bases.

    Formula:
        At the final propagated hyperradius ``rho_f``, construct the asymptotic
        channel data ``Q={epsilon,c_s,c_theta,(a,v,j,K)}``. The directed overlap
        from the final surface basis to the finite-rho channel representative is

        T = C_surface(rho_s)^T P(rho_s,rho_f) c_theta,

        where ``rho_s=result.surface_rho`` is the last sector midpoint and
        ``rho_f=result.rho_final`` is the physical matching boundary.

        The final channel LogD and scattering matrices are then

        Y_channel(E)=T^T Y_surface(E)T,

        S(E)=Match_Pack--Parker[Y_channel(E),Q],

        where ``Match_Pack--Parker`` is implemented by ``get_delves_Smat`` and
        includes BF--SF frame transformations and open/closed channels.

    Inputs:
        result: DelvesPropagationResult - final fixed-grid propagation state
        Etot: EnergyInput - the same total energies used during propagation, in
            Hartree
        basis: DelvesBasis - primitive Delves basis used during propagation
        total_pes: TotalPES - the same total scalar adiabatic PES used to build
            the propagated surfaces

    Returns:
        channels: DelvesAsymptoticBasis - channel labels and thresholds needed
            to interpret each S-matrix row and column
        Smat: tuple[NDArray[np.complex128], ...] - one open-channel BF scattering
            matrix per total energy
    """
    channels = build_delves_asymptotic_basis(basis, total_pes, result.rho_final)
    channel_logD = transform_logD_to_delves_channels(
        basis,
        result.surface_rho,
        result.surface_coefficients,
        result.Y_final,
        channels,
    )
    return channels, get_delves_Smat(channel_logD, Etot, basis, channels)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_delves_frame_transform(
    basis: DelvesBasis,
    channels: DelvesAsymptoticBasis,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Construct ABC's parity-adapted BF-helicity to SF-orbital transformation.

    Formula:
        Within each fixed ``(a,v,j)`` block, the body-fixed matrix of ``L^2`` is

        <K|L^2|K> = J(J+1)+j(j+1)-2K^2,

        <0|L^2|1> = -sqrt[2 J(J+1) j(j+1)],

        <K|L^2|K+1> =
          -sqrt({J(J+1)-K(K+1)}{j(j+1)-K(K+1)}),  K>0.

        Diagonalization gives

        B^T L^2_BF B = diag[L_alpha(L_alpha+1)],
        L_alpha=sqrt(lambda_alpha+1/4)-1/2.

        Each eigenvector is phased so its largest-K component has sign
        ``(-1)^(j+K_max)``, following ABC ``frames``. If the K block is complete,
        numerical ``L_alpha`` values are rounded to their exact integers.

    Inputs:
        basis: DelvesBasis - total angular momentum, parity, and K truncation
        channels: DelvesAsymptoticBasis - BF channels ordered as ``(a,v,j,K)``

    Returns:
        frame_transform: NDArray[np.float64] - BF-to-SF orthogonal matrix,
            shape ``(n_channel,n_channel)``
        orbital_L: NDArray[np.float64] - SF orbital angular momenta aligned with
            the columns of ``frame_transform``, shape ``(n_channel,)``
    """
    n_channel = channels.n_channel
    if n_channel < 1:
        message = "At least one Delves asymptotic channel is required"
        logger.error(message)
        raise ValueError(message)

    transform = np.zeros((n_channel, n_channel), dtype=np.float64)
    orbital_L = np.empty(n_channel, dtype=np.float64)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, (arrangement, v, j, _) in enumerate(channels.qns):
        groups.setdefault((arrangement, v, j), []).append(index)

    total_rotation = basis.Jtot * (basis.Jtot + 1)
    for (_, _, j), positions_list in groups.items():
        positions = np.asarray(positions_list, dtype=np.int64)
        K_values = np.asarray([channels.qns[index][3] for index in positions_list], dtype=np.int64)
        if np.any(np.diff(K_values) != 1):
            message = f"Delves frame block must contain consecutive K values, but got {K_values.tolist()}"
            logger.error(message)
            raise ValueError(message)

        rotor = j * (j + 1)
        L2 = np.diag(total_rotation + rotor - 2.0 * K_values.astype(np.float64) ** 2)
        for local_index, K in enumerate(K_values[:-1]):
            if K == 0:
                coupling = -np.sqrt(2.0 * total_rotation * rotor)
            else:
                K_product = K * (K + 1)
                coupling = -np.sqrt((total_rotation - K_product) * (rotor - K_product))
            L2[local_index, local_index + 1] = coupling
            L2[local_index + 1, local_index] = coupling

        eigenvalues, eigenvectors = np.linalg.eigh(L2)
        if np.min(eigenvalues) < -1.0e-10:
            message = f"Delves L^2 block has a negative eigenvalue {np.min(eigenvalues)}"
            logger.error(message)
            raise ValueError(message)
        phase = (-1) ** (j + int(K_values[-1]))
        eigenvectors *= np.where(phase * eigenvectors[-1] < 0.0, -1.0, 1.0)
        values_L = np.sqrt(np.maximum(eigenvalues, 0.0) + 0.25) - 0.5
        if K_values[-1] == min(j, basis.Jtot):
            values_L = np.rint(values_L)

        transform[np.ix_(positions, positions)] = eigenvectors
        orbital_L[positions] = values_L

    return transform, orbital_L


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_delves_Smat(
    Ymat: ArrayLike,
    Etot: EnergyInput,
    basis: DelvesBasis,
    channels: DelvesAsymptoticBasis,
) -> tuple[NDArray[np.complex128], ...]:
    r"""
    Match final Delves BF LogD matrices to reactive scattering matrices.

    This is ABC's ``frames -> bessel -> smatrx -> frames`` sequence. Unlike the
    ordinary atom--diatom matcher, the translational coordinate
    ``R=rho cos(theta)`` varies across the final hyperangle projection, so the
    propagated ``Y(rho)`` cannot be matched at one fixed Jacobi separation.

    Formula:
        First transform the final channel LogD from BF helicity to SF orbital
        representation,

        Y_SF = B_frame^T Y_BF B_frame.

        On the ``M=basis.n_vib_quad`` midpoint points in
        ``0<theta<theta_max``, let

        s=rho sin(theta),  R=rho cos(theta),

        phi_i(theta)=sum_n c^(theta)_{ni} u_n(theta),
        chi_j(s)=sum_n c^(s)_{nj} u_n(s).

        For channel j with ``p_j=sqrt[2 mu |E-epsilon_j|]``, define open-channel
        energy-normalized Riccati functions

        f_j(R)=j_L(p_j R)/sqrt(p_j),
        g_j(R)=y_L(p_j R)/sqrt(p_j),

        and for a closed channel set ``f_j=0`` and use the decaying modified
        Riccati function for ``g_j``. Their hyperradial derivatives entering the
        Pack--Parker projection are

        d_rho[f chi] = cos(theta) f'_R chi
                       +sin(theta) f chi'_s
                       +f chi/(2 rho),

        and analogously for ``g``. With the selector
        ``delta_(a_i,a_j) delta_(j_i,j_j) delta_(alpha_i,alpha_j)``, where alpha
        is the SF state occupying the original K slot,

        A_ij = sqrt(rho) integral phi_i f_j chi_j dtheta,
        B_ij = sqrt(rho) integral phi_i g_j chi_j dtheta,
        C_ij = sqrt(rho) integral phi_i d_rho[f_j chi_j] dtheta,
        D_ij = sqrt(rho) integral phi_i d_rho[g_j chi_j] dtheta.

        The full open-plus-closed reactance matrix satisfies

        (Y_SF B-D) K = Y_SF A-C.

        Its open--open block is symmetrized before

        S_SF=(I-iK_oo)^(-1)(I+iK_oo).

        Finally ``S_ij`` is multiplied by ``(-i)^(L_i+L_j)``, transformed back
        with ``B_frame``, and multiplied by -1 for negative total parity, exactly
        as in ABC. Each returned matrix follows
        ``flatnonzero(channels.energies < Etot[e])`` in BF channel order.

    Inputs:
        Ymat: ArrayLike - final BF channel LogD energy batch, shape
            ``(n_energy,n_channel,n_channel)``
        Etot: EnergyInput - total scattering energies in Hartree
        basis: DelvesBasis - Delves masses, angular momenta, and quadrature sizes
        channels: DelvesAsymptoticBasis - fixed thresholds and both vibrational
            representations at ``rho_match``

    Returns:
        Smat: tuple[NDArray[np.complex128], ...] - BF scattering matrix for each
            energy, with shape ``(n_open[e],n_open[e])``
    """
    energies = get_Etot(Etot)
    Y = np.asarray(Ymat)
    expected_shape = (energies.size, channels.n_channel, channels.n_channel)
    if Y.shape != expected_shape:
        message = f"Ymat must have shape {expected_shape}, but got {Y.shape}"
        logger.error(message)
        raise ValueError(message)
    if np.iscomplexobj(Y) and np.any(np.abs(np.imag(Y)) > 1.0e-13):
        message = "Delves reactive matching currently requires real LogD matrices"
        logger.error(message)
        raise ValueError(message)
    Y = np.asarray(np.real(Y), dtype=np.float64)
    if not np.all(np.isfinite(Y)):
        message = "Ymat must contain finite values"
        logger.error(message)
        raise ValueError(message)

    frame_transform, orbital_L = get_delves_frame_transform(basis, channels)
    scattering_matrices: list[NDArray[np.complex128]] = []
    for energy_index, energy in enumerate(energies):
        if np.any(np.isclose(energy, channels.energies, rtol=0.0, atol=1.0e-14)):
            message = f"Delves asymptotic matching is undefined at a channel threshold Etot={energy}"
            logger.error(message)
            raise ValueError(message)
        open_indices = np.flatnonzero(energy > channels.energies)
        if open_indices.size == 0:
            scattering_matrices.append(np.empty((0, 0), dtype=np.complex128))
            continue

        Y_SF = frame_transform.T @ Y[energy_index] @ frame_transform
        regular_values, irregular_values, regular_derivatives, irregular_derivatives = _delves_reference_matrices(
            float(energy), basis, channels, orbital_L
        )
        left = Y_SF @ irregular_values - irregular_derivatives
        right = Y_SF @ regular_values - regular_derivatives
        reaction = np.linalg.solve(left, right)
        reaction_open = reaction[np.ix_(open_indices, open_indices)]
        reaction_open = 0.5 * (reaction_open + reaction_open.T)

        identity = np.eye(open_indices.size, dtype=np.complex128)
        scattering_SF = np.linalg.solve(identity - 1.0j * reaction_open, identity + 1.0j * reaction_open)
        open_L = orbital_L[open_indices]
        scattering_SF *= np.exp(-0.5j * np.pi * (open_L[:, None] + open_L[None, :]))
        open_frame = frame_transform[np.ix_(open_indices, open_indices)]
        scattering_BF = open_frame @ scattering_SF @ open_frame.T
        if basis.system_parity == -1:
            scattering_BF *= -1.0
        scattering_matrices.append(np.asarray(scattering_BF, dtype=np.complex128))

    return tuple(scattering_matrices)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _delves_reference_matrices(
    energy: float,
    basis: DelvesBasis,
    channels: DelvesAsymptoticBasis,
    orbital_L: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return Pack--Parker A, B, C, and D projection matrices."""
    reduced_mass, _ = mass_scale(basis.mass)
    rho = channels.rho_match
    limit = float(theta_max(rho, basis.scaled_r_max))
    theta, weights = midpoint_quad(0.0, limit, basis.n_vib_quad)
    theta_primitive = sine_basis(0.0, limit, basis.n_sine, theta)
    s = rho * np.sin(theta)
    R = rho * np.cos(theta)
    s_primitive = sine_basis(0.0, basis.scaled_r_max, basis.n_sine, s)
    s_derivative = sine_basis(0.0, basis.scaled_r_max, basis.n_sine, s, derivative=1)

    local_theta_coefficients = np.empty_like(channels.s_coefficients)
    angular_positions = {label: index for index, label in enumerate(basis.angular_qns)}
    for channel_index, (arrangement, _, j, K) in enumerate(channels.qns):
        position = angular_positions[(arrangement, j, K)]
        block = slice(position * basis.n_sine, (position + 1) * basis.n_sine)
        local_theta_coefficients[:, channel_index] = channels.theta_coefficients[block, channel_index]

    delves_wave = theta_primitive @ local_theta_coefficients
    jacobi_wave = s_primitive @ channels.s_coefficients
    jacobi_derivative = s_derivative @ channels.s_coefficients
    n_channel = channels.n_channel
    translation_regular = np.zeros((theta.size, n_channel), dtype=np.float64)
    translation_irregular = np.empty((theta.size, n_channel), dtype=np.float64)
    radial_regular = np.zeros((theta.size, n_channel), dtype=np.float64)
    radial_irregular = np.empty((theta.size, n_channel), dtype=np.float64)

    for channel_index, (threshold, ell) in enumerate(zip(channels.energies, orbital_L, strict=True)):
        momentum = np.sqrt(2.0 * reduced_mass * abs(energy - threshold))
        arguments = momentum * R
        if energy > threshold:
            for point, argument in enumerate(arguments):
                j_value, y_value, j_prime, y_prime = riccati_bessel_jy(float(ell), float(argument))
                translation_regular[point, channel_index] = j_value / np.sqrt(momentum)
                translation_irregular[point, channel_index] = y_value / np.sqrt(momentum)
                radial_regular[point, channel_index] = np.sqrt(momentum) * j_prime
                radial_irregular[point, channel_index] = np.sqrt(momentum) * y_prime
        else:
            value, derivative = _scaled_modified_riccati_k(float(ell), arguments)
            translation_irregular[:, channel_index] = value / np.sqrt(momentum)
            radial_irregular[:, channel_index] = np.sqrt(momentum) * derivative

    fa = translation_regular * jacobi_wave
    fb = translation_irregular * jacobi_wave
    fc = np.cos(theta)[:, None] * radial_regular * jacobi_wave + np.sin(theta)[:, None] * translation_regular * jacobi_derivative
    fd = np.cos(theta)[:, None] * radial_irregular * jacobi_wave + np.sin(theta)[:, None] * translation_irregular * jacobi_derivative
    fc += fa / (2.0 * rho)
    fd += fb / (2.0 * rho)

    selector = np.zeros((n_channel, n_channel), dtype=np.float64)
    for row, (arrangement_i, _, j_i, K_i) in enumerate(channels.qns):
        for column, (arrangement_j, _, j_j, K_j) in enumerate(channels.qns):
            if arrangement_i == arrangement_j and j_i == j_j and K_i == K_j:
                selector[row, column] = 1.0

    weighted_delves = np.sqrt(rho) * weights[:, None] * delves_wave
    regular_values = (weighted_delves.T @ fa) * selector
    irregular_values = (weighted_delves.T @ fb) * selector
    regular_derivatives = (weighted_delves.T @ fc) * selector
    irregular_derivatives = (weighted_delves.T @ fd) * selector
    return regular_values, irregular_values, regular_derivatives, irregular_derivatives


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _scaled_modified_riccati_k(ell: float, arguments: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a commonly scaled decaying modified Riccati function and derivative."""
    nu = ell + 0.5
    reference_argument = float(arguments[0])
    scaled_K = special.kve(nu, arguments)
    value = np.sqrt(np.pi * arguments / 2.0) * scaled_K * np.exp(reference_argument - arguments)
    previous_K = special.kve(nu - 1.0, arguments)
    log_derivative = -previous_K / scaled_K - nu / arguments + 0.5 / arguments
    derivative = value * log_derivative
    if not np.all(np.isfinite(value)) or not np.all(np.isfinite(derivative)):
        message = f"Modified Riccati functions are non-finite for ell={ell}"
        logger.error(message)
        raise ValueError(message)
    return value, derivative


# ----------------------------------------------------------------------------------------
