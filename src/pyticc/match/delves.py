from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.basis.delves import DelvesBasis, delves_theta_basis, midpoint_quad, sine_basis
from pyticc.matrix.delves import asymptotic_potential, mass_scale
from pyticc.matrix.delves_metric import get_sector_overlap_delves
from pyticc.matrix.delves_reference import get_delves_reference_basis
from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DelvesAsymptoticBasis:
    """
    Fixed Jacobi channels and their finite-hyperradius Delves representatives.

    Members:
        qns: tuple[tuple[int, int, int, int], ...] - channel labels ``(a,v,j,K)``
            ordered as arrangement, vibration, rotation, and helicity; arrangements
            and vibrational levels are one- and zero-based, respectively
        energies: NDArray[np.float64] - fixed Jacobi rovibrational thresholds in
            Hartree, shape ``(n_channel,)``
        s_coefficients: NDArray[np.float64] - fixed-Jacobi vibrational coefficients
            in the normalized sine basis on ``0<s<scaled_r_max``, indexed as
            ``[sine,channel]`` with shape ``(basis.n_sine,n_channel)``
        rho_match: float - matching hyperradius in bohr
        theta_coefficients: NDArray[np.float64] - coefficients of the corresponding
            finite-rho, K-independent Delves functions in the complete primitive
            ``(a,j,K,n)`` basis, shape ``(basis.n_primitive,n_channel)``
        theta_energies: NDArray[np.float64] - finite-rho reference eigenvalues in
            Hartree, shape ``(n_channel,)``; these correlate by state index with
            ``energies`` but are not scattering thresholds
    """

    qns: tuple[tuple[int, int, int, int], ...]
    energies: NDArray[np.float64]
    s_coefficients: NDArray[np.float64]
    rho_match: float
    theta_coefficients: NDArray[np.float64]
    theta_energies: NDArray[np.float64]

    @property
    def n_channel(self) -> int:
        """Return the number of retained asymptotic channels."""
        return len(self.qns)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_delves_asymptotic_basis(
    basis: DelvesBasis,
    total_pes: TotalPES,
    rho_match: float,
) -> DelvesAsymptoticBasis:
    r"""
    Build ABC's fixed Jacobi channels and final finite-rho matching basis.

    This implements the roles of ABC ``basis`` and the final
    ``sbasis(mode=1)``. The fixed Jacobi problem determines the physical channel
    thresholds. A second problem at ``rho_match`` supplies functions in the same
    hyperangle primitive used by the propagated surface basis. Both use the total
    PES only through its arrangement asymptote; no interaction-potential input is
    introduced.

    Formula:
        Let ``mu=sqrt(m_A m_B m_C/(m_A+m_B+m_C))`` and

        u_n(s)=sqrt(2/s_max) sin(n pi s/s_max),  n=1,...,N.

        For every allowed arrangement ``a`` and diatomic rotation ``j``, the
        fixed Jacobi eigenproblem is

        sum_m H^(s,aj)_nm c_m^(avj) = epsilon_avj c_n^(avj),

        H^(s,aj)_nm = delta_nm (n pi/s_max)^2/(2 mu)
          + integral_0^s_max u_n(s)
            [j(j+1)/(2 mu s^2)+V_a^asym(s)]u_m(s) ds.

        Every level with ``epsilon_avj <= E_max`` is replicated over the K values
        present in ``basis.angular_qns``. Channels are sorted as ``(a,v,j,K)``.

        At ``rho=rho_match``, define

        theta_max=asin[min(1,s_max/rho)],
        s=rho sin(theta),

        u_n(theta;rho)=sqrt(2/theta_max)
            sin(n pi theta/theta_max).

        For each retained ``(a,v,j,K)``, the state with the same eigenvalue index
        ``v`` is selected from the K-independent finite-rho reference problem

        H^(theta,aj)_nm = delta_nm
          {[(n pi/theta_max)^2-1/4]/(2 mu rho^2)}
          + integral_0^theta_max u_n(theta;rho)
            [j(j+1)/(2 mu rho^2 sin^2(theta))
             +V_a^asym(rho sin(theta))]u_m(theta;rho)dtheta.

        ``epsilon_avj`` from the fixed-s problem remains the scattering threshold;
        the finite-rho eigenvalue only labels and constructs its Delves-correlated
        partner. All integrals use ``basis.n_vib_quad`` midpoint points.

    Inputs:
        basis: DelvesBasis - resolved Delves basis and energy cutoff; masses in
            electron masses
        total_pes: TotalPES - scalar total adiabatic PES in bohr and Hartree
        rho_match: float - final positive matching hyperradius in bohr

    Returns:
        channels: DelvesAsymptoticBasis - fixed thresholds, channel labels, and
            the two vibrational coefficient representations required by matching
    """
    if not np.isfinite(rho_match) or rho_match <= 0.0:
        message = f"rho_match must be finite and positive, but got {rho_match}"
        logger.error(message)
        raise ValueError(message)

    if basis.n_channel:
        theta_coefficients, theta_energies = get_delves_reference_basis(
            basis,
            total_pes,
            rho_match,
            asymptotic=True,
        )
        return DelvesAsymptoticBasis(
            qns=basis.qns,
            energies=basis.energies.copy(),
            s_coefficients=basis.s_coefficients.copy(),
            rho_match=float(rho_match),
            theta_coefficients=theta_coefficients,
            theta_energies=theta_energies,
        )

    reduced_mass, _ = mass_scale(basis.mass)
    reference = asymptotic_potential(total_pes, basis.mass)
    s_grid, s_weights = midpoint_quad(0.0, basis.scaled_r_max, basis.n_vib_quad)
    s_values = sine_basis(0.0, basis.scaled_r_max, basis.n_sine, s_grid)
    s_kinetic = _sine_kinetic(basis.n_sine, basis.scaled_r_max, reduced_mass, theta=False)

    theta, theta_weights, theta_values = delves_theta_basis(basis, rho_match)
    theta_limit = float(np.sum(theta_weights))
    theta_kinetic = _sine_kinetic(basis.n_sine, theta_limit, reduced_mass * rho_match**2, theta=True)

    angular_labels: dict[tuple[int, int], tuple[int, ...]] = {}
    for arrangement, j, K in basis.angular_qns:
        angular_labels.setdefault((arrangement, j), ())
        angular_labels[(arrangement, j)] += (K,)

    channel_records: list[tuple[tuple[int, int, int, int], float, NDArray[np.float64], float, NDArray[np.float64]]] = []
    for arrangement, j in angular_labels:
        s_hamiltonian = _reference_hamiltonian(
            s_values,
            s_weights,
            s_kinetic,
            j * (j + 1) / (2.0 * reduced_mass * s_grid**2) + reference(arrangement, s_grid),
        )
        energies, s_vectors = np.linalg.eigh(s_hamiltonian)
        s_vectors = _clean_vibrational_phases(s_vectors, s_values)

        scaled_r = rho_match * np.sin(theta)
        theta_hamiltonian = _reference_hamiltonian(
            theta_values,
            theta_weights,
            theta_kinetic,
            j * (j + 1) / (2.0 * reduced_mass * rho_match**2 * np.sin(theta) ** 2) + reference(arrangement, scaled_r),
        )
        theta_energies, theta_vectors = np.linalg.eigh(theta_hamiltonian)
        theta_vectors = _clean_vibrational_phases(theta_vectors, theta_values)

        retained = np.flatnonzero(energies <= basis.E_max)
        for v in retained:
            for K in angular_labels[(arrangement, j)]:
                channel_records.append(
                    (
                        (arrangement, int(v), j, K),
                        float(energies[v]),
                        s_vectors[:, v].copy(),
                        float(theta_energies[v]),
                        theta_vectors[:, v].copy(),
                    )
                )

    if not channel_records:
        message = f"No Delves asymptotic channels lie below E_max={basis.E_max} Hartree"
        logger.error(message)
        raise ValueError(message)
    channel_records.sort(key=lambda item: item[0])

    qns = tuple(record[0] for record in channel_records)
    energies = np.asarray([record[1] for record in channel_records], dtype=np.float64)
    s_coefficients = np.column_stack([record[2] for record in channel_records])
    theta_energies = np.asarray([record[3] for record in channel_records], dtype=np.float64)
    theta_coefficients = np.zeros((basis.n_primitive, len(channel_records)), dtype=np.float64)
    angular_positions = {label: index for index, label in enumerate(basis.angular_qns)}
    for channel_index, (record, *_) in enumerate(channel_records):
        arrangement, _, j, K = record
        primitive_index = angular_positions[(arrangement, j, K)]
        block = slice(primitive_index * basis.n_sine, (primitive_index + 1) * basis.n_sine)
        theta_coefficients[block, channel_index] = channel_records[channel_index][4]

    return DelvesAsymptoticBasis(
        qns=qns,
        energies=energies,
        s_coefficients=s_coefficients,
        rho_match=float(rho_match),
        theta_coefficients=theta_coefficients,
        theta_energies=theta_energies,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def transform_logD_to_delves_channels(
    basis: DelvesBasis,
    rho_surface: float,
    surface_coefficients: ArrayLike,
    Ymat: ArrayLike,
    channels: DelvesAsymptoticBasis,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    r"""
    Transform the propagated surface LogD to ABC's final Delves channel basis.

    Formula:
        Let ``C_surface`` contain the final primitive-to-surface eigenvectors and
        ``C_channel`` the finite-rho coefficients in ``channels``. With the
        directed primitive overlap

        P = <primitive(rho_surface)|primitive(rho_match)>,

        the surface-to-channel overlap and transformed LogD are

        T = C_surface^T P C_channel,

        Y_channel(E) = T^T Y_surface(E) T.

        This is the ``npp>n`` branch of ABC ``logder``. It is a pure basis
        transformation: no fictitious propagation from ``rho_surface`` to
        ``rho_match`` is performed. Normally the two radii are identical, but
        the directed overlap keeps the operation well-defined for a final
        surface centre distinct from the requested boundary.

    Inputs:
        basis: DelvesBasis - primitive Delves basis specification
        rho_surface: float - hyperradius of ``surface_coefficients`` in bohr
        surface_coefficients: ArrayLike - primitive-to-surface coefficients,
            shape ``(basis.n_primitive,n_surface)``
        Ymat: ArrayLike - final surface LogD matrix or energy batch, shape
            ``(...,n_surface,n_surface)``
        channels: DelvesAsymptoticBasis - final channel representation

    Returns:
        Y_channel: NDArray[np.float64] | NDArray[np.complex128] - transformed
            LogD matrix or batch, shape ``(...,n_channel,n_channel)``
    """
    surface = np.asarray(surface_coefficients, dtype=np.float64)
    Y = np.asarray(Ymat)
    if surface.ndim != 2 or surface.shape[0] != basis.n_primitive or surface.shape[1] < 1:
        message = f"surface_coefficients must have shape ({basis.n_primitive},n_surface), but got {surface.shape}"
        logger.error(message)
        raise ValueError(message)
    if Y.ndim < 2 or Y.shape[-2:] != (surface.shape[1], surface.shape[1]):
        message = f"Ymat must end with shape {(surface.shape[1], surface.shape[1])}, but got {Y.shape}"
        logger.error(message)
        raise ValueError(message)
    if channels.theta_coefficients.shape != (basis.n_primitive, channels.n_channel):
        message = "channels.theta_coefficients are incompatible with the supplied Delves basis"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(surface)) or not np.all(np.isfinite(Y)):
        message = "surface_coefficients and Ymat must contain finite values"
        logger.error(message)
        raise ValueError(message)

    primitive_overlap = get_sector_overlap_delves(basis, rho_surface, channels.rho_match)
    transform = surface.T @ primitive_overlap @ channels.theta_coefficients
    result = np.einsum("pi,...pq,qj->...ij", transform, Y, transform, optimize=True)
    return np.asarray(result)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _sine_kinetic(n_sine: int, length: float, mass_factor: float, *, theta: bool) -> NDArray[np.float64]:
    """Return diagonal fixed-s or finite-rho sine kinetic energies."""
    modes = np.arange(1, n_sine + 1, dtype=np.float64)
    eigenvalues = (np.pi * modes / length) ** 2
    if theta:
        eigenvalues -= 0.25
    return eigenvalues / (2.0 * mass_factor)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _reference_hamiltonian(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    kinetic: NDArray[np.float64],
    effective_potential: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return one sine-FBR reference Hamiltonian."""
    hamiltonian = values.T @ (weights[:, None] * effective_potential[:, None] * values)
    hamiltonian += np.diag(kinetic)
    return 0.5 * (hamiltonian + hamiltonian.T)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _clean_vibrational_phases(vectors: NDArray[np.float64], primitive_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply ABC's positive-inner-turning-point phase convention."""
    result = vectors.copy()
    wavefunctions = primitive_values @ result
    for state in range(result.shape[1]):
        maximum = float(np.max(np.abs(wavefunctions[:, state])))
        significant = np.flatnonzero(np.abs(wavefunctions[:, state]) > 0.1 * maximum)
        if significant.size and wavefunctions[significant[0], state] < 0.0:
            result[:, state] *= -1.0
    return result


# ----------------------------------------------------------------------------------------
