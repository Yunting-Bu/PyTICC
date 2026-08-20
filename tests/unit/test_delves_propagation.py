import jax
import numpy as np
import pytest

from pyticc.basis.delves import DelvesBasis
from pyticc.matrix.delves import mass_scale
from pyticc.pes.total import TotalPES
from pyticc.propagation.config import Propagation
from pyticc.propagation.delves import propagate_delves
from pyticc.scattering.delves_hamiltonian import DelvesHamiltonian


def make_basis() -> DelvesBasis:
    return DelvesBasis(
        mass=(1.0, 1.0, 1.0),
        Jtot=0,
        system_parity=1,
        exchange_parity=0,
        jmax=0,
        K_cut=0,
        E_max=1.0,
        rho_min=2.0,
        scaled_r_max=5.0,
        n_sine=1,
        n_vib_quad=4,
        n_gamma_quad=4,
        angular_qns=((1, 0, 0),),
    )


def make_hamiltonian(pes: TotalPES) -> DelvesHamiltonian:
    return DelvesHamiltonian(make_basis(), pes)


def install_scalar_surface(monkeypatch: pytest.MonkeyPatch, energy) -> None:
    import pyticc.scattering.delves_hamiltonian as module

    monkeypatch.setattr(module, "get_surface_matrices_delves", lambda basis, total_pes, rho: (np.array([[energy(rho)]]), np.eye(1)))
    monkeypatch.setattr(
        module,
        "solve_surface_delves",
        lambda H, S, overlap_cut: (np.array([H[0, 0]]), np.eye(1), np.ones(1)),
    )
    monkeypatch.setattr(
        module,
        "get_sector_transform_delves",
        lambda basis, rho_a, coefficients_a, rho_b, coefficients_b: np.eye(1),
    )


def test_delves_uses_piecewise_configured_half_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    sampled_rho: list[float] = []

    def energy(rho: float) -> float:
        sampled_rho.append(rho)
        return 0.5

    install_scalar_surface(monkeypatch, energy)
    basis = make_basis()
    config = Propagation((2.0, 2.6, 3.0), (0.2, 0.1), device="cpu")

    result = propagate_delves(make_hamiltonian(TotalPES(lambda bonds: np.zeros(bonds.shape[1]))), [0.0], config)

    np.testing.assert_allclose(result.radial_points, [2.0, 2.4, 2.6, 2.8, 3.0], atol=1.0e-14)
    np.testing.assert_allclose(sampled_rho, [2.2, 2.5, 2.7, 2.9], atol=1.0e-14)
    assert result.rho_final == pytest.approx(3.0)
    assert result.surface_rho == pytest.approx(2.9)
    assert result.Y_final.shape == (1, 1, 1)
    assert isinstance(result.Y_final, jax.Array)
    assert {device.platform for device in result.Y_final.devices()} == {"cpu"}

    reduced_mass, _ = mass_scale(basis.mass)
    momentum = np.sqrt(2.0 * reduced_mass * 0.5)
    expected_logD = momentum / np.tanh(momentum * (config.boundaries[-1] - config.boundaries[0]))
    np.testing.assert_allclose(result.Y_final[0, 0, 0], expected_logD, rtol=2.0e-14, atol=2.0e-14)


def test_delves_supports_capture_and_energy_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    install_scalar_surface(monkeypatch, lambda rho: -0.5)
    result = propagate_delves(
        make_hamiltonian(TotalPES(lambda bonds: np.zeros(bonds.shape[1]))),
        [0.0, 0.1],
        Propagation((2.0, 2.2), (0.1,), mode="capture", device="cpu"),
    )

    assert result.Y_final.shape == (2, 1, 1)
    assert result.Y_final.dtype == np.complex128
    assert np.all(np.imag(np.asarray(jax.device_get(result.Y_final))[:, 0, 0]) < 0.0)


def test_delves_runs_the_real_multiple_arrangement_surface_chain() -> None:
    mass = (2.0, 3.0, 5.0)
    basis = DelvesBasis(
        mass=mass,
        Jtot=0,
        system_parity=1,
        exchange_parity=0,
        jmax=0,
        K_cut=0,
        E_max=1.0,
        rho_min=6.8,
        scaled_r_max=5.0,
        n_sine=1,
        n_vib_quad=30,
        n_gamma_quad=30,
        angular_qns=((1, 0, 0), (2, 0, 0), (3, 0, 0)),
    )
    pes = TotalPES(lambda bonds: 0.001 * (bonds[0] + 2.0 * bonds[1] + 3.0 * bonds[2]))

    result = propagate_delves(
        DelvesHamiltonian(basis, pes),
        [0.05, 0.08],
        Propagation((6.8, 7.0), (0.1,), device="cpu"),
    )

    assert result.Y_final.shape == (2, 3, 3)
    assert result.surface_energies.shape == (3,)
    assert result.surface_coefficients.shape == (3, 3)
    np.testing.assert_allclose(result.radial_points, [6.8, 7.0], atol=1.0e-14)
    np.testing.assert_allclose(result.Y_final, np.swapaxes(result.Y_final, -1, -2), atol=1.0e-14)
