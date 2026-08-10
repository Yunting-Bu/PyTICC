from pathlib import Path

import numpy as np
import pytest
from scipy.special import roots_legendre

from pyticc.basis.angle import norm_YjK
from pyticc.electric import (
    ElectricResponseTable,
    ElectricResponseValues,
    electric_coefficients,
    load_electric_response_csv,
    rotor_orientation_moment_matrices,
)


def test_load_electric_response_csv_and_natural_spline(tmp_path: Path) -> None:
    path = tmp_path / "response.csv"
    path.write_text(
        "r,mu_z,alpha_xx,alpha_zz,beta_zzz,beta_xxz\n1.0,1.0,2.0,3.0,4.0,5.0\n2.0,2.0,4.0,6.0,8.0,10.0\n3.0,3.0,6.0,9.0,12.0,15.0\n",
        encoding="utf-8",
    )

    response = load_electric_response_csv(path).evaluate(np.array([1.5, 2.5]))

    np.testing.assert_allclose(response.mu_z, [1.5, 2.5])
    np.testing.assert_allclose(response.alpha_xx, [3.0, 5.0])
    np.testing.assert_allclose(response.beta_xxz, [7.5, 12.5])


def test_electric_response_csv_requires_fixed_header(tmp_path: Path) -> None:
    path = tmp_path / "response.csv"
    path.write_text("r,mu,alpha_xx\n1.0,1.0,2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header"):
        load_electric_response_csv(path)


def test_electric_response_rejects_extrapolation() -> None:
    table = ElectricResponseTable(
        r=np.array([1.0, 2.0]),
        mu_z=np.array([1.0, 1.0]),
        alpha_xx=np.array([2.0, 2.0]),
        alpha_zz=np.array([3.0, 3.0]),
        beta_zzz=np.array([4.0, 4.0]),
        beta_xxz=np.array([5.0, 5.0]),
    )

    with pytest.raises(ValueError, match="within"):
        table.evaluate([0.9])


def test_electric_response_sorts_rows_by_bond_length() -> None:
    table = ElectricResponseTable(
        r=np.array([2.0, 1.0, 3.0]),
        mu_z=np.array([20.0, 10.0, 30.0]),
        alpha_xx=np.array([200.0, 100.0, 300.0]),
        alpha_zz=np.array([2000.0, 1000.0, 3000.0]),
        beta_zzz=np.array([4.0, 3.0, 5.0]),
        beta_xxz=np.array([7.0, 6.0, 8.0]),
    )

    np.testing.assert_array_equal(table.r, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(table.mu_z, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(table.alpha_xx, [100.0, 200.0, 300.0])


def test_electric_response_rejects_duplicate_bond_lengths() -> None:
    values = np.ones(3)

    with pytest.raises(ValueError, match="duplicates"):
        ElectricResponseTable(
            r=np.array([1.0, 2.0, 1.0]),
            mu_z=values,
            alpha_xx=values,
            alpha_zz=values,
            beta_zzz=values,
            beta_xxz=values,
        )


def test_electric_coefficients_match_arhf_fortran_power_form() -> None:
    response = ElectricResponseValues(
        mu_z=np.array([0.8]),
        alpha_xx=np.array([5.2]),
        alpha_zz=np.array([9.4]),
        beta_zzz=np.array([18.0]),
        beta_xxz=np.array([2.3]),
    )
    strength = 0.012
    cosine = np.linspace(-1.0, 1.0, 11)

    coefficients = electric_coefficients(response, strength)[:, 0]
    polynomial = sum(coefficients[power] * cosine**power for power in range(4))
    rotated_tensor_form = (
        -response.mu_z[0] * strength * cosine
        - 0.5 * strength**2 * (response.alpha_xx[0] + (response.alpha_zz[0] - response.alpha_xx[0]) * cosine**2)
        - strength**3 * (3.0 * response.beta_xxz[0] * cosine + (response.beta_zzz[0] - 3.0 * response.beta_xxz[0]) * cosine**3) / 6.0
    )

    np.testing.assert_allclose(polynomial, rotated_tensor_form, atol=1.0e-15)


@pytest.mark.parametrize("m", [-2, 0, 1])
def test_rotor_orientation_moment_matrices_match_gauss_legendre_quadrature(m: int) -> None:
    jmax = 5
    nodes, weights = roots_legendre(80)
    j_values = np.arange(abs(m), jmax + 1)
    angular_values = np.stack([norm_YjK(int(j), m, nodes) for j in j_values])

    for power, matrix in enumerate(rotor_orientation_moment_matrices(jmax, m)):
        numerical = np.einsum("iq,q,jq->ij", angular_values, weights * nodes**power, angular_values, optimize=True)
        np.testing.assert_allclose(matrix, numerical, atol=2.0e-13)


def test_cosine_squared_retains_virtual_boundary_contribution() -> None:
    jmax = 2
    matrix_x = rotor_orientation_moment_matrices(jmax, 0)[1]
    matrix_x2 = rotor_orientation_moment_matrices(jmax, 0)[2]

    assert matrix_x2[-1, -1] > (matrix_x @ matrix_x)[-1, -1]
