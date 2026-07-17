import numpy as np
from scipy import special

import pyticc as ticc


def test_riccati_bessel_order_zero_matches_elementary_functions() -> None:
    x = 1.7

    j_value, n_value, j_derivative, n_derivative = ticc.riccati_bessel_jy(0.0, x)

    np.testing.assert_allclose([j_value, n_value], [np.sin(x), -np.cos(x)], rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose([j_derivative, n_derivative], [np.cos(x), np.sin(x)], rtol=1.0e-14, atol=1.0e-14)


def test_riccati_bessel_noninteger_order_has_unit_wronskian() -> None:
    j_value, n_value, j_derivative, n_derivative = ticc.riccati_bessel_jy(2.3, 4.2)

    np.testing.assert_allclose(j_value * n_derivative - j_derivative * n_value, 1.0, rtol=1.0e-13, atol=1.0e-13)


def test_modified_bessel_log_derivatives_match_scipy() -> None:
    nu = 1.7
    x = 3.1

    I_logD, K_logD = ticc.modified_bessel_IK_logD(nu, x)

    np.testing.assert_allclose(I_logD, special.ivp(nu, x) / special.iv(nu, x), rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(K_logD, special.kvp(nu, x) / special.kv(nu, x), rtol=1.0e-13, atol=1.0e-13)
