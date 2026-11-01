"""Fitzpatrick (1999) extinction law, parametrised by R_V.

Implemented directly (no external dependency) following the published
prescription: a cubic spline through optical/IR anchor points joined to the
analytic UV curve. Valid from the far-UV to ~2.7 micron and extrapolated
smoothly to A_lambda -> 0 at 1/lambda -> 0.
"""
import numpy as np
from scipy.interpolate import CubicSpline

_X0, _GAMMA, _C3, _C4 = 4.596, 0.99, 3.23, 0.41


def _poly(coeffs, x):
    return sum(c * x ** i for i, c in enumerate(coeffs))


def _uv(x, rv):
    c2 = -0.824 + 4.717 / rv
    c1 = 2.030 - 3.007 * c2
    x = np.asarray(x, float)
    d = x ** 2 / ((x ** 2 - _X0 ** 2) ** 2 + (x * _GAMMA) ** 2)
    k = c1 + c2 * x + _C3 * d
    fuv = np.where(x >= 5.9, 0.5392 * (x - 5.9) ** 2 + 0.05644 * (x - 5.9) ** 3, 0.0)
    return k + _C4 * fuv + rv  # A(x)/E(B-V)


def _spline(rv):
    x_iropt = np.array([0.0, 1e4 / 26500., 1e4 / 12200., 1e4 / 6000., 1e4 / 5470.,
                        1e4 / 4670., 1e4 / 4110.])
    y_ir = np.array([0.0, 0.26469, 0.82925]) * rv / 3.1
    y_op = np.array([
        _poly([-4.22809e-01, 1.00270, 2.13572e-04], rv),
        _poly([-5.13540e-02, 1.00216, -7.35778e-05], rv),
        _poly([7.00127e-01, 1.00184, -3.32598e-05], rv),
        _poly([1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05], rv),
    ])
    x_uv = np.array([1e4 / 2700., 1e4 / 2600.])
    xk = np.concatenate([x_iropt, x_uv])
    yk = np.concatenate([y_ir, y_op, _uv(x_uv, rv)])
    return CubicSpline(xk, yk, bc_type="natural")


def a_lambda_over_av(wave_um, rv=5.0):
    """A_lambda / A_V for wavelengths in micron."""
    x = 1.0 / np.asarray(wave_um, float)
    out = np.empty_like(x)
    cut = 1e4 / 2700.
    uv = x >= cut
    out[uv] = _uv(x[uv], rv)
    out[~uv] = _spline(rv)(x[~uv])
    out = np.clip(out, 0.0, None)
    return out / rv


def transmission(wave_um, av, rv=5.0):
    """10**(-0.4 A_lambda); ``av`` may be a scalar or 1-D array (-> 2-D output)."""
    alam = a_lambda_over_av(wave_um, rv)
    av = np.atleast_1d(np.asarray(av, float))
    t = 10.0 ** (-0.4 * av[:, None] * alam[None, :])
    return t[0] if t.shape[0] == 1 else t
