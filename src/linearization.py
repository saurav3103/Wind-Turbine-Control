# ══════════════════════════════════════════════════════
#  linearization.py
#  Numerical Jacobian → A, B matrices at operating points
#  Two-mass drivetrain state: [omega_r, omega_g, theta]
# ══════════════════════════════════════════════════════

import numpy as np
from scipy.optimize import brentq

from parameters import (R, rho, w_rated, T_rated, K_opt,
                         J_r, J_g_lss, K_s, B_s)
from aerodynamics import compute_Cp


def T_aero_fn(omega_r, v, beta=0.0):
    lam = max(omega_r * R / v, 0.01)
    Cp  = compute_Cp(lam, beta)
    return (0.5 * rho * np.pi * R**2 * v**3 * Cp) / omega_r


def dTaero_dwr(omega_r, v, beta=0.0, eps=1e-5):
    return (T_aero_fn(omega_r + eps, v, beta)
            - T_aero_fn(omega_r - eps, v, beta)) / (2 * eps)


def dTaero_dv(omega_r, v, beta=0.0, eps=1e-4):
    return (T_aero_fn(omega_r, v + eps, beta)
            - T_aero_fn(omega_r, v - eps, beta)) / (2 * eps)


def find_equilibrium(v, mode='below'):
    """
    Find equilibrium (omega_r, beta) for given wind speed.

    mode='below' : Region 2 — MPPT, beta=0
    mode='above' : Region 3 — rated speed, find beta s.t. T_aero=T_rated
    """
    if mode == 'below':
        wr0 = brentq(lambda w: T_aero_fn(w, v) - K_opt * w**2, 0.1, 2.5)
        return wr0, 0.0
    else:
        fa = T_aero_fn(w_rated, v, 0.0)  - T_rated
        fb = T_aero_fn(w_rated, v, 45.0) - T_rated
        if fa * fb > 0:
            raise ValueError(
                f"No equilibrium beta in [0°, 45°] at v={v} m/s. "
                f"Try a higher wind speed or wider beta range.")
        beta_eq = brentq(lambda b: T_aero_fn(w_rated, v, b) - T_rated,
                         0.0, 45.0)
        return w_rated, beta_eq


def build_A_matrix(v_op, mode='below'):
    """
    Build linearised A matrix and wind-disturbance B vector
    at operating point (v_op, mode).

    State: x = [Δomega_r, Δomega_g, Δtheta]
    """
    wr0, beta0    = find_equilibrium(v_op, mode)
    dTa_dwr       = dTaero_dwr(wr0, v_op, beta0)
    dTgen_dwr     = 2 * K_opt * wr0 if mode == 'below' else 0.0

    A = np.array([
        [(dTa_dwr - B_s - dTgen_dwr) / J_r,  B_s / J_r,       -K_s / J_r     ],
        [ B_s / J_g_lss,                      -B_s / J_g_lss,   K_s / J_g_lss ],
        [ 1.0,                                -1.0,              0.0            ]
    ])

    dTa_dv = dTaero_dv(wr0, v_op, beta0)
    B_vec  = np.array([dTa_dv / J_r, 0.0, 0.0])

    return A, B_vec, wr0, beta0


def print_linearization(v_op, mode='below'):
    A, B_vec, wr0, beta0 = build_A_matrix(v_op, mode)
    eigs = np.linalg.eigvals(A)

    print(f"\n{'═'*55}")
    print(f"  v = {v_op} m/s  [{mode} rated]")
    print(f"{'═'*55}")
    print(f"  Equilibrium  ωr = {wr0:.4f} rad/s,  β₀ = {beta0:.2f}°")
    print(f"\n  A matrix:")
    for row in A:
        print(f"    [{row[0]:12.4f}  {row[1]:12.4f}  {row[2]:12.4f}]")
    print(f"\n  B vector: [{B_vec[0]:.6f}  {B_vec[1]:.6f}  {B_vec[2]:.6f}]")
    print(f"\n  Eigenvalues:")
    for e in sorted(eigs, key=lambda x: abs(x.imag)):
        fn_hz = abs(e.imag) / (2 * np.pi)
        zeta  = -e.real / abs(e) if abs(e) > 1e-10 else 0.0
        flag  = "stable" if e.real < 0 else "*** UNSTABLE ***"
        print(f"    {e.real:9.4f} ± {abs(e.imag):.4f}j"
              f"   f_n={fn_hz:.4f} Hz  ζ={zeta:.4f}  [{flag}]")

    return A, B_vec


if __name__ == '__main__':
    print_linearization(8.0,  'below')
    print_linearization(14.0, 'above')
