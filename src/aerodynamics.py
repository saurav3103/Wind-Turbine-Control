# ══════════════════════════════════════════════════════
#  aerodynamics.py
#  Cp surface, aerodynamic torque, thrust force
# ══════════════════════════════════════════════════════

import numpy as np
from parameters import R, rho


def compute_Cp(lambda_, beta):
    """
    Analytical Cp(lambda, beta) approximation.
    Widely used in wind turbine control research (Heier 2006).

    Parameters
    ----------
    lambda_ : float or np.ndarray — tip speed ratio
    beta    : float or np.ndarray — pitch angle (degrees)

    Returns
    -------
    Cp : float or np.ndarray — power coefficient (≥ 0)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        lambda_i = 1.0 / (1.0 / (lambda_ + 0.08 * beta)
                          - 0.035 / (beta**3 + 1))
    Cp = (0.5176 * (116.0 / lambda_i - 0.4 * beta - 5.0)
          * np.exp(-21.0 / lambda_i)
          + 0.0068 * lambda_)
    return np.maximum(Cp, 0.0)


def compute_CT(lambda_, beta):
    """
    Approximate thrust coefficient CT(lambda, beta).
    Uses the relationship CT ≈ (4/lambda) * Cp for momentum theory.
    """
    Cp = compute_Cp(lambda_, beta)
    lam = np.maximum(lambda_, 0.01)
    return np.where(lam > 0.01, 4.0 * Cp / lam, 0.0)


def aerodynamic_torque(omega_r, v, beta):
    """
    Aerodynamic torque on rotor (N·m).

    Parameters
    ----------
    omega_r : float — rotor angular speed (rad/s)
    v       : float — wind speed (m/s)
    beta    : float — pitch angle (degrees)
    """
    lam    = max(omega_r * R / v, 0.01)
    Cp     = compute_Cp(lam, beta)
    P_aero = 0.5 * rho * np.pi * R**2 * v**3 * Cp
    return P_aero / omega_r


def aerodynamic_thrust(omega_r, v, beta):
    """
    Aerodynamic thrust force on rotor (N).
    """
    lam = max(omega_r * R / v, 0.01)
    CT  = compute_CT(lam, beta)
    return 0.5 * rho * np.pi * R**2 * v**2 * CT


def optimal_tsr_and_cp():
    """
    Numerically find lambda* and Cp_max at beta=0.
    Returns (lambda_opt, Cp_max).
    """
    lam_sweep = np.linspace(1, 15, 5000)
    Cp_sweep  = compute_Cp(lam_sweep, beta=0)
    idx = np.argmax(Cp_sweep)
    return lam_sweep[idx], Cp_sweep[idx]
