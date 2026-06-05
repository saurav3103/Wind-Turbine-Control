# ══════════════════════════════════════════════════════
#  observer.py
#  Luenberger observer for shaft twist estimation
#  Gain-scheduled between below/above rated operating pts
# ══════════════════════════════════════════════════════

import numpy as np
from scipy.signal import place_poles

from parameters import (R, rho, w_rated, T_rated, K_opt,
                         J_r, J_g_lss, K_s, B_s, beta_max)
from aerodynamics import compute_Cp
from linearization import build_A_matrix


# ── Observer output matrix: measure [omega_r, omega_g] ──
C = np.array([[1., 0., 0.],
              [0., 1., 0.]])

# ── Desired observer poles (faster than plant) ──
DESIRED_POLES = [-1.0, -1.2 + 3.89j, -1.2 - 3.89j]


def design_observer_gain(A, label=''):
    """Place observer poles via Ackermann/place_poles."""
    res = place_poles(A.T, C.T, DESIRED_POLES)
    L   = res.gain_matrix.T
    if label:
        print(f"Observer gain [{label}]: placement error = {res.rtol:.2e}")
    return L


def build_observer_gains():
    """Design L_below and L_above at two operating points."""
    A_below = build_A_matrix(8.0,  'below')[0]
    A_above = build_A_matrix(14.0, 'above')[0]
    L_below = design_observer_gain(A_below, 'below rated')
    L_above = design_observer_gain(A_above, 'above rated')
    return L_below, L_above


class LuenbergerObserver:
    """
    Nonlinear Luenberger observer for two-mass drivetrain.

    Estimates the unobservable shaft twist angle theta
    from measurements of omega_r and omega_g.

    Gain scheduling: smooth sigmoid blend between L_below
    and L_above centred at 95% of rated speed.
    """

    def __init__(self, omega_init, L_below, L_above):
        self.L_below = L_below
        self.L_above = L_above
        # State estimate: [omega_r_hat, omega_g_hat, theta_hat]
        self.state = np.array([omega_init, omega_init, np.radians(10.0)])

    def _blend_L(self, omega_r):
        alpha = 1.0 / (1.0 + np.exp(-40.0 * (omega_r - w_rated * 0.95)))
        return (1.0 - alpha) * self.L_below + alpha * self.L_above

    def step(self, omega_r_meas, omega_g_meas,
             T_aero_hat, T_gen, dt):
        """
        One observer update step.

        Parameters
        ----------
        omega_r_meas : float — measured rotor speed (possibly noisy)
        omega_g_meas : float — measured generator speed (possibly noisy)
        T_aero_hat   : float — reconstructed aerodynamic torque (N·m)
        T_gen        : float — generator torque command (N·m)
        dt           : float — timestep (s)
        """
        or_h = self.state[0]
        og_h = self.state[1]
        th_h = self.state[2]
        dw_h = or_h - og_h

        T_sh_h = K_s * th_h + B_s * dw_h

        f_obs = np.array([
            (T_aero_hat - T_sh_h) / J_r,
            (T_sh_h     - T_gen)  / J_g_lss,
            dw_h
        ])

        L     = self._blend_L(or_h)
        innov = np.array([omega_r_meas, omega_g_meas]) - C @ self.state
        self.state = self.state + (f_obs + L @ innov) * dt

    @property
    def theta_hat(self):
        return self.state[2]

    @property
    def T_shaft_hat(self):
        dw = self.state[0] - self.state[1]
        return K_s * self.state[2] + B_s * dw
