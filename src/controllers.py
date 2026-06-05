# ══════════════════════════════════════════════════════
#  controllers.py
#  MPPT torque controller + PID pitch controller
# ══════════════════════════════════════════════════════

import numpy as np
from parameters import (K_opt, T_rated, w_rated, beta_max, Kp, Ki, Kd)


class PitchPID:
    """
    PID controller for rotor speed regulation (Region 3).
    Includes anti-windup integral clamping.
    """

    def __init__(self, kp=Kp, ki=Ki, kd=Kd, dt=0.001,
                 integral_limit=10.0):
        self.kp  = kp
        self.ki  = ki
        self.kd  = kd
        self.dt  = dt
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def step(self, omega_r):
        error             = omega_r - w_rated
        self._integral   += error * self.dt
        self._integral    = np.clip(self._integral,
                                    -self.integral_limit,
                                     self.integral_limit)
        derivative        = (error - self._prev_error) / self.dt
        self._prev_error  = error
        beta_ref = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(beta_ref, 0.0, beta_max))


def smooth_T_gen(omega_r):
    """
    Torque reference that blends smoothly between MPPT and rated torque
    around the rated speed transition, avoiding torque jumps.
    """
    blend  = np.clip((omega_r - w_rated * 0.85) / (w_rated * 0.15),
                     0.0, 1.0)
    T_mppt = min(K_opt * omega_r**2, T_rated)
    return (1.0 - blend) * T_mppt + blend * T_rated


class BaselineController:
    """
    Region-switching controller:
      Region 2 (ω < 0.90·ω_rated) : MPPT via optimal torque law
      Region 3 (ω ≥ 0.90·ω_rated) : Constant torque + PID pitch
    """

    def __init__(self, dt=0.001):
        self.pid      = PitchPID(dt=dt)
        self.w_switch = w_rated * 0.90

    def reset(self):
        self.pid.reset()

    def step(self, omega_r, beta, v_wind=None):
        """
        Parameters
        ----------
        omega_r : float — rotor speed (rad/s)
        beta    : float — current pitch angle (deg), unused here
        v_wind  : float — wind speed (m/s), unused (sensorless MPPT)

        Returns
        -------
        T_gen    : float — generator torque reference (N·m)
        beta_ref : float — pitch angle reference (deg)
        """
        if omega_r < self.w_switch:
            T_gen    = smooth_T_gen(omega_r)
            beta_ref = 0.0
        else:
            T_gen    = T_rated
            beta_ref = self.pid.step(omega_r)
        return T_gen, beta_ref
