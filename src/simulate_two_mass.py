# ══════════════════════════════════════════════════════
#  simulate_two_mass.py
#  Two-mass drivetrain model with baseline PID controller
#
#  State: [omega_r, omega_g_lss, theta_twist, beta]
#
#  NOTE: dt=0.001s required for numerical stability.
#  The drivetrain oscillator at 0.62 Hz needs ~20 steps
#  per period → dt ≤ 0.001s for forward Euler.
# ══════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from parameters import (R, rho, w_rated, v_rated, P_rated, T_rated,
                         K_opt, beta_max, beta_rate, Cp_max, f_n,
                         J_r, J_g_lss, K_s, B_s, Kp, Ki, Kd)
from aerodynamics import compute_Cp
from wind import generate_wind
from controllers import BaselineController


def run_two_mass(t_end=250.0, dt=0.001, wind_profile='smooth_step'):
    """
    Simulate two-mass drivetrain wind turbine model.

    Returns dict of downsampled time-series arrays (every 0.1s).
    """
    t_eval    = np.arange(0.0, t_end, dt)
    v_profile = generate_wind(t_eval, wind_profile)

    # Initial equilibrium at v=5 m/s
    omega_eq = brentq(
        lambda w: (0.5*rho*np.pi*R**2*5.0**3*compute_Cp(w*R/5.0, 0))/w
                  - K_opt*w**2,
        0.1, 2.0
    )

    ctrl  = BaselineController(dt=dt)
    state = [omega_eq, omega_eq, 0.0, 0.0]  # [omega_r, omega_g, theta, beta]

    # Downsample storage
    plot_every = max(1, int(0.1 / dt))
    n_store    = len(t_eval) // plot_every

    t_s       = np.zeros(n_store)
    omega_r_s = np.zeros(n_store)
    omega_g_s = np.zeros(n_store)
    theta_s   = np.zeros(n_store)
    T_shaft_s = np.zeros(n_store)
    beta_s    = np.zeros(n_store)
    P_s       = np.zeros(n_store)
    v_s       = np.zeros(n_store)
    si = 0

    for i in range(len(t_eval)):
        state[0] = max(state[0], 0.01)
        state[1] = max(state[1], 0.01)

        or_  = state[0]
        og   = state[1]
        th   = state[2]
        beta = np.clip(state[3], 0.0, beta_max)
        v    = v_profile[i]

        lam    = max(or_ * R / v, 0.01)
        Cp     = compute_Cp(lam, beta)
        T_aero = (0.5 * rho * np.pi * R**2 * v**3 * Cp) / or_

        T_gen, beta_ref = ctrl.step(or_, beta, v)

        dw      = or_ - og
        T_shaft = K_s * th + B_s * dw

        d_or   = (T_aero  - T_shaft) / J_r
        d_og   = (T_shaft - T_gen)   / J_g_lss
        d_th   = dw
        d_beta = np.clip((beta_ref - beta) / 0.1, -beta_rate, beta_rate)

        if i % plot_every == 0 and si < n_store:
            t_s[si]       = i * dt
            omega_r_s[si] = or_
            omega_g_s[si] = og
            theta_s[si]   = np.degrees(th)
            T_shaft_s[si] = T_shaft / 1e6
            beta_s[si]    = beta
            P_s[si]       = T_gen * og / 1e6
            v_s[si]       = v
            si += 1

        state[0] += d_or   * dt
        state[1] += d_og   * dt
        state[2] += d_th   * dt
        state[3] += d_beta * dt

    return {
        't': t_s, 'v': v_s,
        'omega_r': omega_r_s, 'omega_g': omega_g_s,
        'theta': theta_s, 'T_shaft': T_shaft_s,
        'beta': beta_s, 'P': P_s,
    }


def plot_results(res):
    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    t = res['t']

    axes[0].plot(t, res['v'],       color='steelblue',  lw=1.5)
    axes[0].axhline(v_rated, color='r', ls='--', lw=1, label='v_rated')
    axes[0].set_ylabel('Wind Speed (m/s)'); axes[0].legend(fontsize=8)

    axes[1].plot(t, res['omega_r'], color='darkorange', lw=1.5, label='ω_rotor')
    axes[1].plot(t, res['omega_g'], color='steelblue',  lw=1.0, ls='--',
                 alpha=0.8, label='ω_gen (LSS)')
    axes[1].axhline(w_rated, color='r', ls='--', lw=1, label='ω_rated')
    axes[1].set_ylabel('Speed (rad/s)'); axes[1].legend(fontsize=8)

    axes[2].plot(t, res['theta'],   color='crimson',    lw=1.5)
    axes[2].set_ylabel('Shaft Twist (deg)')

    axes[3].plot(t, res['T_shaft'], color='darkorchid', lw=1.5)
    axes[3].axhline(T_rated/1e6, color='r', ls='--', lw=1, label='T_rated')
    axes[3].set_ylabel('Shaft Torque (MN·m)'); axes[3].legend(fontsize=8)

    axes[4].plot(t, res['beta'],    color='seagreen',   lw=1.5)
    axes[4].set_ylabel('Pitch Angle (deg)')

    axes[5].plot(t, res['P'],       color='purple',     lw=1.5)
    axes[5].axhline(P_rated/1e6, color='r', ls='--', lw=1, label='P_rated')
    axes[5].set_ylabel('Power (MW)'); axes[5].set_xlabel('Time (s)')
    axes[5].legend(fontsize=8)

    for ax in axes:
        ax.grid(True)

    plt.suptitle(f'Two-Mass Drivetrain Wind Turbine (f_n = {f_n} Hz)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"Drivetrain natural freq : {f_n:.3f} Hz")
    print(f"Max rotor speed         : {res['omega_r'].max():.3f} rad/s  (rated: {w_rated})")
    print(f"Max shaft torque        : {res['T_shaft'].max():.3f} MN·m")
    print(f"Max shaft twist         : {res['theta'].max():.3f} deg  (expect <5°)")
    print(f"Max power output        : {res['P'].max():.3f} MW")


if __name__ == '__main__':
    res = run_two_mass()
    plot_results(res)
