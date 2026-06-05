# ══════════════════════════════════════════════════════
#  simulate_single_mass.py
#  Single-mass rotor model with baseline PID controller
#  State: [omega_r, beta]
# ══════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from parameters import (R, rho, J, w_rated, v_rated, P_rated,
                         T_rated, K_opt, beta_max, beta_rate, Cp_max)
from aerodynamics import compute_Cp
from wind import generate_wind
from controllers import BaselineController


def run_single_mass(t_end=250.0, dt=0.1, wind_profile='smooth_step'):
    """
    Simulate single-mass wind turbine model.

    Returns dict of time-series arrays.
    """
    t_eval   = np.arange(0.0, t_end, dt)
    v_profile = generate_wind(t_eval, wind_profile)

    # Initial equilibrium at v=5 m/s
    omega_eq = brentq(
        lambda w: (0.5*rho*np.pi*R**2*5.0**3*compute_Cp(w*R/5.0, 0))/w
                  - K_opt*w**2,
        0.1, 2.0
    )

    ctrl  = BaselineController(dt=dt)
    state = [omega_eq, 0.0]   # [omega_r, beta]

    omega_arr = np.zeros(len(t_eval))
    beta_arr  = np.zeros(len(t_eval))
    T_gen_arr = np.zeros(len(t_eval))
    P_arr     = np.zeros(len(t_eval))
    Cp_arr    = np.zeros(len(t_eval))
    lam_arr   = np.zeros(len(t_eval))

    for i in range(len(t_eval)):
        omega = max(state[0], 0.01)
        beta  = np.clip(state[1], 0.0, beta_max)
        v     = v_profile[i]

        lam    = max(omega * R / v, 0.01)
        Cp     = compute_Cp(lam, beta)
        P_aero = 0.5 * rho * np.pi * R**2 * v**3 * Cp
        T_aero = P_aero / omega

        T_gen, beta_ref = ctrl.step(omega, beta, v)

        beta_rate_cmd = np.clip((beta_ref - beta) / 0.1,
                                -beta_rate, beta_rate)

        d_omega = (T_aero - T_gen) / J
        d_beta  = beta_rate_cmd

        omega_arr[i] = omega
        beta_arr[i]  = beta
        T_gen_arr[i] = T_gen
        P_arr[i]     = T_gen * omega
        Cp_arr[i]    = Cp
        lam_arr[i]   = lam

        state[0] = omega + d_omega * dt
        state[1] = beta  + d_beta  * dt

    return {
        't': t_eval, 'v': v_profile,
        'omega': omega_arr, 'beta': beta_arr,
        'T_gen': T_gen_arr, 'P': P_arr,
        'Cp': Cp_arr, 'lambda': lam_arr,
    }


def plot_results(res):
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    t = res['t']

    axes[0].plot(t, res['v'],     color='steelblue',  lw=1.5)
    axes[0].axhline(v_rated, color='r', ls='--', lw=1, label='v_rated')
    axes[0].set_ylabel('Wind Speed (m/s)');  axes[0].legend(fontsize=8)

    axes[1].plot(t, res['omega'], color='darkorange', lw=1.5)
    axes[1].axhline(w_rated, color='r', ls='--', lw=1, label='ω_rated')
    axes[1].set_ylabel('Rotor Speed (rad/s)'); axes[1].legend(fontsize=8)

    axes[2].plot(t, res['beta'],  color='seagreen',   lw=1.5)
    axes[2].set_ylabel('Pitch Angle (deg)')

    axes[3].plot(t, res['P']/1e6, color='purple',     lw=1.5)
    axes[3].axhline(P_rated/1e6, color='r', ls='--', lw=1, label='P_rated')
    axes[3].set_ylabel('Power (MW)'); axes[3].legend(fontsize=8)

    axes[4].plot(t, res['Cp'],    color='brown',      lw=1.5)
    axes[4].axhline(Cp_max, color='r', ls='--', lw=1, label='Cp_max')
    axes[4].set_ylabel('Cp'); axes[4].set_xlabel('Time (s)')
    axes[4].legend(fontsize=8)

    for ax in axes:
        ax.grid(True)

    plt.suptitle('Single-Mass Wind Turbine — PID Controller',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"Max rotor speed  : {res['omega'].max():.3f} rad/s  (rated: {w_rated})")
    print(f"Max power output : {res['P'].max()/1e6:.3f} MW     (rated: {P_rated/1e6})")
    print(f"Max pitch angle  : {res['beta'].max():.2f} deg")
    region2 = res['v'] < v_rated
    print(f"Mean Cp Region 2 : {res['Cp'][region2].mean():.4f}  (target: {Cp_max:.4f})")


if __name__ == '__main__':
    res = run_single_mass()
    plot_results(res)
