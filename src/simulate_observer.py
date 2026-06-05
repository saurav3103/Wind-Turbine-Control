# ══════════════════════════════════════════════════════
#  simulate_observer.py
#  Runs the Luenberger observer on the two-mass model
#  at three noise levels and plots results.
# ══════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from parameters import (R, rho, w_rated, v_rated, T_rated, P_rated,
                         K_opt, beta_max, beta_rate, J_r, J_g_lss,
                         K_s, B_s)
from aerodynamics import compute_Cp
from wind import generate_wind
from controllers import BaselineController
from observer import LuenbergerObserver, build_observer_gains


NOISE_CASES = [
    (0.000, 'Clean (0.000 rad/s)',      'seagreen'),
    (0.010, 'Realistic (0.010 rad/s)',  'darkorange'),
    (0.050, 'Aggressive (0.050 rad/s)', 'crimson'),
]


def run_with_observer(noise_std, L_below, L_above,
                      t_end=250.0, dt=0.001, seed=42):
    rng       = np.random.default_rng(seed)
    t_eval    = np.arange(0.0, t_end, dt)
    v_profile = generate_wind(t_eval, 'smooth_step')

    omega_eq = brentq(
        lambda w: (0.5*rho*np.pi*R**2*5.0**3*compute_Cp(w*R/5.0, 0))/w
                  - K_opt*w**2, 0.1, 2.0)

    ctrl = BaselineController(dt=dt)
    obs  = LuenbergerObserver(omega_eq, L_below, L_above)
    plant = [omega_eq, omega_eq, 0.0, 0.0]  # [or, og, theta, beta]

    plot_every = max(1, int(0.1 / dt))
    n_store    = len(t_eval) // plot_every

    t_s         = np.zeros(n_store)
    theta_true  = np.zeros(n_store)
    theta_hat   = np.zeros(n_store)
    theta_err   = np.zeros(n_store)
    T_shaft_s   = np.zeros(n_store)
    T_shaft_hat = np.zeros(n_store)
    v_s         = np.zeros(n_store)
    si = 0

    for i in range(len(t_eval)):
        plant[0] = max(plant[0], 0.01)
        plant[1] = max(plant[1], 0.01)

        or_  = plant[0]; og = plant[1]
        th   = plant[2]; beta = np.clip(plant[3], 0.0, beta_max)
        v    = v_profile[i]

        lam   = max(or_ * R / v, 0.01)
        Cp    = compute_Cp(lam, beta)
        T_a   = (0.5 * rho * np.pi * R**2 * v**3 * Cp) / or_
        T_gen, beta_ref = ctrl.step(or_, beta, v)

        dw      = or_ - og
        T_shaft = K_s * th + B_s * dw

        # Noisy measurements
        or_meas = or_ + rng.normal(0, noise_std)
        og_meas = og  + rng.normal(0, noise_std)

        obs.step(or_meas, og_meas, T_a, T_gen, dt)

        if i % plot_every == 0 and si < n_store:
            t_s[si]         = i * dt
            theta_true[si]  = np.degrees(th)
            theta_hat[si]   = np.degrees(obs.theta_hat)
            theta_err[si]   = np.degrees(obs.theta_hat - th)
            T_shaft_s[si]   = T_shaft / 1e6
            T_shaft_hat[si] = obs.T_shaft_hat / 1e6
            v_s[si]         = v
            si += 1

        # Plant ODEs
        d_beta = np.clip((beta_ref - beta) / 0.1, -beta_rate, beta_rate)
        plant[0] += ((T_a - T_shaft) / J_r)    * dt
        plant[1] += ((T_shaft - T_gen) / J_g_lss) * dt
        plant[2] += dw                          * dt
        plant[3] += d_beta                      * dt

    return t_s, theta_true, theta_hat, theta_err, T_shaft_s, T_shaft_hat, v_s


def plot_noise_comparison(results, t_s):
    steady = t_s > 50

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for std, label, color in NOISE_CASES:
        _, _, _, err, _, _, _ = results[std]
        axes[0].plot(t_s, err, color=color, lw=1.0, label=label, alpha=0.85)

    axes[0].axhline(0,     color='gray', ls='--', lw=0.8)
    axes[0].axhline( 0.5,  color='gray', ls=':',  lw=0.8, label='±0.5° band')
    axes[0].axhline(-0.5,  color='gray', ls=':',  lw=0.8)
    axes[0].set_ylabel('Estimation error\nθ̂ − θ  (degrees)')
    axes[0].set_ylim(-12, 12)
    axes[0].legend(fontsize=8); axes[0].grid(True)
    axes[0].set_title('Observer noise robustness — estimation error (full run)')

    for std, label, color in NOISE_CASES:
        _, _, _, err, _, _, _ = results[std]
        axes[1].plot(t_s[steady], err[steady], color=color, lw=1.0,
                     label=label, alpha=0.85)

    axes[1].axhline(0,     color='gray', ls='--', lw=0.8)
    axes[1].axhline( 0.5,  color='gray', ls=':',  lw=0.8, label='±0.5° band')
    axes[1].axhline(-0.5,  color='gray', ls=':',  lw=0.8)
    axes[1].set_ylabel('Estimation error\n(t > 50s)  (degrees)')
    axes[1].legend(fontsize=8); axes[1].grid(True)
    axes[1].set_title('Steady-state error (t > 50 s)')

    _, _, _, _, T_true, T_hat, _ = results[0.010]
    axes[2].plot(t_s, T_true, color='darkorchid', lw=1.5, label='True T_shaft')
    axes[2].plot(t_s, T_hat,  color='darkorange',  lw=1.0, ls='--',
                 label='Estimated T̂_shaft (noise=0.01)')
    axes[2].axhline(T_rated / 1e6, color='r', ls='--', lw=1, label='T_rated')
    axes[2].set_ylabel('Shaft torque (MN·m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(fontsize=8); axes[2].grid(True)
    axes[2].set_title('Shaft torque estimation under realistic noise')

    plt.tight_layout()
    plt.savefig('results/observer_noise_robustness.png', dpi=150,
                bbox_inches='tight')
    plt.show()


def print_summary(results, t_s):
    steady = t_s > 50
    print(f"\n{'─'*65}")
    print(f"  {'Noise (rad/s)':<22} {'Max err (°)':<14} "
          f"{'RMS t>50s (°)':<18} {'Max t>50s (°)'}")
    print(f"{'─'*65}")
    for std, label, _ in NOISE_CASES:
        _, _, _, err, _, _, _ = results[std]
        print(f"  {label:<22} "
              f"{np.abs(err).max():<14.3f} "
              f"{np.sqrt(np.mean(err[steady]**2)):<18.4f} "
              f"{np.abs(err[steady]).max():.3f}")
    print(f"{'─'*65}")


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)

    print("Designing observer gains...")
    L_below, L_above = build_observer_gains()

    results = {}
    for std, label, _ in NOISE_CASES:
        print(f"Running: {label} ...", end=' ', flush=True)
        results[std] = run_with_observer(std, L_below, L_above)
        print("done")

    t_s = results[0.0][0]
    plot_noise_comparison(results, t_s)
    print_summary(results, t_s)
