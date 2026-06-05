# ══════════════════════════════════════════════════════
#  parameters.py
#  NREL 5MW Reference Turbine — all shared constants
# ══════════════════════════════════════════════════════

import numpy as np

# ── Turbine ──
R          = 63.0           # Rotor radius (m)
rho        = 1.225          # Air density (kg/m³)
N          = 97.0           # Gearbox ratio
P_rated    = 5e6            # Rated power (W)
w_rated    = 1.267          # Rated rotor speed (rad/s)
v_rated    = 11.4           # Rated wind speed (m/s)
v_cutin    = 3.0            # Cut-in wind speed (m/s)
v_cutout   = 25.0           # Cut-out wind speed (m/s)
lambda_opt = 7.55           # Optimal tip speed ratio
Cp_max     = 0.482          # Maximum power coefficient
beta_max   = 90.0           # Max pitch angle (deg)
beta_rate  = 8.0            # Max pitch rate (deg/s)

# ── Derived ──
T_rated = P_rated / w_rated
K_opt   = (0.5 * rho * np.pi * R**5 * Cp_max) / lambda_opt**3

# ── Single-mass rotor inertia ──
J = 38_759_228.0            # kg·m²

# ── Two-mass drivetrain ──
J_r        = 35_444_067.0   # Rotor inertia (kg·m²)
J_g_hss    = 534.116        # Generator inertia on HSS (kg·m²)
J_g_lss    = J_g_hss * N**2 # Generator inertia referred to LSS
mu         = (J_r * J_g_lss) / (J_r + J_g_lss)
f_n        = 0.62           # Drivetrain natural frequency (Hz)
K_s        = (2 * np.pi * f_n)**2 * mu   # Shaft stiffness (N·m/rad)
B_s        = 2 * 0.05 * np.sqrt(K_s * mu)  # Shaft damping (N·m·s/rad)

# ── PID gains (pitch controller) ──
Kp     = 40.0
Ki     =  3.0
Kd     =  2.0
dt_pid =  0.1
