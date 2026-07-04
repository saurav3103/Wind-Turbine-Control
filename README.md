# Wind Turbine Control Simulation

A Python simulation of wind turbine control systems built progressively from a single-mass rotor model to a two-mass drivetrain with a Luenberger state observer.

All simulations are based on the **NREL 5MW Reference Turbine** — the standard benchmark in wind energy control research.

---

## What This Covers

### 1. Single-Mass Rotor Model
The simplest physically meaningful model. State: rotor speed `ω` and pitch angle `β`.

- **MPPT controller** (Region 2): Sensorless optimal torque law `T_gen = K_opt · ω²` tracks maximum power coefficient `Cp_max` without wind speed measurement
- **PID pitch controller** (Region 3): Regulates rotor speed to rated when wind exceeds rated speed
- Smooth region switching with torque blending to avoid torque jumps at transition

### 2. Two-Mass Drivetrain Model
Adds drivetrain flexibility between rotor and generator. State: `[ω_rotor, ω_gen, θ_twist]`.

- Drivetrain modelled as spring-damper connecting rotor and generator inertias
- Natural frequency: **0.62 Hz** — requires `dt = 0.001s` for numerical stability
- Shaft torque and twist angle tracked alongside rotor dynamics

### 3. Linearisation
Numerical Jacobian computation at two operating points:
- **Below rated** (v = 8 m/s): MPPT region equilibrium
- **Above rated** (v = 14 m/s): Pitch control region equilibrium

Eigenvalue analysis confirms stability and identifies modal frequencies.

### 4. Luenberger State Observer
Estimates the unobservable shaft twist angle `θ` from measurements of `ω_rotor` and `ω_gen` only.

- Gain-scheduled between below/above rated designs using a sigmoid blend
- Feedforward reconstruction of aerodynamic and generator torques
- Tested at three noise levels: clean (0.000), realistic (0.010), aggressive (0.050) rad/s

---

## Repository Structure

```
wind_turbine_control/
│
├── src/
│   ├── parameters.py           # NREL 5MW constants, derived quantities
│   ├── aerodynamics.py         # Cp(λ,β) surface, T_aero, thrust
│   ├── wind.py                 # Wind profile generators
│   ├── controllers.py          # MPPT torque law + PID pitch controller
│   ├── linearization.py        # Numerical Jacobian, A/B matrices, eigenvalues
│   ├── observer.py             # Luenberger observer, gain scheduling
│   ├── simulate_single_mass.py # Single-mass simulation + plots
│   ├── simulate_two_mass.py    # Two-mass simulation + plots
│   └── simulate_observer.py    # Observer noise robustness study
│
├── results/                    # Output figures saved here
└── requirements.txt
└── notebooks/
    └── Wind_Turbines.ipynb     # Complete .ipynb notebook
└── docs/
    └── writeup.md              # Write up for detailed description
```

---

## Quickstart

```bash
pip install -r requirements.txt
cd src

# Single-mass simulation
python simulate_single_mass.py

# Two-mass drivetrain (takes ~10s, 250k steps at dt=0.001s)
python simulate_two_mass.py

# Linearisation at two operating points
python linearization.py

# Observer noise robustness study
python simulate_observer.py
```

---

## Key Results

| Model | Max ω (rad/s) | Max Power (MW) | Notes |
|---|---|---|---|
| Single-mass, PID | ~1.50 | ~5.94 | Transient overshoot on wind ramps |
| Two-mass, PID | ~1.50 | ~5.90 | Shaft twist < 5°, torsional mode visible |

| Observer noise | RMS error (t > 50s) | Max error (t > 50s) |
|---|---|---|
| Clean (0.000 rad/s) | < 0.01° | < 0.05° |
| Realistic (0.010 rad/s) | < 0.05° | < 0.20° |
| Aggressive (0.050 rad/s) | < 0.15° | < 0.60° |

---

## Control Architecture

```
Wind speed v(t)
      │
      ▼
[Cp(λ, β) surface] ──► T_aero
                              │
                    [Rotor/Drivetrain dynamics]
                              │
                           ω_r(t)
                          ╱       ╲
               ω < 0.9·ω_rated   ω ≥ 0.9·ω_rated
                    │                    │
           [MPPT Controller]    [PID Pitch Controller]
           T_gen = K_opt·ω²      β_ref → actuator
                                         │
                                [Pitch Actuator]
                                β with rate limit ±8°/s
```

---

## Physical Basis

All parameters from the NREL 5MW Reference Turbine technical report (Jonkman et al. 2009).

The `Cp(λ, β)` analytical approximation follows Heier (2006), widely used in controls research. The two-mass drivetrain parameters match those used in the OpenFAST reference simulation.

---

## Next Steps

- [ ] LQR controller on linearised two-mass model — compare with PID
- [ ] Tower fore-aft dynamics (4-state model)
- [ ] Active tower damping via pitch
- [ ] Kalman filter replacing Luenberger observer
