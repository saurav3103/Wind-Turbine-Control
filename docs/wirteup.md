# Wind Turbine Dynamic Modeling, Control, and State Estimation

## Project Overview

This project presents a comprehensive dynamic simulation and control framework for a utility-scale wind turbine based on the NREL 5 MW reference turbine. The objective is to model the aerodynamic, mechanical, and control dynamics of the turbine while investigating drivetrain behavior, controller performance, parameter sensitivity, system linearization, and observer design.

The project progressively develops from basic aerodynamic modeling to advanced control system concepts including state-space linearization and Luenberger observer implementation.

---

# Objectives

- Model the aerodynamic power extraction of a wind turbine.
- Simulate realistic wind speed profiles.
- Implement pitch control using a PID controller.
- Develop a two-mass drivetrain model.
- Analyze drivetrain natural frequency sensitivity.
- Linearize the nonlinear drivetrain dynamics.
- Design a Luenberger state observer.
- Evaluate observer robustness under measurement noise.

---

# Turbine Parameters

The simulation is based on the **NREL 5 MW Reference Wind Turbine**.

Key parameters include:

| Parameter | Value |
|-----------|--------|
| Rotor Radius | 63 m |
| Air Density | 1.225 kg/m³ |
| Rated Power | 5 MW |
| Rated Rotor Speed | 1.267 rad/s |
| Gearbox Ratio | 97 |
| Rotor Inertia | 38,759,228 kg·m² |
| Cut-in Wind Speed | 3 m/s |
| Rated Wind Speed | 11.4 m/s |
| Cut-out Wind Speed | 25 m/s |

---

# 1. Aerodynamic Power Coefficient Model

The aerodynamic efficiency is modeled using the standard analytical approximation of the power coefficient:

\[
C_p = f(\lambda,\beta)
\]

where

- λ = Tip Speed Ratio
- β = Blade Pitch Angle

The implemented analytical model computes the aerodynamic efficiency for different operating conditions while ensuring physically realistic limits.

### Visualization

The notebook generates:

- Cp vs Tip Speed Ratio
- Multiple pitch angles
- Comparison of aerodynamic efficiency

This provides insight into how blade pitch influences power extraction.

---

# 2. Wind Speed Generation

To evaluate controller performance, multiple wind profiles are generated.

Supported profiles include:

- Step Wind
- Ramp Wind
- Turbulent Wind

These allow testing under:

- Region II operation
- Rated operation
- Region III operation

---

# 3. Pitch Controller

A PID controller regulates rotor speed above rated wind speed.

Controller components include:

- Proportional action
- Integral action
- Derivative action
- Integral anti-windup
- Saturation limits

Control objective:

Maintain

\[
\omega_r = \omega_{rated}
\]

by adjusting blade pitch angle.

---

# 4. Two-Mass Drivetrain Model

The drivetrain is modeled as a two-mass torsional system consisting of

- Rotor inertia
- Generator inertia
- Flexible shaft
- Gearbox

The model captures

- Shaft torsion
- Oscillations
- Rotor dynamics
- Generator dynamics

Unlike a rigid drivetrain assumption, this model accurately represents torsional vibrations.

A small integration timestep is used to ensure numerical stability of the stiff drivetrain dynamics.

---

# 5. Sensitivity Analysis

A sensitivity study is performed on the drivetrain natural frequency.

The simulation evaluates the influence of natural frequency on:

- Maximum rotor speed
- Maximum shaft torque
- Maximum shaft twist
- Electrical power output

The notebook automatically determines acceptable natural frequency values that satisfy shaft twist constraints.

This provides useful design insight into drivetrain stiffness selection.

---

# 6. Linearization of the Nonlinear Model

The nonlinear drivetrain model is linearized around an operating point.

The procedure involves:

- Numerical Jacobian computation
- State-space model generation
- Eigenvalue analysis

The resulting system matrix

\[
\dot{x}=Ax+Bu
\]

is used to investigate local stability characteristics.

Eigenvalues provide information regarding:

- Stability
- Oscillation modes
- Damping characteristics

---

# 7. Luenberger State Observer

A full-order Luenberger observer is designed to estimate internal drivetrain states.

The observer reconstructs quantities that cannot be measured directly while using available measurements.

Features include:

- Feedforward reconstruction of aerodynamic torque
- Generator torque estimation
- Gain scheduling
- Pole placement
- Stable observer dynamics

Observer equation:

\[
\dot{\hat{x}}
=
A\hat{x}
+
Bu
+
L(y-C\hat{x})
\]

where

- x̂ = estimated state
- L = observer gain
- y = measured output

---

# 8. Noise Robustness Study

The observer is tested under multiple measurement noise conditions.

Noise levels include:

- Ideal measurements
- Realistic encoder noise
- High industrial noise

Performance metrics include:

- Estimation accuracy
- Tracking capability
- Convergence speed
- Robustness

This demonstrates the observer's practical applicability under noisy measurements.

---

# Results

The notebook successfully demonstrates:

- Accurate aerodynamic power modeling.
- Dynamic wind profile simulation.
- Stable PID pitch control.
- Realistic two-mass drivetrain dynamics.
- Drivetrain parameter sensitivity.
- Linearized state-space representation.
- Successful state estimation using a Luenberger observer.
- Robust observer performance under measurement noise.

---

# Technologies Used

- Python
- NumPy
- Matplotlib
- SciPy
- Pandas

---

# Future Improvements

Possible extensions include:

- Kalman Filter implementation
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Model Predictive Control (MPC)
- Individual blade pitch control
- Generator torque optimization
- Fatigue load estimation
- Wind turbulence using IEC standards
- Integration with OpenFAST
- Digital twin implementation

---

# Conclusion

This project provides a complete workflow for wind turbine dynamic analysis, combining aerodynamic modeling, drivetrain simulation, control system design, linear systems theory, and state estimation. It demonstrates concepts commonly encountered in modern wind energy research and industrial control systems, making it an excellent foundation for advanced studies in renewable energy, systems and control, and observer-based control design.
