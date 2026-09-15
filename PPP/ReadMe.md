# Phase-Field Modeling of Dendritic Solidification in FEniCSx

A finite-element implementation of phase-field models for dendritic
solidification using **FEniCSx / DOLFINx**, **PETSc**, and **MPI**.

The project develops the numerical model progressively from a local
phase-field evolution equation to a coupled temperature–phase-field
formulation with anisotropic interfacial energy capable of reproducing
dendritic growth.

**Keywords:** Phase Field · Dendritic Solidification · Finite Element Method ·
FEniCSx · Nonlinear FEM · PETSc · MPI · Scientific Computing · Materials Science

---

## Overview

Dendritic structures are commonly formed during the solidification of
metals and alloys. Their evolution is governed by the interaction between
phase transformation, heat or solute diffusion, interface energy, and
anisotropic interface kinetics.

Explicitly tracking the moving solid–liquid interface becomes difficult
once branching and complex dendritic morphologies develop. The phase-field
method avoids explicit interface tracking by introducing a continuous
order parameter

\[
\phi =
\begin{cases}
+1, & \text{solid},\\
-1, & \text{liquid},
\end{cases}
\]

with intermediate values describing the diffuse solid–liquid interface.

The objective of this project was to implement and numerically verify a
phase-field formulation for dendritic solidification in **FEniCSx** using
the finite-element method.

The implementation is based primarily on the phase-field formulation
presented in:

> K. Bhagat and S. Rudraraju,  
> *Modeling of dendritic solidification and numerical analysis of the
> phase-field approach to model complex morphologies in alloys*,  
> Engineering with Computers, 39, 2345–2363 (2023).  
> DOI: 10.1007/s00366-022-01767-7

The original work uses finite-element / isogeometric implementations
built on deal.II and PetIGA. This project develops the corresponding
numerical formulation in the **Python-based FEniCSx ecosystem**.

---

## Project Highlights

- Coupled **phase-field and diffusion/temperature model**
- Finite-element discretization with **P1 Lagrange elements**
- Mixed finite-element space for the coupled \(\phi-u\) problem
- Fully implicit **Backward-Euler time integration**
- Monolithic nonlinear solution using **Newton's method**
- PETSc-based nonlinear and linear algebra infrastructure
- Natural homogeneous Neumann / zero-flux boundary conditions
- Diffuse-interface gradient-energy formulation
- Four-fold anisotropic interface energy
- Circular solid-seed initialization
- Dendritic morphology evolution
- Dendrite-tip tracking and velocity calculation
- Energy and thermodynamic consistency checks
- Enthalpy-balance verification
- Mesh/time-step and interface-kinetics verification cases
- MPI-compatible FEniCSx implementation

---

# Final Dendritic Growth Simulation

The final model introduces four-fold interfacial anisotropy and evolves
an initially circular solid seed inside an undercooled liquid domain.

<p align="center">
  <img src="Task_3/outputs/2/t0.png" width="31%">
  <img src="Task_3/outputs/2/t300.png" width="31%">
  <img src="Task_3/outputs/2/t600.png" width="31%">
</p>

<p align="center">
Initial seed → anisotropic interface growth → developed dendritic morphology
</p>

The phase-field interface is represented by the zero level set

\[
\phi = 0,
\]

which separates the bulk solid and liquid regions.

---

## Physical Model

For a pure undercooled melt, the formulation uses the phase-field order
parameter \(\phi\) together with the non-dimensional temperature /
undercooling field \(u\).

The local bulk free-energy density is

\[
f(\phi,u)
=
-\frac{1}{2}\phi^2
+\frac{1}{4}\phi^4
+\zeta u\phi
\left(
1-\frac{2}{3}\phi^2+\frac{1}{5}\phi^4
\right).
\]

The first two terms form the double-well potential associated with the
solid and liquid equilibrium states.

The gradient contribution introduces an energetic penalty for spatial
variation of the phase field and therefore creates a diffuse interface.

A generic free-energy functional can be written as

\[
\Pi[\phi,u]
=
\int_{\Omega}
\left[
f(\phi,u)
+
\frac{1}{2}\lambda^2(\mathbf n)
|\nabla\phi|^2
\right]\,d\Omega.
\]

---

## Coupled Governing Equations

The implemented pure-metal-style coupled system has the form

\[
\tau(\mathbf n)\dot{\phi}
=
-\mu,
\]

with

\[
\mu
=
\frac{\partial f}{\partial\phi}
-
\nabla\cdot\mathbf Q,
\]

and

\[
\dot{u}
=
D\nabla^2u
+
K\dot{\phi}.
\]

Here:

- \(\phi\) — phase-field order parameter
- \(u\) — non-dimensional temperature / undercooling
- \(D\) — diffusion coefficient
- \(\tau\) — phase-field kinetic time scale
- \(K\) — latent-heat / enthalpy coupling coefficient
- \(\mathbf Q\) — anisotropic interfacial flux
- \(\lambda_0\) — characteristic interface thickness

---

## Four-Fold Anisotropy

Dendritic growth requires anisotropic interface properties.

The final implementation uses a four-fold anisotropy function

\[
a(\mathbf n)
=
(1-3\epsilon)
+
4\epsilon
\left(
n_x^4+n_y^4
\right),
\]

where

\[
\mathbf n
=
\frac{\nabla\phi}{|\nabla\phi|}
\]

is the local interface normal.

The kinetic coefficient is scaled as

\[
\tau(\mathbf n)=\tau_0 a^2(\mathbf n).
\]

For

\[
\epsilon = 0,
\]

the model reduces to the isotropic interface formulation.

For non-zero anisotropy, selected crystallographic growth directions
become energetically favorable and the initially circular seed develops
the characteristic dendritic arms.

---

# Numerical Formulation

## Spatial Discretization

The computational domain is discretized using triangular finite elements
and first-order Lagrange interpolation.

For the coupled problem, the finite-element space is

\[
V_h = P_1 \times P_1,
\]

containing both

\[
(\phi,u).
\]

In FEniCSx this is implemented using a mixed finite-element space.

---

## Time Integration

The transient governing equations are discretized using the fully
implicit **Backward-Euler method**.

For example,

\[
\dot{\phi}
\approx
\frac{\phi^{n+1}-\phi^n}{\Delta t}.
\]

Backward Euler was selected for its robustness when integrating the
nonlinear phase-field evolution equations.

---

## Nonlinear Solution

At every time increment, the coupled finite-element residual is solved
using Newton's method.

The residual and consistent Jacobian are assembled through UFL and
solved using the nonlinear solver infrastructure of DOLFINx/PETSc.

The coupled unknowns are solved **monolithically**, rather than solving
the phase and diffusion equations independently.

---

## Boundary Conditions

No essential Dirichlet boundary conditions are imposed in the principal
solidification simulations.

The weak formulation naturally introduces homogeneous Neumann conditions,

\[
\nabla\phi\cdot\mathbf n = 0,
\]

and

\[
\nabla u\cdot\mathbf n = 0,
\]

corresponding to zero phase-field and thermal flux through the external
boundaries.

---

# Progressive Development and Verification

The implementation was developed in a sequence of increasingly complex
test problems. Each stage isolates a specific part of the formulation
before it is introduced into the final dendritic-solidification model.

---

## Task 0 — Local Phase-Field Evolution

**Purpose:** Verify the bulk free-energy implementation independently of
the gradient/interface term.

The spatial gradient contribution is removed, reducing the phase-field
equation to a node-wise nonlinear ODE:

\[
\tau_0\dot{\phi}
=
-
\frac{\partial f}{\partial\phi}.
\]

The test checks that the computed trajectory follows the analytical
bulk free-energy curve and evolves toward the energetically favorable
minimum.

### Main files

- `Task0/task_0.py` — phase-field solver
- `Task0/Testfvsphi.py` — free-energy verification
- `Task0/README.md` — detailed test description

### Selected evolution

<p align="center">
  <img src="Task0/output/t0.png" width="31%">
  <img src="Task0/output/t100.png" width="31%">
  <img src="Task0/output/t400.png" width="31%">
</p>

This first test isolates the nonlinear bulk free-energy contribution
before introducing interface physics.

---

## Task 1 — Gradient Energy and Interface Motion

**Purpose:** Introduce the phase-field gradient term and verify the
formation and propagation of a diffuse interface.

The model becomes

\[
\tau_0\dot{\phi}
=
-
\left[
\frac{\partial f}{\partial\phi}
-
\lambda_0^2\nabla^2\phi
\right].
\]

The verification considers:

- monotonic free-energy evolution,
- flat-interface propagation,
- interface velocity,
- interface thickness,
- sensitivity to spatial and temporal discretization.

### Energy evolution

<p align="center">
  <img src="Task1/Outputs/energy_decay.png" width="60%">
</p>

### Interface propagation

<p align="center">
  <img src="Task1/Outputs/xstar_vs_time_fit.png" width="48%">
  <img src="Task1/Outputs/speed_vs_time.png" width="48%">
</p>

The flat-front problem provides a controlled test of interface kinetics
before the thermal field is coupled to the phase-field equation.

---

## Task 2 — Coupled Phase Field and Temperature

**Purpose:** Introduce the coupled \(\phi-u\) formulation.

A mixed finite-element space is used to solve the phase and diffusion
fields simultaneously.

The phase-field residual contains the nonlinear bulk contribution and
gradient energy,

\[
R_\phi =
\int_\Omega
\tau_0(\phi-\phi_0)w_\phi\,d\Omega
+
\Delta t
\int_\Omega
\frac{\partial f}{\partial\phi}
w_\phi\,d\Omega
+
\Delta t
\int_\Omega
\lambda_0^2
\nabla\phi\cdot\nabla w_\phi\,d\Omega.
\]

The diffusion equation contains the phase-transformation coupling,

\[
R_u =
\int_\Omega
\left[
(u-u_0)
-
\frac{1}{2}(\phi-\phi_0)
\right]
w_u\,d\Omega
+
\Delta t D
\int_\Omega
\nabla u\cdot\nabla w_u\,d\Omega.
\]

### Verification checks

The implementation includes dedicated diagnostics for:

- total free-energy evolution,
- dissipation,
- enthalpy balance,
- phase/temperature coupling,
- homogeneous zero-flux boundary conditions.

### Selected diagnostics

<p align="center">
  <img src="Task_2/out_task2/Energy_vs_time.png" width="48%">
  <img src="Task_2/out_task2/Dissipation_vs_time.png" width="48%">
</p>

<p align="center">
  <img src="Task_2/out_task2/balance.png" width="48%">
  <img src="Task_2/out_task2/misfit_smooth.png" width="48%">
</p>

### Main files

- `Task_2/task_2.py` — coupled monolithic solver
- `Task_2/Test/test2.py` — thermodynamic diagnostics
- `Task_2/Test/test_bc.py` — zero-flux boundary-condition verification
- `Task_2/Test/README.md` — test documentation

---

## Task 3 — Anisotropic Dendritic Growth

**Purpose:** Extend the coupled formulation by introducing anisotropic
interface energy and reproduce dendritic solidification.

The final solver includes:

- diffuse-interface phase-field evolution,
- four-fold anisotropy,
- implicit time integration,
- nonlinear Newton solution,
- circular solid seed,
- optional interface perturbation,
- dendrite-tip tracking.

### Interface evolution

<p align="center">
  <img src="Task_3/outputs/2/output_task_3_2_phi0_contours.png" width="65%">
</p>

### Dendrite-tip velocity

<p align="center">
  <img src="Task_3/outputs/1/v_tip_vs_time.png" width="60%">
</p>

The dendrite tip is obtained by locating the farthest intersection of
the

\[
\phi=0
\]

interface along the selected growth direction and evaluating its
position as a function of time.

---

# Verification Strategy

A major objective of the project was not only to reproduce dendritic
morphologies, but also to verify the numerical implementation
incrementally.

| Stage | Feature introduced | Main verification |
|---|---|---|
| Task 0 | Bulk free energy | Numerical trajectory vs analytical energy landscape |
| Task 1 | Gradient/interface energy | Energy decay, flat-front velocity, interface thickness |
| Task 2 | Phase–temperature coupling | Energy, dissipation, enthalpy balance, zero-flux BC |
| Task 3 | Four-fold anisotropy | Preferred growth direction and dendrite-tip kinetics |

Additional decoupling tests contained in `Task_3/Test/` independently
check:

1. diffusion of \(u\) while \(\phi\) is frozen,
2. phase-field evolution while \(u\) remains constant.

This staged strategy helps distinguish errors in individual model terms
from errors in the fully coupled nonlinear formulation.

---

# Repository Structure

```text
PPP/
│
├── Final.py
│   └── Final anisotropic phase-field solver
│
├── PPP.pdf
│   └── Detailed project report
│
├── Task0/
│   ├── task_0.py
│   ├── Testfvsphi.py
│   ├── README.md
│   └── output/
│
├── Task1/
│   ├── task_1.py
│   ├── Test 1/
│   │   └── verification / post-processing scripts
│   └── Outputs/
│
├── Task_2/
│   ├── task_2.py
│   ├── Test/
│   │   ├── test2.py
│   │   ├── test_bc.py
│   │   └── README.md
│   └── out_task2/
│
└── Task_3/
    ├── Final.py
    ├── Test/
    │   ├── Test_f1.py
    │   ├── Test_f2.py
    │   ├── tipvelocity.py
    │   └── README.md
    └── outputs/
        ├── 1/
        └── 2/
  
