# Phase-Field Modeling of Dendritic Solidification in FEniCSx

Finite-element implementation and numerical verification of phase-field models for
dendritic solidification using **FEniCSx / DOLFINx**, **PETSc**, **MPI**, and **Python**.

**Computational Materials Science · Phase-Field Modeling · Finite Element Method · Nonlinear FEM · Scientific Computing**

---

## Overview

Dendritic microstructures develop during the solidification of metals and alloys as
a result of the interaction between phase transformation, diffusion, interface
energy, and interfacial anisotropy.

Explicitly tracking the solid-liquid interface becomes increasingly difficult when
the interface develops complex morphologies and dendritic branches.

The **phase-field method** avoids explicit interface tracking by introducing a
continuous order parameter `φ`.

```math
\phi =
\begin{cases}
+1, & \text{solid},\\
-1, & \text{liquid}.
\end{cases}
```

Intermediate values of `φ` represent the diffuse solid-liquid interface.

The objective of this project is to implement the phase-field equations using the
**finite element method in FEniCSx**, verify the individual mathematical and
numerical components, and finally simulate anisotropic dendritic growth.

The model development is based primarily on the phase-field formulation presented
in:

> **K. Bhagat and S. Rudraraju**  
> *Modeling of dendritic solidification and numerical analysis of the phase-field
> approach to model complex morphologies in alloys*  
> Engineering with Computers, 39, 2345–2363, 2023.

The implementation is developed progressively rather than directly solving the
complete dendritic-solidification problem.

Each physical contribution is introduced and verified independently before being
combined into the final formulation.

---

# Project Highlights

- Phase-field model implemented directly from the governing equations
- Finite-element discretization using **FEniCSx / DOLFINx**
- First-order **Lagrange finite elements**
- Mixed `P1 × P1` finite-element formulation
- Fully implicit **Backward-Euler time integration**
- Nonlinear solution using **Newton's method**
- Consistent Jacobian generated using **UFL automatic differentiation**
- PETSc-based linear and nonlinear solution infrastructure
- MPI-compatible implementation
- Natural homogeneous Neumann boundary conditions
- Diffuse-interface gradient-energy formulation
- Coupled phase-field and diffusion formulation
- Four-fold interfacial anisotropy
- Dendrite-tip tracking and velocity evaluation
- Energy and dissipation verification
- Enthalpy-balance verification
- Boundary-condition verification
- Decoupling tests for the governing equations
- Numerical post-processing and visualization

---

# Dendritic Growth

The final development stage introduces four-fold interfacial anisotropy to an
initially circular solid seed.

<p align="center">
  <img src="Task_3/outputs/2/t0.png" width="31%">
  <img src="Task_3/outputs/2/t300.png" width="31%">
  <img src="Task_3/outputs/2/t600.png" width="31%">
</p>

<p align="center">
  <b>Evolution from the initial diffuse seed to an anisotropic dendritic morphology</b>
</p>

The solid-liquid interface is approximately represented by the zero level set

```math
\phi = 0.
```

---

# Physical Model

## Phase-Field Variable

The phase-field variable distinguishes the two bulk phases.

```math
\phi \approx +1
```

corresponds to the solid phase, while

```math
\phi \approx -1
```

corresponds to the liquid phase.

The transition between these values occurs over a finite diffuse-interface region.

---

## Bulk Free Energy

The local bulk free-energy density used in the implementation is

```math
f(\phi,u)
=
-\frac{1}{2}\phi^2
+\frac{1}{4}\phi^4
+\zeta u\phi
\left(
1-\frac{2}{3}\phi^2+\frac{1}{5}\phi^4
\right).
```

The first two terms form a double-well energy landscape associated with the two
bulk phases.

The derivative entering the phase-field equation is

```math
\frac{\partial f}{\partial \phi}
=
-\phi
+\phi^3
+\zeta u
\left(
1-2\phi^2+\phi^4
\right).
```

Here:

- `φ` is the phase-field order parameter,
- `u` is the non-dimensional temperature / undercooling field,
- `ζ` controls phase-temperature coupling.

---

## Gradient Energy

A gradient-energy contribution is introduced to assign a finite energetic cost to
the solid-liquid interface.

For the isotropic model, the free-energy functional can be expressed as

```math
\Pi[\phi,u]
=
\int_{\Omega}
\left[
f(\phi,u)
+
\frac{\lambda_0^2}{2}
|\nabla\phi|^2
\right]
\,d\Omega.
```

The parameter `λ₀` controls the characteristic interface thickness.

The gradient term converts the purely local phase evolution into a spatial
phase-field problem and creates a smooth diffuse interface.

---

# Governing Equations

The general phase-field formulation implemented in the project has the structure

```math
\tau(\mathbf n)\dot{\phi}
=
-\mu.
```

The chemical-potential-like driving quantity is

```math
\mu
=
\frac{\partial f}{\partial\phi}
-
\nabla\cdot\mathbf Q.
```

The diffusion field satisfies

```math
\dot{u}
=
D\nabla^2u
+
K\dot{\phi}.
```

The principal variables and parameters are:

| Symbol | Description |
|---|---|
| `φ` | Phase-field order parameter |
| `u` | Non-dimensional temperature / undercooling |
| `D` | Diffusion coefficient |
| `τ₀` | Characteristic phase-field relaxation time |
| `λ₀` | Characteristic interface thickness |
| `K` | Phase-transformation / latent-heat coupling |
| `ζ` | Bulk phase-temperature coupling |
| `Q` | Interfacial flux |
| `n` | Local interface normal |

---

# Four-Fold Interfacial Anisotropy

A purely isotropic interface tends to preserve circular symmetry.

Dendritic growth requires orientation-dependent interface properties.

The final solver introduces the four-fold anisotropy function

```math
a(\mathbf n)
=
(1-3\epsilon)
+
4\epsilon
\left(
n_x^4+n_y^4
\right).
```

The local interface normal is evaluated from the phase-field gradient.

```math
\mathbf n
=
\frac{\nabla\phi}
{\sqrt{\nabla\phi\cdot\nabla\phi+\eta^2}}.
```

A small regularization parameter `η` is introduced to avoid numerical singularities
in regions where the magnitude of the phase-field gradient approaches zero.

The anisotropic kinetic coefficient is defined as

```math
\tau(\mathbf n)
=
\tau_0a^2(\mathbf n).
```

The interfacial flux implemented in the solver has the form

```math
\mathbf Q
=
\lambda_0^2
\left[
a^2\nabla\phi
+
|\nabla\phi|^2
a
\frac{\partial a}{\partial(\nabla\phi)}
\right].
```

When the anisotropy parameter is zero,

```math
\epsilon = 0,
```

the formulation reduces to the isotropic gradient model.

For non-zero anisotropy, preferred growth directions are introduced and the
initially circular interface can develop dendritic arms.

---

# Finite-Element Formulation

## Spatial Discretization

The two-dimensional computational domain is discretized using triangular finite
elements.

The coupled problem uses a mixed finite-element space

```math
V_h
=
P_1 \times P_1.
```

The finite-element unknown can therefore be written as

```math
\mathbf y
=
\begin{bmatrix}
\phi \\
u
\end{bmatrix}.
```

Both fields are approximated using first-order Lagrange basis functions.

---

## Time Discretization

Transient evolution is discretized using the fully implicit
**Backward-Euler scheme**.

For the phase field,

```math
\dot{\phi}
\approx
\frac{\phi^{n+1}-\phi^n}
{\Delta t}.
```

Similarly, for the diffusion field,

```math
\dot{u}
\approx
\frac{u^{n+1}-u^n}
{\Delta t}.
```

The implicit formulation provides numerical robustness for the nonlinear
phase-field evolution problem.

---

# Weak Formulation

## Phase-Field Residual

The phase-field residual implemented in the anisotropic solver is

```math
R_{\phi}
=
\int_{\Omega}
\tau(\mathbf n)
(\phi-\phi_0)
w_{\phi}
\,d\Omega
+
\Delta t
\int_{\Omega}
\frac{\partial f}{\partial\phi}
w_{\phi}
\,d\Omega
+
\Delta t
\int_{\Omega}
\mathbf Q\cdot\nabla w_{\phi}
\,d\Omega.
```

Here `wφ` is the phase-field test function.

---

## Diffusion Residual

The diffusion residual is

```math
R_u
=
\int_{\Omega}
(u-u_0)w_u
\,d\Omega
-
\Delta tK
\int_{\Omega}
(\phi-\phi_0)w_u
\,d\Omega
+
\Delta tD
\int_{\Omega}
\nabla u\cdot\nabla w_u
\,d\Omega.
```

Here `wu` is the diffusion-field test function.

The complete nonlinear residual is

```math
R(\phi,u)
=
R_{\phi}+R_u
=
0.
```

---

# Nonlinear Solution

At every time increment, the nonlinear finite-element equations are solved using
**Newton's method**.

The consistent Jacobian is obtained from the residual using UFL automatic
differentiation.

```math
\mathbf J
=
\frac{\partial\mathbf R}
{\partial\mathbf y}.
```

The linearized Newton problem is handled through PETSc.

The current implementation uses a direct LU factorization and selects
**SuperLU_DIST** or **MUMPS** when these packages are available.

---

# Boundary Conditions

No essential Dirichlet boundary conditions are imposed in the primary
solidification simulations.

The weak formulation naturally introduces homogeneous Neumann boundary conditions.

For the phase field,

```math
\nabla\phi\cdot\mathbf n_b
=
0.
```

For the diffusion field,

```math
\nabla u\cdot\mathbf n_b
=
0.
```

Here `nb` denotes the outward normal to the external domain boundary.

These conditions represent zero flux across the external boundaries.

---

# Progressive Development and Verification

The implementation was developed in four main stages.

Each stage introduces one additional level of physical or numerical complexity.

| Stage | Main feature | Verification objective |
|---|---|---|
| Task 0 | Bulk free energy | Verify local phase evolution |
| Task 1 | Gradient energy | Verify diffuse-interface behavior |
| Task 2 | Coupled `φ-u` problem | Verify coupling and thermodynamic consistency |
| Task 3 | Four-fold anisotropy | Verify anisotropic dendritic evolution |

---

# Task 0 — Local Phase-Field Evolution

## Objective

The first stage verifies the local bulk free-energy implementation independently
of the gradient-energy term.

With the gradient contribution removed, the phase-field equation reduces to a
local nonlinear ODE.

```math
\tau_0\dot{\phi}
=
-
\frac{\partial f}{\partial\phi}.
```

Using the implemented free energy,

```math
\tau_0\dot{\phi}
=
\phi-\phi^3
-
\zeta u
\left(
1-2\phi^2+\phi^4
\right).
```

Because no spatial-gradient term is present, the phase field evolves locally
according to the bulk energy landscape.

The numerical trajectory of a representative node is compared against the
analytical free-energy curve.

---

## Representative Evolution

<p align="center">
  <img src="Task0/output/t0.png" width="31%">
  <img src="Task0/output/t100.png" width="31%">
  <img src="Task0/output/t400.png" width="31%">
</p>

The test verifies that the numerical evolution proceeds toward the energetically
favored phase.

---

## Task 0 Files

- `Task0/task_0.py` — local phase-field solver
- `Task0/Testfvsphi.py` — free-energy verification
- `Task0/README.md` — test documentation
- `Task0/output/` — selected results
- `Task0/output/phiAndBulkData.xlsx` — node-wise phase and energy data

---

# Task 1 — Gradient Energy and Interface Motion

## Objective

The second stage introduces the phase-field gradient-energy term.

The governing equation becomes

```math
\tau_0\dot{\phi}
=
-\mu,
```

with

```math
\mu
=
\frac{\partial f}{\partial\phi}
-
\lambda_0^2\nabla^2\phi.
```

This produces a finite-width diffuse interface.

---

## Verification Quantities

The test examines:

- total free-energy evolution,
- flat-interface propagation,
- interface position,
- interface velocity,
- interface thickness,
- mesh dependence,
- time-step dependence.

---

## Energy Evolution

<p align="center">
  <img src="Task1/Outputs/energy_decay.png" width="65%">
</p>

The total free energy is monitored during phase evolution to assess thermodynamic
consistency.

---

## Interface Position and Velocity

<p align="center">
  <img src="Task1/Outputs/xstar_vs_time_fit.png" width="48%">
  <img src="Task1/Outputs/speed_vs_time.png" width="48%">
</p>

The interface position is tracked during the simulation and used to calculate the
propagation velocity.

---

## Interface Thickness

<p align="center">
  <img src="Task1/Outputs/thickness_vs_time.png" width="60%">
</p>

The diffuse-interface thickness is tracked to verify that the gradient penalty
produces a stable finite-width transition region.

---

## Task 1 Files

- `Task1/task_1.py` — phase-field solver with gradient energy
- `Task1/Test 1/` — verification scripts
- `Task1/Outputs/energy_track.csv`
- `Task1/Outputs/interface_track.csv`
- `Task1/Outputs/thickness_track.csv`
- `Task1/Outputs/` — result figures

---

# Task 2 — Coupled Phase Field and Diffusion

## Objective

The third development stage introduces the diffusion / temperature field `u`.

The two fields are solved simultaneously using a mixed finite-element formulation.

---

## Phase-Field Residual

The phase-field residual used in this stage is

```math
R_{\phi}
=
\int_{\Omega}
\tau_0
(\phi-\phi_0)
w_{\phi}
\,d\Omega
+
\Delta t
\int_{\Omega}
\frac{\partial f}{\partial\phi}
w_{\phi}
\,d\Omega
+
\Delta t\lambda_0^2
\int_{\Omega}
\nabla\phi\cdot\nabla w_{\phi}
\,d\Omega.
```

---

## Diffusion Residual

The coupled diffusion equation is represented by

```math
R_u
=
\int_{\Omega}
\left[
(u-u_0)
-
\frac{1}{2}
(\phi-\phi_0)
\right]
w_u
\,d\Omega
+
\Delta tD
\int_{\Omega}
\nabla u\cdot\nabla w_u
\,d\Omega.
```

The two residuals are assembled into one nonlinear mixed finite-element problem.

---

# Task 2 Verification

The coupled implementation includes dedicated checks for:

- free-energy evolution,
- dissipation,
- enthalpy balance,
- phase-temperature coupling,
- homogeneous Neumann boundary conditions.

---

## Energy and Dissipation

<p align="center">
  <img src="Task_2/out_task2/Energy_vs_time.png" width="48%">
  <img src="Task_2/out_task2/Dissipation_vs_time.png" width="48%">
</p>

---

## Enthalpy / Coupling Balance

<p align="center">
  <img src="Task_2/out_task2/balance.png" width="48%">
  <img src="Task_2/out_task2/misfit_smooth.png" width="48%">
</p>

These diagnostics are used to check whether the coupled numerical implementation is
consistent with the expected integral balance.

---

## Representative Phase and Diffusion Fields

<p align="center">
  <img src="Task_2/out_task2/t0.png" width="31%">
  <img src="Task_2/out_task2/t40.png" width="31%">
  <img src="Task_2/out_task2/u40.png" width="31%">
</p>

---

## Task 2 Files

- `Task_2/task_2.py` — monolithic coupled solver
- `Task_2/Test/test2.py` — thermodynamic diagnostics
- `Task_2/Test/test_bc.py` — boundary-condition checks
- `Task_2/Test/README.md` — verification documentation
- `Task_2/out_task2/diagnostics.csv`
- `Task_2/out_task2/` — selected numerical results

---

# Task 3 — Four-Fold Anisotropy and Dendritic Growth

## Objective

The final development stage introduces four-fold interfacial anisotropy.

This stage investigates the morphological evolution of an initially circular
diffuse-interface seed.

The solver includes:

- mixed `φ-u` finite-element space,
- Backward-Euler integration,
- Newton nonlinear solution,
- anisotropic gradient contribution,
- interface-normal regularization,
- diffuse circular seed,
- small interface perturbation,
- dendrite-tip post-processing.

---

# Important Note About the Current Benchmark

The final solver retains the general `φ-u` formulation.

However, the parameter set used for the current anisotropic benchmark sets

```math
\zeta = 0
```

and

```math
K = 0.
```

Therefore, phase-temperature coupling and latent-heat coupling are disabled for
this particular anisotropy benchmark.

The anisotropic phase-field contribution is consequently examined independently
after the coupled formulation has already been verified in **Task 2**.

---

# Initial Condition

The initial solid phase is represented by a circular diffuse seed.

The phase field is initialized approximately as

```math
\phi(r,0)
=
\tanh
\left(
\frac{r_0-r}
{\sqrt{2}\lambda_0}
\right).
```

where `r₀` is the initial seed radius.

A small perturbation is added primarily near the diffuse interface to allow
anisotropic morphological evolution to develop.

---

# Dendrite Evolution

<p align="center">
  <img src="Task_3/outputs/1/t0.png" width="24%">
  <img src="Task_3/outputs/1/t150.png" width="24%">
  <img src="Task_3/outputs/1/t300.png" width="24%">
  <img src="Task_3/outputs/1/t500.png" width="24%">
</p>

The initially circular interface progressively develops directional growth
associated with the four-fold anisotropic interfacial energy.

---

# Phase-Field Interface

The interface is extracted from the zero contour

```math
\phi = 0.
```

<p align="center">
  <img src="Task_3/outputs/2/output_task_3_2_phi0_contours.png" width="68%">
</p>

This contour provides a convenient representation of the evolving solid-liquid
boundary.

---

# Dendrite-Tip Tracking

The dendrite-tip position is obtained from the `φ = 0` interface.

For each stored simulation state, the post-processing procedure determines the
farthest interface position along the chosen growth direction.

This produces the tip-position history

```math
x_{\mathrm{tip}}(t).
```

The tip velocity can then be estimated from

```math
v_{\mathrm{tip}}
=
\frac{dx_{\mathrm{tip}}}{dt}.
```

---

## Tip-Velocity Result

<p align="center">
  <img src="Task_3/outputs/1/v_tip_vs_time.png" width="65%">
</p>

The corresponding numerical interface coordinates are stored in

```text
Task_3/outputs/1/tip_trace.csv
```

and

```text
Task_3/outputs/2/tip_trace.csv
```

for further analysis.

---

# Decoupling Tests

Additional verification tests are provided in `Task_3/Test/`.

These tests isolate individual components of the governing formulation.

---

## Diffusion-Only Test

The phase field is frozen while the `u` field evolves.

This test isolates:

- the diffusion operator,
- Backward-Euler time integration,
- homogeneous Neumann boundary conditions.

---

## Phase-Field-Only Test

The diffusion field is held constant while `φ` evolves.

This test isolates:

- nonlinear phase evolution,
- gradient-energy contribution,
- curvature-driven interface motion.

---

# Verification Philosophy

The main numerical-development strategy used in this project is:

> **Build → Isolate → Verify → Couple → Verify Again**

The purpose is to avoid treating visually plausible dendritic morphology alone as
proof of numerical correctness.

Instead, individual parts of the formulation are checked before constructing the
final model.

---

# Verification Summary

| Stage | Added feature | Main verification |
|---|---|---|
| **Task 0** | Bulk free energy | Numerical phase trajectory |
| **Task 1** | Gradient energy | Energy, interface speed and thickness |
| **Task 2** | `φ-u` coupling | Energy, dissipation, enthalpy and BCs |
| **Task 3** | Four-fold anisotropy | Morphology and dendrite-tip kinetics |

---

# Parameters of the Current Final Benchmark

The current final simulation uses the following representative parameters.

| Parameter | Value |
|---|---:|
| Domain size | `250 × 250` |
| Mesh divisions | `250 × 250` |
| Time step | `0.04` |
| Total simulation time | `200` |
| `τ₀` | `1.0` |
| `λ₀` | `1.0` |
| `D` | `1.0` |
| Initial `u₀` | `-0.65` |
| Initial seed radius `r₀` | `5.0` |
| Anisotropy `ε` | `0.05` |
| `ζ` | `0` |
| `K` | `0` |

---

# Repository Structure

```text
PPP/
│
├── README.md
├── Final.py
├── PPP.pdf
│
├── Task0/
│   ├── task_0.py
│   ├── Testfvsphi.py
│   ├── README.md
│   └── output/
│       ├── t0.png
│       ├── t100.png
│       ├── t400.png
│       └── phiAndBulkData.xlsx
│
├── Task1/
│   ├── task_1.py
│   │
│   ├── Test 1/
│   │   └── README.MD
│   │
│   └── Outputs/
│       ├── energy_decay.png
│       ├── speed_vs_time.png
│       ├── thickness_vs_time.png
│       ├── xstar_vs_time_fit.png
│       └── CSV diagnostic files
│
├── Task_2/
│   ├── task_2.py
│   │
│   ├── Test/
│   │   ├── test2.py
│   │   ├── test_bc.py
│   │   └── README.md
│   │
│   └── out_task2/
│       ├── Energy_vs_time.png
│       ├── Dissipation_vs_time.png
│       ├── balance.png
│       ├── misfit_smooth.png
│       └── diagnostics.csv
│
└── Task_3/
    ├── Final.py
    │
    ├── Test/
    │   ├── Test_f1.py
    │   ├── Test_f2.py
    │   ├── tipvelocity.py
    │   └── README.md
    │
    └── outputs/
        ├── 1/
        │   ├── t0.png
        │   ├── t150.png
        │   ├── t300.png
        │   ├── t500.png
        │   ├── tip_trace.csv
        │   └── v_tip_vs_time.png
        │
        └── 2/
            ├── t0.png
            ├── t150.png
            ├── t300.png
            ├── t500.png
            ├── t600.png
            ├── output_task_3_2_phi0_contours.png
            ├── tip_trace.csv
            └── v_tip_vs_time.png
```

---

# Software Stack

## Numerical Implementation

- **Python**
- **FEniCSx / DOLFINx**
- **UFL**
- **Basix**
- **PETSc**
- **petsc4py**
- **MPI**
- **mpi4py**
- **NumPy**

---

## Post-Processing and Visualization

- **Matplotlib**
- **PyVista**
- **PyVistaQt**
- **OpenPyXL**
- **ParaView**

---

# Running the Final Solver

A working FEniCSx environment with PETSc and MPI is required.

From the `PPP` directory:

```bash
python Final.py
```

For an MPI-enabled run:

```bash
mpirun -np 4 python Final.py
```

The number of MPI processes can be adjusted according to the available
computational resources.

---

# Running the Development Stages

## Task 0

```bash
cd Task0
python task_0.py
python Testfvsphi.py
```

---

## Task 1

```bash
cd Task1
python task_1.py
```

Additional verification scripts and documentation are contained in

```text
Task1/Test 1/
```

---

## Task 2

```bash
cd Task_2
python task_2.py
```

This stage includes the thermodynamic and boundary-condition diagnostics.

---

## Task 3

```bash
cd Task_3
python Final.py
```

Verification and post-processing utilities are contained in

```text
Task_3/Test/
```

---

# Simulation Output

FEniCSx produces transient field data using XDMF/HDF5 output.

For fine spatial meshes and long transient simulations, the corresponding HDF5
files can become very large.

For this reason, large generated simulation files are intentionally excluded from
this repository.

The repository instead contains selected:

- phase-field snapshots,
- interface contours,
- diffusion-field snapshots,
- energy histories,
- dissipation histories,
- balance diagnostics,
- interface trajectories,
- tip-position data,
- dendrite-tip velocity plots,
- verification CSV files.

This keeps the repository lightweight while retaining the main numerical evidence
required to understand and assess the implementation.

---

# Technical Report

A more detailed description of the mathematical formulation, implementation,
verification procedure, and numerical results is available in:

**[PPP.pdf](PPP.pdf)**

---

# Reference

The numerical model implemented in this project is based primarily on:

**K. Bhagat and S. Rudraraju**

*Modeling of dendritic solidification and numerical analysis of the phase-field
approach to model complex morphologies in alloys*

**Engineering with Computers**, 39, 2345–2363, 2023.

---

# Project Context

This work was developed as a **Personal Programming Project (PPP)** within the

**M.Sc. Computational Materials Science**

program at

**TU Bergakademie Freiberg**.

The objective was not only to generate dendritic morphologies, but to develop,
implement, and systematically verify the numerical phase-field formulation using a
general finite-element framework.

---

# Author

**Kartik Suresh Tandel**

M.Sc. Computational Materials Science  
TU Bergakademie Freiberg

### Areas of Interest

- Computational Mechanics
- Computational Materials Science
- Finite Element Methods
- Phase-Field Modeling
- Microstructure Evolution
- Nonlinear Numerical Methods
- Multiphysics Simulation
- Scientific Computing