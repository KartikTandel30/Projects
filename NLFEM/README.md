# Nonlinear Finite Element Analysis of a Viscoplastic Bar

A self-written **nonlinear finite element solver in Python** for the
incremental analysis of a one-dimensional two-segment bar with
rate-dependent viscoplastic material behavior.

The project implements the finite-element procedure from the element level
through global nonlinear equilibrium, including constitutive integration,
consistent tangent stiffness, numerical integration, assembly, boundary
conditions, and Newton-Raphson iterations.

**Nonlinear FEM · Computational Mechanics · Viscoplasticity · Newton-Raphson · Constitutive Modeling · Python**

---

## Overview

This project develops a nonlinear finite-element implementation for a
one-dimensional bar consisting of two connected sections with different
geometries.

Both ends of the bar are fixed and an external force is applied at the
junction between the two sections.

The response is solved incrementally as the applied load increases.

The implementation includes:

- two-node linear bar elements,
- finite-element strain-displacement formulation,
- two-point Gauss quadrature,
- nonlinear constitutive material response,
- rate-dependent viscoplastic flow,
- internal-force calculation,
- tangent stiffness calculation,
- global finite-element assembly,
- Dirichlet boundary conditions,
- incremental loading,
- Newton-Raphson equilibrium iterations,
- stress-strain post-processing,
- force-displacement response,
- mesh/convergence investigation.

---

# Mechanical Problem

The model consists of two bar segments connected at an internal node.

```text
Fixed                                                Fixed
  |                                                    |
  |------ Bar 1 ------|------ Bar 2 ------------------|
                      ↑
                  Applied force
```

The two bar sections have different lengths and cross-sectional areas.

The current implementation uses:

| Parameter | Bar 1 | Bar 2 |
|---|---:|---:|
| Length | 40 mm | 80 mm |
| Cross-sectional area | 8 mm² | 16 mm² |
| Number of elements | 5 | 5 |

The external force is applied at the junction between the two bars while the
two outer ends are constrained.

Because the two sections have different cross-sectional areas, they experience
different stress levels during loading.

---

# Finite Element Formulation

## Displacement Approximation

Each bar element contains two nodes.

The displacement field inside an element is approximated using linear
shape functions:

```math
u(\xi)
=
N_1(\xi)u_1
+
N_2(\xi)u_2.
```

The shape functions are

```math
N_1(\xi)
=
\frac{1}{2}(1-\xi),
```

and

```math
N_2(\xi)
=
\frac{1}{2}(1+\xi).
```

The element nodal displacement vector is

```math
\mathbf u_e
=
\begin{bmatrix}
u_1 \\
u_2
\end{bmatrix}.
```

---

## Strain-Displacement Relation

For the one-dimensional bar, the strain is

```math
\varepsilon
=
\frac{du}{dx}.
```

Using the finite-element interpolation,

```math
\varepsilon
=
\mathbf B \mathbf u_e,
```

where

```math
\mathbf B
=
\frac{d\mathbf N}{dx}.
```

The transformation between the natural coordinate and physical coordinate is
performed using the element Jacobian.

```math
J
=
\frac{dx}{d\xi}.
```

---

# Constitutive Model

The total strain is decomposed into elastic and viscoplastic contributions.

```math
\varepsilon
=
\varepsilon^e
+
\varepsilon^p.
```

The stress follows the one-dimensional elastic relation

```math
\sigma
=
E
\left(
\varepsilon-\varepsilon^p
\right).
```

The current implementation uses

```math
E = 80000 \; \text{MPa},
```

with a reference/yield stress

```math
\sigma_y = 160 \; \text{MPa}.
```

---

## Rate-Dependent Viscoplastic Flow

The inelastic evolution is governed by an overstress-type flow relation.

A measure of overstress is defined from

```math
\frac{|\sigma|}{\sigma_y}-1.
```

Only positive overstress contributes to viscoplastic deformation.

Using the Macaulay bracket,

```math
\langle x\rangle
=
\max(0,x),
```

the flow rule can be expressed as

```math
\dot{\varepsilon}^{p}
=
\eta
\left\langle
\frac{|\sigma|}{\sigma_y}-1
\right\rangle^m
\operatorname{sign}(\sigma).
```

The current implementation uses

```math
\eta = 50,
```

and

```math
m = 1.
```

Below the reference stress, the material remains elastic.

Once the overstress becomes positive, viscoplastic strain evolves with time.

---

# Incremental Constitutive Update

For every load/time increment, the code first evaluates an elastic trial stress.

```math
\sigma_{\mathrm{trial}}
=
E
\left(
\varepsilon^{n+1}
-
\varepsilon_p^n
\right).
```

If

```math
|\sigma_{\mathrm{trial}}|
\le
\sigma_y,
```

the material response remains elastic.

Therefore,

```math
\varepsilon_p^{n+1}
=
\varepsilon_p^n,
```

and

```math
\sigma^{n+1}
=
\sigma_{\mathrm{trial}}.
```

For an active viscoplastic step, the internal variable is updated
incrementally.

For the implemented case with `m = 1`, the stress update used in the code can
be written as

```math
\sigma^{n+1}
=
\frac{
\left[
\sigma_{\mathrm{trial}}
+
E\Delta t\eta
\operatorname{sign}(\sigma_{\mathrm{trial}})
\right]
\sigma_y
}{
\sigma_y
+
E\Delta t\eta
}.
```

The viscoplastic strain is then updated from

```math
\varepsilon_p^{n+1}
=
\varepsilon_p^n
+
\Delta t
\eta
\left\langle
\frac{|\sigma^{n+1}|}{\sigma_y}-1
\right\rangle^m
\operatorname{sign}(\sigma^{n+1}).
```

---

# Algorithmic Tangent

The nonlinear finite-element solution requires the material tangent.

For the elastic regime,

```math
C_t = E.
```

For the active viscoplastic regime implemented in the current model,

```math
C_t
=
\frac{
E\sigma_y
}{
\sigma_y
+
E\Delta t\eta
}.
```

This tangent is used to construct the element tangent stiffness matrix during
the Newton-Raphson iterations.

---

# Element Formulation

The element internal-force vector is calculated as

```math
\mathbf f_{\mathrm{int}}^e
=
\int_{\Omega_e}
\mathbf B^T
\sigma
A
\,dx.
```

The element tangent stiffness matrix is

```math
\mathbf K_T^e
=
\int_{\Omega_e}
\mathbf B^T
C_t
\mathbf B
A
\,dx.
```

Both quantities are evaluated numerically.

---

# Numerical Integration

The element integrals are calculated using **two-point Gauss quadrature**.

The Gauss points are

```math
\xi_1
=
-\frac{1}{\sqrt{3}},
```

and

```math
\xi_2
=
+\frac{1}{\sqrt{3}}.
```

with weights

```math
w_1 = w_2 = 1.
```

At each Gauss point, the solver evaluates:

- the shape functions,
- shape-function derivatives,
- Jacobian,
- strain,
- constitutive response,
- stress,
- tangent modulus,
- internal force contribution,
- tangent stiffness contribution.

---

# Global Assembly

Each element contribution is mapped into the global system.

The global tangent stiffness is assembled as

```math
\mathbf K_T
=
\sum_e
\mathbf A_e^T
\mathbf K_T^e
\mathbf A_e.
```

The global internal-force vector is assembled as

```math
\mathbf F_{\mathrm{int}}
=
\sum_e
\mathbf A_e^T
\mathbf f_{\mathrm{int}}^e.
```

Here, `A_e` represents the element-to-global degree-of-freedom mapping.

---

# Nonlinear Equilibrium

At every load increment, equilibrium requires

```math
\mathbf R(\mathbf u)
=
\mathbf F_{\mathrm{int}}(\mathbf u)
-
\mathbf F_{\mathrm{ext}}
=
\mathbf 0.
```

Because the constitutive response depends nonlinearly on the current
deformation state, this equation is solved iteratively.

---

# Newton-Raphson Method

For Newton iteration `i`, the linearized problem is

```math
\mathbf K_T^{(i)}
\Delta\mathbf u^{(i)}
=
-
\mathbf R^{(i)}.
```

The displacement is updated using

```math
\mathbf u^{(i+1)}
=
\mathbf u^{(i)}
+
\Delta\mathbf u^{(i)}.
```

The process is repeated until the equilibrium residual or displacement
increment satisfies the specified tolerance.

The implementation monitors both:

```math
\|\mathbf R\|_{\infty},
```

and

```math
\|\Delta\mathbf u\|_{\infty}.
```

---

# Boundary Conditions

The two outer nodes of the bar are constrained.

```math
u_{\mathrm{left}} = 0,
```

and

```math
u_{\mathrm{right}} = 0.
```

The corresponding constrained rows and columns are removed from the global
system before solving for the free degrees of freedom.

The external load is applied at the node joining Bar 1 and Bar 2.

---

# Incremental Loading

The external load is increased linearly over the simulation time.

```math
F(t)
=
\frac{F_{\max}}{t_{\mathrm{tot}}}t.
```

The current model uses

```math
F_{\max}
=
4200 \; \text{N},
```

and

```math
t_{\mathrm{tot}}
=
0.01 \; \text{s}.
```

The response is solved over many small increments so that the evolution of the
viscoplastic internal variable can be captured.

---

# Solution Workflow

The numerical procedure implemented in `PVL.py` follows the sequence:

```text
Input material and geometry parameters
              │
              ▼
Generate mesh and global DOFs
              │
              ▼
Initialize displacement and plastic strain
              │
              ▼
Start load/time increment
              │
              ▼
Start Newton-Raphson iteration
              │
              ▼
Loop over finite elements
              │
              ├── Extract element displacement
              ├── Compute B-matrix
              ├── Evaluate strain
              ├── Call material routine
              ├── Compute stress
              ├── Compute tangent modulus
              ├── Integrate element stiffness
              └── Integrate internal force
              │
              ▼
Assemble global tangent stiffness
and internal-force vector
              │
              ▼
Apply displacement boundary conditions
              │
              ▼
Compute residual
              │
              ▼
Solve for displacement correction
              │
              ▼
Update nodal displacements
              │
              ▼
Check Newton convergence
       │              │
       │ No           │ Yes
       ▼              ▼
Next iteration    Store results
                      │
                      ▼
                Next load step
```

---

# Numerical Results

The repository contains representative plots obtained from the implementation.

## Elastic Response

<p align="center">
  <img src="Elastic_curve.png" width="70%">
</p>

The elastic-response case provides a reference for checking the basic
finite-element assembly and structural response before focusing on the
viscoplastic regime.

---

## Viscoplastic Response

<p align="center">
  <img src="Plastic.png" width="70%">
</p>

The nonlinear calculation tracks the stress-strain behavior as the applied load
increases and viscoplastic deformation becomes active.

---

## Element / Mesh Convergence

<p align="center">
  <img src="Convergense_with_eleme.png" width="70%">
</p>

A discretization study is included to investigate the sensitivity of the computed
response to the number of finite elements.

---

# Quantities Calculated by the Solver

The program stores and evaluates:

- nodal displacement,
- displacement of the loaded junction,
- stress in Bar 1,
- stress in Bar 2,
- strain in Bar 1,
- strain in Bar 2,
- viscoplastic strain,
- tangent material stiffness,
- internal-force vector,
- residual-force vector,
- Newton displacement corrections.

The generated post-processing includes:

- stress-strain response of Bar 1,
- stress-strain response of Bar 2,
- force-displacement response,
- displacement-time response.

---

# Current Model Parameters

The parameter values currently used in `PVL.py` are:

| Parameter | Value |
|---|---:|
| Young's modulus `E` | 80000 MPa |
| Reference/yield stress `σy` | 160 MPa |
| Viscoplastic parameter `η` | 50 |
| Rate exponent `m` | 1 |
| Bar 1 area `A1` | 8 mm² |
| Bar 2 area `A2` | 16 mm² |
| Bar 1 length `L1` | 40 mm |
| Bar 2 length `L2` | 80 mm |
| Maximum applied force | 4200 N |
| Total loading time | 0.01 s |
| Elements in Bar 1 | 5 |
| Elements in Bar 2 | 5 |
| Maximum Newton iterations | 10 |
| Newton tolerance | 0.005 |

---

# Code Structure

The solver is organized into three principal levels.

### Material level

`material()`

Evaluates:

- elastic trial stress,
- activation of viscoplastic flow,
- stress update,
- viscoplastic-strain update,
- material tangent.

### Element level

`elementStiffness()`

Evaluates:

- shape functions,
- Jacobian,
- B-matrix,
- element strain,
- Gauss-point material response,
- element internal force,
- element tangent stiffness.

### Global level

`main()`

Handles:

- time/load stepping,
- element assembly,
- global stiffness matrix,
- internal-force vector,
- boundary conditions,
- Newton-Raphson iterations,
- convergence checks,
- result storage,
- plotting.

This separation reflects the standard hierarchy used in nonlinear
finite-element implementations:

```text
Material routine
      ↓
Element routine
      ↓
Global assembly
      ↓
Nonlinear equilibrium solver
```

---

# Repository Structure

```text
NLFEM/
│
├── README.md
│
├── PVL.py
│   └── Nonlinear finite-element implementation
│
├── NLFEM_report.pdf
│   └── Detailed project report
│
├── Elastic_curve.png
│   └── Elastic-response result
│
├── Plastic.png
│   └── Nonlinear / viscoplastic response
│
└── Convergense_with_eleme.png
    └── Element-convergence study
```

---

# Requirements

The implementation uses:

- Python 3
- NumPy
- Matplotlib

Install the required Python packages with:

```bash
pip install numpy matplotlib
```

---

# Running the Solver

From the `NLFEM` directory:

```bash
python PVL.py
```

The solver performs the complete incremental nonlinear analysis and generates the
stress-strain, force-displacement, and displacement-time plots.

---

# Technical Report

A detailed description of the formulation, implementation, numerical procedure,
and project results is included in:

**[NLFEM_report.pdf](NLFEM_report.pdf)**

---

# Skills Demonstrated

This project demonstrates implementation of fundamental concepts used in
nonlinear computational mechanics:

- finite-element formulation from first principles,
- nonlinear equilibrium,
- Newton-Raphson solution,
- constitutive material integration,
- internal-variable evolution,
- algorithmic tangent stiffness,
- numerical quadrature,
- element-to-global assembly,
- incremental loading,
- convergence assessment,
- scientific programming in Python.

---

# Author

**Kartik Suresh Tandel**

M.Sc. Computational Materials Science  
TU Bergakademie Freiberg

### Areas of Interest

- Computational Mechanics
- Nonlinear Finite Element Analysis
- Constitutive Material Modeling
- Plasticity and Viscoplasticity
- Structural Simulation
- Scientific Computing
- Numerical Methods