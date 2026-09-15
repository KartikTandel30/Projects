# Finite Element Modal Analysis of a 2D Bridge Structure

A Python implementation of **finite element modal analysis** for a two-dimensional
structural model.

The program builds the structural model from external CSV input files, assembles
the global stiffness and mass matrices, solves the generalized eigenvalue problem,
and visualizes the first three vibration mode shapes.

**Structural Dynamics · Finite Element Method · Modal Analysis · Eigenvalue Problems · Python · Scientific Computing**

---

## Overview

Every structure possesses characteristic vibration modes and corresponding
natural frequencies.

Modal analysis determines these properties from the structural stiffness and
mass distributions.

This project implements a finite-element modal-analysis workflow in Python for
a two-dimensional bridge-like structure.

The program performs the complete sequence

```text
Geometry and material input
            │
            ▼
Finite-element discretization
            │
            ▼
Element stiffness and mass matrices
            │
            ▼
Global matrix assembly
            │
            ▼
Boundary conditions
            │
            ▼
Generalized eigenvalue problem
            │
            ▼
Natural frequencies
            │
            ▼
Mode-shape visualization
```

The structural geometry, connectivity, element properties, and material
properties are stored externally in CSV files rather than being hard-coded
directly into the solver.

---

# Project Highlights

- Finite-element modal analysis implemented in Python
- Data-driven model definition using CSV input files
- 2D structural discretization
- Four-node quadrilateral elements
- Two translational degrees of freedom per node
- Bilinear shape functions
- Numerical integration using Gauss quadrature
- Element stiffness calculation
- Element mass calculation
- Global stiffness-matrix assembly
- Global mass-matrix assembly
- Structural boundary conditions
- Generalized eigenvalue solution using SciPy
- Natural-frequency extraction
- Mode-shape visualization
- Matrix symmetry and positive-definiteness checks
- Comparison of undeformed and modal deformation shapes

---

# Structural Model

The finite-element model represents a two-dimensional bridge-like structure.

<p align="center">
  <img src="Bridge_Ele.png" width="75%">
</p>

The current discretization contains:

| Model quantity | Value |
|---|---:|
| Number of nodes | 32 |
| Number of elements | 15 |
| Nodes per element | 4 |
| DOFs per node | 2 |
| Total DOFs before constraints | 64 |

Each node contains two translational degrees of freedom:

```math
\mathbf{u}_i =
\begin{bmatrix}
u_i \\
v_i
\end{bmatrix}
```

where `u` and `v` denote displacement components in the two spatial directions.

---

# Input-Driven Model Definition

The structural model is defined using four CSV files.

```text
node_details.csv
element_connectivity.csv
GeometryDetails.csv
reinforced_concrete_properties.csv
```

This separates the numerical solver from the model data and makes it possible
to modify the geometry and material information without rewriting the matrix
assembly procedure.

---

## Node Coordinates

The nodal coordinates are stored in

```text
node_details.csv
```

with the structure

```text
node_number, x, y
```

The current model contains 32 nodes.

The coordinates define the complete two-dimensional finite-element geometry.

---

## Element Connectivity

Element topology is supplied through

```text
element_connectivity.csv
```

with the format

```text
element_number, n1, n2, n3, n4
```

Each element therefore connects four nodes.

The current mesh contains 15 quadrilateral elements.

---

## Geometric Properties

Element geometrical information is stored in

```text
GeometryDetails.csv
```

including:

```text
element_number
width
height
area
mi
```

For the current model, the listed element geometry uses:

| Property | Value |
|---|---:|
| Width | 2 |
| Height | 1 |
| Area | 2 |
| `mi` | 0.166667 |

The element area is used by the current mass-matrix calculation.

---

## Material Properties

Material information is read from

```text
reinforced_concrete_properties.csv
```

The two properties currently used directly by the solver are:

| Property | Value |
|---|---:|
| Density | 2500 kg/m³ |
| Young's modulus | 30 GPa |

The Young's modulus is converted to SI units in the program before matrix
assembly.

---

# Finite Element Formulation

## Bilinear Shape Functions

Each four-node quadrilateral element uses bilinear interpolation in the natural
coordinates `ξ` and `η`.

The four shape functions are

```math
N_1(\xi,\eta)
=
\frac{1}{4}(1-\xi)(1-\eta)
```

```math
N_2(\xi,\eta)
=
\frac{1}{4}(1+\xi)(1-\eta)
```

```math
N_3(\xi,\eta)
=
\frac{1}{4}(1+\xi)(1+\eta)
```

```math
N_4(\xi,\eta)
=
\frac{1}{4}(1-\xi)(1+\eta)
```

These interpolate the displacement field inside each quadrilateral element.

---

# Coordinate Transformation

The derivatives of the shape functions are initially calculated with respect
to the natural coordinates.

The element Jacobian provides the mapping between the natural and physical
coordinate systems.

```math
\mathbf{J}
=
\frac{\partial(x,y)}
{\partial(\xi,\eta)}
```

The determinant

```math
\det(\mathbf{J})
```

is used during numerical integration.

The inverse Jacobian is used to transform derivatives into the physical
coordinate system required by the element formulation.

---

# Strain-Displacement Matrix

For every integration point, the code constructs an element
strain-displacement matrix `B`.

The finite-element strain field is represented in the general form

```math
\boldsymbol{\varepsilon}
=
\mathbf{B}\mathbf{u}_e
```

where

```math
\mathbf{u}_e
```

contains the eight displacement degrees of freedom associated with the four
nodes of one element.

For four nodes with two displacement components per node,

```math
\mathbf{u}_e
=
\begin{bmatrix}
u_1 &
v_1 &
u_2 &
v_2 &
u_3 &
v_3 &
u_4 &
v_4
\end{bmatrix}^{T}
```

---

# Numerical Integration

The stiffness calculation uses a `2 × 2` Gauss integration scheme.

The Gauss coordinates in each direction are

```math
\xi,\eta
=
\pm \frac{1}{\sqrt{3}}
```

giving four integration points per element.

```text
(-1/√3, -1/√3)
(+1/√3, -1/√3)
(-1/√3, +1/√3)
(+1/√3, +1/√3)
```

At every integration point, the program evaluates:

- shape-function derivatives,
- Jacobian matrix,
- determinant of the Jacobian,
- inverse Jacobian,
- strain-displacement matrix,
- element stiffness contribution.

---

# Element Stiffness

The current implementation builds an element stiffness contribution from the
strain-displacement matrix and Young's modulus.

Its computational structure is

```math
\mathbf{K}_e
=
\sum_{g}
\mathbf{B}_g^{T}
\mathbf{B}_g
E
t
\det(\mathbf{J}_g)
```

where:

- `g` denotes a Gauss point,
- `E` is Young's modulus,
- `t` is the assumed thickness,
- `B` is the element strain-displacement matrix.

The current implementation uses

```math
t = 1
```

for the structural thickness.

---

# Element Mass

The solver also constructs an element mass matrix using the material density,
element area, and assumed thickness.

The current implementation uses a simplified diagonal element mass representation.

```math
\mathbf{M}_e
=
\rho A t \mathbf{I}
```

where:

- `ρ` is the material density,
- `A` is the element area,
- `t` is the assumed thickness,
- `I` is the identity matrix.

The element mass matrix has dimensions

```math
8 \times 8
```

corresponding to the eight element displacement degrees of freedom.

---

# Global Matrix Assembly

The element matrices are assembled into the global system according to the
element connectivity.

The complete structural matrices are

```math
\mathbf{K}
=
\sum_e \mathbf{K}_e
```

and

```math
\mathbf{M}
=
\sum_e \mathbf{M}_e
```

For the current model with 32 nodes and two displacement degrees of freedom
per node, both global matrices have dimensions

```math
64 \times 64
```

before application of the structural constraints.

---

# Boundary Conditions

The current implementation constrains the displacement degrees of freedom
associated with nodes

```text
0, 7, 18, 25
```

For every constrained node, both translational degrees of freedom are treated
as fixed.

The corresponding rows and columns of the global stiffness and mass matrices
are modified before solving the eigenvalue problem.

A very small positive value is assigned to constrained entries of the mass
matrix to preserve numerical positive definiteness.

---

# Matrix Verification

Before solving the modal problem, the program performs several numerical
checks.

These include:

- checking diagonal entries of the mass matrix,
- checking mass-matrix eigenvalues,
- checking mass-matrix symmetry,
- checking positive definiteness after boundary-condition treatment.

If the final mass matrix contains a non-positive eigenvalue, the program
terminates rather than solving an invalid generalized eigenvalue problem.

---

# Free-Vibration Problem

For an undamped structure without external loading, the structural equation of
motion has the form

```math
\mathbf{M}\ddot{\mathbf{u}}
+
\mathbf{K}\mathbf{u}
=
\mathbf{0}
```

Assuming harmonic motion,

```math
\mathbf{u}(t)
=
\boldsymbol{\phi} e^{i\omega t}
```

leads to the generalized eigenvalue problem

```math
\mathbf{K}\boldsymbol{\phi}
=
\lambda
\mathbf{M}\boldsymbol{\phi}
```

with

```math
\lambda = \omega^2
```

Therefore, the natural angular frequency is obtained from

```math
\omega_i
=
\sqrt{\lambda_i}
```

where:

- `λᵢ` is the `i`th eigenvalue,
- `ωᵢ` is the corresponding natural angular frequency,
- `φᵢ` is the associated mode shape.

---

# Eigenvalue Solution

The generalized symmetric eigenvalue problem is solved using

```python
scipy.linalg.eigh(K_global, M_global)
```

The program obtains:

```text
eigenvalues
eigenvectors
```

and calculates the natural angular frequencies from

```python
natural_frequencies = np.sqrt(eigenvalues)
```

The first three natural frequencies are printed to the terminal in

```text
rad/s
```

---

# Mode Shapes

Each eigenvector represents a structural vibration mode.

For visualization, an eigenvector is reshaped into nodal x- and y-displacement
components.

```math
\boldsymbol{\phi}_i
\rightarrow
\left[
(u_1,v_1),
(u_2,v_2),
\ldots
\right]
```

Because eigenvectors provide relative modal amplitudes rather than physical
displacements, the mode shapes are multiplied by a visualization scaling
factor.

The current implementation uses

```text
Scale factor = 100
```

to make the deformation pattern visible.

For each of the first three modes, the program plots:

- the undeformed finite-element mesh,
- the scaled modal deformation,
- the corresponding natural angular frequency.

---

# Modal Analysis Workflow

The complete implementation follows this sequence:

```text
Read CSV input files
        │
        ▼
Read nodal coordinates
        │
        ▼
Read element connectivity
        │
        ▼
Read geometry and material properties
        │
        ▼
Initialize global K and M matrices
        │
        ▼
Loop over quadrilateral elements
        │
        ├── Extract nodal coordinates
        │
        ├── Calculate element mass matrix
        │
        ├── Loop over Gauss points
        │      │
        │      ├── Shape-function derivatives
        │      ├── Jacobian
        │      ├── B-matrix
        │      └── Stiffness contribution
        │
        ├── Assemble element stiffness
        │
        └── Assemble element mass
        │
        ▼
Check global mass matrix
        │
        ▼
Apply boundary conditions
        │
        ▼
Check symmetry / positive definiteness
        │
        ▼
Solve generalized eigenvalue problem
        │
        ▼
Calculate natural frequencies
        │
        ▼
Extract first three eigenvectors
        │
        ▼
Plot structural mode shapes
```

---

# Code Organization

The main solver is contained in

```text
Final_code.py
```

The implementation contains several principal components.

## Shape Functions

```python
bilinear_shape_functions(xi, eta)
```

Defines the four bilinear shape functions associated with the quadrilateral
element.

---

## B-Matrix and Jacobian

```python
compute_B_matrix_and_Jacobian(xi, eta, coords)
```

Calculates:

- natural-coordinate shape-function derivatives,
- element Jacobian,
- determinant of the Jacobian,
- inverse Jacobian,
- strain-displacement matrix.

---

## Element Mass Matrix

```python
compute_local_mass_matrix(density, area, thickness)
```

Constructs the local mass representation used by the current modal solver.

---

## Global Assembly

```python
assemble_matrices()
```

Loops over every element and assembles the element stiffness and mass
contributions into the global matrices.

---

## Eigenvalue Analysis

The assembled matrices are passed to the generalized eigensolver.

```python
eigenvalues, eigenvectors = eigh(K_global, M_global)
```

The corresponding angular frequencies are then calculated and the first three
mode shapes are visualized.

---

# Repository Structure

```text
RSJC/
│
├── README.md
│
├── Final_code.py
│   └── Finite-element modal-analysis solver
│
├── Bridge_Ele.png
│   └── Structural geometry / boundary-condition illustration
│
├── node_details.csv
│   └── Nodal coordinates
│
├── element_connectivity.csv
│   └── Four-node element connectivity
│
├── GeometryDetails.csv
│   └── Element geometric properties
│
└── reinforced_concrete_properties.csv
    └── Material-property input
```

---

# Requirements

The implementation uses:

- Python 3
- NumPy
- pandas
- Matplotlib
- SciPy

Install the dependencies using:

```bash
pip install numpy pandas matplotlib scipy
```

---

# Running the Analysis

The current script contains a directory variable used to locate the CSV files.

```python
directory = "C:/Users/karti/OneDrive/Desktop/RJ&SC"
```

Before running on another computer, change this variable to the directory
containing the four input CSV files.

Then execute:

```bash
python Final_code.py
```

The program will:

1. read the structural model,
2. assemble the stiffness and mass matrices,
3. apply the boundary conditions,
4. solve the generalized eigenvalue problem,
5. print the first three natural angular frequencies,
6. display the first three vibration mode shapes.

---

# Current Implementation Scope

This repository is an educational finite-element implementation intended to
demonstrate the computational workflow of structural modal analysis.

The current formulation uses a **simplified element mass and stiffness
representation** rather than a complete production-level plane-stress or
plane-strain quadrilateral formulation.

The project therefore focuses primarily on understanding and implementing:

- finite-element data structures,
- element-to-global matrix assembly,
- structural mass and stiffness matrices,
- boundary-condition treatment,
- generalized eigenvalue problems,
- natural frequencies,
- structural mode shapes.

This distinction is important when interpreting the numerical frequencies as
engineering predictions.

---

# Possible Extensions

The implementation provides a foundation that can be extended with:

- full plane-stress / plane-strain constitutive matrices,
- consistent or lumped mass formulations,
- systematic treatment of constrained DOFs by matrix reduction,
- frequency conversion and reporting in Hz,
- comparison with analytical or commercial FE results,
- mesh-convergence studies,
- automated result export,
- additional mode-shape visualization.

---

# Skills Demonstrated

This project demonstrates experience with:

- structural dynamics,
- finite-element formulation,
- modal analysis,
- generalized eigenvalue problems,
- quadrilateral finite elements,
- Gaussian numerical integration,
- global matrix assembly,
- structural boundary conditions,
- matrix verification,
- scientific-data input using pandas,
- NumPy matrix operations,
- SciPy eigensolvers,
- engineering visualization with Matplotlib,
- scientific programming in Python.

---

# Author

**Kartik Suresh Tandel**

M.Sc. Computational Materials Science  
TU Bergakademie Freiberg

### Areas of Interest

- Computational Mechanics
- Finite Element Analysis
- Structural Dynamics
- Modal Analysis
- Structural Simulation
- Numerical Methods
- Scientific Computing