

# Test T5 — Verification of Anisotropy & Dendrite Tip Velocity

**Goal.** Verify that four-fold interfacial anisotropy selects the preferred growth direction and that the measured dendrite **tip velocity** matches the expected benchmark for the chosen parameters.

---

## Simulation setup

- **Solver:** monolithic `φ–u` (temperature-coupled) solver: `Final.py`
- **Outputs:** time series written to `output.xdmf` (+ its `.h5`), field name `phi`
- **Anisotropy:** four-fold; preferred growth along **x** unless rotated by a parameter (φ)

After the run, use the post-processor `tipvelocity.py` to extract the tip track and velocity.

---

## What the post-processor does

For each saved time slice:
1. Takes a **centerline** along the chosen growth axis (default: **x** direction).
2. Finds the **farthest interface** point where `phi = 0` along that line.
3. Writes **`tip_trace.csv`** with columns:
4. Saves a **tip velocity vs time** plot (PNG/SVG) in the same folder.

---

## Where to place files

Put `tipvelocity.py` in the same directory with the outputs of order parameter or phase  (e.g., `output.xdmf` and `output.h5`).  
It also works if passed a **full .xdmf path** or a **folder** that contains the pair.

---

## How to run

### Interactive
```bash
python tipvelocity.py

# Test T4 — Verification of Anisotropy & Dendrite Tip Velocity
# Test_f1 — Decoupling Test A (u diffusion with φ frozen)

**Goal:** Verify that the temperature field `u` diffuses under homogeneous Neumann BCs while the phase field stays fixed (`φ ≡ -1`). This tests the diffusion operator, fully implicit Backward–Euler (BE) time stepping, and decoupling with `K = ζ = 0`.

## Requirements
- Python ≥ 3.10
- dolfinx (FEniCSx 0.9.x), ufl, basix
- mpi4py, petsc4py
- numpy

## How to run
```bash
python Test_f1.py



---


# Test_f2 — Decoupling Test B (φ-only evolution; u constant)

**Goal:** Verify curvature-driven collapse of a circular `φ ≈ +1` island in a `φ ≈ -1` matrix with `D=K=ζ=0`. Temperature `u` remains constant in space and time.

## Requirements
- Python ≥ 3.10
- dolfinx (FEniCSx 0.9.x), ufl, basix
- mpi4py, petsc4py
- numpy

## How to run
```bash
python Test_f2.py










