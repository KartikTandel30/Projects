from petsc4py import PETSc
import dolfinx
from mpi4py import MPI

import os
import ufl
from basix.ufl import element
from dolfinx import default_real_type, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, inner
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import time
import matplotlib.pyplot as plt
from openpyxl import Workbook
import pathlib
from dolfinx.io import XDMFFile
import pandas as pd  # <-- added: simple CSV param loader

# ---------------- Param loader (CSV: key,value) ----------------
def _auto_cast(x):
    """Cast strings like '50', '0.01', '-0.6' to int/float when possible; else return str."""
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s

def load_params(csv_name="paraTask0.xlsx"):
    """Load parameters from a local CSV (key,value). Missing file/keys → defaults."""
    defaults = {
        "zet": 1.6,
        "u": -0.5,
        "tau0": 1.0,
        "lambda0": 1.0,
        "dt": 0.01,
        "T": 10.0,
        "Lx": 100.0,
        "Ly": 100.0,
        "nx": 50,
        "ny": 50,
        "init_phi_const": -0.6,   # constant φ(0)
    }
    p = pathlib.Path(csv_name)
    if not p.exists():
        print(f"[param] '{csv_name}' not found → using defaults:\n  {defaults}")
        return defaults
    try:
        dfp = pd.read_csv(p)
        if not {"key", "value"}.issubset(dfp.columns):
            raise ValueError("CSV must have columns: key,value")
        kv = {str(k).strip(): _auto_cast(v) for k, v in zip(dfp["key"], dfp["value"])}
        out = defaults.copy()
        out.update({k: kv[k] for k in kv if k in out})
        print("[param] Loaded parameters from", csv_name)
        for k in out:
            print(f"  {k} = {out[k]}")
        unknown = [k for k in kv if k not in defaults]
        if unknown:
            print("[param] (ignored unknown keys):", unknown)
        return out
    except Exception as e:
        print(f"[param] Failed to parse '{csv_name}' ({e}) → using defaults.")
        return defaults

# ---------------- Start simulation ----------------
t_start = time.time()

# Load parameters from CSV (same folder); print once for reproducibility
params = load_params("paraTask0.csv")
zet      = float(params["zet"])
u        = float(params["u"])
tau_0    = float(params["tau0"])
lamda_0  = float(params["lambda0"])   # kept for consistency if you use it later
dt       = float(params["dt"])
T        = float(params["T"])
Lx       = float(params["Lx"])
Ly       = float(params["Ly"])
nx       = int(params["nx"])
ny       = int(params["ny"])
phi0_c   = float(params["init_phi_const"])

# Create mesh (domain and resolution come from params)
msh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [Lx, Ly]],
    [nx, ny],
    cell_type=CellType.triangle
)

# FE space
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, P1)

# Trial/Test fields
w_phi = ufl.TestFunction(ME)
phi = Function(ME)
phi_0 = Function(ME)

# Initial condition (constant field from params)
def initial_phi(x):
    # creates a constant field for all nodes
    return phi0_c * np.ones(x.shape[1], dtype=default_real_type)

phi.interpolate(initial_phi)
phi_0.interpolate(initial_phi)
phi.x.scatter_forward()
phi_0.x.scatter_forward()

# Free energy derivative df/dφ (bulk only; no gradient term in this test)
df = -phi + phi**3 + zet * u * (1 - 2 * phi**2 + phi**4)

# Weak form (no gradient term)
R0 = (tau_0 * phi * w_phi * dx
      - tau_0 * phi_0 * w_phi * dx
      + dt * inner(df, w_phi) * dx)

# Solver setup
problem = NonlinearProblem(R0, phi)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
solver.atol = 1e-12
solver.max_it = 25
solver.report = True

ksp = solver.krylov_solver
opt = PETSc.Options()
opt_prefix = ksp.getOptionsPrefix()
opt[f"{opt_prefix}ksp_type"] = "preonly"
opt[f"{opt_prefix}pc_type"] = "lu"
opt[f"{opt_prefix}snes_monitor"] = ""
sys = PETSc.Sys()
if sys.hasExternalPackage("superlu_dist"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# Excel workbook setup (stores every node's ϕ and f at each time)
wb_all = Workbook()
ws_all = wb_all.active
ws_all.title = "All Node Data"
header = ["Time (s)"]
for i in range(len(phi.x.array)):
    header.append(f"ϕ_node_{i}")
    header.append(f"f_node_{i}")
ws_all.append(header)

# ---------------- Outputs (XDMF + live PyVista) ----------------

xdmf_path = "phi_task_0.xdmf"
xdmf = XDMFFile(msh.comm, str(xdmf_path), "w")
xdmf.write_mesh(ME.mesh)

# PyVista plot setup (note: requires a working Qt binding)
t = 0.0
topology, cell_types, x = plot.vtk_mesh(ME)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = phi.x.array.real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=True)
plotter.view_xy(True)
plotter.add_text(f"time:{t}", font_size=10, name="timelabel")

# ---------------- Time loop ----------------
while t < T:
    t += dt
    it, res = solver.solve(phi)
    print(f"Step {int(round(t / dt))}: Newton iters = {it}")
    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()

    # Live view update
    pmin = float(phi.x.array.min())
    pmax = float(phi.x.array.max())
    print(f"min(phi): {pmin:.4f}, max(phi): {pmax:.4f}")
    grid.point_data["Phase"] = phi.x.array.real
    plotter.remove_actor("timelabel")
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()

    # Write field to XDMF at current time
    xdmf.write_function(phi, t)

    # Export ϕ and f(ϕ,u) to Excel
    phi_vals_all = phi.x.array.real
    f_vals_all = (
        -0.5 * phi_vals_all**2
        + 0.25 * phi_vals_all**4
        + zet * u * phi_vals_all * (1 - (2/3) * phi_vals_all**2 + 0.2 * phi_vals_all**4)
    )
    row_all = [t]
    for phi_i, f_i in zip(phi_vals_all, f_vals_all):
        row_all.extend([phi_i, f_i])
    ws_all.append(row_all)

    # Early stop if all nodes reached a stable phase (±1 within tolerance)
    if np.allclose(np.abs(phi_vals_all), 1.0, atol=1e-3):
        print("All nodes reached a stable phase (±1). Ending early.")
        break

# Final snapshot
phi.x.scatter_forward()
grid.point_data["Phase"] = phi.x.array.real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = str(results_dir / "phase.png")
pv.plot(grid, show_edges=True, screenshot=screenshot)


wb_all.save("phiAndBulkData.xlsx")
xdmf.close()

if msh.comm.rank == 0:
    print(f"Total runtime: {time.time()-t_start:.2f}s")
    print(f"Saved XDMF to: {xdmf_path}")
    print("Saved Excel")


print("Simulation complete. Close the window to exit.")
plotter.app.exec_()
