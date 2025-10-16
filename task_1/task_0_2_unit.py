from petsc4py import PETSc
import dolfinx
from mpi4py import MPI

import ufl
from basix.ufl import element
from dolfinx import default_real_type, plot
from dolfinx.fem import Function, functionspace, assemble_scalar, form, locate_dofs_geometrical
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, SpatialCoordinate
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import csv                                  # <<< CHG
from pathlib import Path
import time
import csv
import json
import argparse
import pathlib
# Parameters


# Minimal JSON loader: read 'parameter_to.json' located next to this script.
param_file = pathlib.Path(__file__).resolve().parent / "parameter.json"
if not param_file.exists():
    raise RuntimeError(
        f"Parameter file not found: {param_file}\nCreate a JSON file named 'parameter_to.json' next to this script with keys: dt, Lx, Ly, Nx, Ny, T, zet, u, tau_0, lamda_0"
    )

with param_file.open("r", encoding="utf-8") as fh:
    params = json.load(fh)

required = ["dt", "Lx", "Ly", "Nx", "Ny", "T", "zet", "u", "tau_0", "lamda_0"]
missing = [k for k in required if k not in params]
if missing:
    raise RuntimeError(f"Missing parameter keys in {param_file}: {missing}")

# assign parameters (minimal casting)
dt = float(params["dt"]) 
Lx = float(params["Lx"]) 
Ly = float(params["Ly"]) 
Nx = int(params["Nx"]) 
Ny = int(params["Ny"]) 
T = float(params["T"]) 
zet = float(params["zet"]) 
u = float(params["u"]) 
tau_0 = float(params["tau_0"]) 
lamda_0 = float(params["lamda_0"]) 
# =========================
#   HELPERS
# =========================
def favored_phase(u_, zet_):
    return "solid (+1)" if zet_ * u_ < 0 else "liquid (-1)"

# =========================
#   MESH / SPACE
# =========================
msh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny], cell_type=CellType.triangle)

hx = Lx / Nx
edge_margin = 0.0 

P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, P1)
w_phi = ufl.TestFunction(ME)
phi   = Function(ME, name="phi")
phi_0 = Function(ME, name="phi_0")

# =========================
#   INITIALIZATION
# =========================
def init_phi_half(phi_fn, Lx_, Ly_):
    def step(x):
        left = x[0] < 0.5 * Lx_
        return np.where(left, 1.0, -1.0)
    phi_fn.interpolate(step)

def report_fraction(phi_fn, label):
    frac_solid = (phi_fn.x.array > 0).mean()
    frac_solid = 0.0 if np.isnan(frac_solid) else frac_solid
    if msh.comm.rank == 0:
        print(f"{label}: solid fraction ≈ {frac_solid:.3f}")

INIT_MODE = "sharp"
init_phi_half(phi, Lx, Ly)
phi_0.x.array[:] = phi.x.array
phi.x.scatter_forward(); phi_0.x.scatter_forward()

SOLID_ON_LEFT = True  # your IC sets +1 on the LEFT; keep it simple

report_fraction(phi, f"Init ({INIT_MODE})")
if msh.comm.rank == 0:
    print(f"Favored phase for u={u:+.3f}: {favored_phase(u, zet)}")

# =========================
#   VARIATIONAL FORM
# =========================
df = -phi + phi**3 + zet * u * (1 - 2 * phi**2 + phi**4)  # ∂f/∂φ

R0 = ( tau_0 * phi * w_phi * dx
      - tau_0 * phi_0 * w_phi * dx
      + dt * inner(df, w_phi) * dx
      + lamda_0**2 * dt * inner(grad(phi), grad(w_phi)) * dx )

# =========================
#   SOLVER
# =========================
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
sys = PETSc.Sys()
if sys.hasExternalPackage("superlu_dist"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# =========================
#   INTERFACE TRACKING + ENERGY + THICKNESS
# =========================
area = Lx * Ly
X, Y = SpatialCoordinate(msh)
times, fsolids, iface_pos, iface_speed = [], [], [], []
prev_pos = None

def interface_pos(phi_fn, Lx_, Ly_):
    """
    Oriented front position: increases in the direction of advance.
    If solid starts on the LEFT:  x* = As / Ly
    If solid starts on the RIGHT: x* = Lx - As / Ly
    """
    I = 0.5 * (phi_fn + 1.0)               # solid indicator
    As = float(assemble_scalar(form(I * dx)))
    return (As / Ly_) if SOLID_ON_LEFT else (Lx_ - As / Ly_)

def f_bulk(phi_sym):
    return (-0.5 * phi_sym**2 + 0.25 * phi_sym**4
            + zet * u * phi_sym * (1.0 - (2.0/3.0) * phi_sym**2 + (1.0/5.0) * phi_sym**4))

E_form = ( f_bulk(phi) + 0.5 * lamda_0**2 * inner(grad(phi), grad(phi)) ) * dx
energies, times_E = [], []

# centerline for thickness (φ = ±0.8)
gdim = msh.geometry.cmap.dim
coords_all = ME.tabulate_dof_coordinates().reshape((-1, gdim))
xy_all = coords_all[:, :2]
y_target = 0.5 * Ly
row_tol  = 0.5 * (Ly / Ny)
row_dofs = locate_dofs_geometrical(ME, lambda x: np.isclose(x[1], y_target, atol=row_tol))
if row_dofs.size == 0:
    raise RuntimeError("centerline locate_dofs_geometrical returned no dofs; increase row_tol.")
x_row_unsorted = xy_all[row_dofs, 0]
order = np.argsort(x_row_unsorted)
row_dofs_sorted = row_dofs[order]
x_row = x_row_unsorted[order]

thicknesses, times_T = [], []

def threshold_crossing_x(xv, yvals, thresh):
    for i in range(len(xv) - 1):
        y0, y1 = yvals[i], yvals[i + 1]
        if (y0 - thresh) * (y1 - thresh) <= 0:
            alpha = 0.0 if y1 == y0 else (thresh - y0) / (y1 - y0)
            return xv[i] + alpha * (xv[i + 1] - xv[i])
    return None

# MPI-safe helpers & interior mask for saturation tests           # <<< CHG
comm = msh.comm
coords_xy = coords_all[:, :2]
margin = edge_margin
interior_mask = (coords_xy[:, 0] > margin) & (coords_xy[:, 0] < (Lx - margin))
nloc_interior = int(np.count_nonzero(interior_mask))

def global_fraction_close(phi_arr, target=+1.0, tol=1e-3):          # <<< CHG
    ok_local = np.count_nonzero(interior_mask & (np.abs(phi_arr - target) <= tol))
    ok_glob  = comm.allreduce(ok_local, op=MPI.SUM)
    n_glob   = comm.allreduce(nloc_interior, op=MPI.SUM)
    return (ok_glob / max(1, n_glob))

# =========================
#   OUTPUT
# =========================
xdmf = XDMFFile(msh.comm, "phase_grad.xdmf", "w")
xdmf.write_mesh(msh)
t = 0.0
# =========================
#   TIME LOOP
# =========================
while t < T:
    t += dt
    n_it, converged = solver.solve(phi)
    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()
    xdmf.write_function(phi, t)

    # area-based interface position and solid fraction
    mass_solid = assemble_scalar(form(0.5 * (phi + 1.0) * dx))
    f_solid = float(mass_solid / area)
    pos = interface_pos(phi, Lx, Ly)

    # centerline thickness
    phi_vals = phi.x.array[row_dofs_sorted]
    x_m08 = threshold_crossing_x(x_row, phi_vals, -0.8)
    x_p08 = threshold_crossing_x(x_row, phi_vals, +0.8)
    thickness = np.nan if (x_m08 is None or x_p08 is None) else abs(x_p08 - x_m08)

    # total energy
    E = float(assemble_scalar(form(E_form)))

    # log this step (keep last point even if we stop)
    times.append(t); fsolids.append(f_solid); iface_pos.append(pos)
    v = 0.0 if prev_pos is None else (pos - prev_pos) / dt
    iface_speed.append(v); prev_pos = pos
    energies.append(E); times_E.append(t)
    thicknesses.append(thickness); times_T.append(t)

    # ---------- robust early-exit decisions (MPI-safe) ----------   # <<< CHG
    hit_left  = pos <= edge_margin
    hit_right = (Lx - pos) <= edge_margin

    eps_area = 1e-4
    sat_area_solid = (f_solid >= 1.0 - eps_area)
    sat_area_liq   = (f_solid <= eps_area)

    frac_plus  = global_fraction_close(phi.x.array, target=+1.0, tol=1e-3)
    frac_minus = global_fraction_close(phi.x.array, target=-1.0, tol=1e-3)
    sat_point_solid = (frac_plus  >= 0.999)
    sat_point_liq   = (frac_minus >= 0.999)

    reason = None
    if hit_left or hit_right:
        reason = f"interface near {'left' if hit_left else 'right'} edge (margin={edge_margin:.3g})"
    elif sat_area_solid or sat_area_liq:
        reason = f"area saturation: f_solid={f_solid:.6f}"
    elif sat_point_solid or sat_point_liq:
        which = "+1" if sat_point_solid else "-1"
        frac  = frac_plus if sat_point_solid else frac_minus
        reason = f"pointwise saturation in interior: φ≈{which} ({frac:.5f})"

    if reason:
        if comm.rank == 0:
            print(f"[STOP] {reason} at t={t:.4g}, x*={pos:.4g}, E={E:.6e}, δ0.8={thickness:.3g}")
        break
    # --------------------------------------------------------------



xdmf.close()

# =========================
#   SAVE TRACKS
# =========================
data_iface = np.column_stack([times, fsolids, iface_pos, iface_speed])
np.savetxt("interface_track.csv", data_iface,
           header="time,f_solid,interface_pos,interface_speed",
           delimiter=",", comments="")
print("Saved interface_track.csv (time,f_solid,interface_pos,interface_speed)")

data_energy = np.column_stack([np.asarray(times_E), np.asarray(energies)])
np.savetxt("energy_track.csv", data_energy,
           header="time,total_energy", delimiter=",", comments="")
print("Saved energy_track.csv (time,total_energy)")

data_thick = np.column_stack([np.asarray(times_T), np.asarray(thicknesses)])
np.savetxt("thickness_track.csv", data_thick,
           header="time,thickness_phi_pm0.8", delimiter=",", comments="")
print("Saved thickness_track.csv (time,thickness_phi_pm0.8)")



print("Simulation complete. Close the window to exit.")

