from petsc4py import PETSc
import dolfinx
from mpi4py import MPI

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem import assemble_scalar, form  # NEW
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, SpatialCoordinate  # NEW
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import time

# --- Parameters ---
zet = 1.6                 # coupling constant (λ)
u = -0.75                  # Temperature / undercooling (flip sign as you wish)
tau_0 = 1                 # Characteristic time scale
lamda_0 = 1               # Interface thickness ε
dt = 0.04                 # time step

def favored_phase(u, zet):  # NEW
    return "solid (+1)" if zet * u < 0 else "liquid (-1)"

# --- Mesh ---
Lx, Ly = 100.0, 100.0
msh = create_rectangle(MPI.COMM_WORLD,
                       [[0.0, 0.0], [Lx, Ly]],
                       [20, 20],
                       cell_type=CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, P1)

w_phi = ufl.TestFunction(ME)
phi   = Function(ME, name="phi")
phi_0 = Function(ME, name="phi_0")

# =========================
#   INITIALIZATION
# =========================
def init_phi_half_sharp(phi_fn, Lx, Ly, orientation="vertical", solid_on="left"):
    # +1 solid, -1 liquid
    if orientation.lower() == "vertical":
        def step(x):
            left = x[0] < 0.5 * Lx
            if solid_on.lower() in ("left", "+x"):
                return np.where(left, 1.0, -1.0)
            else:
                return np.where(left, -1.0, 1.0)
        phi_fn.interpolate(step)
    else:  # horizontal
        def step(x):
            bottom = x[1] < 0.5 * Ly
            if solid_on.lower() in ("bottom", "-y"):
                return np.where(bottom, 1.0, -1.0)
            else:
                return np.where(bottom, -1.0, 1.0)
        phi_fn.interpolate(step)

def init_phi_half_smooth(phi_fn, Lx, Ly, eps, orientation="vertical", solid_on="left"):
    # Smooth tanh interface; width ~ sqrt(2)*eps (good for Newton)
    root2eps = np.sqrt(2.0) * eps
    if orientation.lower() == "vertical":
        def prof(x):
            s = (x[0] - 0.5 * Lx) / root2eps
            return -np.tanh(s) if solid_on.lower() in ("left", "+x") else np.tanh(s)
        phi_fn.interpolate(prof)
    else:
        def prof(x):
            s = (x[1] - 0.5 * Ly) / root2eps
            return -np.tanh(s) if solid_on.lower() in ("bottom", "-y") else np.tanh(s)
        phi_fn.interpolate(prof)

def report_fraction(phi_fn, label):
    frac_solid = (phi_fn.x.array > 0).mean()
    if np.isnan(frac_solid):  # parallel safety
        frac_solid = 0.0
    if msh.comm.rank == 0:
        print(f"{label}: solid fraction ≈ {frac_solid:.3f}")

# --- choose one init ---
INIT_MODE = "smooth"      # "sharp" or "smooth"
ORI       = "vertical"    # "vertical" or "horizontal"
SOLID_ON  = "right"       # <-- CHANGED: solid on RIGHT (liquid on left)

if INIT_MODE == "sharp":
    init_phi_half_sharp(phi, Lx, Ly, orientation=ORI, solid_on=SOLID_ON)
else:
    init_phi_half_smooth(phi, Lx, Ly, eps=lamda_0, orientation=ORI, solid_on=SOLID_ON)

phi_0.x.array[:] = phi.x.array
phi.x.scatter_forward(); phi_0.x.scatter_forward()
report_fraction(phi, f"Init ({INIT_MODE})")
if msh.comm.rank == 0:
    print(f"Favored phase for u={u:+.3f}: {favored_phase(u, zet)}")  # NEW

# =========================
#   VARIATIONAL FORM
# =========================
# dF/dφ = g'(φ) + λ u h'(φ) with g'(φ)=-φ+φ^3, h'(φ)=1-2φ^2+φ^4
df = -phi + phi**3 + zet * u * (1 - 2*phi**2 + phi**4)

# Backward Euler Allen–Cahn (kept as you had)
R0 = ( tau_0*phi*w_phi*dx 
      - tau_0*phi_0*w_phi*dx
      + dt*inner(df, w_phi)*dx 
      + lamda_0**2*dt*inner(grad(phi), grad(w_phi))*dx )

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
opt[f"{opt_prefix}snes_monitor"] = ""
sys = PETSc.Sys()
if sys.hasExternalPackage("superlu_dist"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# =========================
#   VISUALIZATION  (build once, keep handles)
# =========================
t = 0.0
T = 20.0
VIS_EVERY = 1  # set >1 (e.g., 5 or 10) to throttle rendering

topology, cell_types, x = plot.vtk_mesh(ME)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = np.ascontiguousarray(phi.x.array.real)
grid.set_active_scalars("Phase")

plotter = pvq.BackgroundPlotter(title="Phase", auto_update=False)
actor = plotter.add_mesh(
    grid, scalars="Phase", name="phi",
    clim=[-1, 1], cmap="coolwarm",
    show_edges=False, smooth_shading=True
)
plotter.show_grid(True)
time_txt = plotter.add_text("time: 0.00", font_size=10)
plotter.view_xy(True)
plotter.render()


# =========================
#   INTERFACE TRACKING
# =========================
area = Lx * Ly
X, Y = SpatialCoordinate(msh)
times, fsolids, iface_pos, iface_speed = [], [], [], []
prev_pos = None

def interface_pos(phi_fn, Lx, Ly, orientation="vertical"):
    """
    Estimate flat interface position from moments of the smooth indicator I=(φ+1)/2.
    Works whether solid is on left/right (or bottom/top).
    """
    I = 0.5 * (phi_fn + 1.0)
    M0 = float(assemble_scalar(form(I * dx)))  # ~ solid area
    if M0 <= 0.0:
        return 0.0
    if orientation == "vertical":
        M1 = float(assemble_scalar(form(X * I * dx)))   # ∫ x I dΩ
        xL = 2.0 * M1 / M0                  # assume solid-left
        xR = Lx - (M0 / Ly)                 # assume solid-right
        xL_valid = 0.0 <= xL <= Lx
        xR_valid = 0.0 <= xR <= Lx
        if xL_valid and xR_valid:
            errL = abs(M1 - (Ly * xL**2) / 2.0)
            errR = abs(M1 - (Ly * (Lx**2 - xR**2)) / 2.0)
            return xL if errL <= errR else xR
        return xL if xL_valid else xR
    else:
        M1 = float(assemble_scalar(form(Y * I * dx)))   # ∫ y I dΩ
        yB = 2.0 * M1 / M0
        yT = Ly - (M0 / Lx)
        yB_valid = 0.0 <= yB <= Ly
        yT_valid = 0.0 <= yT <= Ly
        if yB_valid and yT_valid:
            errB = abs(M1 - (Lx * yB**2) / 2.0)
            errT = abs(M1 - (Lx * (Ly**2 - yT**2)) / 2.0)
            return yB if errB <= errT else yT
        return yB if yB_valid else yT

# =========================
#   TIME LOOP
# =========================
while t < T:
    t += dt
    n_it, converged = solver.solve(phi)
    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()

    # interface metrics
    mass_solid = assemble_scalar(form(0.5 * (phi + 1.0) * dx))
    f_solid = float(mass_solid / area)
    pos = interface_pos(phi, Lx, Ly, ORI)
    v = 0.0 if prev_pos is None else (pos - prev_pos) / dt
    prev_pos = pos

    times.append(t); fsolids.append(f_solid); iface_pos.append(pos); iface_speed.append(v)

    print(f"Step {int(round(t/dt))}: iters={n_it}, "
          f"min(phi)={phi.x.array.min():.4f}, max(phi)={phi.x.array.max():.4f}, "
          f"favored={favored_phase(u, zet)}, f_solid={f_solid:.4f}, "
          f"{'x*' if ORI=='vertical' else 'y*'}={pos:.3f}, v={v:.4e}")

    # refresh the window every VIS_EVERY steps
    # refresh the window every VIS_EVERY steps
    step = int(round(t/dt))
    if step % VIS_EVERY == 0:
        new_vals = np.ascontiguousarray(phi.x.array.real)
        grid.point_data["Phase"] = new_vals
        plotter.update_scalars(new_vals, mesh=grid, render=False)  # push to actor
        time_txt.SetText(2, f"time: {t:.2e}")
        plotter.app.processEvents()
        plotter.render()



# =========================
#   SAVE TRACK
# =========================
data = np.column_stack([times, fsolids, iface_pos, iface_speed])
np.savetxt("interface_track.csv", data,
           header="time,f_solid,interface_pos,interface_speed",
           delimiter=",", comments="")
print("Saved interface_track.csv with columns: time, f_solid, interface_pos, interface_speed")

phi.x.scatter_forward()
grid.point_data["Phase"] = phi.x.array.real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = "phase.png"
pv.plot(grid, show_edges=True, screenshot=screenshot)

print("Simulation complete. Close the window to exit.")
while plotter.app.running:
    time.sleep(0.1)
