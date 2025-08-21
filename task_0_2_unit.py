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

# --- Parameters (as you had) ---
zet = 1.6     # coupling constant (λ)
u = -0.75      # Temperature / undercooling (flip sign as you wish)
tau_0 = 1     # Characteristic time scale
lamda_0 = 1   # Interface thickness ε
dt = 0.04     # time step

# Helper: which phase is favored for the current u  # NEW
def favored_phase(u, zet):
    return "solid (+1)" if zet * u < 0 else "liquid (-1)"

# --- Mesh ---
Lx, Ly = 100.0, 100.0
Nx, Ny = 175, 175
msh = create_rectangle(MPI.COMM_WORLD,
                       [[0.0, 0.0], [Lx, Ly]],
                       [Nx,Ny],
                       cell_type=CellType.triangle)

hx = Lx / Nx
edge_tol = 1e-6*hx

P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)

ME = functionspace(msh, P1)

w_phi = ufl.TestFunction(ME)
phi   = Function(ME, name="phi")
phi_0 = Function(ME, name="phi_0")

# =========================
#   INITIALIZATION (NEW)
# =========================

def init_phi_half_sharp(phi_fn, Lx, Ly):
    def step(x):
        left = x[0] < 0.5*Lx
        return np.where(left, 1.0,  -1.0)
    phi_fn.interpolate(step)

def report_fraction(phi_fn, label):
    frac_solid = (phi_fn.x.array > 0).mean()
    if np.isnan(frac_solid):  
        frac_solid = 0.0
    if msh.comm.rank == 0:
        print(f"{label}: solid fraction ≈ {frac_solid:.3f}")

# --- choose one init ---
INIT_MODE = "sharp"   # "sharp" or "smooth"
ORI = "vertical"

init_phi_half_sharp(phi, Lx, Ly)

phi_0.x.array[:] = phi.x.array
phi.x.scatter_forward(); phi_0.x.scatter_forward()
report_fraction(phi, f"Init ({INIT_MODE})")
if msh.comm.rank == 0:
    print(f"Favored phase for u={u:+.3f}: {favored_phase(u, zet)}")  # NEW

# =========================
#   VARIATIONAL FORM
# =========================
df = -phi + phi**3 + zet * u * (1 - 2*phi**2 + phi**4)

# Backward Euler Allen–Cahn 
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
#   VISUALIZATION
# =========================
t = 0.0
T = 40.0
topology, cell_types, x = plot.vtk_mesh(ME)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = phi.x.array.real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=False)
plotter.view_xy(True)
plotter.add_text(f"time:{t}", font_size=10, name="timelabel")

# =========================
#   INTERFACE TRACKING (NEW)
# =========================
area = Lx * Ly
X, Y = SpatialCoordinate(msh)
times, fsolids, iface_pos, iface_speed = [], [], [], []
prev_pos = None

def interface_pos(phi_fn, Lx, Ly):
    """
    Vertical interface; solid (+1) on RIGHT.
    I = (φ+1)/2 is solid indicator. Solid area As = ∫ I dΩ.
    For a flat cut at x* with solid on the right:  As = (Lx - x*) * Ly
    => x* = Lx - As/Ly
    """
    I = 0.5 * (phi_fn + 1.0)
    As = float(assemble_scalar(form(I * dx)))
    return Lx - As / Ly

xdmf = XDMFFile(msh.comm, "phase.xdmf", "w")
xdmf.write_mesh(msh)

# =========================
#   TIME LOOP
# =========================
while t < T:
    t += dt
    n_it, converged = solver.solve(phi)
    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()
    xdmf.write_function(phi, t)
    # — interface metrics (NEW) —
    mass_solid = assemble_scalar(form(0.5 * (phi + 1.0) * dx))
    f_solid = float(mass_solid / area)
    pos = interface_pos(phi, Lx, Ly)
    hit_left  = pos <= edge_tol           # interface near x=0
    hit_right = (Lx - pos) <= edge_tol    # (if you ever grow the other way)

    if hit_left or hit_right:
        side = "left" if hit_left else "right"
        print(f"Stopping: interface reached {side} edge (x*={pos:.4g}, tol={edge_tol:.4g}) at t={t:.4g}")
        break
    v = 0.0 if prev_pos is None else (pos - prev_pos) / dt
    prev_pos = pos

    times.append(t); fsolids.append(f_solid); iface_pos.append(pos); iface_speed.append(v)

    print(f"Step {int(round(t/dt))}: iters={n_it}, "
          f"min(phi)={phi.x.array.min():.4f}, max(phi)={phi.x.array.max():.4f}, "
          f"favored={favored_phase(u, zet)}, f_solid={f_solid:.4f}, "
          f"{'x*' if ORI=='vertical' else 'y*'}={pos:.3f}, v={v:.4e}")

    grid.point_data["Phase"] = phi.x.array.real
    plotter.remove_actor("timelabel")
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()

xdmf.close()
# =========================
#   SAVE TRACK (NEW)
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
