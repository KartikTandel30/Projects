from petsc4py import PETSc
import dolfinx
from mpi4py import MPI

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import time

# --- Parameters (as you had) ---
zet = 1.6     # coupling constant (λ)
u = 0.75     # Temperature 
tau_0 = 1     # Characteristic time scale
lamda_0 = 1   # Interface thickness ε
dt = 0.04     # time step

# --- Mesh ---
Lx, Ly = 100.0, 100.0
msh = create_rectangle(MPI.COMM_WORLD,
                       [[0.0, 0.0], [Lx, Ly]],
                       [50, 50],
                       cell_type=CellType.triangle)                
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, P1)

w_phi = ufl.TestFunction(ME)
phi   = Function(ME, name="phi")
phi_0 = Function(ME, name="phi_0")

# =========================
#   INITIALIZATION (NEW)
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
INIT_MODE = "smooth"   # "sharp" or "smooth"
ORI       = "vertical" # "vertical" or "horizontal"
SOLID_ON  = "left"     # for vertical: left/right, for horizontal: bottom/top

if INIT_MODE == "sharp":
    init_phi_half_sharp(phi, Lx, Ly, orientation=ORI, solid_on=SOLID_ON)
else:
    init_phi_half_smooth(phi, Lx, Ly, eps=lamda_0, orientation=ORI, solid_on=SOLID_ON)

phi_0.x.array[:] = phi.x.array
phi.x.scatter_forward(); phi_0.x.scatter_forward()
report_fraction(phi, f"Init ({INIT_MODE})")

# =========================
#   VARIATIONAL FORM
# =========================
# dF/dφ = g'(φ) + λ u h'(φ) with g'(φ)=-φ+φ^3, h'(φ)=1-2φ^2+φ^4
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
T = 20.0
topology, cell_types, x = plot.vtk_mesh(ME)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = phi.x.array.real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=True)
plotter.view_xy(True)
plotter.add_text(f"time:{t}", font_size=10, name="timelabel")

# =========================
#   TIME LOOP
# =========================
while t < T:
    t += dt
    n_it, converged = solver.solve(phi)
    print(f"Step {int(round(t/dt))}: num iteration: {n_it}")
    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()
    print(f"min(phi): {phi.x.array.min():.4f}, max(phi): {phi.x.array.max():.4f}")
    grid.point_data["Phase"] = phi.x.array.real
    plotter.remove_actor("timelabel")
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()

phi.x.scatter_forward()
grid.point_data["Phase"] = phi.x.array.real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = "phase.png"
pv.plot(grid, show_edges=True, screenshot=screenshot)

print("Simulation complete. Close the window to exit.")
while plotter.app.running:
    time.sleep(0.1)
