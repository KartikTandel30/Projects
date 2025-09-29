from petsc4py import PETSc
from mpi4py import MPI

import os, time, ufl, numpy as np
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, fem
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner
import pyvista as pv
import pyvistaqt as pvq

# -------------------- Parameters --------------------
zet = 1.6
tau_0 = 1.0
lamda_0 = 1.0
dt = 0.04
D = 1.0

# -------------------- Mesh --------------------
Lx, Ly = 500.0, 500.0
Nx, Ny = 1000, 1000   # big; reduce if memory is tight
msh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny], cell_type=CellType.triangle)

P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)   # n+1
com_0 = Function(ME)   # n

phi, u     = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

# -------------------- Initial conditions --------------------
def initial_phi(x):
    r = np.sqrt((x[0] - Lx/2.0)**2 + (x[1] - Ly/2.0)**2)
    return np.where(r < 5.0, 1.0, -1.0)

def initial_u(x):
    return -0.75 * np.ones(x.shape[1], dtype=default_real_type)

com.x.array[:] = 0.0
com.sub(0).interpolate(initial_phi)
com_0.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward()
com_0.x.scatter_forward()

# -------------------- Bulk free-energy derivative --------------------
df = -phi + phi**3 + zet * u * (1 - 2*phi**2 + phi**4)

# -------------------- Anisotropy (4-fold, no rotation) --------------------
eps = 0.05
eta = 1e-12

g   = ufl.variable(ufl.grad(phi))       # ∇φ as UFL variable
g2  = ufl.inner(g, g)                   # |∇φ|^2
rg  = ufl.sqrt(g2 + eta)                # regularized magnitude

nx, ny = g[0]/rg, g[1]/rg
cos4   = 1.0 - 8.0 * nx**2 * ny**2
a_s    = 1.0 + eps * cos4               # |eps| < 1/3

lam_n  = lamda_0 * a_s                  # λ(n) = λ0 a_s
Fgrad  = 0.5 * (lam_n**2) * g2          # ½ λ(n)^2 |∇φ|^2
q      = ufl.diff(Fgrad, g)             # λ^2 ∇φ + |∇φ|^2 λ ∂λ/∂(∇φ)

tau_eff = tau_0 * a_s**2                # anisotropic kinetics (use tau_0 if undesired)

# -------------------- Weak forms --------------------
R0 = (
    tau_eff * (phi - phi_0) * w_phi * dx
  + dt * df * w_phi * dx
  + dt * ufl.inner(q, ufl.grad(w_phi)) * dx
)

R1 = (
    (u - u_0) * w_u * dx
  - 0.5 * (phi - phi_0) * w_u * dx
  + dt * D * inner(grad(u), grad(w_u)) * dx
)

R = R0 + R1
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)

# -------------------- Nonlinear solve --------------------
problem = NonlinearProblem(R, com, bcs=[], J=J)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
solver.atol = 1e-12
solver.max_it = 25
solver.report = True

ksp = solver.krylov_solver
opt = PETSc.Options(); opt_prefix = ksp.getOptionsPrefix()
opt[f"{opt_prefix}ksp_type"] = "preonly"
opt[f"{opt_prefix}pc_type"] = "lu"
opt[f"{opt_prefix}snes_monitor"] = ""
sys = PETSc.Sys()
if sys.hasExternalPackage("superlu_dist"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# -------------------- ParaView I/O (write EVERY step) --------------------
outdir = "results_task3"
if MPI.COMM_WORLD.rank == 0 and not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)

V_phi, map_phi = ME.sub(0).collapse()
V_u,   map_u   = ME.sub(1).collapse()
phi_out = Function(V_phi); phi_out.name = "phi"
u_out   = Function(V_u);   u_out.name   = "u"

xdmf_phi = XDMFFile(msh.comm, os.path.join(outdir, "phi_series.xdmf"), "w")
xdmf_u   = XDMFFile(msh.comm, os.path.join(outdir, "u_series.xdmf"), "w")
xdmf_phi.write_mesh(msh)
xdmf_u.write_mesh(msh)

def write_to_xdmf(t):
    phi_out.x.array[:] = com.x.array[map_phi]
    u_out.x.array[:]   = com.x.array[map_u]
    xdmf_phi.write_function(phi_out, t)
    xdmf_u.write_function(u_out, t)

# initial write (t=0)
write_to_xdmf(0.0)

# -------------------- Live viz (update ONLY every 100 steps) --------------------
P0, dof = ME.sub(0).collapse()
topology, cell_types, x = plot.vtk_mesh(P0)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = com.x.array[dof].real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=False)
plotter.view_xy(True)
plotter.add_text("time: 0.00", font_size=10, name="timelabel")

# -------------------- Time loop --------------------
t = 0.0
T = 20.0
step = 0
VIEW_EVERY = 100  # refresh PyVista every 100 steps

while t < T:
    t += dt
    step += 1
    res = solver.solve(com)
    print(f"Step {int(t/dt)}: num iteration: {res[0]}")
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    write_to_xdmf(t)
    # ---- Update live view ONLY every 100 steps ----
    if step % VIEW_EVERY == 0:
        grid.point_data["Phase"] = com.x.array[dof].real
        plotter.remove_actor("timelabel")
        plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
        plotter.app.processEvents()


write_to_xdmf(t)
xdmf_phi.close(); xdmf_u.close()

# final static plot (optional off-screen)
grid.point_data["Phase"] = com.x.array[dof].real
screenshot = None

if pv.OFF_SCREEN:
    screenshot = os.path.join(outdir, "phase_last.png")
pv.plot(grid, show_edges=True, screenshot=screenshot)

print("Simulation complete. Close the window to exit.")
while plotter.app.running:
    time.sleep(0.1)
