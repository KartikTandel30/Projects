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
from dolfinx.mesh import locate_entities_boundary

# -------------------- Parameters --------------------
zet = 1.9        # Increased coupling parameter for stronger phase-temperature interaction
tau_0 = 1.0
lamda_0 = 1.0    # Back to original interface width
dt = 0.01        # Smaller timestep for stability
D = 1.5          # Original diffusion coefficient

# -------------------- Mesh --------------------
Lx, Ly = 250, 250
Nx, Ny = 125, 125
msh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny], cell_type=CellType.triangle)

# φ: P2, u: P1
Pphi = element("Lagrange", msh.basix_cell(), 2, dtype=default_real_type)  # φ: P2
Pu   = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)  # u: P1
ME   = functionspace(msh, mixed_element([Pphi, Pu]))

w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)   # n+1
com_0 = Function(ME)   # n

phi, u     = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

# -------------------- Initial conditions --------------------
def initial_phi(x):
    r = np.sqrt((x[0] - Lx/2.0)**2 + (x[1] - Ly/2.0)**2)
    return np.where(r < 3, 1.0, -1.0)

def initial_u(x):
    return -0.9 * np.ones(x.shape[1], dtype=default_real_type)  # Increased undercooling

com.x.array[:] = 0.0
com.sub(0).interpolate(initial_phi)
com_0.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward()
com_0.x.scatter_forward()

# -------------------- Bulk free-energy derivative --------------------
df = -phi + phi**3 + zet * u * (1 - 2*phi**2 + phi**4)

# -------------------- Anisotropy (your |φ|⁴ variant, explicit split) --------------------
eps = 0.06    # Back to original anisotropy
eta = 0.1     # Increased regularization

g    = ufl.variable(ufl.grad(phi))
gx, gy = g[0], g[1]
g2   = ufl.inner(g, g)

den_phi = (phi*phi + eta)**2
a_s_phi = (1 - 3*eps) * (1 + (4*eps)/(1 - 3*eps) * (gx**4 + gy**4) / den_phi)

tau_eff = tau_0 * a_s_phi**2
lam_n   = lamda_0 * a_s_phi

qx = 16*eps * gx**3 / den_phi
qy = 16*eps * gy**3 / den_phi
Q  = (lamda_0**2) * g2 * a_s_phi

# -------------------- Weak forms --------------------
dxQ = dx(metadata={"quadrature_degree": 6})
eps_row = PETSc.ScalarType(1e-6)  # Stronger regularization for matrix stability

R0 = (
    tau_eff * (phi - phi_0) * w_phi * dxQ
  + dt * df * w_phi * dxQ
  + dt * (lam_n**2) * ufl.inner(ufl.grad(w_phi), g) * dxQ
  + dt * Q * (w_phi.dx(0)*qx + w_phi.dx(1)*qy) * dxQ
  + eps_row * phi * w_phi * dxQ 
)

R1 = (
    (u - u_0) * w_u * dxQ
  - 0.5 * (phi - phi_0) * w_u * dxQ
  + dt * D * inner(grad(u), grad(w_u)) * dxQ
  + eps_row * u * w_u * dxQ
)

R = R0 + R1
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)

# --- Dirichlet BC on temperature: u = -0.90 on outer boundary ---
tdim = msh.topology.dim
facets = locate_entities_boundary(msh, tdim-1, lambda x: np.full(x.shape[1], True, dtype=np.bool_))
dofs_u = fem.locate_dofs_topological(ME.sub(1), tdim-1, facets).astype(np.int32)
bc_u   = fem.dirichletbc(PETSc.ScalarType(-0.9), dofs_u, ME.sub(1))  # Increased boundary undercooling
if dofs_u.size == 0 and MPI.COMM_WORLD.rank == 0:
    raise RuntimeError("No boundary DOFs found for u. Check locate_entities_boundary().")

# -------------------- Nonlinear solve --------------------
opt = PETSc.Options()
opt["snes_type"] = "newtonls"
opt["snes_linesearch_type"] = "basic"  # Changed to basic linesearch
opt["snes_linesearch_damping"] = "0.5"  # More conservative damping
opt["snes_monitor_short"] = ""   # optional
opt["snes_max_it"] = "50"       # Increased max iterations

problem = NonlinearProblem(R, com, bcs=[bc_u], J=J)
solver  = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "residual"  # Changed to residual-based criterion
solver.rtol = 1e-6  # Relaxed tolerance
solver.atol = 1e-8  # Relaxed tolerance
solver.max_it = 50  # Increased max iterations
solver.report = True

# Linear solver (KSP/PC)
ksp = solver.krylov_solver
p = ksp.getOptionsPrefix()
opt[f"{p}ksp_type"] = "preonly"
opt[f"{p}pc_type"]  = "lu"
sys = PETSc.Sys()
if sys.hasExternalPackage("superlu_dist"):
    opt[f"{p}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{p}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

#------------------- ParaView I/O (write EVERY step) --------------------
outdir = "results_task3"
if MPI.COMM_WORLD.rank == 0 and not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)

# Collapse mixed subspaces
V_phi_high, map_phi = ME.sub(0).collapse()  # φ space is P2
V_u,        map_u   = ME.sub(1).collapse()  # u space is P1

# IO spaces and functions  (FIXED: use functionspace(...), not fem.FunctionSpace)
V_phi_io = functionspace(msh, ("Lagrange", 1))      # P1 "IO" space for φ
phi_high = Function(V_phi_high)                     # holder for φ (P2)
phi_io   = Function(V_phi_io); phi_io.name = "phi"  # what we write (P1)
u_out    = Function(V_u);       u_out.name  = "u"   # u already P1

xdmf_phi = XDMFFile(msh.comm, os.path.join(outdir, "phi_series.xdmf"), "w")
xdmf_u   = XDMFFile(msh.comm, os.path.join(outdir, "u_series.xdmf"), "w")
xdmf_phi.write_mesh(msh)
xdmf_u.write_mesh(msh)

def write_to_xdmf(t: float):
    # fill φ (P2) from mixed vector, then interpolate to P1 for output
    phi_high.x.array[:] = com.x.array[map_phi]
    phi_high.x.scatter_forward()
    phi_io.interpolate(phi_high)

    # fill u (P1) directly
    u_out.x.array[:] = com.x.array[map_u]
    u_out.x.scatter_forward()

    # write
    xdmf_phi.write_function(phi_io, t)
    xdmf_u.write_function(u_out, t)

# initial write (t=0)
write_to_xdmf(0.0)

# -------------------- Live viz --------------------
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
T = 50.0
step = 0
VIEW_EVERY = 100

while t < T:
    t += dt
    step += 1
    res = solver.solve(com)
    print(f"Step {int(t/dt)}: num iteration: {res[0]}")
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    write_to_xdmf(t)

    # live view
    grid.point_data["Phase"] = com.x.array[dof].real
    plotter.remove_actor("timelabel")
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()

# final write and close
write_to_xdmf(t)
xdmf_phi.close(); xdmf_u.close()

# optional static plot
grid.point_data["Phase"] = com.x.array[dof].real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = os.path.join(outdir, "phase_last.png")
pv.plot(grid, show_edges=True, screenshot=screenshot)

print("Simulation complete. Close the window to exit.")
while plotter.app.running:
    time.sleep(0.1)
