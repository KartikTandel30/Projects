from petsc4py import PETSc
from mpi4py import MPI

import os, time, ufl, numpy as np
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, fem
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, create_unit_square
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner
import pyvista as pv
import pyvistaqt as pvq
from dolfinx.mesh import locate_entities_boundary

# -------------------- Parameters --------------------
zet = 1.9        # Increased coupling parameter for stronger phase-temperature interaction
tau_0 = 1.0
lamda_0 = 1.0    # Back to original interface width
dt = 0.001        # Smaller timestep for stability
D = 1.5          # Original diffusion coefficient

# -------------------- Mesh --------------------
Lx, Ly = 100, 100
Nx, Ny = 200, 200
#msh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny], cell_type=CellType.triangle)
msh = create_unit_square(MPI.COMM_WORLD, Lx, Ly, CellType.triangle)
# φ: P2, u: P1
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))


w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)
com_0 = Function(ME)
phi, u     = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

# -------------------- Initial conditions --------------------
'''
def initial_phi(x):
    r = np.sqrt((x[0] - Lx/2.0)**2 + (x[1] - Ly/2.0)**2)
    return np.where(r < 2, 1.0, -1.0)
'''
def initial_phi(x):
    r = np.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2)
    return np.where(r < 0.05, 1.0, -1.0)

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
eps = 0.2    # Back to original anisotropy
eta = 0.00001     # Increased regularization

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
eps_row = PETSc.ScalarType(1e-6)  # Stronger regularization for matrix stability

R0 = (
    tau_eff * (phi - phi_0) * w_phi * dx
  + dt * df * w_phi * dx
  + dt * (lam_n**2) * ufl.inner(ufl.grad(w_phi), g) * dx
  + dt * Q * (w_phi.dx(0)*qx + w_phi.dx(1)*qy) * dx
  #+ eps_row * phi * w_phi * dx 
)

R1 = (
    (u - u_0) * w_u * dx
  - 0.5 * (phi - phi_0) * w_u * dx
  + dt * D * inner(grad(u), grad(w_u)) * dx
  #+ eps_row * u * w_u * dx
)

R = R0 + R1
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)

'''
# --- Dirichlet BC on temperature: u = -0.90 on outer boundary ---
tdim = msh.topology.dim
facets = locate_entities_boundary(msh, tdim-1, lambda x: np.full(x.shape[1], True, dtype=np.bool_))
dofs_u = fem.locate_dofs_topological(ME.sub(1), tdim-1, facets).astype(np.int32)
bc_u   = fem.dirichletbc(PETSc.ScalarType(-0.9), dofs_u, ME.sub(1))  # Increased boundary undercooling
if dofs_u.size == 0 and MPI.COMM_WORLD.rank == 0:
    raise RuntimeError("No boundary DOFs found for u. Check locate_entities_boundary().")
'''
# -------------------- Nonlinear solve --------------------
# Solver
problem = NonlinearProblem(R, com, bcs=[], J=J)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "residual"
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

#------------------- ParaView I/O (write EVERY step) --------------------
file = XDMFFile(MPI.COMM_WORLD, "output_task_3.xdmf", "w")
file.write_mesh(msh)


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
phi_sub = com.sub(0)
file.write_function(phi_sub, 0.0)
while t < T:
    t += dt
    step += 1
    res = solver.solve(com)
    print(f"Step {int(t/dt)}: num iteration: {res[0]}")
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    file.write_function(phi_sub, t)

    # live view
    grid.point_data["Phase"] = com.x.array[dof].real
    plotter.remove_actor("timelabel")
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()

# final write and close
file.close()

# optional static plot
grid.point_data["Phase"] = com.x.array[dof].real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = os.path.join(outdir, "phase_last.png")
pv.plot(grid, show_edges=True, screenshot=screenshot)

print("Simulation complete. Close the window to exit.")
while plotter.app.running:
    time.sleep(0.1)
