from petsc4py import PETSc
import dolfinx
from mpi4py import MPI

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, fem
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import locate_dofs_geometrical  # geometry-based BC
from ufl import dx, grad, inner, Identity, outer, as_vector, sqrt, dot
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import time

# ---------------- Parameters (tuned for growth) ----------------
zet = 1.6        # coupling ξ
tau_0 = 0.30     # faster interface kinetics helps propagation
lamda_0 = 1.0    # interface thickness λ0
dt = 0.02        # smaller step = more robust with anisotropy
D = 1.0          # thermal diffusivity
u_inf = -0.80    # Dirichlet BC at outer boundary (sustained undercooling)
eps_val = 0.05   # anisotropy strength ε4
eta_val = 1e-8   # numerical regularization for |∇φ|
# ---------------------------------------------------------------

# Create mesh
Lx, Ly = 100.0, 100.0
Nx, Ny = 50, 50
msh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny],
                       cell_type=CellType.triangle)

P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

w_phi, w_u = ufl.TestFunctions(ME)  # tests
com   = Function(ME)                # unknowns at n+1
com_0 = Function(ME)                # values at n

phi, u       = ufl.split(com)
phi_0, u_0   = ufl.split(com_0)

# ---------------- Initial conditions ----------------
def initial_phi(x):
    r = np.sqrt((x[0] - 50.0)**2 + (x[1] - 50.0)**2)
    return np.where(r < 5.0, 1.0, -1.0)

rng = np.random.default_rng(42)
def initial_u(x):
    base = u_inf
    noise = 0.01 * (rng.random(x.shape[1]) - 0.5)   # ±0.005
    return (base + noise).astype(default_real_type)

com.x.array[:] = 0
com.sub(0).interpolate(initial_phi)
com_0.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward()
com_0.x.scatter_forward()

# ---------------- Free-energy derivative ∂f/∂φ ----------------
df = -phi + phi**3 + zet*u*(1 - 2*phi**2 + phi**4)

# ---------------- Anisotropy block (cubic/4-fold) --------------
eps_an = fem.Constant(msh, default_real_type(eps_val))
eta    = fem.Constant(msh, default_real_type(eta_val))

gphi = grad(phi)
g2   = inner(gphi, gphi)
ng   = sqrt(g2 + eta**2)
nHat = gphi / ng

d = msh.geometry.dim
I = Identity(d)
P = I - outer(nHat, nHat)

if d == 2:
    a     = (1.0 - 3.0*eps_an) + 4.0*eps_an*(nHat[0]**4 + nHat[1]**4)
    da_dn = as_vector((16.0*eps_an*nHat[0]**3,
                       16.0*eps_an*nHat[1]**3))
else:
    a     = (1.0 - 3.0*eps_an) + 4.0*eps_an*(nHat[0]**4 + nHat[1]**4 + nHat[2]**4)
    da_dn = as_vector((16.0*eps_an*nHat[0]**3,
                       16.0*eps_an*nHat[1]**3,
                       16.0*eps_an*nHat[2]**3))

da_dg = dot(P, da_dn) / ng
q_phi = lamda_0**2 * (a**2 * gphi + g2 * a * da_dg)
F_grad_aniso = inner(q_phi, grad(w_phi)) * dx
# ---------------------------------------------------------------

# ---------------- Weak forms ----------------------------------
R0 = ( tau_0*(phi - phi_0)*w_phi*dx
     + dt*df*w_phi*dx
     + dt*F_grad_aniso )

R1 = ( (u - u_0)*w_u*dx
     - 0.5*(phi - phi_0)*w_u*dx
     + dt*D*inner(grad(u), grad(w_u))*dx )

R = R0 + R1

# ---------------- Geometry-based Dirichlet BC on u -------------
def on_boundary(x):
    return np.logical_or.reduce((
        np.isclose(x[0], 0.0), np.isclose(x[0], Lx),
        np.isclose(x[1], 0.0), np.isclose(x[1], Ly)
    ))

Vu = ME.sub(1)                       # temperature subspace (mixed)
Vu_collapse, _ = Vu.collapse()       # standalone scalar space

# Prescribed boundary value (must live in the collapsed space)
u_bc_fun = fem.Function(Vu_collapse)
u_bc_fun.x.array[:] = u_inf

# KEY FIX: tuple (subspace, collapsed space)
dofs_u = locate_dofs_geometrical((Vu, Vu_collapse), on_boundary)

# Build BC on the subspace
bc_u = fem.dirichletbc(u_bc_fun, dofs_u, Vu)
# ---------------------------------------------------------------

# Jacobian
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)

# Solve
problem = NonlinearProblem(R, com, bcs=[bc_u], J=J)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
solver.atol = 1e-12
solver.max_it = 25
solver.report = True

# Linear solver options
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

# ---------------- Visualization ----------------
t = 0.0
T = 20

P0, dof = ME.sub(0).collapse()
topology, cell_types, x = plot.vtk_mesh(P0)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = com.x.array[dof].real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=True)
plotter.view_xy(True)
plotter.add_text(f"time:{t}", font_size=10, name="timelabel")

# ---------------- Time loop ----------------
while t < T:
    t += dt
    res = solver.solve(com)
    print(f"Step {int(t/dt)}: num iteration: {res[0]}")
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    grid.point_data["Phase"] = com.x.array[dof].real
    plotter.remove_actor("timelabel")
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()

com.x.scatter_forward()
grid.point_data["Phase"] = com.x.array[dof].real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = "phase.png"
pv.plot(grid, show_edges=True, screenshot=screenshot)

print("Simulation complete. Close the window to exit.")
while plotter.app.running:
    time.sleep(0.1)
