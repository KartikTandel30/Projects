from petsc4py import PETSc
import dolfinx
from mpi4py import MPI

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, plot, fem
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from dolfinx.fem import locate_dofs_topological, locate_dofs_geometrical, dirichletbc
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, sqrt, as_vector, atan
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import time

# ---------------- Parameters (paper-faithful defaults) ----------------
zet = 1.6        # coupling ξ
tau_0 = 1.0      # base kinetic time-scale τ0
lamda_0 = 1.0    # λ0
dt = 0.02        # Δt
D = 1.0          # thermal diffusivity
u_inf = -0.75    # far-field undercooling (we'll clamp on boundary)
eps_val = 0.05   # ε4 (anisotropy strength)
eta_val = 1e-8   # regularization for |∇φ|
m_val  = 4       # m-fold symmetry (int; 4 for dendrites)
T = 20.0
# ---------------------------------------------------------------------

# Mesh
Lx, Ly = 100.0, 100.0
Nx, Ny = 175, 175
msh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny],
                       cell_type=CellType.triangle)

P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

# Unknowns / tests
w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)
com_0 = Function(ME)
phi, u     = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

# ---------------- Initial conditions ----------------
def initial_phi(x):
    r = np.sqrt((x[0] - 50.0)**2 + (x[1] - 50.0)**2)
    return np.where(r < 5.0, 1.0, -1.0)

rng = np.random.default_rng(42)
def initial_u(x):
    return -0.75*np.ones(x.shape[1], dtype=default_real_type)

com.x.array[:] = 0
com.sub(0).interpolate(initial_phi)
com_0.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward(); com_0.x.scatter_forward()

# tiny symmetry-breaking perturbation to φ in the interface band
P0_phi, dof_phi = ME.sub(0).collapse()
phi_vals = com.x.array[dof_phi].copy()
mask = np.abs(phi_vals) < 0.9
phi_vals[mask] += 1e-3 * (rng.random(np.count_nonzero(mask)) - 0.5)
com.x.array[dof_phi] = phi_vals
com.x.scatter_forward()

# ---------------- Free-energy derivative ∂f/∂φ ----------------
df = -phi + phi**3 + zet*u*(1 - 2*phi**2 + phi**4)

# ---------------- Angle-form anisotropy (KEEPING YOUR FORM) ----
eps_an = fem.Constant(msh, default_real_type(eps_val))
eta    = fem.Constant(msh, default_real_type(eta_val))
m_int  = int(m_val)

gphi = grad(phi)                 # ∇φ = (φx, φy)
px, py = phi.dx(0), phi.dx(1)    # your style
g2   = inner(gphi, gphi)         # |∇φ|^2
ng   = sqrt(g2 + eta*eta)        # |∇φ|_η
theta = atan(py / px)            # <-- unchanged (your choice)

sin_m = ufl.sin(m_int * theta)
cos_m = ufl.cos(m_int * theta)
# ∂a/∂(∇φ) using chain rule with |∇φ|_η^2 in the denominator
da_dg_x = eps_an * m_int * sin_m * py / (ng**2)
da_dg_y = -eps_an * m_int * sin_m * px / (ng**2)

a       = 1.0 + eps_an * cos_m
tau_n   = tau_0 * a**2

# ---- Three separate integrals exactly like the paper ----
F1_lambda_sq = (lamda_0**2) * (a**2) * inner(grad(w_phi), grad(phi)) * dx
F2_chain_x   = (lamda_0**2) * g2 * a * ( w_phi.dx(0) * da_dg_x ) * dx
F3_chain_y   = (lamda_0**2) * g2 * a * ( w_phi.dx(1) * da_dg_y ) * dx
F_grad_aniso = F1_lambda_sq + F2_chain_x + F3_chain_y

# ---------------- Weak forms -----------------------
R0 = ( tau_n*(phi - phi_0)*w_phi*dx
     + dt*df*w_phi*dx
     + dt*F_grad_aniso )

R1 = ( (u - u_0)*w_u*dx
     - 0.5*(phi - phi_0)*w_u*dx
     + dt*D*inner(grad(u), grad(w_u))*dx )

R = R0 + R1

# ---------------- Dirichlet BC on u at outer boundary (robust) -----------
bcs = []
fdim = msh.topology.dim - 1
def on_box_coords(x):
    return np.logical_or.reduce((
        np.isclose(x[0], 0.0), np.isclose(x[0], Lx),
        np.isclose(x[1], 0.0), np.isclose(x[1], Ly)
    ))

# 1) mark boundary facets topologically
facets = locate_entities_boundary(msh, fdim, on_box_coords)
print(f"[BC] facets on outer box: {facets.size if hasattr(facets,'size') else len(facets)}")

Vu = ME.sub(1)
Vu_c, _ = Vu.collapse()

# 2) map facets -> u dofs (topological)
dofs_u = locate_dofs_topological((Vu, Vu_c), fdim, facets)
dofs_u = np.asarray(dofs_u, dtype=np.int32).ravel()
print(f"[BC] u Dirichlet DoFs (topological): {len(dofs_u)}")

# 3) fallback: geometrical locator if none found
if len(dofs_u) == 0:
    def on_box_geom(x):
        return np.logical_or.reduce((
            np.isclose(x[0], 0.0), np.isclose(x[0], Lx),
            np.isclose(x[1], 0.0), np.isclose(x[1], Ly)
        ))
    dofs_u = locate_dofs_geometrical((Vu, Vu_c), on_box_geom)
    dofs_u = np.asarray(dofs_u, dtype=np.int32).ravel()
    print(f"[BC] u Dirichlet DoFs (geometrical): {len(dofs_u)}")

# 4) last resort: pin ONE u DoF at a corner to kill constant mode
if len(dofs_u) == 0:
    def at_corner(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    dofs_u = locate_dofs_geometrical((Vu, Vu_c), at_corner)
    dofs_u = np.asarray(dofs_u, dtype=np.int32).ravel()[:1]
    print(f"[BC] u pinned DoFs (corner): {len(dofs_u)}")

# --- Use Constant for BC value to match dolfinx signature ---
u_const = fem.Constant(msh, default_real_type(u_inf))
bc_u = dirichletbc(u_const, dofs_u, Vu)
bcs = [bc_u]
# -----------------------------------------------------------------------

# Jacobian & solver
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)
problem = NonlinearProblem(R, com, bcs=bcs, J=J)

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
opt[f"{opt_prefix}pc_type"]  = "lu"
sys = PETSc.Sys()
if sys.hasExternalPackage("superlu_dist"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# ---------------- Visualization ----------------
t = 0.0
P0, dof = ME.sub(0).collapse()
topology, cell_types, x = plot.vtk_mesh(P0)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = com.x.array[dof].real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=False)
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
