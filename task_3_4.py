from petsc4py import PETSc
import dolfinx
from mpi4py import MPI
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, sqrt
import numpy as np
import time

# ---------------- Parameters (paper-faithful defaults) ----------------
zet = 1.6        # coupling ξ
tau_0 = 1.0      # kinetic time-scale τ0
lamda_0 = 1.0    # λ0
dt = 0.01        # Δt
D = 1.0          # thermal diffusivity
u_inf = -0.9     # initial undercooling
eps_val = 0.05   # ε4 (anisotropy strength)
eta_val = 1e-8   # regularization for |∇φ|
T = 500.0        # total simulation time

# ---------------- Mesh ----------------
Lx, Ly = 500.0, 500.0
Nx, Ny = 1000, 1000  # High resolution for arms!
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
    r = np.sqrt((x[0] - Lx/2)**2 + (x[1] - Ly/2)**2)
    return np.where(r < 10.0, 1.0, -1.0)  # r < 10 for larger seed

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
com.x.scatter_forward(); com_0.x.scatter_forward()

# Tiny symmetry-breaking perturbation to φ in the interface band
P0_phi, dof_phi = ME.sub(0).collapse()
phi_vals = com.x.array[dof_phi].copy()
mask = np.abs(phi_vals) < 0.9
phi_vals[mask] += 1e-3 * (rng.random(np.count_nonzero(mask)) - 0.5)
com.x.array[dof_phi] = phi_vals
com.x.scatter_forward()

# ---------------- Free-energy derivative ∂f/∂φ ----------------
df = -phi + phi**3 + zet*u*(1 - 2*phi**2 + phi**4)

# ---------------- Anisotropy and Chain Rule Terms ----------------
gphi = grad(phi)
g2 = inner(gphi, gphi)
ng = sqrt(g2 + eta_val)
nx, ny = gphi[0]/ng, gphi[1]/ng
theta = ufl.atan2(ny, nx)
a = 1.0 + eps_val * ufl.cos(4 * theta)
lam_n = lamda_0 * a
tau_n = tau_0 * a**2

# Chain rule derivatives (see finite difference code structure)
# cos(4θ), sin(4θ) in terms of (nx,ny)
cos4 = nx**4 - 6*nx**2*ny**2 + ny**4
sin4 = 4*nx*ny*(nx**2 - ny**2)
# Derivatives
dcos4_dn = ufl.as_vector((4*nx**3 - 12*nx*ny**2, 4*ny**3 - 12*ny*nx**2))
dsin4_dn = ufl.as_vector((12*nx**2*ny - 4*ny**3, 4*nx**3 - 12*nx*ny**2))
da_dn = eps_val * dcos4_dn  # θ₀ = 0, so sin4 term vanishes

I  = ufl.Identity(msh.geometry.dim)
P  = I - ufl.outer(as_vector([nx, ny]), as_vector([nx, ny]))
da_dg_x = (P[0, 0]*da_dn[0] + P[0, 1]*da_dn[1]) / ng
da_dg_y = (P[1, 0]*da_dn[0] + P[1, 1]*da_dn[1]) / ng

# Three terms
F1_lambda_sq = (lamda_0**2) * (a**2) * inner(grad(w_phi), grad(phi)) * dx
F2_chain_x   = (lamda_0**2) * g2 * a * ( w_phi.dx(0) * da_dg_x ) * dx
F3_chain_y   = (lamda_0**2) * g2 * a * ( w_phi.dx(1) * da_dg_y ) * dx
F_grad_aniso = F1_lambda_sq + F2_chain_x + F3_chain_y

# ---------------- Weak forms (using τ(n)) -----------------------
R0 = ( tau_n*(phi - phi_0)*w_phi*dx
     + dt*df*w_phi*dx
     + dt*F_grad_aniso )

R1 = ( (u - u_0)*w_u*dx
     - 0.5*(phi - phi_0)*w_u*dx
     + dt*D*inner(grad(u), grad(w_u))*dx )

R = R0 + R1

# Natural Neumann BCs (no Dirichlet BCs, as per the paper)
bcs = []

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
opt[f"{opt_prefix}snes_monitor"] = ""
sys = PETSc.Sys()
if sys.hasExternalPackage("superlu_dist"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# ---------------- Output to XDMF for ParaView ----------------
outdir = "results_dendrite"
if MPI.COMM_WORLD.rank == 0:
    import os
    os.makedirs(outdir, exist_ok=True)

V_phi, map_phi = ME.sub(0).collapse()
V_u,   map_u   = ME.sub(1).collapse()
phi_out = Function(V_phi)
u_out   = Function(V_u)
phi_out.name = "phi"
u_out.name = "u"

xdmf_phi = XDMFFile(msh.comm, f"{outdir}/phi.xdmf", "w")
xdmf_u = XDMFFile(msh.comm, f"{outdir}/u.xdmf", "w")
xdmf_phi.write_mesh(msh)
xdmf_u.write_mesh(msh)

def write_xdmf(t):
    phi_out.x.array[:] = com.x.array[map_phi]
    phi_out.x.scatter_forward()
    u_out.x.array[:] = com.x.array[map_u]
    u_out.x.scatter_forward()
    xdmf_phi.write_function(phi_out, t)
    xdmf_u.write_function(u_out, t)

write_xdmf(0.0)  # initial state

# ---------------- Time loop ----------------
t = 0.0
step = 0
while t < T:
    t += dt
    step += 1
    res = solver.solve(com)
    print(f"Step {step}: time={t:.2f}, Newton iters={res[0]}")
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    write_xdmf(t)
    # Optional: print progress every 100 steps
    if step % 10 == 0 and MPI.COMM_WORLD.rank == 0:
        print(f"Saved output at t={t:.2f}")

# Final write and close
write_xdmf(t)
xdmf_phi.close(); xdmf_u.close()

if MPI.COMM_WORLD.rank == 0:
    print("Simulation complete. Visualize 'results_dendrite/phi.xdmf' in ParaView or similar.")