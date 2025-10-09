from petsc4py import PETSc
import dolfinx
from mpi4py import MPI
import os
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, plot, fem
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, Identity, outer, as_vector, sqrt, sin, cos, atan2, dot
from dolfinx.io import XDMFFile
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import time
from numpy.random import default_rng  # ADD THIS
rng = default_rng(12345)

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

t_start = time.time()
# ---------------- Parameters (paper-faithful defaults) ----------------
zet = 1.6        # coupling ξ
tau_0 = 1.0      # base kinetic time-scale τ0
lamda_0 = 1.0    # λ0
       # Δt
D = 1.0          # thermal diffusivity
t = 0.0
T = 500.0
step = 0
dt0 = 0.02

dt_min = 1e-3
dt_max = dt0
STRIDE = 10       # save every STRIDE time steps
# ---------------------------------------------------------------------

# Mesh
Lx, Ly = 250.0, 250.0
Nx, Ny = 250, 250
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
    return np.where(r < 5.0, 1.0, -1.0)

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

# ---------------- Anisotropy via angle a(θ)=1+ε cos(mθ) --------
eps_an = fem.Constant(msh, default_real_type(0.05))
eta    = fem.Constant(msh, default_real_type(1e-6))
K =  fem.Constant(msh, default_real_type(0.5))
m =  fem.Constant(msh, default_real_type(6))# reuse your 'm' variable: set m = 4 or 6 above
dt  = fem.Constant(msh, default_real_type(dt0))
theta_c = 0.0           # rotate arms by this angle

gphi = grad(phi)
g2   = inner(gphi, gphi)
ng   = sqrt(g2 + eta**2)
nHat = gphi / ng
nx, ny = nHat[0]+ eta, nHat[1] + eta  # avoid exact zeros

d = msh.geometry.dim
I = Identity(d)
P = I - outer(nHat, nHat)

# Angle and anisotropy (general m)
theta   = ufl.atan2(ny, nx)                      # only used to evaluate a
a       = 1.0 + eps_an * ufl.cos(m * (theta - theta_c))
da_dn = -eps_an * m * ufl.sin(m * (theta - theta_c)) * as_vector((-ny, nx))

# Chain rule to ∂a/∂(∇φ): da/dg = (∂n/∂g)^T·(da/dn) = (P/|∇φ|_η)·da_dn
da_dg = dot(P, da_dn) / ng

# Gradient contribution and kinetic prefactor
q_phi = lamda_0**2 * (a**2 * gphi + g2 * a * da_dg)
F_grad_aniso = inner(q_phi, grad(w_phi)) * dx
tau = tau_0 * a**2


# ---------------- Weak forms (using τ(n)) -----------------------
R0 = ( tau*(phi - phi_0)*w_phi*dx
     + dt*df*w_phi*dx
     + dt*F_grad_aniso )

R1 = ( (u - u_0)*w_u*dx
     - K*(phi - phi_0)*w_u*dx
     + dt*D*inner(grad(u), grad(w_u))*dx )

R = R0 + R1

# Natural Neumann BCs (paper case): no Dirichlet BCs
bcs = []

# Jacobian & solver
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)
problem = NonlinearProblem(R, com, bcs=bcs, J=J)

solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
solver.atol = 1e-12
solver.max_it = 200
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


out = "output_task_3_angle.xdmf"
with XDMFFile(MPI.COMM_WORLD, out, "w") as xf:
    xf.write_mesh(msh)
    xf.write_function(com.sub(0), 0.0)   # phi at t=0

def save_frame(t_save: float):
    with XDMFFile(MPI.COMM_WORLD, out, "a") as xf:
        xf.write_function(com.sub(0), t_save)


# ---------- Time loop with adaptive dt (shrink/grow) ----------
t = 0.0
step = 0
dt_min = 1e-3
dt_max = dt0
grow_factor = 1.5
grow_every  = 3
its_ok      = 4
success_streak = 0

while t < T:
    retries = 0
    while True:
        try:
            its, converged = solver.solve(com)
            if not converged:
                raise RuntimeError("Newton not converged")
            break  # success
        except Exception as e:
            new_dt = 0.5 * float(dt.value)
            if new_dt < dt_min:
                # write last good and exit cleanly
                t_good = t
                if msh.comm.rank == 0:
                    print(f"[FAIL] {e}. dt<{dt_min}. Writing last good frame t={t_good:.4g} and exiting.")
                save_frame(t_good)
                raise
            if msh.comm.rank == 0:
                print(f"[retry] Newton failed → dt {float(dt.value):.4g} → {new_dt:.4g}")
            dt.value = new_dt
            # revert to last good state
            com.x.array[:] = com_0.x.array
            com.x.scatter_forward()
            success_streak = 0
            retries += 1

    # success → advance time and state
    t += float(dt.value)
    step += 1
    if msh.comm.rank == 0:
        print(f"Step {step}: Newton iterations = {its} (OK)  dt={float(dt.value):.4g}")

    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()

    # grow dt back when solves are easy
    if its <= its_ok:
        success_streak += 1
    else:
        success_streak = 0

    if success_streak >= grow_every and float(dt.value) < dt_max:
        old = float(dt.value)
        dt.value = min(grow_factor * old, dt_max)
        success_streak = 0
        if msh.comm.rank == 0:
            print(f"[grow] dt {old:.4g} → {float(dt.value):.4g}")

    if step % STRIDE == 0:
        save_frame(t)


if msh.comm.rank == 0:
    print("Open in ParaView → output_task_3_angle.xdmf → Apply → time slider.")
    print(f"Total runtime: {time.time()-t_start:.2f}s")