from petsc4py import PETSc
from mpi4py import MPI
import os, time, numpy as np, ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, default_real_type
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from ufl import dx, grad, inner, Identity, outer, as_vector, sqrt, dot, atan2

# -------------------- run parameters (stable defaults) --------------------
# KR98-like discretization on your 250x250 grid
Lx = Ly = 250.0
Nx = Ny = 250

# Physical/nondimensional knobs
Delta   = 0.65                         # undercooling (u = -Delta)
tau_0   = 1.0                          # base kinetic scale
lamda_0 = 2.5                          # W0; with dx=1 → dx/W0 = 0.4
zet_val = 1.596                        # thin-interface coupling λ (IVF)
D_th    = 1.0                          # thermal diffusivity
K_lat   = 0.5                          # coupling in heat equation
mfold   = 6.0                          # 6-fold dendrite
eps_val = 0.022                        # anisotropy (safe for m=6)
theta_c = 0.0                          # orientation offset (radians)

# Time control
dt0 = 0.016                            # base Δt (KR98 used 0.016 τ0)
dt_min = 2e-4                          # allow halving down here
grow_factor, grow_every, its_ok = 1.5, 3, 4

# Seed
R0 = 5.0                               # initial radius
# -------------------------------------------------------------------------

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
t_start = time.time()

# -------------------- mesh & function spaces --------------------
msh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny],
                       cell_type=CellType.triangle)

P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)
com_0 = Function(ME)
phi, u     = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

# -------------------- initial conditions --------------------
def initial_phi(x):
    xc, yc = x[0]-Lx/2.0, x[1]-Ly/2.0
    r = np.sqrt(xc**2 + yc**2)
    # equilibrium tanh profile; half-width ≈ sqrt(2)*W0
    w_eq = np.sqrt(2.0) * lamda_0
    return np.tanh((R0 - r)/w_eq)

def initial_u(x):
    return -Delta*np.ones(x.shape[1], dtype=default_real_type)

com.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(0).interpolate(initial_phi)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward(); com_0.x.scatter_forward()

# -------------------- constants --------------------
eps_an = fem.Constant(msh, default_real_type(eps_val))
eta    = fem.Constant(msh, default_real_type(1e-7))
K      = fem.Constant(msh, default_real_type(K_lat))
zet    = fem.Constant(msh, default_real_type(zet_val))
dt     = fem.Constant(msh, default_real_type(dt0))
mC     = fem.Constant(msh, default_real_type(mfold))

# -------------------- free-energy derivative --------------------
df = -phi + phi**3 + zet*u*(1 - 2*phi**2 + phi**4)

# -------------------- anisotropy a(θ)=1+ε cos[m(θ-θc)] --------------------
gphi = grad(phi)
g2   = inner(gphi, gphi)
ng   = sqrt(g2 + eta**2)

# Unit normal n and projector P
nHat  = gphi / ng
normn = sqrt(nHat[0]*nHat[0] + nHat[1]*nHat[1] + default_real_type(1e-14))
n     = as_vector((nHat[0]/normn, nHat[1]/normn))
I     = Identity(msh.geometry.dim)
P     = I - outer(n, n)

theta = atan2(n[1], n[0])
a     = 1.0 + eps_an * ufl.cos(mC * (theta - theta_c))

# da/d(∇φ) = (P/|∇φ|_η) · da/dn  with  dθ/dn = (-ny, nx)
tangent = as_vector((-n[1], n[0]))
da_dn   = -eps_an * mC * ufl.sin(mC * (theta - theta_c)) * tangent
da_dg   = dot(P, da_dn) / ng

q_phi = lamda_0**2 * (a**2 * gphi + g2 * a * da_dg)
F_grad_aniso = inner(q_phi, grad(w_phi)) * dx
tau = tau_0 * a**2

# -------------------- residuals --------------------
R0 = ( tau*(phi - phi_0)*w_phi*dx
     + dt*df*w_phi*dx
     + dt*F_grad_aniso )

R1 = ( (u - u_0)*w_u*dx
     - K*(phi - phi_0)*w_u*dx
     + dt*D_th*inner(grad(u), grad(w_u))*dx )

R = R0 + R1

# -------------------- solver --------------------
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)
problem = NonlinearProblem(R, com, bcs=[], J=J)

solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.relaxation_parameter  = 0.6
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
solver.atol = 1e-12
solver.max_it = 200
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

# -------------------- output (append-and-close) --------------------
phi_sub = com.sub(0); phi_sub.name = "phi"
u_sub   = com.sub(1); u_sub.name   = "u"

out = "output_m6_angle.xdmf"
with XDMFFile(MPI.COMM_WORLD, out, "w") as xf:
    xf.write_mesh(msh)
    xf.write_function(phi_sub, 0.0)
    # xf.write_function(u_sub, 0.0)  # add u if you want

def save_frame(t_save: float):
    with XDMFFile(MPI.COMM_WORLD, out, "a") as xf:
        xf.write_function(phi_sub, t_save)
        # xf.write_function(u_sub, t_save)

# -------------------- time loop with adaptive dt --------------------
t = 0.0
step = 0
success_streak = 0

while t < 500.0:
    # Newton with halve-and-retry
    while True:
        try:
            its, converged = solver.solve(com)
            if not converged:
                raise RuntimeError("Newton not converged")
            break
        except Exception as e:
            new_dt = 0.5 * float(dt.value)
            if new_dt < dt_min:
                if msh.comm.rank == 0:
                    print(f"[FAIL] {e}. dt<{dt_min}. Writing last good frame at t={t:.4g} and exiting.")
                save_frame(t)
                raise
            if msh.comm.rank == 0:
                print(f"[retry] Newton failed → dt {float(dt.value):.4g} → {new_dt:.4g}")
            dt.value = new_dt
            # revert to last good state
            com.x.array[:] = com_0.x.array
            com.x.scatter_forward()
            success_streak = 0

    # success → advance
    t += float(dt.value)
    step += 1
    

    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()

    # grow dt back if solves are easy
    if its <= its_ok:
        success_streak += 1
    else:
        success_streak = 0
    if success_streak >= grow_every and float(dt.value) < dt0:
        old = float(dt.value)
        dt.value = min(grow_factor * old, dt0)
        success_streak = 0
        if msh.comm.rank == 0:
            print(f"[grow] dt {old:.4g} → {float(dt.value):.4g}")

    # periodic save
    if step % 10 == 0:
        save_frame(t)
        if msh.comm.rank == 0:
            print(f"Step {step}: Newton iterations = {its} (OK)  dt={float(dt.value):.4g}")

if msh.comm.rank == 0:
    print("Open in ParaView → output_m6_angle.xdmf → Apply → time slider.")
    print(f"Total runtime: {time.time()-t_start:.2f}s")
