from petsc4py import PETSc
import os
import dolfinx
from mpi4py import MPI
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
import random
from ufl import dx, grad, inner, Identity, outer, as_vector, sqrt, dot
import numpy as np
import time
from numpy.random import default_rng  # ADD THIS
rng = default_rng(12345) 

t_start = time.time()
# ---------------- Parameters ----------------
zet = 1.6
tau_0 = 1
lamda_0 = 1
dt = 0.04
D = 1
STRIDE = 10  # save every STRIDE time steps
# Mesh
Lx, Ly = 500, 500
Nx, Ny = 250,250
msh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny],
                       cell_type=CellType.triangle)

P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)
com_0 = Function(ME)
phi, u     = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

'''
m = 4             # set to your anisotropy (2,4,6,...)
R0 = 5        # base radius (your seed)
epsR = 0.02       # 1–3% wobble
theta0 = 0.0      # rotation; use np.pi/4 for 45°

def initial_phi(x):
    xc = x[0] - Lx/2.0
    yc = x[1] - Ly/2.0
    r = np.sqrt(xc**2 + yc**2)
    theta = np.arctan2(yc, xc)
    R = R0 * (1.0 + epsR * np.cos(m * (theta - theta0)))
    return np.where(r < R, 1.0, -1.0)
'''
def initial_phi(x):
    r = np.sqrt((x[0] - Lx/2)**2 + (x[1] - Ly/2)**2)
    return np.where(r < 5, 1.0, -1.0)


def initial_u(x):
    return -0.75*np.ones(x.shape[1], dtype=default_real_type)

com.x.array[:] = 0
com.sub(0).interpolate(initial_phi)
com_0.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward()
com_0.x.scatter_forward()

phi_vec = com.sub(0).x.array
mask = np.clip(1.0 - phi_vec**2, 0.0, 1.0)
phi_vec += (5e-4) * mask * rng.standard_normal(phi_vec.shape)  # smaller amp and masked
np.clip(phi_vec, -1.0, 1.0, out=phi_vec)                       # keep in [-1,1]
com.sub(0).x.array[:] = phi_vec
com.x.scatter_forward()
com_0.x.array[:] = com.x.array
com_0.x.scatter_forward()

# Free-energy derivative
df = -phi + phi**3 + zet*u*(1 - 2*phi**2 + phi**4)

# ----------------- ANISOTROPY -----------------
eps_an = fem.Constant(msh, default_real_type(0.1))
eta    = fem.Constant(msh, default_real_type(1e-8))
K =  fem.Constant(msh, default_real_type(0.5))

gphi = grad(phi)
g2   = inner(gphi, gphi)
ng   = sqrt(g2 + eta**2)
nHat = gphi / ng

d = msh.geometry.dim
I = Identity(d)
P = I - outer(nHat, nHat)


a     = (1.0 - 3.0*eps_an) + 4.0*eps_an*(nHat[0]**4 + nHat[1]**4)
da_dn = as_vector((16.0*eps_an*nHat[0]**3, 16.0*eps_an*nHat[1]**3))

da_dg = dot(P, da_dn) / ng
q_phi = lamda_0**2 * (a**2 * gphi + g2 * a * da_dg)
F_grad_aniso = inner(q_phi, grad(w_phi)) * dx
tau = tau_0 * a**2
# ----------------------------------------------

# Weak forms
R0 = ( tau*(phi - phi_0)*w_phi*dx
     + dt*df*w_phi*dx
     + dt*F_grad_aniso )

R1 = ( (u - u_0)*w_u*dx
     - K*(phi - phi_0)*w_u*dx
     + dt*D*inner(grad(u), grad(w_u))*dx )

R = R0 + R1
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)

# --------- Solver — Field-split Schur (φ|u) with explicit index sets ---------
problem = NonlinearProblem(R, com, bcs=[], J=J)
solver = NewtonSolver(msh.comm, problem)

# You asked for "incremental" — add safe fallback to "residual" (0.9.0 only)
try:
    solver.convergence_criterion = "incremental"
except Exception:
    solver.convergence_criterion = "residual"

solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 200
solver.report = True

ksp = solver.krylov_solver
pc  = ksp.getPC()

# --- Build PETSc index sets for each subspace (needed in dolfinx 0.9.0) ---
Vphi, map_phi = ME.sub(0).collapse()
Vu,   map_u   = ME.sub(1).collapse()
is_phi = PETSc.IS().createGeneral(np.asarray(map_phi, dtype=np.int32))
is_u   = PETSc.IS().createGeneral(np.asarray(map_u,   dtype=np.int32))

# Tell PETSc how to split the mixed system
pc.setType(PETSc.PC.Type.FIELDSPLIT)
pc.setFieldSplitIS(("phi", is_phi), ("u", is_u))

# PETSc options
opt = PETSc.Options()
opt_prefix = ksp.getOptionsPrefix()

# Outer Krylov
opt[f"{opt_prefix}ksp_type"]    = "gmres"
opt[f"{opt_prefix}ksp_rtol"]    = 1e-8
opt[f"{opt_prefix}ksp_max_it"]  = 200

# Schur complement preconditioner
opt[f"{opt_prefix}pc_fieldsplit_type"]           = "schur"
opt[f"{opt_prefix}pc_fieldsplit_schur_fact_type"] = "lower"

# Block solvers:
#   φ-block (stiff interface operator): LU
opt[f"{opt_prefix}fieldsplit_phi_ksp_type"] = "preonly"
opt[f"{opt_prefix}fieldsplit_phi_pc_type"]  = "lu"

#   u-block (diffusion): AMG (hypre)
opt[f"{opt_prefix}fieldsplit_u_ksp_type"] = "preonly"
opt[f"{opt_prefix}fieldsplit_u_pc_type"]  = "hypre"

# Ensure matrix type compatible with AMG
opt[f"{opt_prefix}mat_type"] = "aij"

print("\n>>> Using Field-split Schur preconditioner (phi: LU, u: HYPRE)\n")
ksp.setFromOptions()


# ---------------- Output dir ----------------
file = XDMFFile(MPI.COMM_WORLD, "output_task_3_3.xdmf", "w")
file.write_mesh(msh)

# Time
t = 0.0
T = 500.0
step = 0
# Initial output fields (t=0)
phi_sub = com.sub(0)
file.write_function(phi_sub, 0.0)


while t < T:
    t += dt
    step += 1

    its, converged = solver.solve(com)
    print(f"Step {step}: Newton iterations = {its} ({'OK' if converged else 'NOT CONV'})")

    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    if step % STRIDE == 0:
        file.write_function(phi_sub, t)
        

if msh.comm.rank == 0:
    print("Open in ParaView. File -> Open -> output_task_3.xdmf")
if msh.comm.rank == 0:
    print(f"Total runtime: {time.time()-t_start:.2f}s")