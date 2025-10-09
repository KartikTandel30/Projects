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
D = 1.5
STRIDE = 10  # save every STRIDE time steps
# Mesh
Lx, Ly = 250, 250
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

R0 = 5.0
w_eq = np.sqrt(2.0) * lamda_0 

def initial_phi(x):
    xc = x[0] - Lx/2.0
    yc = x[1] - Ly/2.0
    r  = np.sqrt(xc**2 + yc**2)
    return np.tanh((R0 - r) / w_eq)

'''
def initial_phi(x):
    r = np.sqrt((x[0] - Lx/2)**2 + (x[1] - Ly/2)**2)
    return np.where(r < random.uniform(4.9, 5.1), 1.0, -1.0)
'''

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
eps_an = fem.Constant(msh, default_real_type(0.08))
eta    = fem.Constant(msh, default_real_type(1e-8))
K =  fem.Constant(msh, default_real_type(1.0))

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

# Solver---lu 
problem = NonlinearProblem(R, com, bcs=[], J=J)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-8
solver.atol = 1e-10
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

print("\n>>> Using Direct LU solver (SuperLU / MUMPS)\n")
ksp.setFromOptions()

# ---------------- Output dir ----------------
file = XDMFFile(MPI.COMM_WORLD, "output_task_3_4.xdmf", "w")
file.write_mesh(msh)

# Time
t = 0.0
T = 500.0
step = 0
# Initial output fields (t=0)
phi_sub = com.sub(0)
phi_sub.name = "phi"
file.write_function(phi_sub, 0.0)

print("Starting time-simulation...")
while t < T:
    t += dt
    step += 1

    its, converged = solver.solve(com)
    #print(f"Step {step}: Newton iterations = {its} ({'OK' if converged else 'NOT CONV'})")

    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    if step % STRIDE == 0:
        file.write_function(phi_sub, t)
        

if msh.comm.rank == 0:
    print("Open in ParaView. File -> Open -> output_task_3_4.xdmf")
if msh.comm.rank == 0:
    print(f"Total runtime: {time.time()-t_start:.2f}s")