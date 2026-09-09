#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task3_f1.py — Monolithic φ–u phase-field (2D), isotropic (ε=0) by default

Model (non-dimensional, Backward–Euler in time):
    τ0 φ̇ = -μ,   μ = ∂f/∂φ(φ,u) - ∇·Q(∇φ)       (anisotropy optional)
    u̇  = D ∇²u + K φ̇                           (latent-heat coupling)

Weak residuals each step (tests w_φ, w_u):
    R_φ = ∫ [ τ( n̂ )(φ-φ0) w_φ ] dx
          + Δt ∫ [ (∂f/∂φ)(φ,u) w_φ ] dx
          + Δt ∫ [ Q(∇φ) · ∇w_φ ] dx

    R_u = ∫ [ (u - u0) - ΔtK(φ - φ0) ] w_u dx
          + Δt ∫ [ D ∇u · ∇w_u ] dx

Here:
  • ζ=0 ⇒ bulk f is independent of u (decouples thermodynamics from u)
  • K=0 ⇒ no latent-heat coupling (set K≈0.5 to enable)
  • ε=0 ⇒ isotropic interface penalty; set ε>0 to turn on four-fold anisotropy

I/O: writes φ,u time series to OUT_DIR every STRIDE steps (XDMF+HDF5).
BCs: natural (zero-flux) for φ and u.
"""

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
from numpy.random import default_rng  #for reproducible noise
rng = default_rng(12345) 

OUT_DIR = "out_task3_f1"
comm = MPI.COMM_WORLD
rank = comm.rank
t_start = time.time()


# ---------------- Parameters ----------------
zet = 0     # coupling ξ
tau_0 = 1.0        # characteristic time-scale
lambda_0 = 1.0      # interface width
dt = 0.01          # time step size
D = 1              # diffusion coeff
u0 = 0         # initial undercooling
T = 40.0          # total simulation time
STRIDE = 5  # save every STRIDE time steps
# Mesh
Lx, Ly = 100, 100    #  domain size
Nx, Ny = 100, 100    # number of elements 
r0 = 3           # initial solid seed  radius
eps_an = 0.00
K = 0.0

#--------Mesh-------------

msh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny],
                       cell_type=CellType.triangle)


#-------Function spaces --------------    
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)
com_0 = Function(ME)
phi, u     = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

#----------------   Initial conditions --------------
w_eq = np.sqrt(2.0) * lambda_0
def initial_phi(x):
    r = np.sqrt((x[0]-Lx/2.0)**2 + (x[1]-Ly/2.0)**2)
    return np.tanh((r0 - r) / w_eq)

def initial_u(x):
    # nice & smooth to visualize
    return np.cos(np.pi * x[0] / Lx) * np.cos(np.pi * x[1] / Ly)


#--------------- Initialize fields --------------
com.x.array[:] = 0
com.sub(0).interpolate(initial_phi)
com_0.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward()
com_0.x.scatter_forward()

# Add small random noise to phi to trigger dynamics

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
epsan = fem.Constant(msh, default_real_type(eps_an))
eta    = fem.Constant(msh, default_real_type(1e-8))
k =  fem.Constant(msh, default_real_type(K))

gphi = grad(phi)
g2   = inner(gphi, gphi)
ng   = sqrt(g2 + eta**2)
nHat = gphi / ng

d = msh.geometry.dim
I = Identity(d)
P = I - outer(nHat, nHat)


a     = (1.0 - 3.0*epsan) + 4.0*epsan*(nHat[0]**4 + nHat[1]**4)
da_dn = as_vector((16.0*epsan*nHat[0]**3, 16.0*epsan*nHat[1]**3))

da_dg = dot(P, da_dn) / ng
q_phi = lambda_0**2 * (a**2 * gphi + g2 * a * da_dg)
F_grad_aniso = inner(q_phi, grad(w_phi)) * dx
tau = tau_0 * a**2
# ----------------------------------------------

#--------------------Weak forms---------------------
R0 = ( tau*(phi - phi_0)*w_phi*dx
     + dt*df*w_phi*dx
     + dt*F_grad_aniso )

R1 = ( (u - u_0)*w_u*dx
     - dt*k*(phi - phi_0)*w_u*dx
     + dt*D*inner(grad(u), grad(w_u))*dx )

R = R0 + R1
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)


# -----------------solver setup -----------------
# Solver---lu 
problem = NonlinearProblem(R, com, bcs=[], J=J)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 100
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
phi_series = XDMFFile(comm, f"{OUT_DIR}/phi.xdmf", "w")
u_series   = XDMFFile(comm, f"{OUT_DIR}/u.xdmf", "w")
phi_series.write_mesh(msh)
u_series.write_mesh(msh)

V_phi, map_phi = ME.sub(0).collapse()
V_u,   map_u   = ME.sub(1).collapse()
phi_out = fem.Function(V_phi)
u_out   = fem.Function(V_u)

# ---------------- Time-stepping ----------------
# Time
t = 0.0

step = 0

if msh.comm.rank == 0:
    print("[run] Starting simulation…")

while t < T:
    t += dt
    step += 1
    solver.solve(com)  # silent

    # advance "previous" state and scatter
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    
    if STRIDE > 0 and step % STRIDE == 0:
        phi_out.x.array[:] = com.x.array[map_phi]
        u_out.x.array[:]   = com.x.array[map_u]
        phi_series.write_function(phi_out, t)
        u_series.write_function(u_out, t)
    '''
    phi_vals_all = com.x.array[map_phi]
    if np.allclose(np.abs(phi_vals_all), 1.0, atol=1e-3):
        print("All nodes reached a stable phase (±1). Ending early.")
        break
    '''


#----------------- Finalize output -----------------
phi_series.close()
u_series.close()
com.x.scatter_forward()
phi_fn = com.sub(0)
u_fn   = com.sub(1)


#----------------- Done -----------------
if msh.comm.rank == 0:
    print("Simulation finished successfully.")
    print(time.time() - t_start, "seconds elapsed.")
