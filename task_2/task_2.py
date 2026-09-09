#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task_2.py — Coupled phase-field (φ) + diffusion (u) with implicit time stepping (dolfinx)

Overview
--------
Solves a standard coupled Allen–Cahn / diffusion system on a 2D rectangle using
P1 Lagrange elements with a mixed space [φ, u]. The scheme is fully implicit and
includes a gradient term for φ (interface energy) and a linear diffusion term for u.

Weak forms (implicit Euler, test functions w_φ, w_u):
  Let df/dφ = -φ + φ^3 + ζ u (1 - 2 φ^2 + φ^4). Then:
    R_φ = ( τ0(φ^{n+1} - φ^{n}), w_φ )
        + ( dt * df/dφ(φ^{n+1}, u^{n+1}), w_φ )
        + ( dt * λ0^2 ∇φ^{n+1} · ∇w_φ ) = 0

    R_u = ( u^{n+1} - u^n, w_u )
        - ( 0.5 (φ^{n+1} - φ^n), w_u )
        + ( dt * D ∇u^{n+1} · ∇w_u ) = 0

Boundary conditions
-------------------
No essential (Dirichlet) BCs are imposed; the gradient terms imply natural
zero-flux Neumann BCs for both φ and u.

Outputs
-------
- XDMF time series for φ and u in folder OUT_DIR (created if missing).
- Optional diagnostics via CoupledDiagnostics (CSV/PNGs under its own folder).

Notes
-----
- PETSc linear solve uses direct LU (MUMPS or SuperLU_DIST if available).
- All parameters are hard-coded below.
- Uses WRITE_EVERY to thin XDMF output frequency.

"""

from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np

from basix.ufl import element, mixed_element
from dolfinx import default_real_type
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner
from dolfinx import fem
from test2 import CoupledDiagnostics
from test_bc import bc_check


'''
# ---------------- parameters ----------------
'''

zet = 1.6    # coupling constant
tau_0 = 1.0  # characteristic time scale
lamda_0 = 1.0  # characteristic interface thickness
dt = 0.04   # time step
D = 1.0     # diffusion coefficient for u
Lx = 200.0   # domain size
Ly = 200.0          
Nx = 150  # mesh resolution
Ny = 150   
T = 40.0   # total simulation time

WRITE_EVERY = 10         # write XDMF every N steps (set 0 to disable)
OUT_DIR = "out_task2"    # where XDMFs go

comm = MPI.COMM_WORLD
rank = comm.rank

'''
# ---------------- mesh & spaces -------------
'''
msh = create_rectangle(
    comm,
    [[0.0, 0.0], [Lx, Ly]],
    [Nx, Ny],
    cell_type=CellType.triangle,
)

P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)               # unknowns at n+1: [φ, u]
com_0 = Function(ME)               # previous state: [φⁿ, uⁿ]
phi, u = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

'''
# ---------------- initials ------------------
'''
def initial_phi(x):
    r = np.sqrt((x[0] - Lx/2.0)**2 + (x[1] - Ly/2.0)**2)
    return np.where(r < 2.5, 1.0, -1.0)

def initial_u(x):
    return -0.75*np.ones(x.shape[1], dtype=default_real_type)

com.x.array[:] = 0.0
com.sub(0).interpolate(initial_phi)
com_0.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward()
com_0.x.scatter_forward()

'''
# ---------------- weak forms ----------------
'''
df = -phi + phi**3 + zet*u*(1 - 2*phi**2 + phi**4)

R0 = (
    tau_0*phi*w_phi*dx
  - tau_0*phi_0*w_phi*dx
  + dt*df*w_phi*dx
  + lamda_0**2*dt*inner(grad(phi), grad(w_phi))*dx
)

R1 = (
    (u - u_0)*w_u*dx
  - dt*0.5*(phi - phi_0)*w_u*dx
  + dt*D*inner(grad(u), grad(w_u))*dx
)

R   = R0 + R1
dcom = ufl.TrialFunction(ME)
J    = ufl.derivative(R, com, dcom)

'''
# ---------------- diagnostics check  ----------------
'''
# Silence runtime prints by not passing print_fn
diag = CoupledDiagnostics(msh, phi, u, lamda_0, zet, tau=tau_0, D=D, plot_every=999999)
diag.start()  # no printing

'''
# ---------------- solver ---------------------
'''
problem = NonlinearProblem(R, com, bcs=[], J=J)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
solver.atol = 1e-12
solver.max_it = 25
solver.report = False   # <- disable per-step solver report output

# Use direct LU via PETSc
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

'''
# ---------------- XDMF writers ---------------
'''

phi_series = XDMFFile(comm, f"{OUT_DIR}/phi_series.xdmf", "w")
u_series   = XDMFFile(comm, f"{OUT_DIR}/u_series.xdmf", "w")
phi_series.write_mesh(msh)
u_series.write_mesh(msh)


'''
# Prepare subspaces for output
'''
V_phi, map_phi = ME.sub(0).collapse()
V_u,   map_u   = ME.sub(1).collapse()
phi_out = fem.Function(V_phi)
u_out   = fem.Function(V_u)

'''
# ---------------- time loop ------------------
'''
t = 0.0

step = 0

if rank == 0:
    print("[run] Starting simulation…")

while t < T:
    t += dt
    step += 1
    solver.solve(com)
    diag.update(t)  # diagnostic call

    # Update previous time step
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()

    
    phi_out.x.array[:] = com.x.array[map_phi]
    u_out.x.array[:]   = com.x.array[map_u]
    phi_series.write_function(phi_out, t)
    u_series.write_function(u_out, t)

'''
# ---------------- closing all calls --------------------
'''
diag.finish()  # writes CSV + PNGs under diag_out/

phi_series.close()
u_series.close()
com.x.scatter_forward()
phi_fun = com.sub(0)
u_fun   = com.sub(1)

'''
#---------------- boundary condition check --------------------
'''
bc_check(msh, phi_fun, u_fun, tol=1e-8)
if rank == 0:
    print("[run] Simulation finished. XDMF written to:", OUT_DIR)
