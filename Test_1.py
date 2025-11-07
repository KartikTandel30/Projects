# -*- coding: utf-8 -*-
"""
Dendrite Phase-Field (φ–u) — monolithic Backward–Euler solver (FEniCSx)

This driver reads parameters from `params_task3_dendrite.json`, builds a mixed P1×P1
finite-element model, and solves the coupled φ–u system *monolithically* with Newton.

Model (non-dimensional, pure-metal style):
    τ0 * φ̇ = -μ,
    μ = ∂f/∂φ(φ, u) - ∇·Q,                    (anisotropic gradient penalty via Q)
    u̇  = D ∇²u + K * φ̇                      (enthalpy coupling; here K≈1/2)

Bulk energy density:
    f(φ,u) = -½ φ² + ¼ φ⁴ + ζ u φ ( 1 - (2/3)φ² + (1/5)φ⁴ )

Anisotropy (four-fold) enters the gradient energy with
    a(n) = (1 - 3ε) + 4ε (n_x^4 + n_y^4),  where n = ∇φ / |∇φ|_ε
and the interfacial flux Q(∇φ) is constructed so that the weak form is:
    R_φ = ∫ τ(n) (φ - φ0) w_φ dx
          + Δt ∫ [ ∂f/∂φ(φ,u) w_φ ] dx
          + Δt ∫ [ Q(∇φ) · ∇w_φ ] dx,
    R_u = ∫ [ (u - u0) - K (φ - φ0) ] w_u dx
          + Δt ∫ [ D ∇u · ∇w_u ] dx.

Here τ(n) = τ0 a(n)^2 (kinetic anisotropy); K≈0.5 is the latent-heat factor.

Boundary conditions: homogeneous Neumann (natural) for both φ and u.

I/O: writes XDMF series for φ and u in OUT_DIR every `save_stride` steps.

Usage:
    mpirun -np 4 python Final.py
    (Ensure `params_task3_dendrite.json` exists next to this file; see schema below.)

JSON schema (required keys):
    dt, Lx, Ly, Nx, Ny, T, zet, tau_0, lambda_0, D, R0, u0
Optional: save_stride (defaults to 10)
"""
#---------------Imports----------------#
from petsc4py import PETSc
import dolfinx
from mpi4py import MPI
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from ufl import dx, grad, inner, Identity, outer, as_vector, sqrt, dot
import numpy as np
import time
import pathlib
import json
from numpy.random import default_rng

rng = default_rng(12345)

OUT_DIR = "out_task3_FT2_2"
comm = MPI.COMM_WORLD
rank = comm.rank
t_start = time.time()

# ---------------- Parameters ----------------#
param_file = pathlib.Path(__file__).resolve().parent / "params_task3_dendrite.json"
if not param_file.exists():
    raise RuntimeError(
        f"Parameter file not found: {param_file}\n"
        "Create 'params_task3_dendrite.json' with keys: "
        "dt, Lx, Ly, Nx, Ny, T, zet, tau_0, lambda_0, D, R0, u0, eps_an, K"
    )

with param_file.open("r", encoding="utf-8") as fh:
    params = json.load(fh)

required = ["dt", "Lx", "Ly", "Nx", "Ny", "T", "zet", "tau_0", "lambda_0", "D", "R0", "u0", "eps_an", "K"]
missing = [k for k in required if k not in params]
if missing:
    raise RuntimeError(f"Missing parameter keys in {param_file}: {missing}")

# assign parameters
dt       = float(params["dt"])
Lx       = float(params["Lx"])
Ly       = float(params["Ly"])
Nx       = int(params["Nx"])
Ny       = int(params["Ny"])
T        = float(params["T"])
zet      = float(params["zet"])
tau_0    = float(params["tau_0"])
lambda_0 = float(params["lambda_0"])
D        = float(params["D"])
R0       = float(params["R0"])
u0       = float(params["u0"])
eps_an   = float(params["eps_an"])
K        = float(params["K"])
STRIDE   = int(params.get("save_stride", 10))  # optional

# --- To view Parameters taken from JSON file ---
def show_run_params(params, **used):
    if MPI.COMM_WORLD.rank != 0:
        return
    import json as _json
    print("\n===== Parameters (from JSON file) =====", flush=True)
    print(_json.dumps(params, indent=2, sort_keys=True), flush=True)
    print("===== Parameters actually used (cast) =====", flush=True)
    print(_json.dumps(used, indent=2, sort_keys=True), flush=True)
    print("==========================================\n", flush=True)

show_run_params(
    params,
    dt=dt, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, T=T,
    zet=zet, tau_0=tau_0, lambda_0=lambda_0,
    D=D, R0=R0, u0=u0, save_stride=STRIDE
)


# ---------------- Mesh ----------------#
msh = create_rectangle(MPI.COMM_WORLD,
                       [[0.0, 0.0], [Lx, Ly]],
                       [Nx, Ny],
                       cell_type=CellType.triangle)

# ---------------- Function Spaces ----------------#
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)
com_0 = Function(ME)
phi, u     = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

# ---------------- Initial conditions ----------------#
'''
w_eq = np.sqrt(2.0) * lambda_0

def initial_phi(x):
    xc = x[0] - Lx/2.0
    yc = x[1] - Ly/2.0
    r  = np.sqrt(xc**2 + yc**2)
    return np.tanh((R0 - r) / w_eq)

def initial_u(x):
    return np.full(x.shape[1], u0, dtype=default_real_type)
'''
def initial_phi(x):
    return -np.ones(x.shape[1], dtype=default_real_type) 



def initial_u(x):
    # Neumann eigenmode on [0,Lx]×[0,Ly]; has closed-form decay
    return np.cos(np.pi * x[0] / Lx) * np.cos(np.pi * x[1] / Ly)

# ---------------- Initializing fields with noise ----------------#
com.x.array[:] = 0
com.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.x.array[:] = com.x.array
com.x.scatter_forward(); com_0.x.scatter_forward()

# small masked noise on phi only (kept in [-1,1])
phi_vec = com.sub(0).x.array
mask = np.clip(1.0 - phi_vec**2, 0.0, 1.0)
phi_vec += (5e-4) * mask * rng.standard_normal(phi_vec.shape)
np.clip(phi_vec, -1.0, 1.0, out=phi_vec)
com.sub(0).x.array[:] = phi_vec
com.x.scatter_forward()
com_0.x.array[:] = com.x.array
com_0.x.scatter_forward()

# ---------------- Anisotropy ----------------#
df = -phi + phi**3 + zet*u*(1 - 2*phi**2 + phi**4)

eps_an = fem.Constant(msh, default_real_type(eps_an))
eta    = fem.Constant(msh, default_real_type(1e-8))
K      = fem.Constant(msh, default_real_type(K))

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

q_phi = lambda_0**2 * (a**2 * gphi + g2 * a * da_dg)
F_grad_aniso = inner(q_phi, grad(w_phi)) * dx
tau = tau_0 * a**2

# ---------------- Weak form ----------------#
R0 = ( tau*(phi - phi_0)*w_phi*dx
     + dt*df*w_phi*dx
     + dt*F_grad_aniso )

R1 = ( (u - u_0)*w_u*dx
     - K*(phi - phi_0)*w_u*dx
     + dt*D*inner(grad(u), grad(w_u))*dx )

R = R0 + R1
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)

# ---------------- Solver ----------------#
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

# ---------------- Output + time loop ----------------#

# ---------------- Output dir ----------------
phi_series = XDMFFile(comm, f"{OUT_DIR}/phi.xdmf", "w")
u_series   = XDMFFile(comm, f"{OUT_DIR}/u.xdmf", "w")
phi_series.write_mesh(msh)
u_series.write_mesh(msh)

V_phi, map_phi = ME.sub(0).collapse()
V_u,   map_u   = ME.sub(1).collapse()
phi_out = fem.Function(V_phi)
u_out   = fem.Function(V_u)

phi_out.x.array[:] = com.x.array[map_phi]
u_out.x.array[:]   = com.x.array[map_u]
phi_out.x.scatter_forward()
u_out.x.scatter_forward()

# Time
t = 0.0
step = 0

if msh.comm.rank == 0:
    print("[run] Starting simulation…")

# (Optional) quick sanity print: confirm φ frozen and u evolving
if rank == 0:
    up0 = com.sub(1).x.array.copy()
    print(f"[init] ||u||₂={np.linalg.norm(up0):.6e}")
    p0 = com.sub(0).x.array.copy()
    print(f"[init] ||φ||₂={np.linalg.norm(p0):.6e}")

u_L2 = np.sqrt(fem.assemble_scalar(fem.form(u_out*u_out*dx)))
phi_L2 = np.sqrt(fem.assemble_scalar(fem.form(phi_out*phi_out*dx)))
if rank == 0:
    print(f"[init] ||u||_L2={u_L2:.6e}  ||phi||_L2={phi_L2:.6e}")

uL2_0_FE = np.sqrt(fem.assemble_scalar(fem.form(u*u*dx)))   # L2 at t=0
lam = D * ((np.pi/Lx)**2 + (np.pi/Ly)**2)                    # cos(pi x)cos(pi y) on [0,L]^2
area = Lx * Ly
if rank == 0:
    print(f"[verify:init] L2_FE(u)={uL2_0_FE:.6e}")

while t < T:
    t += dt
    step += 1
    solver.solve(com)  # silent
    com.x.scatter_forward()

    if rank == 0 and (step % max(1, (STRIDE or 10)) == 0):
        uL2_FE = np.sqrt(fem.assemble_scalar(fem.form(u*u*dx)))
        ratio_meas = uL2_FE / (uL2_0_FE if uL2_0_FE != 0 else 1.0)
        ratio_th   = np.exp(-lam * t)
        mean_u     = fem.assemble_scalar(fem.form(u*dx)) / area
        print(f"[verify:t={t:.3f}] L2_ratio={ratio_meas:.6e} (theory {ratio_th:.6e}); "
            f"mean(u)={mean_u:.3e}")
    
    if rank == 0 and (step % max(1, (STRIDE or 10)) == 0):
        phi_curr = com.x.array[map_phi]
        u_curr   = com.x.array[map_u]
        phi_prev = com_0.x.array[map_phi]
        u_prev   = com_0.x.array[map_u]
        dphi = np.linalg.norm(phi_curr - phi_prev)
        du   = np.linalg.norm(u_curr   - u_prev)
        print(f"[t={t:.3f}] ||u||₂={np.linalg.norm(u_curr):.6e}  "
              f"||Δu||₂={du:.2e}  ||Δφ||₂={dphi:.2e}")

    # writes (optional)
    if STRIDE > 0 and step % STRIDE == 0:
        phi_out.x.array[:] = com.x.array[map_phi]
        u_out.x.array[:]   = com.x.array[map_u]
        phi_series.write_function(phi_out, t)
        u_series.write_function(u_out, t)

    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()

    '''
    phi_vals_all = com.x.array[map_phi]
    if np.allclose(np.abs(phi_vals_all), 1.0, atol=1e-3):
        print("All nodes reached a stable phase (±1). Ending early.")
        break
    '''



phi_series.close()
u_series.close()
com.x.scatter_forward()


if msh.comm.rank == 0:
    print("Simulation finished successfully.")
    print(time.time() - t_start, "seconds elapsed.")