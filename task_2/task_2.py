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
import json
import argparse
import pathlib
# Parameters
from test2 import CoupledDiagnostics
from test_bc import bc_check

# Minimal JSON loader: read 'parameter_to.json' located next to this script.
param_file = pathlib.Path(__file__).resolve().parent / "parameter.json"
if not param_file.exists():
    raise RuntimeError(
        f"Parameter file not found: {param_file}\nCreate a JSON file named 'parameter_to.json' next to this script with keys: dt, Lx, Ly, Nx, Ny, T, zet, u, tau_0, lamda_0"
    )

with param_file.open("r", encoding="utf-8") as fh:
    params = json.load(fh)

required = ["dt", "Lx", "Ly", "Nx", "Ny", "T", "zet", "u", "tau_0", "lamda_0"]
missing = [k for k in required if k not in params]
if missing:
    raise RuntimeError(f"Missing parameter keys in {param_file}: {missing}")

# assign parameters (minimal casting)
dt = float(params["dt"]) 
Lx = float(params["Lx"]) 
Ly = float(params["Ly"]) 
Nx = int(params["Nx"]) 
Ny = int(params["Ny"]) 
T = float(params["T"]) 
zet = float(params["zet"]) 
u = float(params["u"]) 
tau_0 = float(params["tau_0"]) 
lamda_0 = float(params["lamda_0"]) 
D = float(params.get("D", 1.0))  # default to 1.0 if not provided

WRITE_EVERY = 50         # write XDMF every N steps (set 0 to disable)
OUT_DIR = "out_task2"    # where XDMFs go

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------- mesh & spaces -------------
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

# ---------------- initials ------------------
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



# ---------------- weak forms ----------------
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

# ---------------- diagnostics ----------------
# Silence runtime prints by not passing print_fn
diag = CoupledDiagnostics(msh, phi, u, lamda_0, zet, tau=tau_0, D=D, plot_every=999999)
diag.start()  # no printing

# ---------------- solver ---------------------
problem = NonlinearProblem(R, com, bcs=[], J=J)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
solver.atol = 1e-12
solver.max_it = 25
solver.report = False   # <- disable per-step solver report output

# Direct LU (kept as-is; switch to GMRES+AMG later for speed if needed)
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

# ---------------- XDMF writers ---------------


phi_series = XDMFFile(comm, f"{OUT_DIR}/phi_series.xdmf", "w")
#u_series   = XDMFFile(comm, f"{OUT_DIR}/u_series.xdmf", "w")
phi_series.write_mesh(msh)
#u_series.write_mesh(msh)

# Prepare collapsed subspaces for output

V_phi, map_phi = ME.sub(0).collapse()
V_u,   map_u   = ME.sub(1).collapse()
phi_out = fem.Function(V_phi)
u_out   = fem.Function(V_u)

# ---------------- time loop ------------------
t = 0.0
T = 180.0
step = 0

if rank == 0:
    print("[run] Starting simulation…")

while t < T:
    t += dt
    step += 1
    solver.solve(com)
    diag.update(t)  # silent

    # advance "previous" state and scatter
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    
    if WRITE_EVERY > 0 and step % WRITE_EVERY == 0:
        phi_out.x.array[:] = com.x.array[map_phi]
        u_out.x.array[:]   = com.x.array[map_u]
        phi_series.write_function(phi_out, t)

    phi_vals_all = com.x.array[map_phi]
    if np.allclose(np.abs(phi_vals_all), 1.0, atol=1e-3):
        print("All nodes reached a stable phase (±1). Ending early.")
        break
    #u_series.write_function(u_out, t)

# ---------------- wrap-up --------------------
diag.finish()  # writes CSV + PNGs under diag_out/

phi_series.close()
#u_series.close()
bc_check(msh, phi, u, tol=1e-8)
if rank == 0:
    print("[run] Simulation finished. XDMF written to:", OUT_DIR)
