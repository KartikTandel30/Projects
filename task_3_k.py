from petsc4py import PETSc
from mpi4py import MPI

import os, sys, time, ufl, numpy as np
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, fem
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner
import pyvista as pv
import pyvistaqt as pvq

# ====================================================
# Kobayashi (1993) anisotropic dendrite — FEniCSx 0.9.0
# Angle-based anisotropy (θ = atan2(φ_y, φ_x)) as in Kobayashi
# ====================================================

# -------------------- Parameters --------------------
# Phase-field/thermal parameters (nondimensional)
zet     = 1.9     # thermal/latent-heat coupling (λ)
tau_0   = 1     # kinetic coefficient τ0
lamda_0 = 1.0     # interface thickness scale W
dt      = 0.001   # timestep
D       = 1.5     # thermal diffusivity

# Anisotropy (Kobayashi): a(θ) = 1 + eps * cos(m*(θ - θ0))
eps      = 0.20   # anisotropy strength (|eps| ≲ 0.25)
mfold    = 4      # fold symmetry (4 or 6)
theta0   = 0.0    # rotation (radians)
EPS_ATAN = PETSc.ScalarType(1e-12)  # small epsilon for atan2 safety

# -------------------- Mesh --------------------
Lx, Ly = 250.0, 250.0
Nx, Ny = 500, 500   # reduce if memory is tight
msh = create_rectangle(MPI.COMM_WORLD,
                       [[0.0, 0.0], [Lx, Ly]],
                       [Nx, Ny],
                       cell_type=CellType.triangle)

P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME)   # n+1
com_0 = Function(ME)   # n

phi, u     = ufl.split(com)
phi_0, u_0 = ufl.split(com_0)

# -------------------- Initial conditions --------------------
# Seed a solid disk (phi=+1) in an undercooled melt (u<0)

def initial_phi(x):
    r = np.sqrt((x[0] - Lx/2.0)**2 + (x[1] - Ly/2.0)**2)
    # small noise to trigger side-branching (optional, keep modest)
    noise = 0.01 * (2.0 * np.random.random(x.shape[1]) - 1.0)
    return np.where(r < 2.5, 1.0 + noise, -1.0)

def initial_u(x):
    # Far-field undercooling (keeping natural BCs in this template)
    return -1.0 * np.ones(x.shape[1], dtype=default_real_type)

com.x.array[:] = 0.0
com.sub(0).interpolate(initial_phi)
com_0.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward()
com_0.x.scatter_forward()

# -------------------- Bulk term (double-well + coupling) --------------------
# τ φ_t = W^2 ∇·A + φ − φ^3 − λ u (1 − φ^2)^2
# In residual form we use df = (−φ + φ^3 + λ u (1 − φ^2)^2)

df = -phi + phi**3 + zet * u * (1 - 2*phi**2 + phi**4)

# -------------------- Anisotropy (angle-based Kobayashi) -------------------
# θ = atan2(φ_y, φ_x),
# a(θ) = 1 + eps cos(m(θ − θ0)),  a'(θ) = −eps m sin(m(θ − θ0))
# Flux A = a(θ)^2 ∇φ + [ −a a' φ_y,  a a' φ_x ]^T

g = grad(phi)
gx, gy = g[0], g[1]

theta = ufl.atan2(gy + EPS_ATAN, gx + EPS_ATAN)
ang = mfold * (theta - theta0)
a  = 1.0 + eps * ufl.cos(ang)
ap = - eps * mfold * ufl.sin(ang)
A  = a*a*g + ufl.as_vector((-a*ap*gy, a*ap*gx))

# Optional anisotropic kinetics (Kobayashi often uses τ(θ)=τ0 a(θ)^2)
tau_eff = tau_0 * a*a

# -------------------- Weak forms (Backward Euler) --------------------------
# τ(θ)(φ−φ⁰) + dt[ −φ + φ³ + λu(1−φ²)² ] + dt W² ⟨∇w, A⟩ = 0
R0 = (
    tau_eff * (phi - phi_0) * w_phi * dx
  + dt * df * w_phi * dx
  + dt * (lamda_0**2) * inner(grad(w_phi), A) * dx
)

# u_t = D Δu + 0.5 φ_t  ⇒  (u−u⁰) − 0.5(φ−φ⁰) + dt D ⟨∇u,∇w⟩ = 0
R1 = (
    (u - u_0) * w_u * dx
  - 0.5 * (phi - phi_0) * w_u * dx
  + dt * D * inner(grad(u), grad(w_u)) * dx
)

R = R0 + R1
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)

# -------------------- Nonlinear solve --------------------
problem = NonlinearProblem(R, com, bcs=[], J=J)
solver  = NewtonSolver(msh.comm, problem)
# Use valid 0.9.0 criterion keyword (lowercase)
solver.convergence_criterion = "residual"
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 40
solver.report = True

# (dolfinx 0.9.0) NewtonSolver has no public .snes; use defaults or set
# SNES options via prefix if needed. We keep defaults here to avoid API errors.
# Linear solver (KSP/PC)
ksp = solver.krylov_solver
p = ksp.getOptionsPrefix()                # prefix applies ONLY to KSP/PC
opt = PETSc.Options()
opt[f"{p}ksp_type"] = "preonly"
opt[f"{p}pc_type"]  = "lu"
# opt[f"{p}ksp_monitor_short"] = ""       # optional linear monitor
sys = PETSc.Sys()
if sys.hasExternalPackage("superlu_dist"):
    opt[f"{p}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{p}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# -------------------- ParaView I/O (write EVERY step) --------------------
file = XDMFFile(MPI.COMM_WORLD, "output_task_3.xdmf", "w")
file.write_mesh(msh)



# Time
t = 0.0
T = 50.0
step = 0
# Initial output fields (t=0)
phi_sub = com.sub(0)
file.write_function(phi_sub, 0.0)

# initial write (t=0)
write_to_xdmf(0.0)

# -------------------- Live viz (update ONLY every 100 steps) --------------------
Vphi_viz, dof = ME.sub(0).collapse()
topology, cell_types, x = plot.vtk_mesh(Vphi_viz)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = com.x.array[dof].real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=False)
plotter.view_xy(True)
plotter.add_text("time: 0.00", font_size=10, name="timelabel")

# -------------------- Time loop --------------------
T = 100.0      # allow dendrites to develop
step = 0
VIEW_EVERY = 100  # refresh PyVista every 100 steps

t = 0.0
while t < T - 1e-14:
    t += dt
    step += 1
    its, converged = solver.solve(com)
    print(f"Step {step}: Newton iterations = {its} ({'OK' if converged else 'FAIL'})")
    if not converged:
        raise RuntimeError("Newton did not converge — reduce dt or eps, or increase lamda_0.")

    # roll state forward
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward(); com_0.x.scatter_forward()

    # write output
    file.write_function(phi_sub, t)

    # ---- Update live view ONLY every VIEW_EVERY steps ----
    if step % VIEW_EVERY == 0:
        grid.point_data["Phase"] = com.x.array[dof].real
        if plotter is not None:
            try:
                plotter.remove_actor("timelabel")
            except Exception:
                pass
            plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
            plotter.app.processEvents()

# final write & close files


# final static plot (optional off-screen)
try:
    grid.point_data["Phase"] = com.x.array[dof].real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = os.path.join(outdir, "phase_last.png")
    pv.plot(grid, show_edges=True, screenshot=screenshot)
except Exception as e:
    if MPI.COMM_WORLD.rank == 0:
        print(f"[viz] final static plot skipped: {e}")

if MPI.COMM_WORLD.rank == 0:
    print("Simulation complete. Close the window to exit.")

# keep UI responsive if user wants to pan/zoom
while plotter.app.running:
    time.sleep(0.1)
