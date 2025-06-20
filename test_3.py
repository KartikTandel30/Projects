import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, io, log, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from basix.ufl import element
from ufl import dx, grad, inner
import petsc4py.PETSc as PETSc

# --- PyVista Setup ---
try:
    import pyvista as pv
    import pyvistaqt as pvqt
    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualize the solution.")
    have_pyvista = False

# --- Parameters ---
epsilon = 0.01  # Interface width parameter
lambda_c = 1.0    # Coupling constant
nu = -0.75         # Constant temperature undercooling
dt = 0.01         # Time step
theta = 0.5       # Crank-Nicolson
T = 100           # Final time

# --- Mesh and function space ---
domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
V = fem.functionspace(domain, element("Lagrange", domain.basix_cell(), 1))

# --- Trial and test functions ---
phi = fem.Function(V)
phi_n = fem.Function(V)
w = ufl.TestFunction(V)

# --- Initial condition: solid seed in center ---
def initial_phi(x):
    r = np.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2)
    return np.where(r < 0.05, 1.0, -1.0)

phi.interpolate(initial_phi)
phi_n.interpolate(initial_phi)
phi.x.scatter_forward()
phi_n.x.scatter_forward()

# --- Free energy derivative from paper ---
phi_mid = (1 - theta) * phi_n + theta * phi

gp = -phi_mid + phi_mid**3
hp = 1 - 2 * phi_mid**2 + phi_mid**4

dfdphi = gp + lambda_c * nu * hp

F = ((phi - phi_n) / dt) * w * dx + epsilon**2 * inner(grad(phi_mid), grad(w)) * dx + dfdphi * w * dx

# --- Nonlinear solver ---
problem = NonlinearProblem(F, phi)
solver = NewtonSolver(domain.comm, problem)
solver.rtol = 1e-6
solver.convergence_criterion = "incremental"

# Optional: direct solver
ksp = solver.krylov_solver
opts = PETSc.Options()
prefix = ksp.getOptionsPrefix()
opts[f"{prefix}ksp_type"] = "preonly"
opts[f"{prefix}pc_type"] = "lu"
ksp.setFromOptions()

# --- Output ---
xdmf = io.XDMFFile(domain.comm, "task1_output.xdmf", "w")
xdmf.write_mesh(domain)
xdmf.write_function(phi, 0.0)

# --- PyVista visualization setup ---
if have_pyvista:
    topology, cell_types, x = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, x)
    grid.point_data["phi"] = phi.x.array.real
    p = pvqt.BackgroundPlotter(title="Phase-Field Evolution", auto_update=True)
    p.add_mesh(grid, clim=[-1, 1])
    p.view_xy(True)
    p.add_text("time: 0.0", name="time_label", font_size=10)

# --- Time stepping ---
t = 0.0
step = 0
while t < T:
    t += dt
    step += 1
    n, converged = solver.solve(phi)
    print(f"Step {step}: converged = {converged} in {n} iterations")
    phi_n.x.array[:] = phi.x.array
    phi_n.x.scatter_forward()
    xdmf.write_function(phi, t)

    # Update PyVista plot
    if have_pyvista:
        grid.point_data["phi"] = phi.x.array.real
        p.add_text(f"time: {t:.3f}", name="time_label")
        p.app.processEvents()

xdmf.close()
