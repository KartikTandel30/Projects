from petsc4py import PETSc
import dolfinx
from mpi4py import MPI

import os
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import time

# Parameter constants used, taken from the ref. paper
zet = 1.6
u = -0.75
tau_0 = 1  # Characteristic time scale 
lamda_0 = 0.01  # characteristic interface thickness
dt = 0.001  # time step

# Create mesh
msh = create_unit_square(MPI.COMM_WORLD, 100, 100, CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, P1)

w_phi = ufl.TestFunction(ME)  # test function for order parameter
phi = Function(ME)  # trial function n+1
phi_0 = Function(ME)  # previous value

def initial_phi(x):
    r = np.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2)
    return np.where(r < 0.05, 1.0, -1.0)

phi.interpolate(initial_phi)
phi_0.interpolate(initial_phi)
phi.x.scatter_forward()
phi_0.x.scatter_forward()

# Free energy derivative
df = -phi + phi**3 + zet*u*(1 - 2*phi**2 + phi**4)

# Weak or variational form for the task-1
R0 = ( tau_0*phi*w_phi*dx 
      - tau_0*phi_0*w_phi*dx
      + dt*inner(df, w_phi)*dx
      + lamda_0*dt*inner(grad(phi), grad(w_phi))*dx )

# Solving the nonlinear problem 
problem = NonlinearProblem(R0, phi)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
solver.atol = 1e-12
solver.max_it = 25
solver.report = True

# Setting the type of the solver
ksp = solver.krylov_solver
opt = PETSc.Options()
opt_prefix = ksp.getOptionsPrefix()

opt[f"{opt_prefix}ksp_type"] = "preonly"
opt[f"{opt_prefix}pc_type"] = "lu"
opt[f"{opt_prefix}snes_monitor"] = ""
sys = PETSc.Sys()

if sys.hasExternalPackage("superlu_dist"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

t = 0.0
T = 5

# Visualize using PyVista
topology, cell_types, x = plot.vtk_mesh(ME)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = phi.x.array.real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=True)
plotter.view_xy(True)
plotter.add_text(f"time:{t}", font_size=10, name="timelabel")

# Time loop
while t < T:
    t += dt
    res = solver.solve(phi)
    print(f"Step {int(t/dt)}: num iteration: {res[0]}")
    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()
    print(f"min(phi): {phi.x.array.min():.4f}, max(phi): {phi.x.array.max():.4f}")
    grid.point_data["Phase"] = phi.x.array.real
    plotter.remove_actor("timelabel")
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()

phi.x.scatter_forward()
grid.point_data["Phase"] = phi.x.array.real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = "phase.png"
pv.plot(grid, show_edges=True, screenshot=screenshot)


print("Simulation complete. Close the window to exit.")
while plotter.app.running:
    time.sleep(0.1)