from petsc4py import PETSc
import dolfinx
from mpi4py import MPI

import os
import ufl
from basix.ufl import element
from dolfinx import default_real_type, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, inner
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters
zet = 1.6
u =-0.75
tau_0 = 1
lamda_0 = 1
dt = 0.04

# Create mesh
msh = create_rectangle(MPI.COMM_WORLD,
                      [[0.0, 0.0], [100,100]],
                       [50,50],
                       cell_type=CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, P1)

w_phi = ufl.TestFunction(ME)
phi = Function(ME)
phi_0 = Function(ME)

'''
def initial_phi(x):
    r = np.sqrt((x[0] - 50.0)**2 + (x[1] - 50.0)**2)
    return np.where(r < 5.0, 1.0, -1.0)

'''
def initial_phi(x):
    return -0.60* np.ones(x.shape[1], dtype=default_real_type)


phi.interpolate(initial_phi)
phi_0.interpolate(initial_phi)
phi.x.scatter_forward()
phi_0.x.scatter_forward()

# Free energy derivative
df = -phi + phi**3 + zet * u * (1 - 2 * phi**2 + phi**4)

# Weak form without gradient term
R0 = (tau_0 * phi * w_phi * dx
      - tau_0 * phi_0 * w_phi * dx
      + dt * inner(df, w_phi) * dx)

# Solver setup
problem = NonlinearProblem(R0, phi)
solver = NewtonSolver(msh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
solver.atol = 1e-12
solver.max_it = 25
solver.report = True

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

# PyVista plot setup
t = 0.0
T = 20
topology, cell_types, x = plot.vtk_mesh(ME)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = phi.x.array.real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=True)
plotter.view_xy(True)
plotter.add_text(f"time:{t}", font_size=10, name="timelabel")

# Time loop with free energy plotting
while t < T:
    t += dt
    res = solver.solve(phi)
    print(f"Step {int(t / dt)}: num iteration: {res[0]}")
    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()

    print(f"min(phi): {phi.x.array.min():.4f}, max(phi): {phi.x.array.max():.4f}")
    grid.point_data["Phase"] = phi.x.array.real
    plotter.remove_actor("timelabel")
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()
    """
    # Plot f(ϕ) vs ϕ every 25 steps
    if int(t / dt) % 25 == 0:
        phi_vals = phi.x.array.real
        f_vals = -phi_vals + phi_vals**3 + zet * u * (1 - 2 * phi_vals**2 + phi_vals**4)
        plt.figure()
        plt.plot(phi_vals, f_vals, '.', alpha=0.3)
        plt.xlabel("ϕ")
        plt.ylabel("f(ϕ)")
        plt.title(f"Free energy at time t={t:.2f}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"f_vs_phi_t{int(t*100):04d}.png", dpi=300)
        plt.close()
    """

# Final plot
phi.x.scatter_forward()
grid.point_data["Phase"] = phi.x.array.real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = "phase.png"
pv.plot(grid, show_edges=True, screenshot=screenshot)

print("Simulation complete. Close the window to exit.")
while plotter.app.running:
    time.sleep(0.1)
