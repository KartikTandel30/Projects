"""task_0.py

FEM phase-field Implicit time-stepping example using dolfinx + PETSc.

This script solves a time-dependent phase-field evolution on a 2D
rectangular mesh. In code paramter reading, The script writes XDMF/HDF5
output for visualization, produces an Excel spreadsheet with node-wise
phase and free-energy values over time, and saves a final plot of free
energy vs phase.

High-level behaviour
- Create a triangular mesh and Lagrange P1 function space.
- Initialize phi with a constant value via `initial_phi`.
- Formulate a simple (local) free-energy derivative and solve the
    nonlinear residual with a Newton solver at each time step.
- Save results to XDMF and an Excel workbook; display a live PyVista
    visualization during the run.

Requirements
- Python packages: dolfinx, petsc4py, mpi4py, pyvista, pyvistaqt, openpyxl,
    matplotlib, basix, ufl, numpy
- A working MPI/PETSc environment configured for dolfinx.

Usage
- Run the script with

Outputs
- `phase_output/phi.xdmf` (XDMF/HDF5 containing phi over time)
- `phiAndBulkData.xlsx` (Excel workbook with node-wise phi and free-energy)
- `f_vs_phi_final.png` (final free-energy vs phi scatter)

"""

#--------------------Import necessary modules

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
from openpyxl import Workbook
from dolfinx.io import XDMFFile
from pathlib import Path


t_start = time.time()


# Parameter constants used, taken from the ref. paper
'''
-----------------Parmater section 

'''

zet = 1.6   # coupling constant
u = -0.75   # Temparature
tau_0 = 1   # Characteristic time scale 
dt = 0.04   # time step

# Create mesh
Lx, Ly = 100.0, 100.0
Nx = 50
Ny = 50
T = 40.0  # total simulation time


'''
#---------------Mesh Creation
'''
msh = create_rectangle(MPI.COMM_WORLD,
                      [[0.0, 0.0], [Lx, Ly]],
                       [Nx, Ny],
                       cell_type=CellType.triangle)



'''
Function space setup
'''
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, P1)

w_phi = ufl.TestFunction(ME)
phi = Function(ME)
phi_0 = Function(ME)


'''
The initial condition defination for phi
'''
def initial_phi(x):
    return -0.6* np.ones(x.shape[1], dtype=default_real_type)


'''
#------------------------ Initialize phi and phi_0-------
'''
phi.interpolate(initial_phi)
phi_0.interpolate(initial_phi)
phi.x.scatter_forward()
phi_0.x.scatter_forward()


'''
#------------------------ Weak form -------
'''
#--------------------  Free energy derivative-------
df = -phi + phi**3 + zet * u * (1 - 2 * phi**2 + phi**4)

#------------------- Weak form without gradient term----------


R0 = (tau_0 * phi * w_phi * dx
      - tau_0 * phi_0 * w_phi * dx
      + dt * inner(df, w_phi) * dx)



'''
#------------------------ Solver setup -------
'''
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
opt[f"{opt_prefix}snes_monitor"] = None
sys = PETSc.Sys()
if sys.hasExternalPackage("superlu_dist"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()


'''
#------------------------ Output defination -------
'''
wb_all = Workbook()
ws_all = wb_all.active
ws_all.title = "All Node Data"
header = ["Time (s)"]
for i in range(len(phi.x.array)):
    header.append(f"ϕ_node_{i}")
    header.append(f"f_node_{i}")
ws_all.append(header)


'''
#----------------- ParaView output (XDMF/HDF5)-----------
'''
t = 0.0
step = 0
results_dir = Path("phase_output")
results_dir.mkdir(parents=True, exist_ok=True)

# Open once, write mesh once, then write functions per time step
xdmf_path = results_dir / "phi.xdmf"
xdmf = XDMFFile(msh.comm, str(xdmf_path), "w")
xdmf.write_mesh(msh)

#--------------------- PyVista plot setup------------------
topology, cell_types, x = plot.vtk_mesh(ME)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = phi.x.array.real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=True)
plotter.view_xy(True)
plotter.add_text(f"time:{t}", font_size=10, name="timelabel")


'''
#------------------------ Time loop -------
'''

# Time loop with free energy calculation and Excel export
while t < T:
    t += dt
    it , res = solver.solve(phi)
    #print(f"Step {int(t / dt)}: num iteration: {it}")
    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()

    #print(f"min(phi): {phi.x.array.min():.4f}, max(phi): {phi.x.array.max():.4f}")
    grid.point_data["Phase"] = phi.x.array.real
    try:
        plotter.remove_actor("timelabel")
    except Exception:
        pass
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()

    xdmf.write_function(phi, t)
    # Export data to Excel
    phi_vals_all = phi.x.array.real
    f_vals_all = (
    -0.5 * phi_vals_all**2
    + 0.25 * phi_vals_all**4
    + zet * u * phi_vals_all * (1 - (2/3) * phi_vals_all**2 + 0.2 * phi_vals_all**4)
    )
    row_all = [t]
    for phi_i, f_i in zip(phi_vals_all, f_vals_all):
        row_all.extend([phi_i, f_i])
    ws_all.append(row_all)

    # Early stop if fully solid (within tolerance)
    pmin = float(phi.x.array.min())
    pmax = float(phi.x.array.max())
    

    if np.allclose(phi_vals_all, 1.0, atol=1e-3):
        print("All nodes have reached solid phase. Ending early.")
        break
    

'''
#------------------------ closing the simulation -------
'''

phi.x.scatter_forward()
grid.point_data["Phase"] = phi.x.array.real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = "phase.png"
pv.plot(grid, show_edges=True, screenshot=screenshot)

# Save Excel file
wb_all.save("phiAndBulkData.xlsx")
xdmf.close()
if msh.comm.rank == 0:
    print(f"Total runtime: {time.time()-t_start:.2f}s")


print("Simulation complete. Close the window to exit.")
plotter.app.exec_()



