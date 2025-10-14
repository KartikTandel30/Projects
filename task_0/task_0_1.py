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
import pathlib
from dolfinx.io import XDMFFile

t_start = time.time()
# Parameters
zet = 1.6
u = -0.75
tau_0 = 1
lamda_0 = 1
dt = 0.01

# Create mesh
msh = create_rectangle(MPI.COMM_WORLD,
                      [[0.0, 0.0], [100, 100]],
                       [50, 50],
                       cell_type=CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, P1)

w_phi = ufl.TestFunction(ME)
phi = Function(ME)
phi_0 = Function(ME)

'''
def initial_phi(x):
    r = np.sqrt((x[0] - 50.0)**2 + (x[1] - 50.0)**2)
    return np.where(r < 5.0, -1.0, 1.0)
'''
def initial_phi(x):
    return -0.6* np.ones(x.shape[1], dtype=default_real_type)


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

# Excel workbook setup
wb_all = Workbook()
ws_all = wb_all.active
ws_all.title = "All Node Data"
header = ["Time (s)"]
for i in range(len(phi.x.array)):
    header.append(f"ϕ_node_{i}")
    header.append(f"f_node_{i}")
ws_all.append(header)

# =============================
# ParaView output (XDMF/HDF5)
# =============================
results_dir = pathlib.Path("phase_output")
results_dir.mkdir(parents=True, exist_ok=True)

# Open once, write mesh once, then write functions per time step
xdmf_path = results_dir / "phi.xdmf"
xdmf = XDMFFile(msh.comm, str(xdmf_path), "w")
xdmf.write_mesh(ME.mesh)

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

step = 0
# Time loop with free energy plotting
while t < T:
    t += dt
    it , res = solver.solve(phi)
    print(f"Step {int(t / dt)}: num iteration: {it}")
    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()

    print(f"min(phi): {phi.x.array.min():.4f}, max(phi): {phi.x.array.max():.4f}")
    grid.point_data["Phase"] = phi.x.array.real
    plotter.remove_actor("timelabel")
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
    print(f"min(phi): {pmin:.4f}, max(phi): {pmax:.4f}")

    if np.allclose(phi_vals_all, 1.0, atol=1e-3):
        print("All nodes have reached solid phase. Ending early.")
        break
    
# Final plot
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
# Final f vs phi plot
phi_vals_final = phi.x.array.real
f_vals_final = (
    -0.5 * phi_vals_final**2
    + 0.25 * phi_vals_final**4
    + zet * u * phi_vals_final * (1 - (2/3) * phi_vals_final**2 + 0.2 * phi_vals_final**4)
    )
plt.figure()
plt.plot(phi_vals_final, f_vals_final, '.', alpha=0.4)
plt.xlabel("ϕ")
plt.ylabel("f(ϕ)")
plt.title(f"Final Free Energy Distribution at t = {t:.2f}")
plt.grid(True)
plt.tight_layout()
plt.savefig("f_vs_phi_final.png", dpi=300)
plt.show()

print("Simulation complete. Close the window to exit.")
plotter.app.exec_()
    