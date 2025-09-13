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
from dolfinx.mesh import CellType, create_unit_square, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner
import pyvista as pv
import pyvistaqt as pvq
import numpy as np
import time

# Parameter constants used, taken from the ref. paper
zet = 1.6  # coupling constant
tau_0 = 1   # Characteristic time scale 
lamda_0 = 1   # characteristic interface thickness
dt = 0.04   # time step
D = 1

# Create mesh
'''msh = create_unit_square(MPI.COMM_WORLD, 10, 10, CellType.triangle)'''
Lx, Ly = 100.0, 100.0
Nx, Ny = 175, 175
msh = create_rectangle(MPI.COMM_WORLD,
                       [[0.0, 0.0], [Lx, Ly]],
                       [Nx,Ny],
                       cell_type=CellType.triangle) 
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh,mixed_element( [P1, P1]))

w_phi, w_u = ufl.TestFunctions(ME)  # test function for order parameter and temperature
com = Function(ME)  # trial function n+1 combined 
com_0 = Function(ME)  # previous value combined

phi , u  = ufl.split(com)
phi_0, u_0, = ufl.split(com_0)


def initial_phi(x):
    r = np.sqrt((x[0] - 50.0)**2 + (x[1] - 50.0)**2)  # Center at (250, 250)
    return np.where(r < 5.0, 1.0, -1.0)

'''
def initial_phi(x):
    return -1* np.ones(x.shape[1], dtype=default_real_type)
'''
def initial_u(x):
    return -0.75*np.ones(x.shape[1], dtype=default_real_type)

com.x.array[:] = 0
com.sub(0).interpolate(initial_phi)  # for the bond order parameter
com_0.sub(0).interpolate(initial_phi)
com.sub(1).interpolate(initial_u)
com_0.sub(1).interpolate(initial_u)
com.x.scatter_forward()  # for parallelization 
com_0.x.scatter_forward()

# Free energy derivative
df = -phi + phi**3 + zet*u*(1 - 2*phi**2 + phi**4)

# Weak or variational form of phase filed PDE
R0 = ( tau_0*phi*w_phi*dx 
      - tau_0*phi_0*w_phi*dx
      + dt*df*w_phi*dx 
      + lamda_0**2*dt*inner(grad(phi), grad(w_phi))*dx 
      + 
       
     ) 

# Weak or variational form of Temperature PDE
R1 = (
        (u - u_0)*w_u*dx - 0.5*(phi- phi_0)*w_u*dx + 
        dt*D*inner(grad(u),grad(w_u))*dx
     )


R = R0 + R1
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)


# Solving the nonlinear problem 
problem = NonlinearProblem(R, com,bcs=[], J=J)
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
T = 20

P0, dof = ME.sub(0).collapse()
# Visualize using PyVista
topology, cell_types, x = plot.vtk_mesh(P0)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = com.x.array[dof].real
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=True)
plotter.view_xy(True)
plotter.add_text(f"time:{t}", font_size=10, name="timelabel")

# Time loop
while t < T:
    t += dt
    res = solver.solve(com)
    print(f"Step {int(t/dt)}: num iteration: {res[0]}")
    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()
    #print(f"min(phi): {phi.x.array.min():.4f}, max(phi): {phi.x.array.max():.4f}")
    grid.point_data["Phase"] = com.x.array[dof].real
    plotter.remove_actor("timelabel")
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    plotter.app.processEvents()

com.x.scatter_forward()
grid.point_data["Phase"] = com.x.array[dof].real
screenshot = None
if pv.OFF_SCREEN:
    screenshot = "phase.png"
pv.plot(grid, show_edges=True, screenshot=screenshot)


print("Simulation complete. Close the window to exit.")
while plotter.app.running:
    time.sleep(0.1)