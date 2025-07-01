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
import ufl.differentiation

#Parameter constants used, taken from the ref. paper
zet = 1.6
u = -0.75
tau_0 = 1 # Characteristic time scale 
lamda_0 = 1 #characteristic interface thickness
dt = 0.01 #time step

#  Create mesh
#msh = create_unit_square(MPI.COMM_WORLD, 100, 100, CellType.triangle)
msh = create_rectangle(MPI.COMM_WORLD,
                       [[0.0, 0.0], [10.0, 10.0]],  # New domain corners
                       [50, 50],               # More elements for resolution
                       cell_type=CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh,P1)

w_phi = ufl.TestFunction(ME) # test function for order parameter
phi = Function(ME) # trial function n+1
phi_0 = Function(ME) # previous value


# Boundary and Initial condition application
#intial condition
def initial_phi(x):
    r = np.sqrt((x[0] - 5)**2 + (x[1] - 5)**2)
    print("radius=",r)
    return np.where(r < 0.05, 1.0, -1.0)

phi.interpolate(initial_phi)
phi_0.interpolate(initial_phi)
phi.x.scatter_forward()
phi_0.x.scatter_forward()

# No flux boundary condition should be applied 


#Weak form and ealuation of equation
'''
ph = ufl.variable(phi) 
f = -0.5*ph**2 + 0.25*ph**4 + zet*u*ph*(1-(2/3)*ph**2+0.2*ph**4)
df = ufl.derivative(f, ph)'''
#df = (-phi + phi**3 + zet*u*(1 - 2*phi + phi**4)) - (-phi_0 + phi_0**3 + zet*u*(1 - 2*phi_0 + phi_0**4))
df = -phi + phi**3 + zet*u*(1- 2*phi**2 + phi**4)
#print(df)
#print(ufl.algorithms.expand_derivatives(df))

#  weak or variational form for the task-1
R0 = ( tau_0*inner(phi, w_phi)*dx 
      - tau_0*inner(phi_0, w_phi)*dx
      + dt*inner(df,w_phi)*dx
      + lamda_0**2*dt*inner(grad(phi),grad(w_phi))*dx 
)
print(R0)
# solving the nonlinear problem 
problem = NonlinearProblem(R0, phi)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

opt = PETSc.Options()
opt["snes_monitor"] = ""  # to print the  newton residue
opt["ksp_type"] = "preonly"
opt["pc_type"] = "lu"

solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
solver.atol = 1e-12
solver.max_it = 25
solver.report = True

solver.krylov_solver.setFromOptions()

t = 0.0
T = 10
''' to view the mesh created 
# outputing the mesh file to view in other source( paraview )
file = XDMFFile(MPI.COMM_WORLD, "demo_ch/output.xdmf", "w")
file.write_mesh(msh)
file.close()
'''
#  Visualize using PyVista
#topology, cell_types, x = plot.vtk_mesh(msh,msh.topology.dim)
topology, cell_types, x = plot.vtk_mesh(ME)
grid = pv.UnstructuredGrid(topology, cell_types, x)
grid.point_data["Phase"] = phi.x.array
grid.set_active_scalars("Phase")
plotter = pvq.BackgroundPlotter(title="Phase", auto_update = True)
plotter.add_mesh(grid, clim =[-1,1], cmap= "coolwarm", show_edges=True)
#contours = grid.contour(isosurfaces=[0.0])
#plotter.add_mesh(contours, color="black", line_width=2, name="contour")
plotter.view_xy(True)
plotter.add_text(f"time:{t}", font_size=10, name="timelable")
plotter.show()

phi_0.x.array[:]= phi.x.array

#time loop
while t < T:
    t += dt
    res = solver.solve(phi)
    print(f"Step {int(t/dt)}: num iteration: {res[0]}")
    phi_0.x.array[:] = phi.x.array
    print(f"min(phi): {phi.x.array.min():.4f}, max(phi): {phi.x.array.max():.4f}")
    #plotting the updated phase
    plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
    grid.point_data["Phase"] = phi.x.array
    plotter.app.processEvents()

phi.x.scatter_forward()
grid.point_data["Phase"] = phi.x.array
screenshot = None
if pv.OFF_SCREEN:
    screenshot = "phase.png"
pv.plot(grid, show_edges = True, screenshot=screenshot)











