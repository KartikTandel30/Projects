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

#Parameter constants used, taken from the ref. paper
zet = 1.6
u = -0.75
tau_0 = 1 # Characteristic time scale 
lamda_0 = 1 #characteristic interface thickness
dt = 0.01 #time step

#  Create mesh
msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh,P1)

w_phi = ufl.TestFunctions(ME) # test function for order parameter
phi = Function(ME) # trial function n+1
phi_0 = Function(ME) # previous value

# Boundary condition application
#Weak form and ealuation of equation
ph = ufl.variable(phi) 
f = -0.5*ph**2 + 0.25*ph**4 + zet*u*ph*(1-(2/3)*ph**2+0.2*ph**4)
df = ufl.diff(f , ph)
print(ufl.algorithms.expand_derivatives(df))

#  weak or variational form for the task-1
R0 = ( tau_0*inner(phi, w_phi)*dx - tau_0*inner(phi_0, w_phi)
      + dt*inner(df,w_phi)*dx
      + lamda_0*dt*inner(grad(phi),grad(w_phi))*dx 
)

# solving the nonlinear problem 
problem = NonlinearProblem(R0, phi)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
solver.atol = 1e-12
solver.max_it = 25
solver.report = True

# setting the type of the solver
ksp = solver.krylov_solver
opt = PETSc.Options()
opt_prefix = ksp.getOptionsPrefix
opt[f"{opt_prefix}ksp_type"] = "preonly"
opt[f"{opt_prefix}pc_type"] = "lu"
sys = PETSc.Sys()

if sys.hasExternalPackage("superlu_dist"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opt[f"{opt_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()



''' to view the mesh created 
# outputing the mesh file to view in other source( paraview )
file = XDMFFile(MPI.COMM_WORLD, "demo_ch/output.xdmf", "w")
file.write_mesh(msh)
file.close()
'''

topology, cell_types, x = plot.vtk_mesh(msh,msh.topology.dim)

#  Visualize using PyVista 
if MPI.COMM_WORLD.rank == 0:
    grid = pv.UnstructuredGrid(topology, cell_types, x)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, color="white")
    plotter.view_xy()
    plotter.add_text("Mesh grid", font_size=10)
    plotter.show()













