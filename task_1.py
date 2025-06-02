from petsc4py import PETSc
import dolfinx
from mpi4py import MPI

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


# Step 1: Create mesh
msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh,P1)

w_phi = ufl.TestFunctions(ME) # test function for order parameter

phi = Function(ME) # trial function n+1

phi_0 = Function(ME) # previous value

# Boundary condition application




''' to view the mesh created '''
# outputing the mesh file to view in other source( paraview )
file = XDMFFile(MPI.COMM_WORLD, "demo_ch/output.xdmf", "w")
file.write_mesh(msh)
file.close()


topology, cell_types, x = plot.vtk_mesh(msh,msh.topology.dim)

#  Visualize using PyVista 
if MPI.COMM_WORLD.rank == 0:
    grid = pv.UnstructuredGrid(topology, cell_types, x)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, color="white")
    plotter.view_xy()
    plotter.add_text("Mesh grid", font_size=10)
    plotter.show()













