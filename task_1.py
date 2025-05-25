import numpy as np
import dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import pyvista as pv

from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner

mesh=create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)  #creating mesh basic
P1 = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type) #slecting the elemnt type
ME = functionspace(mesh, mixed_element([P1, P1])) #







