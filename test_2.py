import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTKFile

# Parameters
domain_size = 200.0
mesh_resolution = 100

dt = 0.01  # Time step
t_final = 1.0  # Total simulation time

# Constants
lambda_0 = 1.0
xi = 1.60
T_0 = 1.0
u = -0.75  # scaled non-dimensional temperature

# Create mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [
    [0.0, 0.0], [domain_size, domain_size]], [mesh_resolution, mesh_resolution], mesh.CellType.triangle)

element = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
V = fem.FunctionSpace(domain, element)

# Initial condition
phi_n = fem.Function(V)
phi_0 = -1.0
phi_n.x.array[:] = phi_0

# Solid nucleus
x = domain.geometry.x
center = np.array([domain_size / 2, domain_size / 2])
radius = 5.0
initial_seed = np.linalg.norm(x - center, axis=1) < radius
phi_n.x.array[initial_seed] = 1.0

# Define trial and test functions
phi = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Free energy
def f(phi):
    return -0.5 * phi**2 + 0.25 * phi**4 + xi * nu * phi * (1 - (2/3)*phi**2 + (1/5)*phi**4)

# Weak form
F = T_0 * (phi - phi_n) / dt * v * ufl.dx \
    + ufl.derivative(f(phi), phi, v) * ufl.dx \
    + lambda_0**2 * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx

a, L = ufl.lhs(F), ufl.rhs(F)

# Define solver
problem = LinearProblem(a, L, bcs=[], petsc_options={})

# Solution function
t = 0.0
phi = fem.Function(V)
vtk = VTKFile(domain.comm, "results/phi.pvd", "w")

while t < t_final:
    phi.x.array[:] = problem.solve()
    phi_n.x.array[:] = phi.x.array
    vtk.write_function(phi, t)
    t += dt
    print(f"Time step completed: t = {t:.3f}")
