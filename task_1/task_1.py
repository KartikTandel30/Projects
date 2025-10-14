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
from dolfinx.io import XDMFFile
import pathlib
from ufl import SpatialCoordinate, conditional, lt, gt, as_ufl

t_start = time.time()
# Parameter constants used, taken from the ref. paper
zet = 1.6   # coupling constant
u = -0.75   # Temparature
tau_0 = 1   # Characteristic time scale 
lamda_0 = 1   # characteristic interface thickness
dt = 0.04   # time step

# Create mesh
Lx, Ly = 100.0, 100.0
msh = create_rectangle(MPI.COMM_WORLD,
                       [[0.0, 0.0], [Lx, Ly]],
                       [50, 50],
                       cell_type=CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, P1)

w_phi = ufl.TestFunction(ME)  # test function for order parameter
phi = Function(ME)  # trial function n+1
phi_0 = Function(ME)  # previous value


'''
R0 = 5.0
w_eq = np.sqrt(2.0) * lamda_0 

def initial_phi(x):
    xc = x[0] - Lx/2.0
    yc = x[1] - Ly/2.0
    r  = np.sqrt(xc**2 + yc**2)
    return np.tanh((R0 - r) / w_eq)
'''
w0= 1 
def initial_phi(x):
    phi0_expr = conditional(lt(x[1], w0), 1.0, -1.0)
    return phi0_expr

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
      + lamda_0**2*dt*inner(grad(phi), grad(w_phi))*dx ) 

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


results_dir = pathlib.Path("phase_output_T1")
results_dir.mkdir(parents=True, exist_ok=True)

# Open once, write mesh once, then write functions per time step
xdmf_path = results_dir / "phi.xdmf"
xdmf = XDMFFile(msh.comm, str(xdmf_path), "w")
xdmf.write_mesh(ME.mesh)

t = 0.0
T = 200 


while t < T:
    t += dt

    its, converged = solver.solve(phi)
    #print(f"Step {step}: Newton iterations = {its} ({'OK' if converged else 'NOT CONV'})")

    phi_0.x.array[:] = phi.x.array
    phi.x.scatter_forward()
    xdmf.write_function(phi, t)
    phi_vals_all = phi.x.array[:]
    if (np.allclose(phi_vals_all,  1.0, atol=1e-3) or np.allclose(phi_vals_all, -1.0, atol=1e-3)):
        print("Domain fully saturated at ±1. Ending early.")
        break


xdmf.close()  
if msh.comm.rank == 0:
    print("Open in ParaView. File -> Open -> output_task_1.xdmf")
if msh.comm.rank == 0:
    print(f"Total runtime: {time.time()-t_start:.2f}s")
    