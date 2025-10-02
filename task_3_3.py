from petsc4py import PETSc
from mpi4py import MPI
import os, time, ufl, numpy as np

from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner

# ---- live viz (optional) ----
ENABLE_LIVE = True
try:
    import pyvista as pv
    import pyvistaqt as pvq
except Exception:
    ENABLE_LIVE = False

# -------------------- Parameters (updated) --------------------
dt       = 0.005
T_final  = 100.0
D        = 2.0
lambda_u = 5.0            # ↑ stronger coupling
eps0     = 0.8            # ↓ slightly thinner interface
delta    = 0.12
theta0   = 0.0
alpha    = 2.5            # ensures m(u)>0 for u∈[-1,0]
gamma    = 1.0
u_bc     = -1.1           # ↑ more undercooling

# stabilizers
etaG     = 1e-12
eps_row  = PETSc.ScalarType(1e-10)
eps_grad = PETSc.ScalarType(1e-12)

# -------------------- Mesh --------------------
Lx, Ly = 300.0, 300.0
Nx, Ny = 300, 300
mesh   = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]],
                          [Nx, Ny], cell_type=CellType.triangle)

# φ : P2, u : P1
Pphi = element("Lagrange", mesh.basix_cell(), 2, dtype=default_real_type)
Pu   = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
ME   = functionspace(mesh, mixed_element([Pphi, Pu]))

w_phi, w_u = ufl.TestFunctions(ME)
com   = Function(ME); com_0 = Function(ME)
phi, u       = ufl.split(com)
phi_0, u_0   = ufl.split(com_0)

# -------------------- Initial conditions (updated seed radius) --------------------
def init_phi(x):
    r = np.sqrt((x[0] - Lx/2.0)**2 + (x[1] - Ly/2.0)**2)
    return np.where(r < 8.0, 1.0, -1.0)   # was 3.0

def init_u(x):
    ux = u_bc * np.ones(x.shape[1], dtype=default_real_type)
    # OPTIONAL: give the seed a slight local cooling "kick" (uncomment if needed)
    # x0, y0 = Lx/2.0, Ly/2.0
    # r2 = (x[0]-x0)**2 + (x[1]-y0)**2
    # ux -= 0.2 * np.exp(-r2 / (8.0**2))  # 0.2 colder in a radius ~8
    return ux

com.x.array[:] = 0.0
com.sub(0).interpolate(init_phi);   com_0.sub(0).interpolate(init_phi)
com.sub(1).interpolate(init_u);     com_0.sub(1).interpolate(init_u)
com.x.scatter_forward();            com_0.x.scatter_forward()

# -------------------- Model terms --------------------
# kinetics m(u) = alpha/pi + atan(gamma u), clamped positive
m_u = alpha/np.pi + ufl.atan(gamma * u)
m_u = ufl.max_value(m_u, PETSc.ScalarType(0.1))

# anisotropy via unit normal (fourfold)
g  = ufl.variable(ufl.grad(phi))
g2 = ufl.inner(g, g)
rg = ufl.sqrt(g2 + etaG)
nx, ny = g[0]/rg, g[1]/rg
c0, s0 = ufl.cos(theta0), ufl.sin(theta0)
nxr =  c0*nx + s0*ny
nyr = -s0*nx + c0*ny
cos4 = 1.0 - 8.0*nxr**2 * nyr**2
eps_n = eps0 * (1.0 + delta * cos4)

# gradient energy and flux
Fgrad = 0.5 * (eps_n**2) * g2
q     = ufl.diff(Fgrad, g)

# bulk derivative (classic double-well + linear thermal coupling)
df = -phi + phi**3 - lambda_u * u * (1 - phi**2)

# -------------------- Weak forms --------------------
dxQ = dx(metadata={"quadrature_degree": 6})

R_phi = (
    m_u * (phi - phi_0) * w_phi * dxQ
  + dt * ufl.inner(q, ufl.grad(w_phi)) * dxQ
  + dt * df * w_phi * dxQ
  + eps_row  * phi * w_phi * dxQ
  + eps_grad * inner(grad(phi), grad(w_phi)) * dxQ
)

R_u = (
    (u - u_0) * w_u * dxQ
  - 0.5 * (phi - phi_0) * w_u * dxQ    # negative latent-heat sign
  + dt * D * inner(grad(u), grad(w_u)) * dxQ
  + eps_row * u * w_u * dxQ
)

R = R_phi + R_u
dcom = ufl.TrialFunction(ME)
J = ufl.derivative(R, com, dcom)

# -------------------- BC: u = u_bc on boundary --------------------
tdim   = mesh.topology.dim
facets = locate_entities_boundary(mesh, tdim-1, lambda x: np.full(x.shape[1], True, dtype=np.bool_))
dofs_u = fem.locate_dofs_topological(ME.sub(1), tdim-1, facets).astype(np.int32)
bc_u   = fem.dirichletbc(PETSc.ScalarType(u_bc), dofs_u, ME.sub(1))

# -------------------- Solver --------------------
opt = PETSc.Options()
opt["snes_type"] = "newtonls"
opt["snes_linesearch_type"] = "bt"
opt["snes_linesearch_damping"] = "0.8"

problem = NonlinearProblem(R, com, bcs=[bc_u], J=J)
solver  = NewtonSolver(mesh.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 40
solver.report = True

ksp = solver.krylov_solver
p = ksp.getOptionsPrefix()
opt[f"{p}ksp_type"]   = "gmres"
opt[f"{p}ksp_rtol"]   = "1e-8"
opt[f"{p}ksp_max_it"] = "200"
opt[f"{p}pc_type"]    = "gamg"
opt[f"{p}mat_type"]   = "aij"
ksp.setFromOptions()

# -------------------- XDMF output (φ P2 -> P1) --------------------
outdir = "results_from_matlab_style"
if mesh.comm.rank == 0 and not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)

V_phi_hi, map_phi = ME.sub(0).collapse()
V_u,       map_u  = ME.sub(1).collapse()
V_phi_io = functionspace(mesh, ("Lagrange", 1))
phi_hi   = Function(V_phi_hi)
phi_io   = Function(V_phi_io); phi_io.name = "phi"
u_out    = Function(V_u);      u_out.name  = "u"

xdmf_phi = XDMFFile(mesh.comm, os.path.join(outdir, "phi.xdmf"), "w")
xdmf_u   = XDMFFile(mesh.comm, os.path.join(outdir, "u.xdmf"), "w")
xdmf_phi.write_mesh(mesh); xdmf_u.write_mesh(mesh)

def write_xdmf(t):
    phi_hi.x.array[:] = com.x.array[map_phi]; phi_hi.x.scatter_forward()
    phi_io.interpolate(phi_hi)
    u_out.x.array[:]  = com.x.array[map_u];   u_out.x.scatter_forward()
    xdmf_phi.write_function(phi_io, t)
    xdmf_u.write_function(u_out, t)

write_xdmf(0.0)

# -------------------- Live viz (optional) --------------------
if ENABLE_LIVE:
    P0, dof = ME.sub(0).collapse()
    topology, cell_types, x = plot.vtk_mesh(P0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)
    grid.point_data["Phase"] = com.x.array[dof].real
    grid.set_active_scalars("Phase")
    plotter = pvq.BackgroundPlotter(title="Phase", auto_update=True)
    plotter.add_mesh(grid, clim=[-1, 1], cmap="coolwarm", show_edges=False)
    plotter.view_xy(True)
    plotter.add_text("time: 0.00", font_size=10, name="timelabel")

# -------------------- Time loop --------------------
t, step = 0.0, 0
WRITE_EVERY = 20

while t < T_final:
    t += dt; step += 1
    its, converged = solver.solve(com)

    # simple diagnostic to ensure φ is moving
    dphi = com.x.array[map_phi] - com_0.x.array[map_phi]
    if mesh.comm.rank == 0:
        print(f"[{step:05d}] Newton iters: {its} | ||Δphi|| ≈ {np.linalg.norm(dphi):.3e}")

    com_0.x.array[:] = com.x.array
    com.x.scatter_forward()

    if (step % WRITE_EVERY == 0) or (t >= T_final):
        write_xdmf(t)

    if ENABLE_LIVE:
        grid.point_data["Phase"] = com.x.array[dof].real
        plotter.remove_actor("timelabel")
        plotter.add_text(f"time: {t:.2e}", font_size=10, name="timelabel")
        plotter.app.processEvents()

xdmf_phi.close(); xdmf_u.close()
print("Done; view XDMF files in ParaView.")

if ENABLE_LIVE:
    grid.point_data["Phase"] = com.x.array[dof].real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = os.path.join(outdir, "phase_last.png")
    pv.plot(grid, show_edges=True, screenshot=screenshot)
    print("Simulation complete. Close the window to exit.")
    while plotter.app.running:
        time.sleep(0.1)
