#!/usr/bin/env python3
"""
Task 3 (phase-field + temperature), FEniCSx 0.9.0
- Reads a CSV once (same folder by default) or falls back to built-in defaults.
- Copies parameters into typed locals for clarity and validation.
- Uses short two-word helper names.
- Adds optional masked Gaussian noise to the initial phase field.
"""

import csv
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.random import default_rng
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from ufl import dx, grad, inner, Identity, outer, as_vector, sqrt, dot  # use ufl.TestFunctions / ufl.TrialFunction

from dolfinx import default_real_type
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.fem import Constant, Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from basix.ufl import element, mixed_element


# -------------------------- Parameter handling -------------------------- #
def loadParams(csv_path: str) -> Tuple[Dict[str, float | int | str], str]:
    """
    Loads parameters from a key,value CSV if present; otherwise returns built-in defaults.

    Behavior
    --------
    - If `csv_path` is empty, the function searches 'params_task3.csv' in the same folder as this script.
    - If the file is missing, defaults are used and source="defaults" is returned.
    - If the file exists, defaults are overridden by CSV entries and source is the file path.

    Returns
    -------
    params : dict
        Final parameter dictionary (defaults possibly overridden by CSV values).
    source : str
        "defaults" when using the built-in set, otherwise the resolved CSV file path.
    """
    DEFAULTS = {
        # physics
        "zet": 1.6, "tau_0": 1.0, "lamda_0": 1.0, "D": 1.0, "K": 0.5,
        "eps_an": 0.05, "eta": 1e-8, "u0": -0.75,
        # mesh / time
        "Lx": 1.0, "Ly": 1.0, "Nx": 500, "Ny": 500,
        "dt": 0.04, "T": 100.0, "save_stride": 10,
        # seed
        "R0": 0.0125,
        # noise on phi0
        "noise_amp": 5e-4, "noise_seed": 12345, "noise_masked": 1,
        # IO
        "out_file": "output_task_3.xdmf",
        # Newton / linear solver
        "newton_rtol": 1e-8, "newton_atol": 1e-10, "newton_max_it": 100,
    }

    path = Path(csv_path) if csv_path else Path(__file__).resolve().parent / "params_task3.csv"

    if not path.exists():
        print(f"[INFO] No parameter file found at: {path}")
        print("[INFO] Using built-in defaults.\n")
        return DEFAULTS.copy(), "defaults"

    params = DEFAULTS.copy()

    def _auto(v: str):
        v = v.strip()
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return v  # keep as string

    with open(path, "r", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#") or len(row) < 2:
                continue
            params[row[0].strip()] = _auto(row[1])

    print(f"[INFO] Loaded parameters from: {path}\n")
    return params, str(path)


# -------------------------- Problem construction -------------------------- #
def meshCreation(comm: MPI.Comm, Lx: float, Ly: float, Nx: int, Ny: int):
    """
    Creates a 2D triangular mesh on [0, Lx] × [0, Ly] with (Nx, Ny) intervals.

    Returns
    -------
    msh : dolfinx.mesh.Mesh
        The constructed mesh.
    """
    return create_rectangle(
        comm,
        [[0.0, 0.0], [float(Lx), float(Ly)]],
        n=(Nx, Ny),
        cell_type=CellType.triangle
    )


def functionSpaces(msh):
    """
    Builds the mixed function space ME = P1(phi) × P1(u).

    Returns
    -------
    ME : dolfinx.fem.FunctionSpace
        Mixed space for (phi, u).
    P1 : basix.ufl.element
        The scalar P1 element used to form the mixed space.
    """
    P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
    ME = functionspace(msh, mixed_element([P1, P1]))
    return ME, P1


def initial_phi_tanh_factory(Lx: float, Ly: float, R0: float, lamda_0: float):
    """
    Builds a tanh-profile initializer for phi: tanh((R0 - r)/w_eq), with w_eq = sqrt(2)*lamda_0.

    Returns
    -------
    f : callable
        Function mapping coordinates x (shape (2, npts)) to the initial phi values.
    """
    w_eq = np.sqrt(2.0) * lamda_0

    def _phi0(x):
        xc = x[0] - Lx / 2.0
        yc = x[1] - Ly / 2.0
        r = np.sqrt(xc * xc + yc * yc)
        return np.tanh((R0 - r) / w_eq)

    return _phi0


def initial_u_constant_factory(u0: float):
    """
    Builds a constant initializer for u that returns u0 everywhere.

    Returns
    -------
    f : callable
        Function mapping coordinates x to an array filled with u0.
    """
    def _u0(x):
        return u0 * np.ones(x.shape[1], dtype=default_real_type)
    return _u0


def initFields(
    ME,
    Lx: float, Ly: float, R0: float, lamda_0: float,
    u0: float, noise_amp: float, noise_seed: int, noise_masked: int
):
    """
    Initializes mixed states com/com_0 (phi,u) with a tanh seed and optional masked noise.

    Returns
    -------
    com : Function
        Current mixed state (sub(0)=phi, sub(1)=u).
    com_0 : Function
        Previous mixed state initialized equal to com.
    phi, u : UFL symbols
        The UFL split of the current state used for weak-form assembly.
    """
    com   = Function(ME, name="com")
    com_0 = Function(ME, name="com_0")

    # Initial conditions
    phi0_f = initial_phi_tanh_factory(Lx, Ly, R0, lamda_0)
    u0_f   = initial_u_constant_factory(u0)
    com.x.array[:] = 0.0
    com.sub(0).interpolate(phi0_f)
    com.sub(1).interpolate(u0_f)
    com.x.scatter_forward()

    # Optional masked Gaussian noise on phi
    if float(noise_amp) > 0.0:
        rng = default_rng(int(noise_seed))
        phi_vec = com.sub(0).x.array
        mask = np.clip(1.0 - phi_vec**2, 0.0, 1.0) if int(noise_masked) == 1 else 1.0
        phi_vec += float(noise_amp) * mask * rng.standard_normal(phi_vec.shape)
        np.clip(phi_vec, -1.0, 1.0, out=phi_vec)
        com.sub(0).x.array[:] = phi_vec
        com.x.scatter_forward()

    # Previous ← current
    com_0.x.array[:] = com.x.array
    com_0.x.scatter_forward()

    phi, u = ufl.split(com)
    return com, com_0, phi, u


def buildForms(
    msh, ME, phi, u, com, com_0,
    zet: float, tau_0: float, lamda_0: float, dt: float, D: float,
    eps_an_val: float, eta_val: float, K_val: float
):
    """
    Assembles the nonlinear residual R and Jacobian J for the coupled anisotropic system.

    Returns
    -------
    R : UFL form
        The total residual for the (phi,u) system.
    J : UFL form
        The Jacobian of R with respect to the mixed unknown.
    (w_phi, w_u) : tuple
        The test functions used in the weak forms.
    """
    w_phi, w_u = ufl.TestFunctions(ME)
    phi_0, u_0 = ufl.split(com_0)

    eps_an = Constant(msh, default_real_type(eps_an_val))
    eta    = Constant(msh, default_real_type(eta_val))
    Kcoup  = Constant(msh, default_real_type(K_val))

    # Free-energy derivative
    df = -phi + phi**3 + zet * u * (1 - 2 * phi**2 + phi**4)

    # Anisotropy (four-fold) with regularization
    gphi = grad(phi)
    g2   = inner(gphi, gphi)
    ng   = sqrt(g2 + eta**2)
    nHat = gphi / ng

    d = msh.geometry.dim
    I = Identity(d)
    P = I - outer(nHat, nHat)

    a     = (1.0 - 3.0 * eps_an) + 4.0 * eps_an * (nHat[0]**4 + nHat[1]**4)
    da_dn = as_vector((16.0 * eps_an * nHat[0]**3, 16.0 * eps_an * nHat[1]**3))
    da_dg = dot(P, da_dn) / ng

    q_phi = lamda_0**2 * (a**2 * gphi + g2 * a * da_dg)
    F_grad_aniso = inner(q_phi, grad(w_phi)) * dx
    tau = tau_0 * a**2

    # Weak forms
    R_phi = ( tau * (phi - phi_0) * w_phi * dx
              + dt * df * w_phi * dx
              + dt * F_grad_aniso )

    R_u = ( (u - u_0) * w_u * dx
            - Kcoup * (phi - phi_0) * w_u * dx
            + dt * D * inner(grad(u), grad(w_u)) * dx )

    R = R_phi + R_u
    dcom = ufl.TrialFunction(ME)
    J = ufl.derivative(R, com, dcom)
    return R, J, (w_phi, w_u)


def solverSetup(msh, R, J, newton_rtol: float, newton_atol: float, newton_max_it: int) -> NewtonSolver:
    """
    Configures a Newton solver using preonly+LU (SuperLU_DIST/MUMPS if available).

    Returns
    -------
    solver : NewtonSolver
        The assembled nonlinear solver ready to advance the system.
    """
    problem = NonlinearProblem(R, None, bcs=[], J=J)
    solver = NewtonSolver(msh.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = float(newton_rtol)
    solver.atol = float(newton_atol)
    solver.max_it = int(newton_max_it)
    solver.report = True

    ksp = solver.krylov_solver
    opt = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    opt[f"{prefix}ksp_type"] = "preonly"
    opt[f"{prefix}pc_type"]  = "lu"

    sys = PETSc.Sys()
    if sys.hasExternalPackage("superlu_dist"):
        opt[f"{prefix}pc_factor_mat_solver_type"] = "superlu_dist"
    elif sys.hasExternalPackage("mumps"):
        opt[f"{prefix}pc_factor_mat_solver_type"] = "mumps"

    if msh.comm.rank == 0:
        print("\n>>> Using Direct LU solver (SuperLU_DIST / MUMPS if available)\n")
    ksp.setFromOptions()
    return solver


def timeLoop(
    msh, ME, solver: NewtonSolver, com: Function, com_0: Function,
    dt: float, T: float, save_stride: int, out_file: str,
    on_step=None, on_save=None
):
    """
    Advances the solution in time and writes phi snapshots to an XDMF file.

    Returns
    -------
    stats : dict
        A summary with keys {"steps", "saves", "wall_s"}.
    """
    phi_sub = com.sub(0); phi_sub.name = "phi"
    t, step = 0.0, 0
    t0 = time.time()
    n_saves, n_steps = 0, 0

    with XDMFFile(msh.comm, out_file, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_sub, 0.0)

        if msh.comm.rank == 0:
            print("Starting time-simulation...")

        while t < T:
            t += dt; step += 1
            its, converged = solver.solve(com)

            if on_step:
                on_step(step, t, its, converged, com)

            com_0.x.array[:] = com.x.array
            com.x.scatter_forward()

            if step % save_stride == 0:
                if on_save:
                    on_save(step, t, com)
                xdmf.write_function(phi_sub, t)
                n_saves += 1
            n_steps += 1

    return {"steps": n_steps, "saves": n_saves, "wall_s": time.time() - t0}


# ---------------------------------- Main ---------------------------------- #
def main():
    """
    Serves as the CLI entry-point:
    - Reads './params_task3.csv' (same folder) when available, otherwise uses defaults.
    - Copies values into typed locals, builds the problem, prints a one-time run configuration,
      and executes the time-stepping loop.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Task 3 (locals) — same-folder params, FEniCSx 0.9.0")
    parser.add_argument("--params", type=str, default="", help="(Optional) path to key,value CSV")
    args = parser.parse_args()

    # Parameters
    p, p_source = loadParams(args.params)

    # Locals (typed)
    Lx = float(p["Lx"]);   Ly = float(p["Ly"])
    Nx = int(p["Nx"]);     Ny = int(p["Ny"])
    dt = float(p["dt"]);   T  = float(p["T"])
    save_stride = int(p["save_stride"])
    out_file = str(p["out_file"])

    zet = float(p["zet"])
    tau_0 = float(p["tau_0"])
    lamda_0 = float(p["lamda_0"])
    D = float(p["D"]);     K = float(p["K"])
    eps_an_val = float(p["eps_an"])
    eta_val = float(p["eta"])
    u0 = float(p["u0"]);   R0 = float(p["R0"])

    noise_amp = float(p["noise_amp"])
    noise_seed = int(p["noise_seed"])
    noise_masked = int(p["noise_masked"])

    newton_rtol = float(p["newton_rtol"])
    newton_atol = float(p["newton_atol"])
    newton_max_it = int(p["newton_max_it"])

    # Basic checks
    assert Nx > 0 and Ny > 0, "Nx, Ny must be positive"
    assert Lx > 0 and Ly > 0, "Lx, Ly must be positive"
    assert dt > 0 and T >= 0, "dt must be > 0 and T >= 0"

    # One-time summary (covers defaults and CSV cases)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[RUN CONFIG] Source: {p_source}")
        print(f"  Domain: Lx={Lx}, Ly={Ly}, Nx={Nx}, Ny={Ny}")
        print(f"  Time:   dt={dt}, T={T}, save_stride={save_stride}")
        print(f"  Phys:   lamda_0={lamda_0}, tau_0={tau_0}, zet={zet}, D={D}, K={K}")
        print(f"  Aniso:  eps_an={eps_an_val}, eta={eta_val}")
        print(f"  Seed:   R0={R0}, u0={u0}")
        print(f"  Noise:  amp={noise_amp}, seed={noise_seed}, masked={noise_masked}")
        print(f"  Solver: rtol={newton_rtol}, atol={newton_atol}, max_it={newton_max_it}")
        print(f"  Output: {out_file}\n")

    # Build problem
    comm = MPI.COMM_WORLD
    msh = meshCreation(comm, Lx, Ly, Nx, Ny)
    ME, _ = functionSpaces(msh)
    com, com_0, phi, u = initFields(ME, Lx, Ly, R0, lamda_0, u0, noise_amp, noise_seed, noise_masked)
    R, J, _ = buildForms(msh, ME, phi, u, com, com_0,
                         zet, tau_0, lamda_0, dt, D, eps_an_val, eta_val, K)
    solver = solverSetup(msh, R, J, newton_rtol, newton_atol, newton_max_it)

    # Time stepping
    stats = timeLoop(msh, ME, solver, com, com_0, dt, T, save_stride, out_file)
    if comm.rank == 0:
        print(f"[done] steps={stats['steps']} saves={stats['saves']} wall={stats['wall_s']:.2f}s")


if __name__ == "__main__":
    main()
