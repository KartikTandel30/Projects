# test_bc.py
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem
'''
def bc_check(msh, phi, u, tol=1e-8, print_fn=print):
    """
    Post-run boundary-condition check:
      - Verifies zero normal flux (homogeneous Neumann) for phi and u:
            ∫_{∂Ω} ∇phi · n ds  ≈ 0
            ∫_{∂Ω} ∇u   · n ds  ≈ 0
    Prints PASS/FAIL with the absolute residuals.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
    phi : ufl/Coefficient (the phase-field Function)
    u   : ufl/Coefficient (the temperature Function)
    tol : float   tolerance for PASS/FAIL
    print_fn : callable  (defaults to print or PETSc.Sys.Print)
    """
    n  = ufl.FacetNormal(msh)
    ds = ufl.ds(domain=msh)

    # Local boundary fluxes
    F_phi_loc = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(phi), n) * ds))
    F_u_loc   = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u),   n) * ds))

    # Global reductions (MPI)
    F_phi = msh.comm.allreduce(F_phi_loc, op=MPI.SUM)
    F_u   = msh.comm.allreduce(F_u_loc,   op=MPI.SUM)

    ok = (abs(F_phi) < tol) and (abs(F_u) < tol)

    if msh.comm.rank == 0:
        print_fn("[BC test] Checking homogeneous Neumann (zero-flux) BCs")
        print_fn(f"[BC test] ∫∂Ω ∇φ·n ds = {F_phi:+.3e}")
        print_fn(f"[BC test] ∫∂Ω ∇u ·n ds = {F_u:+.3e}")
        print_fn(f"[BC test] {'PASSED' if ok else 'FAILED'} (tol={tol:g})")

    return ok, F_phi, F_u
'''

def bc_check(msh, phi, u, tol=1e-8, print_fn=print):
    n  = ufl.FacetNormal(msh)
    ds = ufl.ds(domain=msh)

    # Net flux (can cancel)
    F_phi = msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(phi), n) * ds)), op=MPI.SUM)
    F_u   = msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u),   n) * ds)), op=MPI.SUM)

    # L1-like residuals without abs(): use sqrt(q^2 + eps)
    eps = 0.0  # or small like 1e-30 if you want strict positivity
    q_phi = ufl.inner(ufl.grad(phi), n)
    q_u   = ufl.inner(ufl.grad(u),   n)

    r1_phi = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.sqrt(q_phi*q_phi + eps) * ds)), op=MPI.SUM)
    r1_u   = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.sqrt(q_u*q_u   + eps) * ds)), op=MPI.SUM)

    # L2 residuals
    r2_phi = (msh.comm.allreduce(
        fem.assemble_scalar(fem.form((q_phi*q_phi) * ds)), op=MPI.SUM))**0.5
    r2_u   = (msh.comm.allreduce(
        fem.assemble_scalar(fem.form((q_u*q_u) * ds)), op=MPI.SUM))**0.5

    ok = (abs(F_phi) < tol) and (abs(F_u) < tol)

    if msh.comm.rank == 0:
        print_fn("[BC test] Checking homogeneous Neumann (zero-flux) BCs")
        print_fn(f"[BC test] Net flux:  ∫∂Ω ∇φ·n ds = {F_phi:+.3e},  ∫∂Ω ∇u·n ds = {F_u:+.3e}")
        print_fn(f"[BC test] L1-resid:  ∫∂Ω |∇φ·n| ds ≈ {r1_phi:.3e},  ∫∂Ω |∇u·n| ds ≈ {r1_u:.3e}")
        print_fn(f"[BC test] L2-resid:  (∫∂Ω (∇φ·n)^2 ds)^½ = {r2_phi:.3e},  (∫∂Ω (∇u·n)^2 ds)^½ = {r2_u:.3e}")
        print_fn(f"[BC test] {'PASSED' if ok else 'FAILED'} (net-flux tol={tol:g})")

    return ok, (F_phi, F_u), (r1_phi, r1_u), (r2_phi, r2_u)
