
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# bc_check.py — sanity checks for homogeneous Neumann (zero-flux) BCs
#
# Purpose
#   Compute boundary flux diagnostics for scalar fields φ and u on a dolfinx mesh:
#     • Net flux (can cancel):  ∫_∂Ω ∇φ·n ds,  ∫_∂Ω ∇u·n ds
#     • L¹-like residuals:     ∫_∂Ω |∇φ·n| ds, ∫_∂Ω |∇u·n| ds
#     • L² residuals:          (∫_∂Ω (∇φ·n)² ds)½, (∫_∂Ω (∇u·n)² ds)½
#
# Use
#   ok, (Fφ, Fu), (r1φ, r1u), (r2φ, r2u) = bc_check(mesh, phi, u, tol=1e-8)
#
# Notes
#   • For natural BCs arising from gradient terms, the *net* flux should be ~0.
#   • L¹/L² residuals quantify “how zero” the flux is regardless of cancellation.
#   • Pass a custom ds measure (e.g. marked boundary) via ds_measure if needed.
#   • Ensure ghosted DoFs are up-to-date (set ensure_scatter=True if unsure).
# ---------------------------------------------------------------------------

from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem


def bc_check(msh, phi, u, tol=1e-8, print_fn=print):
    """
    Check zero-flux (homogeneous Neumann) boundary conditions for φ and u.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        The mesh (used for facet normals and communicator).
    phi, u : dolfinx.fem.Function
        Scalar fields to be checked.
    tol : float, default 1e-8
        Tolerance for the net-flux pass/fail check.
    print_fn : callable, default print
        Logger for rank-0 output.
    ds_measure : ufl.Measure, optional
        Custom boundary measure (e.g., Measure('ds', domain=msh, subdomain_data=...)).
        If None, uses ufl.ds(domain=msh) over the whole exterior boundary.
    ensure_scatter : bool, default False
        If True, calls .x.scatter_forward() on both phi and u before assembly.

    Returns
    -------
    ok : bool
        True iff both net fluxes are below `tol`.
    (F_phi, F_u) : tuple of floats
        Net flux integrals over ∂Ω.
    (r1_phi, r1_u) : tuple of floats
        L¹-like boundary residuals ∫|∇•·n| ds.
    (r2_phi, r2_u) : tuple of floats
        L² boundary residuals (∫(∇•·n)² ds)^½.
    """
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
