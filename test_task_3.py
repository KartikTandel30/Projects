#!/usr/bin/env python3
"""
Postprocess FEniCSx XDMF snapshots to reproduce 'test2.py' diagnostics & plots.

Outputs (in the XDMF folder):
  - diagnostics.csv        (t, Energy, DissipationProxy, DeltaIntU, HalfDeltaIntPhi, misfit)
  - energy.png
  - misfit.png
  - misfit_smooth.png

Requirements: numpy, matplotlib, meshio, h5py (meshio usually pulls h5py)
"""

import argparse
import math
from pathlib import Path
import numpy as np
import csv

# Matplotlib (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Try meshio; give a friendly message if missing
try:
    import meshio
except Exception as e:
    raise SystemExit(
        "\n[error] This script needs 'meshio' (and numpy, matplotlib). "
        "Install in your environment (e.g., 'pip install meshio') and rerun.\n"
    )

def triangle_areas(points, tri_conn):
    """
    Compute areas of triangles and the constant gradients of the P1 shape functions.
    points: (N,2), tri_conn: (M,3) int
    Returns:
      areas: (M,)
      gradN: (M,3,2) where gradN[k,i,:] = grad N_i on triangle k
    """
    X = points[tri_conn]  # (M,3,2)
    x1, y1 = X[:,0,0], X[:,0,1]
    x2, y2 = X[:,1,0], X[:,1,1]
    x3, y3 = X[:,2,0], X[:,2,1]

    # Twice area = (x2-x1)(y3-y1) - (x3-x1)(y2-y1)
    twoA = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
    areas = 0.5 * np.abs(twoA)

    # For linear P1 triangle, gradients of shape functions are constant:
    # grad N1 = [ (y2 - y3), (x3 - x2) ] / (2A), etc.
    denom = 2.0*areas
    # Avoid /0 for degenerate elements
    denom_safe = np.where(denom == 0.0, 1.0, denom)

    g1x = (y2 - y3) / denom_safe
    g1y = (x3 - x2) / denom_safe
    g2x = (y3 - y1) / denom_safe
    g2y = (x1 - x3) / denom_safe
    g3x = (y1 - y2) / denom_safe
    g3y = (x2 - x1) / denom_safe

    gradN = np.stack([np.stack([g1x, g1y], axis=1),
                      np.stack([g2x, g2y], axis=1),
                      np.stack([g3x, g3y], axis=1)], axis=1)  # (M,3,2)
    return areas, gradN

def cellwise_grad_scalar_p1(phi, gradN, tri_conn):
    """
    Compute cellwise constant gradients of a P1 scalar field.
    phi: (N,) nodal
    gradN: (M,3,2)
    tri_conn: (M,3)
    returns grad_phi: (M,2)
    """
    phi_cells = phi[tri_conn]  # (M,3)
    # grad(phi) = sum_i phi_i * gradN_i
    grad_phi = np.einsum("m i, m i a -> m a", phi_cells, gradN)
    return grad_phi  # (M,2)

def cell_integrate_p1_scalar(f_nodes, tri_conn, areas):
    """
    Integrate a scalar P1 function over each triangle using vertex average * area.
    Returns total integral over domain.
    """
    vals = f_nodes[tri_conn]  # (M,3)
    avg_on_cell = np.mean(vals, axis=1)  # (M,)
    return float(np.sum(avg_on_cell * areas))

def energy_density_per_cell(phi_nodes, u_nodes, grad_phi_cells, lam, zeta, tri_conn, areas):
    """
    Compute ∫_cell [ (lam^2/2)*|grad phi|^2 + W(phi,u) ] dΩ for each cell and sum.
    W(phi,u) = -1/2 φ^2 + 1/4 φ^4 + ζ u ( φ - 2/3 φ^3 + 1/5 φ^5 )
    For the potential part, use vertex-averaged value; for gradient term, constant per cell.
    """
    # Gradient term (constant per cell)
    grad2 = np.sum(grad_phi_cells**2, axis=1)        # (M,)
    grad_term = 0.5*(lam**2) * grad2 * areas        # (M,)

    # Potential term: vertex average on the cell
    phi_c = phi_nodes[tri_conn]                     # (M,3)
    u_c   = u_nodes[tri_conn]                       # (M,3)

    W_v = (-0.5*phi_c**2 + 0.25*phi_c**4
           + zeta*u_c*(phi_c - (2.0/3.0)*phi_c**3 + (1.0/5.0)*phi_c**5))  # (M,3)
    W_avg = np.mean(W_v, axis=1)                    # (M,)
    pot_term = W_avg * areas                         # (M,)

    return float(np.sum(grad_term + pot_term))

def read_timeseries_xdmf(xdmf_path, phi_name="phi", u_name="u"):
    """
    Read a time series XDMF with point_data for phi and u.
    Returns: points(N,2), tri_conn(M,3), times(K,), list_phi[K](N,), list_u[K](N,)
    Notes: designed for 2D triangles and point fields named phi/u.
    """
    xdmf_path = Path(xdmf_path)
    with meshio.xdmf.TimeSeriesReader(str(xdmf_path)) as ts:
        points, cells = ts.read_points_cells()
        # Find triangle connectivity
        tri_conn = None
        for c in cells:
            if c.type in ("triangle",):
                tri_conn = c.data
                break
        if tri_conn is None:
            raise RuntimeError("No 'triangle' cells found in XDMF.")

        times = []
        list_phi = []
        list_u = []

        # Some writers store data under point_data; others under ts.read_data(k)
        num_steps = ts.num_steps
        for k in range(num_steps):
            t, point_data, _ = ts.read_data(k)
            times.append(float(t))

            # Try exact names; otherwise try to guess first scalar
            def pick(name, fallback="__first__"):
                if name in point_data:
                    arr = point_data[name]
                    return np.ravel(arr.astype(float))
                if fallback == "__first__" and len(point_data):
                    # Pick first 1D array
                    for key, arr in point_data.items():
                        a = np.asarray(arr)
                        if a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1):
                            return np.ravel(a.astype(float))
                raise KeyError(f"Field '{name}' not found in point_data keys {list(point_data.keys())}")

            phi_k = pick(phi_name)
            u_k   = pick(u_name)
            list_phi.append(phi_k)
            list_u.append(u_k)

    P = np.asarray(points, float)
    if P.shape[1] != 2:
        raise RuntimeError(f"This script currently supports 2D only, got points shape {P.shape}.")
    T = np.asarray(tri_conn, dtype=int)
    return P, T, np.array(times), list_phi, list_u

def moving_average(y, win=7):
    if win < 2 or win > len(y):
        return y.copy()
    c = np.convolve(y, np.ones(win)/win, mode="valid")
    # Pad to original length
    padL = (len(y) - len(c))//2
    padR = len(y) - len(c) - padL
    return np.pad(c, (padL, padR), mode="edge")

def main():
    ap = argparse.ArgumentParser(description="Postprocess XDMF snapshots into diagnostics (energy, misfit, etc.)")
    ap.add_argument("xdmf", type=str, help="Path to time-series XDMF (dolfinx output)")
    ap.add_argument("--lam", type=float, default=1.0, help="Interface thickness λ")
    ap.add_argument("--zeta", type=float, default=1.0, help="Coupling ζ (a.k.a. xi)")
    ap.add_argument("--phi-name", type=str, default="phi", help="Point field name for phase field")
    ap.add_argument("--u-name", type=str, default="u", help="Point field name for temperature/undercooling")
    ap.add_argument("--out-dir", type=str, default="", help="Output directory (default: alongside XDMF)")
    args = ap.parse_args()

    xdmf_path = Path(args.xdmf)
    if not xdmf_path.exists():
        raise SystemExit(f"[error] XDMF not found: {xdmf_path}")

    out_dir = Path(args.out_dir) if args.out_dir else xdmf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[i] reading {xdmf_path.name} ...")
    P, T, times, list_phi, list_u = read_timeseries_xdmf(xdmf_path, args.phi_name, args.u_name)
    print(f"[ok] steps: {len(times)}, points: {P.shape[0]}, triangles: {T.shape[0]}")

    # Precompute geometric data
    areas, gradN = triangle_areas(P, T)

    H_vals = []           # Energy Π(t)
    IntU_vals = []        # ∫ u
    IntPhi_vals = []      # ∫ φ
    DissProxy_vals = []   # ΔΠ per step (<=0 ideally for AC)
    times_list = list(times)

    prev_H = None
    for k, (phi_k, u_k) in enumerate(zip(list_phi, list_u)):
        # Gradients
        grad_phi = cellwise_grad_scalar_p1(phi_k, gradN, T)  # (M,2)

        # Energy
        H = energy_density_per_cell(phi_k, u_k, grad_phi, args.lam, args.zeta, T, areas)
        H_vals.append(H)

        # Simple domain integrals using P1 vertex average
        IntU = cell_integrate_p1_scalar(u_k, T, areas)
        IntPhi = cell_integrate_p1_scalar(phi_k, T, areas)
        IntU_vals.append(IntU)
        IntPhi_vals.append(IntPhi)

        if prev_H is None:
            DissProxy_vals.append(0.0)
        else:
            # “DissipationProxy”: energy drop per step (negative for monotone decay).
            # If your original code used ∫(τ0 φ̇^2) dt or similar, replace this with that definition.
            DissProxy_vals.append(H - prev_H)
        prev_H = H

    # Build deltas vs initial snapshot (k=0)
    IntU0, IntPhi0 = IntU_vals[0], IntPhi_vals[0]
    DeltaIntU = [u - IntU0 for u in IntU_vals]
    HalfDeltaIntPhi = [0.5*(p - IntPhi0) for p in IntPhi_vals]
    misfit = [du - hdp for du, hdp in zip(DeltaIntU, HalfDeltaIntPhi)]

    # Write CSV (same header as your test2.py)
    csv_path = out_dir / "diagnostics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "Energy", "DissipationProxy", "DeltaIntU", "HalfDeltaIntPhi", "misfit"])
        for row in zip(times_list, H_vals, DissProxy_vals, DeltaIntU, HalfDeltaIntPhi, misfit):
            w.writerow(row)
    print(f"[ok] wrote {csv_path}")

    # === PLOTS ===

    # 1) Energy vs time
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(times_list, H_vals, lw=1.2, label="Π(t)")
    ax.set_xlabel("time")
    ax.set_ylabel(r"Energy $\Pi$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "energy.png", dpi=300)
    plt.close(fig)

    # 2) Misfit (tight axis)
    mis = np.array(misfit, dtype=float)
    t  = np.array(times_list, dtype=float)

    absmax = float(np.max(np.abs(mis))) if mis.size else 1.0
    pad = 0.05*absmax
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(t, mis, lw=1.1, label="misfit")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax.set_ylim(-absmax - pad, absmax + pad)
    ax.set_xlabel("time")
    ax.set_ylabel(r"misfit $=\ \Delta\!\int u - 0.5\,\Delta\!\int\phi$")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set_title("Misfit (tight axis)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "misfit.png", dpi=300)
    plt.close(fig)

    # 3) Optional smoothed misfit
    if mis.size >= 20:
        mis_s = moving_average(mis, win=max(5, mis.size//50))
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(t, mis, lw=0.7, alpha=0.35, label="raw")
        ax.plot(t, mis_s, lw=1.4, label="smoothed (MA)")
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("time")
        ax.set_ylabel(r"misfit $=\ \Delta\!\int u - 0.5\,\Delta\!\int\phi$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "misfit_smooth.png", dpi=300)
        plt.close(fig)

    print("[done] Plots saved: energy.png, misfit.png", "(+ misfit_smooth.png)" if mis.size >= 20 else "")

if __name__ == "__main__":
    main()
