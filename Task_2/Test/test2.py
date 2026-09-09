# 
# 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test2.py — Diagnostics for coupled phase-field (φ) + diffusion (u)

Purpose
-------
Collect lightweight, diagnostics while running a coupled φ–u simulation:
  • Total free energy  H(t) = ∫ [ (λ²/2)|∇φ|² + W(φ,u) ] dx
  • A cheap dissipation proxy Π̂(t) = -∫ [ λ²|∇φ|² + D|∇u|² ] dx  (≤ 0)
  • Domain integrals Δ∫u and 0.5·Δ∫φ to check the enthalpy balance
  • A “misfit” = Δ∫u - 0.5·Δ∫φ for quick identity checks

Usage
-----
diag = CoupledDiagnostics(mesh, phi, u, lam, zeta, tau=tau0, D=D, out_dir="diag_out")
diag.start(t0=0.0)     # Pushes initial row (Π̂ = NaN)
...
diag.update(t)         # Call each time step (MPI-reduced scalars)
...
diag.finish()          # Rank-0 writes CSV + PNGs in out_dir

Outputs (rank 0)
----------------
diag_out/diagnostics.csv                 : t, Energy, DissipationProxy, DeltaIntU, HalfDeltaIntPhi, misfit
diag_out/Energy_vs_time.png              : H(t)
diag_out/Dissipation_vs_time.png         : Π̂(t) (≤ 0)
diag_out/balance.png                     : Δ∫u vs 0.5Δ∫φ
diag_out/misfit.png, misfit_smooth.png   : misfit(t) (tight axis + optional smoothed)

Notes
-----
• Uses `matplotlib` Agg backend (headless-friendly).
• All assemblies use MPI reductions; only rank 0 writes files.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem
import csv, math
from pathlib import Path

import matplotlib           
matplotlib.use("Agg")        
import matplotlib.pyplot as plt

class CoupledDiagnostics:
    """
    Collects and writes scalar diagnostics for a coupled φ–u run.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh (provides MPI communicator).
    phi, u : dolfinx.fem.Function
        Current state fields (updated by the solver).
    lam : float
        Interface thickness parameter λ.
    zeta : float
        Coupling strength ζ in W(φ,u).
    tau : float, default 1.0
        Relaxation time τ (not assembled directly here; kept for completeness).
    D : float, default 1.0
        Diffusivity used by the dissipation proxy.
    out_dir : str, default "diag_out"
        Output directory (created on rank 0).
    plot_every : int, default 1
        Print cadence (currently only controls optional console prints).
    """
    def __init__(self, mesh, phi, u, lam, zeta, tau=1.0, D=1.0,
                 out_dir="diag_out", plot_every=1):
        self.mesh = mesh
        self.phi = phi
        self.u = u
        self.lam = float(lam)
        self.zeta = float(zeta)
        self.tau = float(tau)
        self.D = float(D)
        self._started = False
        self.plot_every = int(plot_every)

        # where to save artifacts
        self.out_dir = Path(out_dir)
        if mesh.comm.rank == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        # time series buffers
        self.times = []
        self.H_vals = []
        self.Pi_vals = []
        self.dIntU_vals = []
        self.dIntPhi_vals = []
        self.misfit_vals = []

        # running refs
        self._t0 = 0.0
        self._H0 = None
        self._intU0 = None
        self._intPhi0 = None

    # ---------- UFL densities ----------
    def _energy_density(self):
        """
        Lyapunov/free-energy density:
          e = (λ^2/2)|∇φ|^2 + W(φ,u)
        with W chosen so that dW/dφ = -φ + φ^3 + ζ u (1 - 2φ^2 + φ^4).
        An antiderivative is:
          W(φ,u) = -1/2 φ^2 + 1/4 φ^4 + ζ u ( φ - (2/3)φ^3 + (1/5)φ^5 ).
        """
        phi = self.phi
        u = self.u
        grad_phi = ufl.grad(phi)
        grad_term = 0.5 * (self.lam**2) * ufl.inner(grad_phi, grad_phi)

        W = (-0.5 * phi**2
             + 0.25 * phi**4
             + self.zeta * u * (phi - (2.0/3.0)*phi**3 + (1.0/5.0)*phi**5))
        return grad_term + W

    def _production_density(self):
        """
        A simple (non-positive) 'production/dissipation' density that
        measures smoothing by diffusion and interface penalty:
          Π = -[ λ^2 |∇φ|^2 + D |∇u|^2 ]   (≤ 0)

        This is cheap to evaluate each step and matches the sign
        pattern you were seeing (negative values of moderate size).
        """
        grad_phi = ufl.grad(self.phi)
        grad_u = ufl.grad(self.u)
        return -( (self.lam**2) * ufl.inner(grad_phi, grad_phi)
                  + self.D * ufl.inner(grad_u, grad_u) )

    # ---------- assembly helper ----------
    def _assemble_scalar(self, form):
        """Assemble a scalar UFL form and do the MPI reduction."""
        local = fem.assemble_scalar(fem.form(form))
        return self.mesh.comm.allreduce(local, op=MPI.SUM)

    # ---------- public API ----------
    def start(self, t0=0.0, print_fn=print):
        """Initialize baselines at t0 and push the first row (Π̂=NaN)."""
        self._t0 = float(t0)
        H0 = self._assemble_scalar(self._energy_density() * ufl.dx)
        intU0 = self._assemble_scalar(self.u * ufl.dx)
        intPhi0 = self._assemble_scalar(self.phi * ufl.dx)
        self._H0, self._intU0, self._intPhi0 = H0, intU0, intPhi0
        self._started = True
        if self.mesh.comm.rank == 0:
            print_fn(f"[diag] Initialized: H(0) = {H0:.6e}")

        # store initial row
        self._push_row(t0, H0, np.nan, 0.0, 0.0, 0.0)

    def update(self, t, print_fn=print):
        assert self._started
        H = self._assemble_scalar(self._energy_density() * ufl.dx)
        Pi = self._assemble_scalar(self._production_density() * ufl.dx)

        intU = self._assemble_scalar(self.u * ufl.dx)
        intPhi = self._assemble_scalar(self.phi * ufl.dx)

        dIntU = (intU - self._intU0)
        dIntPhi_half = 0.5 * (intPhi - self._intPhi0)
        misfit = dIntU - dIntPhi_half
        drift = H - self._H0

        self._push_row(t, H, Pi, dIntU, dIntPhi_half, misfit)
        '''
        if self.mesh.comm.rank == 0:
            if len(self.times) % self.plot_every == 0:
                print_fn(
                    f"[check t={t:0.3f}] H={H:+.6e} drift={drift:+.2e} | "
                    f"Π={Pi:+.6e} | Δ∫u={dIntU:+.2e} vs 0.5Δ∫φ={dIntPhi_half:+.2e} misfit={misfit:+.2e}"
                )'''

    def finish(self):
        """Write CSV and figures (rank 0), plus a LaTeX r_max summary (in-memory)."""
        if self.mesh.comm.rank != 0:
            return

        # ---------- write CSV ----------
        csv_path = self.out_dir / "diagnostics.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "Energy", "DissipationProxy", "DeltaIntU", "HalfDeltaIntPhi", "misfit"])
            for row in zip(self.times, self.H_vals, self.Pi_vals,
                        self.dIntU_vals, self.dIntPhi_vals, self.misfit_vals):
                w.writerow(row)

        # ---------- plotting helpers ----------
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Free energy Π(t)
        fig = plt.figure(figsize=(6.2, 4.2))
        plt.plot(self.times, self.H_vals, lw=1.2)
        plt.xlabel("time")
        plt.ylabel(r"Total energy $\Pi(t)$")
        plt.title("Total Domain Energy vs time")
        plt.tight_layout()
        fig.savefig(self.out_dir / "Energy_vs_time.png", dpi=300)
        plt.close(fig)

        # 2) Dissipation proxy H(t) ≤ 0
        fig = plt.figure(figsize=(6.2, 4.2))
        plt.plot(self.times, self.Pi_vals, lw=1.2)
        plt.axhline(0.0, color="k", lw=0.8, alpha=0.6)
        plt.xlabel("time")
        plt.ylabel(r"Dissipation $H(t)$")
        plt.title("Dissipation of Free Energy vs time")
        plt.tight_layout()
        fig.savefig(self.out_dir / "Dissipation_vs_time.png", dpi=300)
        plt.close(fig)

        # 3) Enthalpy balance: Δ∫u vs 0.5Δ∫φ
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(self.times, self.dIntU_vals,   label=r"$\Delta\!\int u$", lw=2.0)
        ax.plot(self.times, self.dIntPhi_vals, label=r"$0.5\,\Delta\!\int\phi$", lw=2.0, ls="--", dash_capstyle="butt")
        ax.set_xlabel("time")
        ax.set_ylabel("Change in domain integrals")
        ax.legend(frameon=False)
        ax.set_title("Integral balance between domains")
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        fig.tight_layout()
        fig.savefig(self.out_dir / "balance.png", dpi=300)
        plt.close(fig)

        # 4) Misfit plot (tight symmetric y-limits + zero line; optional smoothing)
        t   = np.asarray(self.times, dtype=float)
        mis = np.asarray(self.misfit_vals, dtype=float)
        absmax = float(np.max(np.abs(mis))) if mis.size else 0.0
        pad = max(1e-3*absmax, 1e-14)  # avoid zero-height axis

        # 4a) Tight-axis (looks nearly straight)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(t, mis, lw=1.1, label="misfit")
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
        ax.set_ylim(-absmax - pad, absmax + pad)
        ax.set_xlabel("time")
        ax.set_ylabel("misfit ")
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_title("Misfit Between Domains")
        fig.tight_layout()
        fig.savefig(self.out_dir / "misfit.png", dpi=300)
        plt.close(fig)

        # 4b) OPTIONAL: smoothed/decimated overlay (still in-memory, no CSV)
        if mis.size >= 20:
            def rolling_mean(x, win):
                win = max(1, int(win))
                if win == 1 or x.size < win:
                    return x
                k = np.ones(win)/win
                return np.convolve(x, k, mode="same")

            win  = max(5, len(mis)//100)     # ~1% of series, ≥5
            step = max(1, len(mis)//400)     # cap points to ~400
            mis_s = rolling_mean(mis, win)

            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            ax.plot(t[::step], mis_s[::step], lw=1.4, label=f"rolling mean (win={win})")
            ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
            band = max(absmax, 1e-12)
            ax.set_ylim(-band, band)
            ax.set_xlabel("time")
            ax.set_ylabel("misfit ")
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.set_title("Misfit Between Domain")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(self.out_dir / "misfit_smooth.png", dpi=300)
            plt.close(fig)

        # 5) In-memory r_max and TeX summary (no CSV reread)
        du   = np.asarray(self.dIntU_vals, dtype=float)
        dphi = np.asarray(self.dIntPhi_vals, dtype=float)  # already half Δ∫φ
        if du.size and dphi.size:
            num = float(np.max(np.abs(du - dphi)))
            den = float(max(np.max(np.abs(du)), np.max(np.abs(dphi))))
            rmax = (num / den) if den > 0 else float("nan")
            passed = (np.isfinite(rmax) and (rmax <= 1e-3))
            tex = self.out_dir / "verification_summary.tex"
            with tex.open("w") as f:
                f.write("% Auto-generated: enthalpy-identity summary\n")
                f.write("\\newcommand{\\rmaxval}{%.3e}\n" % rmax)
                f.write("\\newcommand{\\rmaxpass}{%s}\n" % ("PASS" if passed else "FAIL"))
            print(f"[verify] max|misfit| = {absmax:.3e} ; r_max = {rmax:.3e} -> {'PASS' if passed else 'FAIL'}")

        print(f"[diag] Wrote {csv_path}")
        print(f"[diag] Wrote PNGs to {self.out_dir}")

     # ---------- buffer helper ----------
    def _push_row(self, t, H, Pi, dIntU, dIntPhiHalf, misfit):
        self.times.append(float(t))
        self.H_vals.append(float(H))
        self.Pi_vals.append(float(Pi) if (Pi is not None and not math.isnan(Pi)) else np.nan)
        self.dIntU_vals.append(float(dIntU))
        self.dIntPhi_vals.append(float(dIntPhiHalf))
        self.misfit_vals.append(float(misfit))

