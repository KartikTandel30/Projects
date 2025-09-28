# test2.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem
import csv, math
from pathlib import Path
import matplotlib.pyplot as plt

class CoupledDiagnostics:
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

        if self.mesh.comm.rank == 0:
            if len(self.times) % self.plot_every == 0:
                print_fn(
                    f"[check t={t:0.3f}] H={H:+.6e} drift={drift:+.2e} | "
                    f"Π={Pi:+.6e} | Δ∫u={dIntU:+.2e} vs 0.5Δ∫φ={dIntPhi_half:+.2e} misfit={misfit:+.2e}"
                )

    def finish(self):
        """Write CSV and figures (rank 0)."""
        if self.mesh.comm.rank != 0:
            return

        csv_path = self.out_dir / "diagnostics.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "H", "Pi", "DeltaIntU", "HalfDeltaIntPhi", "misfit"])
            for row in zip(self.times, self.H_vals, self.Pi_vals,
                           self.dIntU_vals, self.dIntPhi_vals, self.misfit_vals):
                w.writerow(row)

        # Plots
        fig1 = plt.figure(); plt.plot(self.times, self.H_vals)
        plt.xlabel("time"); plt.ylabel("Energy H"); plt.tight_layout()
        fig1.savefig(self.out_dir / "H_vs_time.png", dpi=200); plt.close(fig1)

        fig2 = plt.figure(); plt.plot(self.times, self.Pi_vals)
        plt.xlabel("time"); plt.ylabel("Production Π"); plt.tight_layout()
        fig2.savefig(self.out_dir / "Pi_vs_time.png", dpi=200); plt.close(fig2)

        fig3 = plt.figure()
        plt.plot(self.times, self.dIntU_vals, label="Δ∫u")
        plt.plot(self.times, self.dIntPhi_vals, label="0.5 Δ∫φ")
        plt.xlabel("time"); plt.ylabel("change in integrals")
        plt.legend(); plt.tight_layout()
        fig3.savefig(self.out_dir / "balance.png", dpi=200); plt.close(fig3)

        fig4 = plt.figure(); plt.plot(self.times, self.misfit_vals)
        plt.xlabel("time"); plt.ylabel("misfit = Δ∫u - 0.5Δ∫φ")
        plt.tight_layout()
        fig4.savefig(self.out_dir / "misfit.png", dpi=200); plt.close(fig4)

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
