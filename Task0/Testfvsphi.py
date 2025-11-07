#!/usr/bin/env python3
"""
Bulk free-energy plotter for a phase-field dendrite model.

This script plots the theoretical bulk free energy density f(φ, u)
against the order parameter φ and overlays a single simulation node's
trajectory (φ_node_N(t), f_node_N(t)) loaded from an Excel file.

Panels:
  - TOP: f(φ, u) for one or more undercoolings u (theory only)
  - BOTTOM: light-gray theory + black dashed simulation trajectory with
            a direction arrow and the final point highlighted

Expected Excel columns (by default with a Greek phi in the header):
  - "ϕ_node_{N}" : φ time series at node N
  - "f_node_{N}" : f time series at node N
If those are missing, the loader will try ASCII fallbacks:
  - "phi_node_{N}" and "f_node_{N}"

Dependencies:
  numpy, pandas, matplotlib, openpyxl (for reading .xlsx)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, AutoMinorLocator

# =========================
# Config
# =========================
node_number = 100    # which node to plot from the Excel file
file_path   = "phiAndBulkData.xlsx"  # input Excel file path

zeta        = 1.6                # model parameter ζ 
u_list      = [-0.75]                       # list of undercoolings u to plot theory curves for  
phi_range   = np.linspace(-1.5, 1.5, 1201)    # φ range for theory curves
outname     = f"f_vs_phi_node{node_number}_stacked.png"   # output figure filename

# =========================
# Load data
# =========================
df = pd.read_excel(file_path)     # radinf Excel file

phi_col = f"ϕ_node_{node_number}"      # try Greek phi first
f_col   = f"f_node_{node_number}"        # corresponding f column     

if phi_col not in df.columns or f_col not in df.columns:      # Error mesage if missing
    raise ValueError(f"Node {node_number} not found in the Excel file! "
                     f"Missing columns: {phi_col} and/or {f_col}")

phi_values = df[phi_col].to_numpy()    # extract data arrays 
f_values   = df[f_col].to_numpy()       

# Clean NaNs from data for plotting 
mask = np.isfinite(phi_values) & np.isfinite(f_values)
phi_values = phi_values[mask]
f_values   = f_values[mask]


# =========================
# Model helpers
# =========================
def f_bulk(phi, zeta, u):
    
    """f(φ,u) = -1/2 φ^2 + 1/4 φ^4 + ζ u φ (1 - 2/3 φ^2 + 1/5 φ^4)
    Bulk free energy density as a function of order parameter φ and undercooling u.
    ϕ : order parameter (can be array)
    ζ : model parameter
    u : undercooling

    Returns: f(φ, u)
    """
    return (-0.5*phi**2 + 0.25*phi**4
            + zeta*u*phi*(1.0 - (2.0/3.0)*phi**2 + (1.0/5.0)*phi**4))

# =========================
# Figure (two stacked subplots) + one overall title
# =========================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7.6, 8.4), sharex=True,
    gridspec_kw={'hspace': 0.18}, dpi=300
)

# ---- TOP: theoretical curve ----
for uu in u_list:
    F_u = f_bulk(phi_range, zeta, uu)
    ax1.plot(phi_range, F_u, lw=2, label=fr"$u={uu:g}$")

ax1.set_ylabel(r"$f(\phi, u)$", fontsize=12)
ax1.grid(True, which="major", alpha=0.25)
ax1.grid(True, which="minor", alpha=0.10)
ax1.legend(ncol=min(3, len(u_list)), fontsize=12, frameon=False, loc="best")

# ---- BOTTOM: theoretical (light) + simulation trajectory ----
for uu in u_list:
    F_u = f_bulk(phi_range, zeta, uu)
    ax2.plot(phi_range, F_u, lw=1.2, alpha=0.6, color="#888888")

ax2.plot(
    phi_values, f_values,
    linestyle='--', marker='o',
    markevery=max(1, len(phi_values)//25),
    markersize=3.5, linewidth=1.3, color='black',
    label=f"Simulation", zorder=6
)

# emphasizing last point + direction arrow
if len(phi_values) > 2:
    ax2.scatter(phi_values[-1], f_values[-1], s=36,
                facecolors='none', edgecolors='black', linewidths=1.3, zorder=7)
    k0 = max(0, int(0.9*len(phi_values))-1)
    ax2.annotate("", xy=(phi_values[-1], f_values[-1]),
                 xytext=(phi_values[k0], f_values[k0]),
                 arrowprops=dict(arrowstyle="->", lw=1.1, color='black'))

ax2.set_xlabel(r"$\phi$", fontsize=1)
ax2.set_ylabel(r"$f(\phi, u)$", fontsize=12)
ax2.grid(True, which="major", alpha=0.25)
ax2.grid(True, which="minor", alpha=0.10)
ax2.legend(loc="best", fontsize=12, frameon=False)

# ---- Ticks: shared x, fixed major ticks + minor ticks ----
xticks = [-1.5, -1.0,-0.5, 0.0,0.5, 1.0, 1.5]   # or [-1, 0, 1]
for ax in (ax1, ax2):
    ax.set_xlim(phi_range.min(), phi_range.max())
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(labelsize=10)

# ---- One overall title (suptitle), no per-axes titles ----
fig.suptitle(r"Theory and Simulation Plot of Bulk function($f(\phi, u)$) and Order parameter($\phi$)",y=0.94, fontsize=12)
 # leave room for suptitle

# ---- Save + show ----
fig.savefig(outname, bbox_inches="tight")
print(f"[OK] saved {outname}")
plt.show()
