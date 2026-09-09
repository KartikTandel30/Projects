#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tip_velocity_from_xdmf.py — Extract dendrite tip position/velocity from XDMF

Purpose
-------
Reads a dolfinx-generated XDMF (+ HDF5) time series (phi/u field), picks a
centerline (x- or y-directed), finds the zero crossing of the chosen field
(φ≈0 front) by linear interpolation just beyond the domain center, and
reports:
  • tip position vs time
  • tip velocity vs time (central differences)
It saves a CSV and velocity plots (PNG + SVG).

How it works
------------
1) Opens XDMF via meshio.TimeSeriesReader
2) Selects a centerline row/column of nodes based on `direction` and `center`
3) For each time step, finds the first sign change ( + → − ) beyond the center
   and interpolates the crossing location.
4) Differentiates positions to get velocity; plots/saves results.

Inputs
------
- A directory containing XDMF + matching HDF5 (same stem), or a path to the XDMF.

Outputs (written next to the XDMF)
----------------------------------
- tip_trace.csv              (t; tip_pos; V_tip)
- v_tip_vs_time.png/.svg     (velocity-only panel)

User settings (edit in file)
----------------------------
- SKIP_FIRST: discard initial samples before plotting velocity
- YMIN/YMAX:  set y-axis bounds for velocity panel
- V_THEORY:   draw a horizontal reference line at the theoretical speed

Notes
-----
- Assumes a 2D mesh (x,y). For y-directed growth, set direction='y'.
- Uses Agg backend (headless-safe). Interactive prompts can be disabled
  by passing a path via CLI wrapper or editing defaults.
- This script should be placed next to the XDMF/HDF5 files.
"""

import os, sys, numpy as np, meshio



# Headless plotting
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

PREFERRED_XDMF = ("phi_series.xdmf", "u_series.xdmf")  # preference order
# ================== USER SETTINGS (edit here) ==================
SKIP_FIRST = 0          # number of initial samples to drop from the plot (e.g., 2 or 3)
YMIN, YMAX = None, None # e.g., YMIN=0.045, YMAX=0.075 for a wider range; set to None to auto
V_THEORY = 0.0469         # e.g., 0.050  -> draws a green horizontal line at 0.050
# ===============================================================

def _has_pair_h5(xdmf_path: Path) -> bool:
    """Require a sibling HDF5 with the same stem, e.g. phi_series.xdmf + phi_series.h5."""
    h5 = xdmf_path.with_suffix(".h5")
    if not h5.exists():
        print(f"  Warning: missing HDF5 pair '{h5.name}' for '{xdmf_path.name}'")
        return False
    return True

def find_xdmf_in_folder(folder: str) -> str:
    """
    Find a usable .xdmf inside 'folder'.
    Preference:
      1) 'phi_series.xdmf' then 'u_series.xdmf' if present with matching .h5
      2) Else newest *.xdmf that has a matching .h5
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"'{folder}' is not a directory.")

    # Preferred names first
    for name in PREFERRED_XDMF:
        cand = folder / name
        if cand.is_file() and _has_pair_h5(cand):
            print(f"Selected XDMF: {cand} (preferred)")
            return str(cand)

    # Else newest *.xdmf with a matching .h5
    cands = sorted(folder.glob("*.xdmf"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands = [p for p in cands if _has_pair_h5(p)]
    if not cands:
        raise FileNotFoundError(f"No usable .xdmf (+matching .h5) found in '{folder}'.")
    print(f"Selected XDMF: {cands[0]} (newest)")
    return str(cands[0])

def ask(prompt, default=None, checker=None):
    while True:
        val = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not val and default is not None:
            val = default
        if checker is None or checker(val):
            return val
        print("  → please enter a valid value.")

def is_file(path):
    return os.path.isfile(path)

def run(xdmf_path, field=None, direction="x", center=0.5):
    ts = meshio.xdmf.TimeSeriesReader(xdmf_path)
    base_dir = os.path.dirname(xdmf_path)
    out_csv  = os.path.join(base_dir, "tip_trace.csv")
    out_png_vel = os.path.join(base_dir, "v_tip_vs_time.png")
    out_svg_vel = os.path.join(base_dir, "v_tip_vs_time.svg")

    with ts:
        points, cells = ts.read_points_cells()
        if points.shape[1] < 2:
            print("This looks like a 1D/3D file; need 2D coordinates (x,y).")
            return
        x, y = points[:,0], points[:,1]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        Lx, Ly = xmax - xmin, ymax - ymin
        cx = xmin + center * Lx
        cy = ymin + center * Ly

        ux = np.unique(np.round(x, 12))
        uy = np.unique(np.round(y, 12))
        dx_h = float(np.min(np.diff(ux))) if len(ux) > 1 else Lx/100.0
        dy_h = float(np.min(np.diff(uy))) if len(uy) > 1 else Ly/100.0

        if direction.lower() == "x":
            mask = np.isclose(y, cy, atol=dy_h/2)
            idx = np.where(mask)[0]
            s_line = x[idx]
            order = np.argsort(s_line); idx = idx[order]; s_line = s_line[order]
            s0 = cx
        else:
            mask = np.isclose(x, cx, atol=dx_h/2)
            idx = np.where(mask)[0]
            s_line = y[idx]
            order = np.argsort(s_line); idx = idx[order]; s_line = s_line[order]
            s0 = cy

        def zero_cross(vals):
            sgn = np.sign(vals); s_tip = np.nan
            for k in range(len(vals) - 1):
                if s_line[k] <= s0:
                    continue
                if sgn[k] > 0.0 and sgn[k+1] < 0.0:
                    v1, v2 = vals[k], vals[k+1]
                    s1, s2 = s_line[k], s_line[k+1]
                    a = v1 / (v1 - v2)   # linear interpolation
                    s_tip = s1 + a * (s2 - s1)
            return s_tip

        tip_t, tip_s = [], []
        for k in range(ts.num_steps):
            t, point_data, _ = ts.read_data(k)
            key = field
            if key is None:
                for cand in ("phi","f","phi_0","f_0","phi_sub"):
                    if cand in point_data:
                        key = cand; break
                if key is None:
                    for nm, arr in point_data.items():
                        if getattr(arr, "ndim", 1) == 1:
                            key = nm; break
            if key not in point_data:
                raise KeyError(f"Field '{key}' not found. Available: {list(point_data.keys())}")
            vals = point_data[key][idx]
            s_tip = zero_cross(vals)
            tip_t.append(float(t))
            tip_s.append(float(s_tip) if np.isfinite(s_tip) else np.nan)

        tip_t = np.asarray(tip_t, dtype=float)
        tip_s = np.asarray(tip_s, dtype=float)

        # Central-difference velocity
        V_tip = np.full_like(tip_s, np.nan, dtype=float)
        if len(tip_t) >= 2:
            V_tip[0]  = (tip_s[1] - tip_s[0]) / (tip_t[1] - tip_t[0]) if (tip_t[1]-tip_t[0])!=0 else np.nan
            V_tip[-1] = (tip_s[-1]-tip_s[-2]) / (tip_t[-1]-tip_t[-2]) if (tip_t[-1]-tip_t[-2])!=0 else np.nan
            if len(tip_t) > 2:
                V_tip[1:-1] = (tip_s[2:] - tip_s[:-2]) / (tip_t[2:] - tip_t[:-2])

        # Save CSV
        np.savetxt(out_csv, np.c_[tip_t, tip_s, V_tip], delimiter=";", fmt="%.6g",
                   header="t; tip_pos; V_tip", comments="")

        # Plot ONLY velocity
        m = np.isfinite(tip_t) & np.isfinite(V_tip)
        t_plot = tip_t[m]
        v_plot = V_tip[m]

        # Skip initial transient points (no re-scaling)
        if SKIP_FIRST > 0 and SKIP_FIRST < len(t_plot):
            t_plot = t_plot[SKIP_FIRST:]
            v_plot = v_plot[SKIP_FIRST:]

        fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=180)

        # simulation line (blue) + markers
        sim_line, = ax.plot(t_plot, v_plot, lw=1.8, color="C0", label="Simulation")
        ax.plot(t_plot, v_plot, ms=3.2, color="C0")

        # theory as a thin green horizontal line (only if provided)
        if V_THEORY is not None:
            label_theory = f"Theory = {V_THEORY:.5g}"
            th_line = ax.axhline(V_THEORY, color="black", lw=1.8, label=label_theory)

        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\nu_{\mathrm{tip}}$")

        # widen y-range if you set YMIN/YMAX at the top
        if YMIN is not None and YMAX is not None:
            ax.set_ylim(YMIN, YMAX)

        ax.grid(True, alpha=0.3)

        # compact legend (only if theory is drawn)
        if V_THEORY is not None:
            ax.legend(loc="upper right",
                    fontsize=8,        # smaller text
                    frameon=False,     # no box
                    handlelength=2.6,  # shorter handles
                    borderpad=0.2, labelspacing=0.2)

        fig.tight_layout()
        fig.savefig(out_png_vel)
        fig.savefig(out_svg_vel)
        plt.close(fig)

        print("\nSaved CSV :", out_csv)
        print("Saved plot:", out_png_vel)
        print("Saved plot:", out_svg_vel)
        print(f"Last: t={tip_t[-1]:.6g}, tip={tip_s[-1]:.6g}, V_tip≈{V_tip[-1]:.6g}")

if __name__ == "__main__":

    print("=== Tip Velocity (XDMF) ===")
    # Default to the folder where THIS script lives
    script_dir = Path(__file__).expanduser().resolve().parent
    print(f"(Default folder = {script_dir})")
    print("Give me a FOLDER (with .xdmf + .h5) OR a full path to an .xdmf.")
    print("Press Enter to use the script's folder.")

    # Let the user override; default is the script's folder
    path_in = ask("Folder OR .xdmf path", default=str(script_dir))

    p = Path(path_in).expanduser().resolve()

    # Resolve to an XDMF file path
    if p.is_dir():
        xdmf_path = find_xdmf_in_folder(p)   # uses your helper from earlier
    elif p.is_file() and p.suffix.lower() == ".xdmf":
        if not _has_pair_h5(p):
            print("\nERROR: matching HDF5 is missing "
                  f"('{p.with_suffix('.h5').name}'). Put it next to the .xdmf.")
            sys.exit(1)
        xdmf_path = str(p)
    else:
        print("\nERROR: please provide a folder or a .xdmf file path.")
        sys.exit(1)

    print(f"Using XDMF: {xdmf_path}")  # <-- helpful confirmation
    

    direction = ask("Growth direction (x/y)", default="x",
                    checker=lambda s: s.lower() in ("x","y"))
    field = ask("Field name (Enter for auto-detect)", default="")
    field = field if field else None

    def chk_center(v):
        try:
            f = float(v); return 0.0 <= f <= 1.0
        except: return False
    center = float(ask("Centerline position from 0..1 (0.5 = midline)",
                       default="0.5", checker=chk_center))

    try:
        run(xdmf_path, field=field, direction=direction, center=center)
    except Exception as e:
        print("\nERROR:", e)
        print("Tip: ensure the .h5 sits next to the .xdmf and the field name is correct.")
        sys.exit(1)
