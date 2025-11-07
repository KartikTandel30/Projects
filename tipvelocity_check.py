import os, sys
import numpy as np
import meshio
from pathlib import Path

# ================== USER SETTINGS ==================
SKIP_FIRST = 0              # drop first N points from plot & averaging
YMIN, YMAX = None, None     # set axis limits or None for auto
V_THEORY = 0.0469           # theory line (set None to hide)
AUTO_ALIGN = True           # auto-pick x or y each step from tip orientation
DEFAULT_FIELD = "phi"       # hard-lock to phi unless you pass another name
PREFERRED_XDMF = ("phi_series.xdmf", "u_series.xdmf")
# ===================================================

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------- small io helpers -----------------
def ask(prompt, default=None, checker=None):
    while True:
        val = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not val and default is not None:
            val = default
        if checker is None or checker(val):
            return val
        print("  → please enter a valid value.")

def _has_pair_h5(xdmf_path: Path) -> bool:
    h5 = xdmf_path.with_suffix(".h5")
    if not h5.exists():
        print(f"  Warning: missing HDF5 pair '{h5.name}' for '{xdmf_path.name}'")
        return False
    return True

def find_xdmf_in_folder(folder: str) -> str:
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"'{folder}' is not a directory.")

    # 1) try preferred stems first
    for name in PREFERRED_XDMF:
        cand = folder / name
        if cand.is_file() and _has_pair_h5(cand):
            print(f"Selected XDMF: {cand} (preferred)")
            return str(cand)

    # 2) newest *.xdmf that has a sibling .h5
    cands = sorted(folder.glob("*.xdmf"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands = [p for p in cands if _has_pair_h5(p)]
    if not cands:
        raise FileNotFoundError(f"No usable .xdmf (+matching .h5) found in '{folder}'.")
    print(f"Selected XDMF: {cands[0]} (newest)")
    return str(cands[0])


# ----------------- triangle helpers -----------------
def _tri_connectivity(cells):
    """
    meshio TimeSeriesReader.read_points_cells() returns a list like:
      [('triangle', ndarray(...)), ('line', ...), ...]
    Return the triangle connectivity ndarray (nT x 3, dtype=int).
    """
    # handle both tuple API and meshio CellBlock API
    for blk in cells:
        if isinstance(blk, tuple):
            ctype, conn = blk
            if ctype in ("triangle", "tri"):
                return np.asarray(conn, dtype=int)
        else:
            # CellBlock
            if getattr(blk, "type", "") in ("triangle", "tri"):
                return np.asarray(blk.data, dtype=int)
    raise RuntimeError("No triangular cells found in this XDMF mesh.")


def _tri_centroids(points, tri):
    # average of the three vertex coordinates
    return points[tri].mean(axis=1)  # (nT, 3?) -> (nT,2) for 2D


def _grad_on_triangle(xy, values):
    """
    Gradient of a P1 scalar over a single triangle.
    xy: (3,2) coords; values: (3,) nodal values at the triangle vertices
    Returns (dfdx, dfdy)
    """
    (x1, y1), (x2, y2), (x3, y3) = xy
    f1, f2, f3 = values
    # 2A = det
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if det == 0.0:
        return 0.0, 0.0
    inv2A = 1.0 / det
    dfdx = inv2A * (f1 * (y2 - y3) + f2 * (y3 - y1) + f3 * (y1 - y2))
    dfdy = inv2A * (f1 * (x3 - x2) + f2 * (x1 - x3) + f3 * (x2 - x1))
    return float(dfdx), float(dfdy)


# ----------------- main computation -----------------
def run(xdmf_path, field=DEFAULT_FIELD, direction="x", center=0.5,
        plateau=None):
    """
    plateau: (t_a, t_b) or None (auto last third)
    AUTO_ALIGN: if True, per-step pick 'x' or 'y' closer to gradient direction at the tip (one-step lag).
    """
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
        x, y = points[:, 0], points[:, 1]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        Lx, Ly = xmax - xmin, ymax - ymin
        cx = xmin + center * Lx
        cy = ymin + center * Ly

        # grid steplike tolerances to select a "centerline" band
        ux = np.unique(np.round(x, 12))
        uy = np.unique(np.round(y, 12))
        dx_h = float(np.min(np.diff(ux))) if len(ux) > 1 else Lx / 200.0
        dy_h = float(np.min(np.diff(uy))) if len(uy) > 1 else Ly / 200.0

        # precompute line index caches for x- and y-probes
        def _line_cache(dir_char):
            if dir_char == "x":
                mask = np.isclose(y, cy, atol=dy_h / 2)
                idx = np.where(mask)[0]
                s = x[idx]
                order = np.argsort(s)
                idx = idx[order]
                s = s[order]
                s0 = cx
            else:
                mask = np.isclose(x, cx, atol=dx_h / 2)
                idx = np.where(mask)[0]
                s = y[idx]
                order = np.argsort(s)
                idx = idx[order]
                s = s[order]
                s0 = cy
            return idx, s, s0

        cache = {
            "x": _line_cache("x"),
            "y": _line_cache("y"),
        }

        # mesh topology for orientation estimate near the tip
        tri = _tri_connectivity(cells)
        tri_cent = _tri_centroids(points[:, :2], tri)

        def zero_cross(vals, s_line, s0):
            """Return first zero-crossing beyond s0 (linear interp). Python float or np.nan."""
            sgn = np.sign(vals)
            s_tip = np.nan
            for k in range(len(vals) - 1):
                if s_line[k] <= s0:
                    continue
                if sgn[k] > 0.0 and sgn[k + 1] < 0.0:
                    v1, v2 = float(vals[k]), float(vals[k + 1])
                    s1, s2 = float(s_line[k]), float(s_line[k + 1])
                    if (v1 - v2) == 0.0:
                        continue
                    a = v1 / (v1 - v2)  # linear interpolation fraction
                    s_tip = s1 + a * (s2 - s1)
                    break
            return float(s_tip) if np.isfinite(s_tip) else np.nan

        # storage
        tip_t, tip_s, tip_dir = [], [], []   # times, 1D position, 'x'/'y'
        V_tip = []                           # instantaneous velocity
        theta_hist = []                      # orientation angle at tip (radians)

        # per-step direction (starts with user's choice)
        cur_dir = direction.lower()

        for k in range(ts.num_steps):
            t, point_data, _ = ts.read_data(k)

            # field selection (hard default to 'phi' unless you pass something else)
            key = field
            if key not in point_data:
                # try a few common fallbacks
                for cand in ("phi", "f", "phi_0", "f_0", "phi_sub"):
                    if cand in point_data:
                        key = cand
                        break
            if key not in point_data:
                raise KeyError(f"Field '{field}' not in this XDMF. Available: {list(point_data.keys())}")

            phi_vals_all = np.asarray(point_data[key]).reshape(-1)

            # choose line indices for current step
            idx, s_line, s0 = cache[cur_dir]
            vals = phi_vals_all[idx]
            s_tip = zero_cross(vals, s_line, s0)

            # position of the tip point in (x,y)
            if np.isfinite(s_tip):
                if cur_dir == "x":
                    x_tip, y_tip = s_tip, cy
                else:
                    x_tip, y_tip = cx, s_tip

                # find nearest triangle centroid to tip point
                d2 = (tri_cent[:, 0] - x_tip) ** 2 + (tri_cent[:, 1] - y_tip) ** 2
                iti = int(np.argmin(d2))
                tri_nodes = tri[iti]
                dfdx, dfdy = _grad_on_triangle(points[tri_nodes, :2],
                                               phi_vals_all[tri_nodes])
                theta = np.arctan2(dfdy, dfdx)  # orientation of grad(phi)
            else:
                theta = np.nan

            # AUTO_ALIGN: choose next step's direction by angle (one-step lag)
            if AUTO_ALIGN and np.isfinite(theta):
                # axis-aligned choice: pick axis closer to gradient direction
                # If |cos(theta)| >= |sin(theta)| -> closer to x-axis
                next_dir = "x" if abs(np.cos(theta)) >= abs(np.sin(theta)) else "y"
            else:
                next_dir = cur_dir

            # store
            tip_t.append(float(t))
            tip_s.append(float(s_tip) if np.isfinite(s_tip) else np.nan)
            tip_dir.append(cur_dir)
            theta_hist.append(float(theta) if np.isfinite(theta) else np.nan)

            # update direction for the next frame
            cur_dir = next_dir

        # compute instantaneous velocity (central difference)
        tip_t = np.asarray(tip_t, dtype=float)
        tip_s = np.asarray(tip_s, dtype=float)
        V_tip = np.full_like(tip_s, np.nan, dtype=float)
        if len(tip_t) >= 2:
            dt0 = tip_t[1] - tip_t[0]
            dtN = tip_t[-1] - tip_t[-2]
            V_tip[0]  = (tip_s[1] - tip_s[0]) / dt0 if dt0 != 0 else np.nan
            V_tip[-1] = (tip_s[-1] - tip_s[-2]) / dtN if dtN != 0 else np.nan
            if len(tip_t) > 2:
                dt_mid = tip_t[2:] - tip_t[:-2]
                ok = dt_mid != 0
                V_tip[1:-1][ok] = (tip_s[2:][ok] - tip_s[:-2][ok]) / dt_mid[ok]

        # -------- plateau averaging --------
        # remove NaNs and SKIP_FIRST for averaging/plot
        m_all = np.isfinite(tip_t) & np.isfinite(V_tip)
        t_all = tip_t[m_all]
        v_all = V_tip[m_all]

        if SKIP_FIRST > 0 and SKIP_FIRST < len(t_all):
            t_all = t_all[SKIP_FIRST:]
            v_all = v_all[SKIP_FIRST:]

        if plateau is None:
            # auto window: last third of available points
            n = len(t_all)
            j0 = int(np.floor(2 * n / 3))
            t_a, t_b = (float(t_all[j0]), float(t_all[-1])) if n > 0 else (np.nan, np.nan)
            m_plateau = np.ones_like(t_all, dtype=bool)
            m_plateau[:j0] = False
        else:
            t_a, t_b = plateau
            m_plateau = (t_all >= t_a) & (t_all <= t_b)

        v_plateau = v_all[m_plateau]
        V_num = float(np.nanmean(v_plateau)) if v_plateau.size else np.nan
        V_std = float(np.nanstd(v_plateau)) if v_plateau.size else np.nan

        # -------- save CSV --------
        np.savetxt(os.path.join(base_dir, "tip_trace.csv"),
                   np.c_[tip_t, tip_s, V_tip],
                   delimiter=";", fmt="%.6g",
                   header="t; tip_pos; V_tip", comments="")

        # -------- plot velocity --------
        # filter for plotting (finite, with SKIP_FIRST)
        m_plot = np.isfinite(tip_t) & np.isfinite(V_tip)
        t_plot = tip_t[m_plot]
        v_plot = V_tip[m_plot]
        if SKIP_FIRST > 0 and SKIP_FIRST < len(t_plot):
            t_plot = t_plot[SKIP_FIRST:]
            v_plot = v_plot[SKIP_FIRST:]

        fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=180)
        ax.plot(t_plot, v_plot, lw=1.8, color="C0", label="Simulation")

        # theory line with numeric label
        if V_THEORY is not None:
            ax.axhline(V_THEORY, color="black", lw=1.6,
                       label=f"Theory = {V_THEORY:.5g}")

        # plateau mean line
        if np.isfinite(V_num):
            ax.axhline(V_num, color="C2", lw=1.6, linestyle="--",
                       label=f"Mean (plateau) = {V_num:.5g}")

        # show plateau window
        if np.isfinite(t_a) and np.isfinite(t_b):
            ax.axvspan(t_a, t_b, color="C2", alpha=0.10, label="Plateau window")

        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\nu_{\mathrm{tip}}$")

        if YMIN is not None and YMAX is not None:
            ax.set_ylim(YMIN, YMAX)

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=False, handlelength=2.6,
                  borderpad=0.2, labelspacing=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, "v_tip_vs_time.png"))
        fig.savefig(os.path.join(base_dir, "v_tip_vs_time.svg"))
        plt.close(fig)

        print("\nSaved:", os.path.join(base_dir, "tip_trace.csv"))
        print("Saved:", os.path.join(base_dir, "v_tip_vs_time.png"))
        print("Saved:", os.path.join(base_dir, "v_tip_vs_time.svg"))
        print(f"Last frame: t={tip_t[-1]:.6g}, tip={tip_s[-1]:.6g}, V_tip≈{V_tip[-1]:.6g}")
        if np.isfinite(V_num):
            print(f"Plateau [{t_a:.3g}, {t_b:.3g}] → V_num={V_num:.6g} (std={V_std:.2g})")


# ----------------- CLI -----------------
if __name__ == "__main__":
    print("=== Tip Velocity (XDMF) ===")
    # default to folder where this script lives
    script_dir = Path(__file__).expanduser().resolve().parent
    print(f"(Default folder = {script_dir})")
    print("Give a FOLDER (with .xdmf + .h5) OR a full path to an .xdmf.")
    print("Press Enter to use the script's folder.")

    path_in = ask("Folder OR .xdmf path", default=str(script_dir))
    p = Path(path_in).expanduser().resolve()

    # resolve XDMF path
    if p.is_dir():
        xdmf_path = find_xdmf_in_folder(p)
    elif p.is_file() and p.suffix.lower() == ".xdmf":
        if not _has_pair_h5(p):
            print("\nERROR: matching HDF5 missing "
                  f"('{p.with_suffix('.h5').name}'). Put it next to the .xdmf.")
            sys.exit(1)
        xdmf_path = str(p)
    else:
        print("\nERROR: provide a folder or an .xdmf path.")
        sys.exit(1)

    direction = ask("Initial growth direction (x/y)", default="x",
                    checker=lambda s: s.lower() in ("x", "y")).lower()
    # field is hard-defaulted to "phi"; change here if needed
    print(f"Using field: {DEFAULT_FIELD}")

    # plateau window input: "ta,tb" or blank for auto
    win = ask("Plateau window ta,tb (blank=auto last third)", default="")
    plateau = None
    if win.strip():
        try:
            ta_str, tb_str = win.split(",")
            plateau = (float(ta_str), float(tb_str))
        except Exception:
            print("  → could not parse window; falling back to auto.")
            plateau = None

    center = float(ask("Centerline position 0..1 (0.5 = midline)", default="0.5",
                       checker=lambda v: (lambda f: 0.0 <= f <= 1.0)(
                           float(v)) if v else False))

    try:
        run(xdmf_path, field=DEFAULT_FIELD, direction=direction,
            center=center, plateau=plateau)
    except Exception as e:
        print("\nERROR:", e)
        print("Tip: ensure the .h5 sits next to the .xdmf and the field name is correct.")
        sys.exit(1)
