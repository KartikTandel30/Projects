import os, sys, numpy as np, meshio

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
    out_csv  = os.path.join(base_dir, "tip_trace_from_meshio.csv")

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

        # estimate spacing for picking a centerline
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
            # find rightmost φ=0 crossing to the + direction
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
                # try common names; otherwise first scalar
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

        tip_t = np.asarray(tip_t)
        tip_s = np.asarray(tip_s)
        V_tip = np.full_like(tip_s, np.nan, dtype=float)
        if len(tip_t) >= 2:
            V_tip[1:] = (tip_s[1:] - tip_s[:-1]) / (tip_t[1:] - tip_t[:-1])

        np.savetxt(out_csv, np.c_[tip_t, tip_s, V_tip], delimiter=";", fmt="%.6g",
           header="t; tip_pos; V_tip", comments="")
        print("\nSaved:", out_csv)
        print(f"Last: t={tip_t[-1]:.6g}, tip={tip_s[-1]:.6g}, V_tip≈{V_tip[-1]:.6g}")

if __name__ == "__main__":
    print("=== Tip Velocity (XDMF) ===")
    print("Paste the full path to your .xdmf (its .h5 must sit next to it).")
    xdmf_path = ask("XDMF path", checker=is_file)

    # Optional: direction, field name, centerline position
    direction = ask("Growth direction (x/y)", default="x",
                    checker=lambda s: s.lower() in ("x","y"))
    field = ask("Field name (Enter for auto-detect)", default="")
    field = field if field else None

    def chk_center(v):
        try:
            f = float(v); return 0.0 <= f <= 1.0
        except: return False
    center = float(ask("Centerline position from 0..1 (0.5 = midline)", default="0.5", checker=chk_center))

    try:
        run(xdmf_path, field=field, direction=direction, center=center)
    except Exception as e:
        print("\nERROR:", e)
        print("Tip: ensure the .h5 is in the same folder as the .xdmf, and the field name is correct.")
        sys.exit(1)
