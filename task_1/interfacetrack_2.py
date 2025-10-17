# flat_interface_plot.py
# Reads: interface_track.csv [time,f_solid,interface_pos,interface_speed]
#        energy_track.csv    [time,total_energy]            
#        thickness_track.csv [time,thickness_phi_pm0.8]     

import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------
# Config (edit if needed)
# -----------------------
Lx, Ly = 50.0, 50.0         # domain size used in the run
n_lines = 4                   # how many interface lines to draw
out_dpi = 300                 # figure DPI
font_main = 11                # base font size
plt.rcParams.update({
    "font.size": font_main,
    "axes.titlesize": font_main+1,
    "axes.labelsize": font_main,
    "legend.fontsize": font_main-1,
    "xtick.labelsize": font_main-1,
    "ytick.labelsize": font_main-1,
})

# -----------------------
# Load interface tracking
# -----------------------
if not os.path.exists("interface_track.csv"):
    raise FileNotFoundError("Missing interface_track.csv")

iface = np.loadtxt("interface_track.csv", delimiter=",", skiprows=1)
t, f_solid, xstar, v = iface.T

# Indices for the vertical lines plot
if len(t) < n_lines:
    idx = np.arange(len(t))
else:
    idx = np.linspace(0, len(t)-1, n_lines, dtype=int)

# -----------------------
# 1) Flat interface lines
# -----------------------
fig = plt.figure(figsize=(6, 6))
for k in idx:
    plt.plot([xstar[k], xstar[k]], [0, Ly], linewidth=2, color="k", alpha=0.9)
    #plt.text(xstar[k], 0.02*Ly, f"{t[k]:.2f}", ha="center", va="bottom", color="k")

plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"$\phi=0$ (area-based flat interface) at selected times")
plt.grid(True, alpha=0.20)
#plt.tight_layout()
plt.savefig("flat_interface_lines.png", dpi=out_dpi)
#plt.close(fig)
plt.show()

# ---------------------------------------
# 2) x*(t) vs time + linear-fit for speed
# ---------------------------------------
A = np.vstack([np.ones_like(t), t]).T
coef, *_ = np.linalg.lstsq(A, xstar, rcond=None)
a, v_fit = coef
x_fit = a + v_fit * t
ss_res = np.sum((xstar - x_fit)**2)
ss_tot = np.sum((xstar - xstar.mean())**2)
R2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

fig = plt.figure(figsize=(6.2, 4.2))
plt.plot(t, xstar, lw=1.6, label=r"$x^{*}(t)$ data")
plt.plot(t, x_fit, lw=1.6, linestyle="--", label=fr"linear fit")
plt.xlabel("time")
plt.ylabel(r"interface position $x^{*}(t)$")
plt.title(r"Interface position vs time (constant-speed test)")
# put R^2 in a corner box
txt = fr"$R^2 = {R2:.5f}$"
plt.text(0.90, 0.98, txt, transform=plt.gca().transAxes,
         va="top", ha="right", bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.7"))
plt.legend(loc="best", frameon=False)
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("xstar_vs_time_fit.png", dpi=out_dpi)
plt.close(fig)
plt.show()

# -----------------------
# 3) Energy decay (if any)
# -----------------------
energy_fig_done = False
if os.path.exists("energy_track.csv"):
    Edat = np.loadtxt("energy_track.csv", delimiter=",", skiprows=1)
    tE, E = Edat.T

    # monotonicity check (allow visual; you can enforce in code if desired)
    dE = np.diff(E)
    worst_rel_increase = 0.0
    if len(E) > 1:
        worst_rel_increase = float(np.max(np.maximum(0.0, dE) / np.maximum(1.0, np.abs(E[:-1]))))

    fig = plt.figure(figsize=(6.0, 4.2))
    plt.plot(tE, E, lw=1.8)
    plt.xlabel("time")
    plt.ylabel(r"total energy($\Pi(t)$)")
    plt.title("Energy decay")
    plt.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("energy_decay.png", dpi=out_dpi)
    plt.close(fig)
    plt.show()
    energy_fig_done = True

# -----------------------------------------
# 4) Thickness vs time (if thickness exists)
# -----------------------------------------
thick_fig_done = False
if os.path.exists("thickness_track.csv"):
    Tdat = np.loadtxt("thickness_track.csv", delimiter=",", skiprows=1)
    tT, Th = Tdat.T

    fig = plt.figure(figsize=(6.0, 4.2))
    plt.plot(tT, Th, lw=1.8)
    plt.xlabel("time")
    plt.ylabel(r"interface thickness $\delta_{0.8}(t)$")
    plt.title(r"Thickness vs time (centerline $\phi=\pm 0.8$)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("thickness_vs_time.png", dpi=out_dpi)
    plt.close(fig)
    plt.show()
    thick_fig_done = True

# -----------------------------------------
# 5) Optional: instantaneous speed v(t)
# -----------------------------------------
speed_fig_done = False
if np.any(np.isfinite(v)):
    fig = plt.figure(figsize=(6.0, 4.2))
    plt.plot(t, v, lw=1.4)
    plt.axhline(v_fit, linestyle="--", linewidth=1.2, color="0.3", label=fr"mean fit $v={v_fit:.4f}$")
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.title("Interface velocity")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("speed_vs_time.png", dpi=out_dpi)
    plt.close(fig)
    speed_fig_done = True

# -----------------------
# Print/save a tiny summary
# -----------------------
lines = []
lines.append("=== Flat-front post summary ===")
lines.append(f"n_samples: {len(t)}; time in [{t.min():.3f}, {t.max():.3f}]")
lines.append(f"x* fit speed v = {v_fit:.6f}, R^2 = {R2:.6f}")
if energy_fig_done:
    lines.append("energy decay figure: energy_decay.pdf/png")
if thick_fig_done:
    # if you want a single number: final thickness or median
    lines.append(f"thickness final/median: {Th[-1]:.6f} / {np.nanmedian(Th):.6f}")
if speed_fig_done:
    lines.append("speed figure: speed_vs_time.pdf/png")
summary = "\n".join(lines)
print(summary)
with open("post_summary.txt", "w") as fh:
    fh.write(summary + "\n")

print("[OK] wrote figures: flat_interface_lines, xstar_vs_time_fit"
      + (", energy_decay" if energy_fig_done else "")
      + (", thickness_vs_time" if thick_fig_done else "")
      + (", speed_vs_time" if speed_fig_done else ""))
