# flat_interface_plot.py
import numpy as np
import matplotlib.pyplot as plt

# --- set your domain size used in the run ---
Lx, Ly = 100.0, 100.0

# --- load csv written by your solver ---
data = np.loadtxt("interface_track.csv", delimiter=",", skiprows=1)
t, f_solid, xstar, v = data.T   # time, solid fraction, interface x*, speed

# pick a few times to draw (6 lines)
idx = np.linspace(0, len(t)-1, 6, dtype=int)

plt.figure(figsize=(5.6, 5.6))
for k in idx:
    # vertical ϕ=0 line at x = x*(t_k)
    plt.plot([xstar[k], xstar[k]], [0, Ly], linewidth=2, color="k")
    plt.text(xstar[k], 0.02*Ly, f"{t[k]:.2f}", ha="center", va="bottom", color="k")

plt.xlim(0, Lx); plt.ylim(0, Ly)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("X"); plt.ylabel("Y")
plt.title("ϕ = 0 (flat interface) at selected times")
plt.tight_layout()
plt.savefig("flat_interface_lines.png", dpi=200)
plt.show()
