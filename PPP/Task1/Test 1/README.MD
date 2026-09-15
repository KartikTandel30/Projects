# Test T2 — Gradient penalty & flat-front kinetics (energy decay, constant-speed motion, scaling)

**Aim**  
Verify that enabling the gradient penalty produces a physically consistent diffuse interface:
1) **Monotone decay** of the **total** free energy Π(t).  
2) **Constant-speed** motion of a flat front under a uniform tilt \(u\).  
3) Correct **scaling** of front speed with \(\lambda_0\) and \(\tau_0\) (empirically \(v \propto \lambda_0/\tau_0\) at fixed tilt).  
4) **Mesh/time-step independence** of measured speed and interface thickness.

---

## Setup

**Governing equations (Allen–Cahn with gradient penalty)**
\[
\tau_0\,\dot{\phi} = -\,\mu, \qquad
\mu = \frac{\partial f}{\partial \phi}(\phi,u) - \lambda_0^2\,\nabla^2\phi,
\]
with bulk energy
\[
f(\phi,u) = -\tfrac12\phi^2 + \tfrac14\phi^4
+ \zeta\,u\,\phi\!\Bigl(1-\tfrac{2}{3}\phi^2+\tfrac{1}{5}\phi^4\Bigr).
\]

**Total (Lyapunov) free energy**
\[
\Pi[\phi] = \int_\Omega \Bigl(f(\phi,u) + \tfrac{\lambda_0^2}{2}\,|\nabla\phi|^2\Bigr)\,d\Omega,
\qquad \text{with } \tfrac{d}{dt}\Pi \le 0\ \text{for fixed } u \text{ and no-flux/periodic BC.}
\]

**Flat interface test**
- **Domain / mesh:** rectangle \(L_x \times L_y\) (e.g., \(50 \times 50\)), structured; baseline \(\Delta x = 1\).
- **BCs:** periodic **or** homogeneous Neumann on all sides.
- **Initial condition (partitioned):**
  \[
    \phi(x,0)=
    \begin{cases}
      +1, & x < L_x/2,\\
      -1, & x \ge L_x/2,
    \end{cases}
  \]
  giving a single vertical, flat front at \(x \approx L_x/2\).
- **Baseline parameters:** \(\zeta=1.6,\ u=-0.75,\ \tau_0=1.0,\ \lambda_0=1.0,\ \Delta t=0.04\).
- **Front tracking:** compute solid area
  \[
  A_s(t) = \int_\Omega \tfrac{\phi+1}{2}\,d\Omega, \quad
  x^*(t) = L_x - \frac{A_s(t)}{L_y}.
  \]
  Estimate speed from a linear fit of \(x^*(t)\) vs. \(t\) away from boundaries.
- **Thickness tracking:** e.g., for a 1D profile (slice at mid-height), define
  \[
  \delta_{0.9} = x(\phi=+0.9) - x(\phi=-0.9)
  \]
  (any consistent percentile pair is fine). Expect \(\delta \propto \lambda_0\) and weak dependence on \(\tau_0\).

**Folder contents**
- `task_1.py` — main solver (AC + gradient penalty).  
- `task_0_2.py` — flat-front IC + trackers (writes CSV logs).   
- `interface_track.py` — post-processing (speed, thickness, plots).
- other simulation results files are there 

The domain size should be same as the simulation ran for the interface_track.py to track it properly, if changes in the also change in this code to, to have proper interface lines.
---

## How to run

```bash
# 1) Baseline run (writes: interface_track.csv, energy_track.csv, thickness_track.csv)
python task_0_2.py  #  writes csv files as otutput which will be used in interface_track.py

# 2) Post-process: plots + summary to terminal
python interface_track.py
