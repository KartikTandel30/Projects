# Test — φ–T Coupling (energy, enthalpy balance, and zero-flux BCs)

This folder verifies that the **monolithic φ–T solver** (mixed `P1×P1`, fully implicit
Backward-Euler) is coded consistently:

1) **Enthalpy coupling identity**  
   \(\Delta\!\int u \stackrel{?}{=} \tfrac12\,\Delta\!\int \phi\) under homogeneous Neumann BCs.

2) **Thermodynamic consistency**  
   **Total free energy** \(\Pi(t)\) decreases in time; a dissipation proxy \(H(t)\le0\).

3) **Boundary conditions**  
   Homogeneous Neumann (zero-flux) BCs for both fields hold to tight tolerance.

---

## Model–code alignment (what the solver assembles)

- **Temperature**
  \[
  R_1 = \int_\Omega [(u-u_0) - \Delta t\tfrac12(\phi-\phi_0)]\,w_u\,dx
        \;+\;\Delta t\,D\!\int_\Omega \nabla u\!\cdot\!\nabla w_u\,dx
  \]
- **Phase field**
  \[
  R_0 = \int_\Omega \frac{\phi-\phi_0}{\Delta t}\,w_\phi\,dx
          \;+\;\int_\Omega (-\phi+\phi^3+\zeta u(1-2\phi^2+\phi^4))\,w_\phi\,dx
          \;+\;\lambda_0^2 \!\int_\Omega \nabla\phi\!\cdot\!\nabla w_\phi\,dx
  \]

BCs: homogeneous Neumann (zero flux) for both \(u\) and \(\phi\).

---

## Files

- `task_2.py` — main φ–T solver (monolithic, BE).
- `test2.py` — **`CoupledDiagnostics`**: energy \(\Pi(t)\), dissipation \(H(t)\le0\), enthalpy balance.
- `test_bc.py` — **`bc_check`**: zero-flux checks.
- `diag_out/` — auto-created: CSV + PNG diagnostics.

> Note: variable name in code is `lamda_0`; in formulas we write \(\lambda_0\).

---

## How to run

### A) One-shot (solver already calls the tests)
```bash
python task_2.py

### B) If you want to wire the checks yourself

Add the diagnostics and BC checks directly inside your solver (e.g., `task_2.py`) so you can reuse the **test codes** without changing your workflow.

```python
# --- at the top of task_2.py (imports) ---
from test2 import CoupledDiagnostics
from test_bc import bc_check

# --- after you create mesh/fields/params (before the time loop) ---
diag = CoupledDiagnostics(
    msh,        # dolfinx mesh
    phi, u,     # Function objects used in the solve
    lamda_0,    # NOTE: variable is 'lamda_0' in code (φ-interface length)
    zet,        # zeta
    tau=tau_0,  # kinetics parameter τ0
    D=D,        # thermal diffusivity
    plot_every=10  # save plots every N updates
)
diag.start()

# --- time loop ---
for step in range(nsteps):
    # assemble/solve the monolithic φ–T system ...
    # after a successful Newton solve:
    t += dt
    diag.update(t)   # logs Π(t), H(t), ∫u, ∫φ, misfit; writes PNG/CSV when due
# --- end time loop ---

diag.finish()        # final flush of CSV/plots to diag_out/

# --- boundary-condition check (run at end; optional per-step) ---
bc_check(msh, phi, u, tol=1e-8)   # prints PASS/FAIL + L1/L2 residuals


