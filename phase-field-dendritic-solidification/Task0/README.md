# Test T1 — No-gradient phase-field (node-wise ODE verification)

**Aim**  
Verifies that, with the gradient term removed, the phase field evolves **homogeneously** and **node-wise** as an ODE driven only by the bulk free-energy density. Confirm that a representative node’s trajectory \((\phi_i(t),\,f(\phi_i(t),u))\) lies on the analytical \(f(\phi,u)\) curve and moves downhill toward the favored well (tilt set by \(\zeta u\)).

---

## Setup

**Model (no-gradient Allen–Cahn):**
\[
\tau_0\,\dot\phi \;=\; -\,\frac{\partial f}{\partial\phi}(\phi,u)
\;=\; \phi - \phi^3 - \zeta u\,(1-2\phi^2+\phi^4).
\]

**Bulk free energy:**
\[
f(\phi,u) \;=\; -\tfrac12\phi^2 + \tfrac14\phi^4 \;+\; \zeta\,u\,\phi\!\Bigl(1-\tfrac{2}{3}\phi^2+\tfrac{1}{5}\phi^4\Bigr).
\]

**Initial & fixed fields**  
- \(\phi(\mathbf{x},0)=-0.6\) (uniform in \(\Omega\)).  
- \(u(\mathbf{x},t)\equiv -0.75\) (kept fixed for this verification).  
- \(\zeta = 1.6\), pick an interior node (e.g. `node_id=100`) that undergoes a clean liquid→solid transition.

**Folder contents**  
- `task_0.py` — produces the node-time data (Excel/CSV) used here.  
- `Testfvsphi.py` — plotting/overlay script for \(f(\phi,u)\) vs node trajectory.  
- Data file produced by `task_0.py` with columns:
  - `phi_node` — \(\phi_i(t_k)\) at the chosen node,
  - `f_node` — \(f(\phi_i(t_k),u)\) (or the script can recompute it).

---

## How to run

> Keep `Testfvsphi.py`, `task_0.py`,    and the produced data file in the **same folder**.

1) Generate the node trajectory (from your simulation):
```bash

python task_0.py      # run this to obtain the output excel file
python Testfvsphi.py  # reads phi_node, f_node and u = -0.75, zeta = 1.6, node data is set at 100
