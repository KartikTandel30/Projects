import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Fixed node number with full liquid-to-solid transition ===
node_number = 1347  # node with good evolution behavior

# === Load Excel file ===
file_path = "all_nodes_phi_fphi_data.xlsx"
df = pd.read_excel(file_path)

# === Construct column names ===
phi_col = f"ϕ_node_{node_number}"
f_col = f"f_node_{node_number}"

# === Sanity check ===
if phi_col not in df.columns or f_col not in df.columns:
    raise ValueError(f"Node {node_number} not found in the Excel file!")

# === Extract data ===
phi_values = df[phi_col]
f_values = df[f_col]

# Parameters for the energy function
u = np.array([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
zet = 1.6
phi_range = np.linspace(-1.5, 1.5, 100)

# Free energy calculation components (f1 part of energy function)
f1 = -0.5 * phi_range**2 + 0.25 * phi_range**4

# Plot f(ϕ) for all values of u
plt.figure(figsize=(8, 6))

# Loop to calculate and store the energy function for each u
for val_u in u:
    f2 = zet * val_u * phi_range * (1 - (2/3) * phi_range**2 + 0.2 * phi_range**4)  # Second part of the energy function for each u
    f = f1 + f2  # Full energy function
    plt.plot(phi_range, f, label=f'u = {val_u}')  # Plot the result for this u

# Plot the data from Excel file
plt.plot(phi_values, f_values, 'o-', color='blue', markersize=3, label=f'Node {node_number}')

# Customize the plot
plt.xlabel("ϕ", fontsize=12)
plt.ylabel("f(ϕ)", fontsize=12)
plt.title(f"f(ϕ) vs ϕ for node {node_number} and different u values", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"f_vs_phi_node{node_number}_all_u_values.png", dpi=300)
plt.show()
