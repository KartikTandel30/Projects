# -*- coding: utf-8 -*-
"""
RSJC - Vibration Analysis of Mechanical Structures (Modal Analysis)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
# Load input files
directory = "C:/Users/karti/OneDrive/Desktop/RJ&SC"
nodes = pd.read_csv(f"{directory}/node_details.csv", encoding='ISO-8859-1')
elements = pd.read_csv(f"{directory}/element_connectivity.csv", encoding='ISO-8859-1')
geometry = pd.read_csv(f"{directory}/GeometryDetails.csv", encoding='ISO-8859-1')
materials = pd.read_csv(f"{directory}/reinforced_concrete_properties.csv", encoding='ISO-8859-1')

# Function to clean and convert properties
def clean_and_convert(value):
    try:
        cleaned_value = ''.join(c for c in value if c.isdigit() or c == '.')
        return float(cleaned_value)
    except ValueError:
        return None

# Extract properties
density = clean_and_convert(materials.loc[materials['Property'] == 'Density', 'Value'].values[0])
young_modulus = clean_and_convert(materials.loc[materials['Property'] == "Young's Modulus", 'Value'].values[0]) * 1e9
thickness = 1  # Using a thickness of 1 for the calculation

# Initialize global matrices
num_nodes = len(nodes)
dof = num_nodes * 2
K_global = np.zeros((dof, dof))
M_global = np.zeros((dof, dof))

# Define helper functions for matrix assembly
def bilinear_shape_functions(xi, eta):
    return np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ])

def compute_B_matrix_and_Jacobian(xi, eta, coords):
    N_dxi = np.array([
        [-0.25 * (1 - eta),  0.25 * (1 - eta),  0.25 * (1 + eta), -0.25 * (1 + eta)],
        [-0.25 * (1 - xi), -0.25 * (1 + xi),   0.25 * (1 + xi),   0.25 * (1 - xi)]
    ])
    J = N_dxi @ coords
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    
    B = np.zeros((3, 8))
    B[0, 0:8:2] = invJ[0, 0] * N_dxi[0, :]  # dN/dx
    B[0, 1:8:2] = invJ[1, 0] * N_dxi[0, :]  # dN/dx
    B[1, 0:8:2] = invJ[0, 1] * N_dxi[1, :]  # dN/dy
    B[1, 1:8:2] = invJ[1, 1] * N_dxi[1, :]  # dN/dy
    B[2, 0:8:2] = B[1, 1:8:2]  # dN/dy
    B[2, 1:8:2] = B[0, 0:8:2]  # dN/dx
    return B, detJ

def compute_local_mass_matrix(density, area, thickness):
    local_mass_matrix = density * area * thickness * np.eye(8)
    return local_mass_matrix

def assemble_matrices():
    for _, elem in elements.iterrows():
        node_indices = elem[['n1', 'n2', 'n3', 'n4']].astype(int).tolist()
        coords = nodes.loc[node_indices, ['x', 'y']].values
        area = geometry.loc[geometry['element_number'] == elem['element_number'], 'area'].values[0]
        local_mass = compute_local_mass_matrix(density, area, thickness)
        # Add local mass matrix to global matrix
        for i, ni in enumerate(node_indices):
            for j, nj in enumerate(node_indices):
                M_global[2*ni:2*ni+2, 2*nj:2*nj+2] += local_mass[2*i:2*i+2, 2*j:2*j+2]
                
        gauss_points = [(xi, eta) for xi in [-1/np.sqrt(3), 1/np.sqrt(3)] for eta in [-1/np.sqrt(3), 1/np.sqrt(3)]]
        K_local = np.zeros((8, 8))  # Initialize local stiffness matrix
        for xi, eta in gauss_points:
            B, detJ = compute_B_matrix_and_Jacobian(xi, eta, coords)
            weight = 1
            K_local += weight * (B.T @ B) * young_modulus * thickness * detJ
        # Add local stiffness matrix to global matrix
        for i in range(4):
            for j in range(4):
                K_global[2*node_indices[i]:2*node_indices[i]+2, 2*node_indices[j]:2*node_indices[j]+2] += K_local[2*i:2*i+2, 2*j:2*j+2]
           
# Assemble the global matrices
assemble_matrices()

# Example check before boundary conditions (this is implemented while rectifying the code)
if not np.all(np.diag(M_global) > 0):
    print("Diagonal elements of the mass matrix are not all positive.")

# Debugging output before applying boundary conditions
eigenvalues_m = np.linalg.eigvals(M_global)
if np.any(eigenvalues_m <= 0):
    print("Mass matrix has non-positive eigenvalues before boundary conditions:", eigenvalues_m)

# Apply boundary conditions
fixed_nodes = [0, 7,18,25]
for node in fixed_nodes:
    indices = [2 * node, 2 * node + 1]
    for idx in indices:
        K_global[idx, :] = 0
        K_global[:, idx] = 0
        K_global[idx, idx] = 1  # Maintain structural integrity of the matrix
        M_global[idx, :] = 0
        M_global[:, idx] = 0
        M_global[idx, idx] = 1e-10  # Small value to keep matrix positive definite

# Regularization for stability
np.fill_diagonal(M_global, np.diag(M_global) + 1e-10)

# Verify matrix properties after applying boundary conditions
if not np.allclose(M_global, M_global.T):
    print("Mass matrix is not symmetric post-boundary conditions.")
if not np.all(np.linalg.eigvals(M_global) > 0):
    print("Mass matrix is not positive definite post-boundary conditions.")

# Final check before eigenvalue problem
if np.any(np.linalg.eigvals(M_global) <= 0):
    raise ValueError("Mass matrix is still not positive definite at computation time.")

# Eigenvalue problem
eigenvalues, eigenvectors = eigh(K_global, M_global)
natural_frequencies = np.sqrt(eigenvalues)

# Output natural frequencies
print("Natural Frequencies (rad/s):", natural_frequencies[:3])

# Plotting the mode shapes
scale_factor = 100  # Change this factor to scale the deformations for better visibility
num_modes_to_plot = 3  # Number of modes to plot

# Create a figure with subplots for each mode
for i in range(num_modes_to_plot):
    fig, ax = plt.subplots(figsize=(12, 12))
    mode_shape = eigenvectors[:, i].reshape(-1, 2) * scale_factor
    
    # Plot undeformed structure
    for _, element in elements.iterrows():
        node_indices = [element['n1'], element['n2'], element['n3'], element['n4'], element['n1']]
        x_coords = [nodes.iloc[idx]['x'] for idx in node_indices]
        y_coords = [nodes.iloc[idx]['y'] for idx in node_indices]
        ax.plot(x_coords, y_coords, 'k--', label='Undeformed' if element.name == 0 else "")  # Black dashed for undeformed
    
    # Plot deformed structure
    for _, element in elements.iterrows():
        node_indices = [element['n1'], element['n2'], element['n3'], element['n4'], element['n1']]
        x_coords = [nodes.iloc[idx]['x'] + mode_shape[idx, 0] for idx in node_indices]
        y_coords = [nodes.iloc[idx]['y'] + mode_shape[idx, 1] for idx in node_indices]
        ax.plot(x_coords, y_coords, 'r-', label='Deformed' if element.name == 0 else "")  # Red for deformed

    # Set plot limits and titles
    ax.set_title(f'Mode {i + 1} Shape (Natural Frequency: {natural_frequencies[i]:.2f} rad/s)', fontsize=14)
    ax.set_xlabel('Displacement (m)', fontsize=12)
    ax.set_ylabel('Length(m)', fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2)  # Move legend outside the plot
    ax.grid(True)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
