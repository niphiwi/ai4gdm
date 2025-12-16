import numpy as np
import scipy.sparse as sp
from dolfinx import mesh, fem
from mpi4py import MPI
import ufl

def create_dares_mesh_64x64(output_dir='.', 
                             nx=63, ny=63,  # 63+1 = 64 vertices per dimension
                             x_range=(0, 64), 
                             y_range=(0, 64)):
    """
    Create DARES mesh files using FEniCS-DOLFINx for 64×64 grid
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the mesh files
    nx, ny : int
        Number of cells (vertices will be nx+1, ny+1)
    x_range, y_range : tuple
        Physical domain boundaries
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create rectangular mesh
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
        [nx, ny],
        cell_type=mesh.CellType.triangle
    )
    
    # Function space (P1 = linear elements)
    import basix.ufl
    element = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, element)
    
    # Get vertex coordinates
    # In DOLFINx, geometry is accessed differently
    x = domain.geometry.x
    vertices = x[:, :2]  # Take only x, y coordinates (ignore z if present)
    n_vertices = vertices.shape[0]
    
    print(f"Created mesh with {n_vertices} vertices")
    print(f"Expected: approximately {(nx+1) * (ny+1)} vertices")
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Assemble Laplacian operator: ∫ ∇u · ∇v dx
    print("Assembling Laplacian...")
    a_laplace = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = fem.form(a_laplace)
    L_matrix = fem.assemble_matrix(L_form)
    
    # Convert to scipy sparse matrix
    L_sparse = L_matrix.to_scipy()
    
    # Assemble gradient operators
    # Gx: ∫ v * ∂u/∂x dx
    print("Assembling gradient operators...")
    a_grad_x = v * u.dx(0) * ufl.dx
    Gx_form = fem.form(a_grad_x)
    Gx_matrix = fem.assemble_matrix(Gx_form)
    Gx_sparse = Gx_matrix.to_scipy()
    
    # Gy: ∫ v * ∂u/∂y dx
    a_grad_y = v * u.dx(1) * ufl.dx
    Gy_form = fem.form(a_grad_y)
    Gy_matrix = fem.assemble_matrix(Gy_form)
    Gy_sparse = Gy_matrix.to_scipy()
    
    # Save files
    print("Saving files...")
    np.save(os.path.join(output_dir, 'vertices.npy'), vertices)
    sp.save_npz(os.path.join(output_dir, 'laplace.npz'), L_sparse)
    sp.save_npz(os.path.join(output_dir, 'grad_x.npz'), Gx_sparse)
    sp.save_npz(os.path.join(output_dir, 'grad_y.npz'), Gy_sparse)
    
    print(f"\n✓ Successfully saved mesh files to '{output_dir}/'")
    print(f"  - vertices.npy: shape {vertices.shape}")
    print(f"  - laplace.npz: shape {L_sparse.shape}")
    print(f"  - grad_x.npz: shape {Gx_sparse.shape}")
    print(f"  - grad_y.npz: shape {Gy_sparse.shape}")
    
    # Verify mesh properties
    print(f"\nMesh properties:")
    print(f"  - x range: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
    print(f"  - y range: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
    print(f"  - Number of cells: {domain.topology.index_map(2).size_local}")
    
    return vertices, L_sparse, Gx_sparse, Gy_sparse

def visualize_mesh(vertices, output_file='mesh_64x64.png'):
    """Visualize the generated mesh"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 10))
    plt.plot(vertices[:, 0], vertices[:, 1], 'k.', markersize=2, alpha=0.5)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'DARES Mesh: {len(vertices)} vertices')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Saved mesh visualization to '{output_file}'")
    plt.show()

if __name__ == "__main__":
    # Create mesh files for 64×64 grid
    vertices, L, Gx, Gy = create_dares_mesh_64x64(
        output_dir='./mesh',
        nx=63,  # Results in 64 vertices in x
        ny=63,  # Results in 64 vertices in y
        x_range=(0, 64),
        y_range=(0, 64)
    )
    
    # Visualize
    visualize_mesh(vertices, 'mesh_64x64.png')
    
    # Print some statistics
    print("\nMatrix statistics:")
    print(f"  - Laplacian sparsity: {L.nnz / (L.shape[0] * L.shape[1]) * 100:.2f}%")
    print(f"  - Gx sparsity: {Gx.nnz / (Gx.shape[0] * Gx.shape[1]) * 100:.2f}%")
    print(f"  - Gy sparsity: {Gy.nnz / (Gy.shape[0] * Gy.shape[1]) * 100:.2f}%")