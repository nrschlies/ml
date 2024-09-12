import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_vectors(ax, vectors, title, color='r'):
    """
    Plot 3D vectors on a given Axes object.
    """
    ax.set_title(title)
    for i in range(vectors.shape[1]):
        # Handle 2D or 3D vectors
        x, y, z = (vectors[0, i], vectors[1, i], 0) if vectors.shape[0] == 2 else (vectors[0, i], vectors[1, i], vectors[2, i])
        ax.quiver(0, 0, 0, x, y, z, color=color, label=f'Basis Vector {i+1}')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()

def visualize_ftla(A):
    # Perform Singular Value Decomposition
    U, S, VT = np.linalg.svd(A)

    # Rank of A
    r = np.linalg.matrix_rank(A)

    # Column Space (Range of A)
    col_space = U[:, :r]

    # Row Space (Range of A^T)
    row_space = VT[:r, :].T  # Correctly transpose to align with basis vectors

    # Null Space (Null space of A)
    null_space = VT[r:, :].T  # Remaining vectors form the null space

    # Left Null Space (Null space of A^T)
    left_null_space = U[:, r:]

    fig = plt.figure(figsize=(10, 10))

    # Plot Column Space (Range of A)
    ax1 = fig.add_subplot(221, projection='3d')
    plot_vectors(ax1, col_space, 'Column Space (Range of A)', color='r')

    # Plot Row Space (Range of A^T)
    ax2 = fig.add_subplot(222, projection='3d')
    plot_vectors(ax2, row_space, 'Row Space (Range of A^T)', color='b')

    # Plot Null Space (Null(A))
    ax3 = fig.add_subplot(223, projection='3d')
    if null_space.size > 0:
        plot_vectors(ax3, null_space, 'Null Space (Null(A))', color='g')
    else:
        ax3.text(0, 0, 0, 'Null Space is Trivial', color='red')

    # Plot Left Null Space (Null(A^T))
    ax4 = fig.add_subplot(224, projection='3d')
    if left_null_space.size > 0:
        plot_vectors(ax4, left_null_space, 'Left Null Space (Null(A^T))', color='g')
    else:
        ax4.text(0, 0, 0, 'Left Null Space is Trivial', color='red')

    plt.tight_layout()
    plt.show()

# Example usage with a rank-deficient matrix
np.random.seed(0)
A = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])  # Example rank-deficient matrix
visualize_ftla(A)
