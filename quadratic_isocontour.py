import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Function to generate a positive definite matrix
def generate_positive_definite_matrix(n):
    M = np.random.rand(n, n)
    A = np.dot(M, M.T) + n * np.eye(n)  # Adding n*I to ensure positive definiteness
    return A

# Function to plot the ellipsoid
def plot_ellipsoid(A, ax):
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    if np.any(eigenvalues <= 0):
        print("Matrix is not positive definite!")
        return
    
    # Calculate rotation angle and axes lengths
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
    width, height = 2 * np.sqrt(1 / eigenvalues)

    # Plot the ellipse
    ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle, edgecolor='r', facecolor='none', label='Ellipsoid (Isocontour)')
    ax.add_patch(ellipse)
    
    # Plot eigenvectors for reference
    for i in range(len(eigenvalues)):
        vector = eigenvectors[:, i]
        ax.plot([0, vector[0]], [0, vector[1]], 'k-', lw=2, label=f'Eigenvector {i+1}' if i == 0 else None)  # Label only the first to avoid repetition

# Main function
def main():
    n = 2
    A = generate_positive_definite_matrix(n)
    print(f"Positive Definite Matrix A:\n{A}\n")

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal', 'box')
    ax.set_title('Isocontour of Quadratic Form (Ellipsoid)')
    
    plot_ellipsoid(A, ax)
    
    # Add legend
    ax.legend()
    
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

