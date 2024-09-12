import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to generate a positive definite matrix
def generate_positive_definite_matrix(n):
    M = np.random.rand(n, n)
    A = np.dot(M, M.T) + n * np.eye(n)  # Adding n*I to ensure positive definiteness
    return A

# Function to rotate a vector by an angle
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# Function to interpolate transformations smoothly
def interpolate_transform(frame, U, S, Vt, total_frames):
    # Generate a unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Rotation interpolation for Vt
    phase1_fraction = min(frame / (total_frames / 3), 1)  # Normalized time for Phase 1
    angle_vt = phase1_fraction * np.arccos(np.clip(np.trace(Vt.T) / 2.0, -1.0, 1.0))  # Interpolated angle
    R_vt = rotation_matrix(angle_vt)
    rotated_circle = R_vt @ circle

    # Scaling interpolation using Sigma
    phase2_fraction = min(max((frame - total_frames / 3) / (total_frames / 3), 0), 1)  # Normalized time for Phase 2
    scaling_factors = 1 + phase2_fraction * (S - 1)  # Interpolating singular values
    scaled_ellipse = np.diag(scaling_factors) @ rotated_circle

    # Rotation interpolation for U
    phase3_fraction = min(max((frame - 2 * total_frames / 3) / (total_frames / 3), 0), 1)  # Normalized time for Phase 3
    angle_u = phase3_fraction * np.arccos(np.clip(np.trace(U) / 2.0, -1.0, 1.0))  # Interpolated angle
    R_u = rotation_matrix(angle_u)
    transformed_ellipse = R_u @ scaled_ellipse

    return transformed_ellipse

# Function to plot the interpolated ellipsoid
def plot_ellipsoid(ax, ellipse):
    # Clear previous plot and draw updated ellipsoid
    ax.clear()
    ax.plot(ellipse[0, :], ellipse[1, :], 'r')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    ax.set_title('Ellipsoid Transformation via SVD')
    ax.grid()

# Animation function
def animate_svd(A, ax, fig):
    # Perform SVD
    U, S, Vt = np.linalg.svd(A)

    # Number of frames for a fluid animation
    total_frames = 150

    # Function to update the plot
    def update(frame):
        transformed_ellipse = interpolate_transform(frame, U, S, Vt, total_frames)
        plot_ellipsoid(ax, transformed_ellipse)

    # Create animation
    ani = FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=False)
    plt.show()

# Main function
def main():
    n = 2
    A = generate_positive_definite_matrix(n)
    print(f"Positive Definite Matrix A:\n{A}\n")

    fig, ax = plt.subplots()  # Define 'fig' here
    animate_svd(A, ax, fig)  # Pass 'fig' as an argument

if __name__ == "__main__":
    main()
