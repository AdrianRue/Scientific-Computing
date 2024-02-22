import numpy as np
import matplotlib.pyplot as plt

class DiffusionSolver:
    def __init__(self, N=50):
        self.N = N
        self.grid = np.zeros((N, N))
        self.setup_boundary_conditions()

    def setup_boundary_conditions(self):
        self.grid[0, :] = 1  # Top boundary set to 1
        self.grid[-1, :] = 0   

    def update_with_periodic_conditions(self, new_grid):
        # Apply periodic boundary conditions in the x-direction
        new_grid[:, 0] = new_grid[:, -2]  
        new_grid[:, -1] = new_grid[:, 1]

    def jacobi_solve(self, epsilon=1e-5):
        iteration_count = 0
        while True:
            iteration_count += 1
            new_grid = np.copy(self.grid)
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    new_grid[i, j] = 0.25 * (self.grid[i+1, j] + self.grid[i-1, j] +
                                             self.grid[i, j+1] + self.grid[i, j-1])
            self.update_with_periodic_conditions(new_grid)
            if np.max(np.abs(new_grid - self.grid)) < epsilon:
                break
            self.grid = new_grid
        print(f"Jacobi method converged in {iteration_count} iterations.")

    def gauss_seidel_solve(self, epsilon=1e-5):
        iteration_count = 0
        while True:
            iteration_count += 1
            old_grid = np.copy(self.grid)
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    self.grid[i, j] = 0.25 * (self.grid[i+1, j] + self.grid[i-1, j] +
                                              self.grid[i, j+1] + self.grid[i, j-1])
            self.update_with_periodic_conditions(self.grid)
            if np.max(np.abs(self.grid - old_grid)) < epsilon:
                break
        print(f"Gauss-Seidel method converged in {iteration_count} iterations.")

    def sor_solve(self, omega, epsilon=1e-5):
        iteration_count = 0
        while True:
            iteration_count += 1
            old_grid = np.copy(self.grid)
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    new_val = 0.25 * (self.grid[i+1, j] + self.grid[i-1, j] +
                                      self.grid[i, j+1] + self.grid[i, j-1])
                    self.grid[i, j] = omega * new_val + (1 - omega) * self.grid[i, j]
            self.update_with_periodic_conditions(self.grid)
            if np.max(np.abs(self.grid - old_grid)) < epsilon:
                break
        print(f"SOR method converged in {iteration_count} iterations.")

    def plot_solution(self):
        plt.imshow(self.grid, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Solution Plot")
        plt.show()

# Example usage
N = 50
solver = DiffusionSolver(N)

print("Solving with Jacobi...")
solver.jacobi_solve()
solver.plot_solution()

solver = DiffusionSolver(N)  # Reset for Gauss-Seidel
print("Solving with Gauss-Seidel...")
solver.gauss_seidel_solve()
solver.plot_solution()

solver = DiffusionSolver(N)  # Reset for SOR
omega = 1.8  
print("Solving with SOR, omega =", omega)
solver.sor_solve(omega)
solver.plot_solution()
