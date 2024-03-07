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

    def reset_grid(self):
        self.grid = np.zeros((self.N, self.N))
        self.setup_boundary_conditions()

    def update_with_periodic_conditions(self, new_grid):
        # Apply periodic boundary conditions in the x-direction
        new_grid[:, 0] = new_grid[:, -2]  
        new_grid[:, -1] = new_grid[:, 1]

    def jacobi_solve(self, epsilon=1e-5, track_delta=False):
        iteration_count = 0
        delta_history = []
        while True:
            iteration_count += 1
            new_grid = np.copy(self.grid)
            max_change = 0
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    temp = 0.25 * (self.grid[i+1, j] + self.grid[i-1, j] +
                                   self.grid[i, j+1] + self.grid[i, j-1])
                    max_change = max(max_change, abs(temp - self.grid[i, j]))
                    new_grid[i, j] = temp
            self.update_with_periodic_conditions(new_grid)
            if track_delta:
                delta_history.append(max_change)
            if max_change < epsilon:
                break
            self.grid = new_grid
        if track_delta:
            self.delta_history = delta_history
        print(f"Jacobi method converged in {iteration_count} iterations.")

    def gauss_seidel_solve(self, epsilon=1e-5, track_delta=False):
        iteration_count = 0
        delta_history = []
        while True:
            iteration_count += 1
            old_grid = np.copy(self.grid)
            max_change = 0
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    temp = 0.25 * (self.grid[i+1, j] + self.grid[i-1, j] +
                                   self.grid[i, j+1] + self.grid[i, j-1])
                    max_change = max(max_change, abs(temp - self.grid[i, j]))
                    self.grid[i, j] = temp
            self.update_with_periodic_conditions(self.grid)
            if track_delta:
                delta_history.append(max_change)
            if max_change < epsilon:
                break
        if track_delta:
            self.delta_history = delta_history
        print(f"Gauss-Seidel method converged in {iteration_count} iterations.")

    def sor_solve(self, omega, epsilon=1e-5, max_iterations=1000, track_delta=False, testing_mode=False):
        iteration_count = 0
        delta_history = []
        while True:
            iteration_count += 1
            old_grid = np.copy(self.grid)
            max_change = 0
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):
                    temp = 0.25 * (self.grid[i+1, j] + self.grid[i-1, j] +
                                   self.grid[i, j+1] + self.grid[i, j-1])
                    new_val = omega * temp + (1 - omega) * self.grid[i, j]
                    max_change = max(max_change, abs(new_val - self.grid[i, j]))
                    self.grid[i, j] = new_val
            self.update_with_periodic_conditions(self.grid)
            if track_delta:
                delta_history.append(max_change)
            if max_change < epsilon:
                break
            if testing_mode and iteration_count >= max_iterations:
                if track_delta:
                    self.delta_history = delta_history
                return iteration_count, False
        if track_delta:
            self.delta_history = delta_history
        print(f"SOR method converged in {iteration_count} iterations.")
        return iteration_count, True

    def generate_analytical_solution(self):
        y = np.linspace(1, 0, self.N)  # Linear array from 0 to 1
        analytical_solution = np.tile(y, (self.N, 1)).T  # Repeat this array across all columns
        return analytical_solution

    def calculate_rmse(self, numerical_solution):
        analytical_solution = self.generate_analytical_solution()
        error = numerical_solution - analytical_solution
        mse = np.mean(np.square(error))
        rmse = np.sqrt(mse)
        return rmse

    def plot_solution(self, numerical_solution, title, compare_with_analytical=False):
        plt.figure(figsize=(12, 6))
    
        # Plot numerical solution
        plt.subplot(1, 2, 1)
        plt.imshow(numerical_solution, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(title)
    
        if compare_with_analytical:
            # Plot analytical solution
            plt.subplot(1, 2, 2)
            analytical_solution = self.generate_analytical_solution()
            plt.imshow(analytical_solution, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Analytical Solution c(y, t) = y")

        plt.show()

# Example usage
# N = 50
# solver = DiffusionSolver(N)

# print("Solving with Jacobi...")
# solver.jacobi_solve()
# jacobi_solution = np.copy(solver.grid)
# jacobi_rmse = solver.calculate_rmse(jacobi_solution)
# print(f"RMSE Jacobi: {jacobi_rmse}")

# solver = DiffusionSolver(N)  # Reset for Gauss-Seidel
# print("Solving with Gauss-Seidel...")
# solver.gauss_seidel_solve()
# gauss_seidel_solution = np.copy(solver.grid)
# gauss_seidel_rmse = solver.calculate_rmse(gauss_seidel_solution)
# print(f"RMSE Gauss-Seidel: {gauss_seidel_rmse}")

# solver = DiffusionSolver(N)  # Reset for SOR
# omega = 1.8
# print("Solving with SOR, omega =", omega)
# solver.sor_solve(omega)
# sor_solution = np.copy(solver.grid)
# sor_rmse = solver.calculate_rmse(sor_solution)
# print(f"RMSE SOR (omega={omega}): {sor_rmse}")

# solver.jacobi_solve(track_delta=True)
# jacobi_delta_history = solver.delta_history

# solver = DiffusionSolver(N)
# solver.gauss_seidel_solve(track_delta=True)
# gauss_seidel_delta_history = solver.delta_history

# solver = DiffusionSolver(N)
# solver.sor_solve(omega=1.8, track_delta=True)
# sor_delta_history = solver.delta_history

# plt.figure(figsize=(10, 6))
# plt.semilogy(jacobi_delta_history, label='Jacobi Method')
# plt.semilogy(gauss_seidel_delta_history, label='Gauss-Seidel Method')
# plt.semilogy(sor_delta_history, label=f'SOR Method (omega={1.8})')
# plt.xlabel('Iteration ($k$)')
# plt.ylabel('Max Change ($\delta$)')
# plt.title('Convergence Measure $\delta$ vs. Number of Iterations $k$')
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()


def find_optimal_omega(N, omega_start, omega_end, omega_step, epsilon=1e-5, max_iterations=1000):
    solver = DiffusionSolver(N)
    optimal_omega = omega_start
    min_iterations = float('inf')
    
    for omega in np.arange(omega_start, omega_end, omega_step):
        solver.reset_grid()
        iterations, converged = solver.sor_solve(omega, epsilon, max_iterations, testing_mode=True)
        
        if converged and iterations < min_iterations:
            min_iterations = iterations
            optimal_omega = omega
            
        print(f"Omega: {omega:.2f}, Iterations: {iterations}")

    return optimal_omega, min_iterations

N = 256
omega_start = 1.0
omega_end = 2.0
omega_step = 0.05

optimal_omega, min_iterations = find_optimal_omega(N, omega_start, omega_end, omega_step)
print(f"The optimal omega is {optimal_omega} with {min_iterations} iterations.")