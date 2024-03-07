import numpy as np
import matplotlib.pyplot as plt



class DiffusionSolver:
    def __init__(self, N=50):
        self.N = N
        self.grid = np.zeros((N, N))
        self.setup_boundary_conditions()

    def setup_boundary_conditions(self):
        self.grid[0, :] = 1  
        self.grid[-1, :] = 0   

    def reset_grid(self):
        self.grid = np.zeros((self.N, self.N))
        self.setup_boundary_conditions()

    def update_with_periodic_conditions(self, new_grid):
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
        # print(f"SOR method converged in {iteration_count} iterations.")
        return iteration_count, True

    def generate_analytical_solution(self):
        """Generate analytical solution"""
        y = np.linspace(1, 0, self.N)  
        analytical_solution = np.tile(y, (self.N, 1)).T  
        return analytical_solution

    def calculate_rmse(self, numerical_solution):
        """Calculates the RMSE for different methods"""
        analytical_solution = self.generate_analytical_solution()
        error = numerical_solution - analytical_solution
        mse = np.mean(np.square(error))
        rmse = np.sqrt(mse)
        return rmse

    def plot_solution(self, numerical_solution, title, compare_with_analytical=False):
        """To see if the simulated results looks like the analytical one"""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(numerical_solution, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(title)
    
        if compare_with_analytical:
            plt.subplot(1, 2, 2)
            analytical_solution = self.generate_analytical_solution()
            plt.imshow(analytical_solution, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Analytical Solution c(y, t) = y")

        plt.show()


N = 50
solver = DiffusionSolver(N)

# print("Solving with Jacobi...")
# solver.jacobi_solve()
# jacobi_solution = np.copy(solver.grid)
# jacobi_rmse = solver.calculate_rmse(jacobi_solution)
# print(f"RMSE Jacobi: {jacobi_rmse}")


