import numpy as np
import matplotlib.pyplot as plt
from diffusion_steady_state import DiffusionSolver


def solve_diffusion(N, omega, epsilon=1e-5, testing_mode=False, max_iterations=1000):
    solver = DiffusionSolver(N)
    solver.reset_grid()
    iteration_count, converged = solver.sor_solve(omega, epsilon, track_delta=True, testing_mode=testing_mode, max_iterations=max_iterations)
    return iteration_count, converged

def find_optimal_omega_per_N(N_values, omega_start, omega_end, increment, max_iterations=1000):
    optimal_omega_per_N = []  

    for N in N_values:
        best_omega = None
        best_iterations = max_iterations
        for omega in np.arange(omega_start, omega_end + increment, increment):
            iteration_count, converged = solve_diffusion(N, omega, testing_mode=True, max_iterations=max_iterations)
            if converged and (best_omega is None or iteration_count < best_iterations):
                best_omega = omega
                best_iterations = iteration_count
        if best_omega is not None:
            optimal_omega_per_N.append((N, best_omega))
            print(f"Optimal omega for N={N} is {best_omega} with {best_iterations} iterations.")
        else:
            print(f"No optimal omega found for N={N}.")

    return optimal_omega_per_N


def plot_optimal_omegas(optimal_omega_per_N):
    Ns, omegas = zip(*optimal_omega_per_N)  # Unpack N values and their corresponding optimal omegas
    
    plt.figure(figsize=(10, 6))
    plt.plot(Ns, omegas, marker='o', linestyle='-', color='blue')
    plt.xlabel('Grid Size N')
    plt.ylabel('Optimal Omega (ω)')
    plt.title('Optimal Omega vs. Grid Size N')
    plt.grid(True)
    plt.show()


