import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from diffusion_steady_state import DiffusionSolver

def find_optimal_omega(N_omega_max_iterations):
    N, omega, max_iterations = N_omega_max_iterations
    solver = DiffusionSolver(N)
    optimal_omega = {'omega': omega, 'iterations': float('inf'), 'rmse': float('inf'), 'exceeded': False}
    
    solver.reset_grid()
    solver.sor_solve(omega, epsilon=1e-5, track_delta=True)
    iterations = len(solver.delta_history)
    
    if iterations > max_iterations:
        optimal_omega['exceeded'] = True  # Mark that iterations exceeded the threshold
        return optimal_omega
    
    rmse = solver.calculate_rmse(solver.grid)
    optimal_omega.update({'omega': omega, 'iterations': iterations, 'rmse': rmse})
    
    return optimal_omega

def perform_omega_sweep(args):
    N, omega_start, omega_end, num_points, max_iterations = args
    omega_values = np.linspace(omega_start, omega_end, num_points)
    with multiprocessing.Pool() as pool:
        results = pool.map(find_optimal_omega, [(N, omega, max_iterations) for omega in omega_values])
    
    # Filter out results where iterations exceeded the threshold
    filtered_results = [result for result in results if not result['exceeded']]
    
    if not filtered_results:  # If all results were filtered out
        return {'omega': None, 'iterations': float('inf'), 'rmse': float('inf'), 'exceeded': True}
    
    optimal_omega_info = min(filtered_results, key=lambda x: x['iterations'])
    return optimal_omega_info

def plot_omega_iterations(N, omega_start, omega_end, num_points, max_iterations):
    omega_values = np.linspace(omega_start, omega_end, num_points)
    with multiprocessing.Pool() as pool:
        results = pool.map(find_optimal_omega, [(N, omega, max_iterations) for omega in omega_values])
    
    iterations = [result['iterations'] for result in results if not result['exceeded']]
    omegas = [result['omega'] for result in results if not result['exceeded']]

    plt.figure(figsize=(10, 6))
    plt.plot(omegas, iterations, marker='o')
    plt.xlabel('Omega (ω)')
    plt.ylabel('Number of Iterations to Converge')
    plt.title(f'Number of Iterations to Converge vs. Omega (ω) for N={N}')
    plt.grid(True)
    plt.show()

def analyze_omega_dependency(N_values, omega_start, omega_end, num_points, max_iterations):
    results = []
    for N in N_values:
        optimal_omega_info = perform_omega_sweep((N, omega_start, omega_end, num_points, max_iterations))
        results.append((N, optimal_omega_info['omega']))
        print(f"For N={N}, the optimal omega is {optimal_omega_info['omega']} with {optimal_omega_info['iterations']} iterations.")
    return results

def main():
    N_values = [40]
    omega_start = 1.72
    omega_end = 1.9
    increment = 0.02
    max_iterations = 1000  # Set a reasonable threshold to avoid excessive iterations
    num_points = int((omega_end - omega_start) / increment + 1)

    # Collect optimal omega for each N
    optimal_omega_results = analyze_omega_dependency(N_values, omega_start, omega_end, num_points, max_iterations)

    # Plot the dependency of optimal omega on N
    plt.figure(figsize=(10, 6))
    Ns, optimal_omegas = zip(*optimal_omega_results)  # Unpack N values and corresponding optimal omegas
    plt.plot(Ns, optimal_omegas, marker='o', linestyle='-', color='blue')
    plt.xlabel('Grid Size N')
    plt.ylabel('Optimal Omega (ω)')
    plt.title('Dependency of Optimal Omega on Grid Size N')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()