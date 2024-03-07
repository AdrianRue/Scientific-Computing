import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange


grid_size = 100
grid = np.zeros((grid_size, grid_size))
concentration = np.zeros((grid_size, grid_size))

seed_position_x = grid_size // 2
seed_position_y = grid_size - 1
grid[seed_position_y][seed_position_x] = 1

# setting boundary condition
def boundary_condition():
    concentration[0, :] = 1
    concentration[-1, :] = 0

@jit(nopython=True)
def update_with_periodic_conditions(concentration):
    concentration[:, 0] = concentration[:, -2]
    concentration[:, -1] = concentration[:, 1]

def concentration_gradient():
    c_max = 1.0
    c_min = 0.0
    for y in range(grid_size):
        concentration_y = c_max - (y / (grid_size - 1)) * (c_max - c_min)
        concentration[y, :] = concentration_y
    grid[seed_position_y][seed_position_x] = 1

def initialize_concentration():
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 1:
                concentration[i, j] = 0
            else:
                concentration[i, j] = 1 - i / (grid_size - 1)
@jit(nopython=True)
def solve_laplace_sor(concentration, grid, omega=1.9, epsilon=1e-5, max_iteration=10000):
    grid_size = concentration.shape[0]
    iteration_count = 0
    while True:
        delta = 0
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                if grid[i, j] == 1:
                    continue
                old_value = concentration[i, j]
                north = concentration[i - 1, j]
                south = concentration[i + 1, j]
                east = concentration[i, j + 1]
                west = concentration[i, j - 1]
                new_value = omega * 0.25 * (north + south + east + west) + (1 - omega) * old_value
                concentration[i, j] = new_value
                delta += abs(new_value - old_value)
        update_with_periodic_conditions(concentration)
        iteration_count += 1
        if delta < epsilon or iteration_count > max_iteration:
            break

def growth_candidates():
    candidates = []
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            if grid[i, j] == 0 and (grid[i-1, j] == 1 or grid[i+1, j] == 1 or grid[i, j-1] == 1 or grid[i, j+1] == 1):
                candidates.append((i, j))
    print(f"Number of growth candidates: {len(candidates)}")
    return candidates

def growth_probabilities(candidates, concentration, eta):
    probabilities = []
    for i, j in candidates:
        c = max(concentration[i, j], 1e-10)
        probabilities.append(c**eta)
    probabilities = np.array(probabilities)
    total = np.sum(probabilities)
    if total == 0:
        total = 1e-10
    probabilities /= total
    print(f"Growth probabilities (first 5): {probabilities[:5]} Total: {total}")
    return probabilities

def select_growth_candidate(candidates, probabilities):
    selected_index = np.random.choice(len(candidates), p=probabilities)
    selected_candidate = candidates[selected_index]
    print(f"Selected growth candidate: {selected_candidate}")
    return selected_candidate

def add_to_cluster(selected_candidate):
    i, j = selected_candidate
    grid[i, j] = 1
    concentration[i, j] = 0 

def visualize(grid, concentration):
    plt.figure(figsize=(6, 6))
    plt.imshow(concentration, cmap='hsv', interpolation='nearest', alpha=0.8)
    cluster = np.ma.masked_where(grid == 0, grid)
    plt.imshow(cluster, cmap='twilight', interpolation='nearest', alpha=1.0)
    plt.title('Diffusion-Limited Aggregation with Concentration Field')
    plt.show()

boundary_condition()
concentration_gradient()

for iteration in range(365):
    solve_laplace_sor(concentration, grid, omega=1.9, epsilon=1e-5, max_iteration=10000)
    candidates = growth_candidates()
    if not candidates:
        break
   # print(f"Number of growth candidates: {len(candidates)}")
    probabilities = growth_probabilities(candidates, concentration, eta=2.0)
    # print(f"Growth probabilities (first 5): {probabilities[:5]} Total: {total}")
    selected_candidate = select_growth_candidate(candidates, probabilities)
    add_to_cluster(selected_candidate)
    initialize_concentration()
    solve_laplace_sor(concentration, grid, omega=1.9, epsilon=1e-5, max_iteration=10000)

visualize(grid, concentration)


