import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import erfc


def analytical_solution(y, t, D, terms=50):
    summation = 0
    for i in range(terms):
        summation += erfc((1 - y + 2*i) / (2 * np.sqrt(D * t)))
        summation -= erfc((1 + y + 2*i) / (2 * np.sqrt(D * t)))
    return summation 


def Calc_Value(Copy_Lattice, i, j, D, dx, dt, N):
    # Calculate the new concentration value for a given grid point using the finite difference method
    # Handle periodic boundary conditions in the x-direction
    left_i = (i - 1) % N
    right_i = (i + 1) % N
    up_j = (j + 1) % N
    down_j = (j - 1) % N
    
    Value = Copy_Lattice[i,j] + (D*dt/(dx**2)) * (Copy_Lattice[right_i,j] + Copy_Lattice[left_i,j] + Copy_Lattice[i,up_j] + Copy_Lattice[i,down_j] - 4*Copy_Lattice[i,j])
    return Value


def Diffusion(dx, dt, D):
    N = int(round(1 / dx))
    Lattice = np.zeros((N,N))
    Lattice[:, N-1] = 1

    y_values = np.linspace(0, 1, N)
    analytical_results = {}

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axs = axs.flatten()
    time_points = [0.001, 0.01, 0.1, 1]
    plot_number = 0 


    for t in np.arange(0, 1 + dt, dt):
        Copy_Lattice = np.copy(Lattice)  

        for i in range(N):
            for j in range(1, N - 1):  
                Lattice[i,j] = Calc_Value(Copy_Lattice, i, j, D, dx, dt, N)

    #     if t in [0.001, 0.01, 0.1, 1]:
    #         analytical_results[t] = [analytical_solution(y, t, D) for y in y_values]
    #         values_y = [Lattice[0,j] for j in range(N)]
    #         values_x = np.arange(0, 1, dx)
    #         plt.plot(y_values, values_y, label=f"Numerical t={t:.3f}")

    # for t, c_values in analytical_results.items():
    #     plt.plot(y_values, c_values, label=f"Analytical t={t:.3f}", linestyle='--')
        if t in time_points:
            ax = axs[plot_number]
            im = ax.imshow(np.flipud(Lattice.T), cmap='hot', extent=[0, 1, 1, 0])  # Set origin to 'upper'
            ax.set_title(f"t={t:.3f}")
            fig.colorbar(im, ax=ax)
            plot_number += 1

    # plt.xlabel('y')
    # plt.ylabel('c')
    # plt.legend()
    # plt.show()
    plt.tight_layout()
    plt.show()


# Parameters
dx = 0.05
dt = 0.0001
D = 1

# Run the diffusion simulation
Diffusion(dx, dt, D)
