import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

class DiffusionSimulation:
    def __init__(self, D=1, dx=0.05, dt=0.0001, L=1):
        self.D = D  # Diffusion coefficient
        self.dx = dx  # Spatial step size
        self.dt = dt  # Time step size
        self.L = L  # Domain length
        self.initialize_lattice()  # Initialize lattice with boundary conditions

    def initialize_lattice(self):
        """Initialize or reset the lattice to the initial condition."""
        self.N = int(round(self.L / self.dx))  
        self.Lattice = np.zeros((self.N, self.N)) 
        self.Lattice[:, self.N - 1] = 1  
    
    def analytical_solution(self, y, t, terms=50):
        """Calculate the analytical solution for the concentration."""
        summation = 0
        for i in range(terms):
            summation += erfc((1 - y + 2*i) / (2 * np.sqrt(self.D * t)))
            summation -= erfc((1 + y + 2*i) / (2 * np.sqrt(self.D * t)))
        return summation

    def calc_value(self, Copy_Lattice, i, j):
        """Calculate the new concentration value for a given grid point."""
        left_i = (i - 1) % self.N
        right_i = (i + 1) % self.N
        up_j = (j + 1) % self.N
        down_j = (j - 1) % self.N

        Value = Copy_Lattice[i,j] + (self.D * self.dt / (self.dx ** 2)) * \
                (Copy_Lattice[right_i,j] + Copy_Lattice[left_i,j] + \
                 Copy_Lattice[i,up_j] + Copy_Lattice[i,down_j] - 4*Copy_Lattice[i,j])
        return Value

    def simulate_to_time(self, t):
        """Simulate the system up to time t, starting from the initial condition."""
        self.initialize_lattice()  
        steps = int(t / self.dt)
        for _ in range(steps):
            Copy_Lattice = np.copy(self.Lattice)
            for i in range(self.N):
                for j in range(1, self.N - 1):
                    self.Lattice[i, j] = self.calc_value(Copy_Lattice, i, j)

    def compare_with_analytical(self, time_points):
        """Compare the simulation results with the analytical solution at multiple time points."""
        plt.figure(figsize=(10, 6))
        y_values = np.linspace(0, 1, self.N)
        
        for t in time_points:
            self.simulate_to_time(t)
            simulation_results = self.Lattice[0, :]
            analytical_results = [self.analytical_solution(y, t) for y in y_values]
            plt.plot(y_values, simulation_results, label=f'Simulation t={t}')
            plt.plot(y_values, analytical_results, label=f'Analytical t={t}', linestyle="--")
        
        plt.title('Comparison of Simulation with Analytical Solution')
        plt.xlabel('y')
        plt.ylabel('Concentration')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_2d_concentration(self, time_points):
        """Visualize the 2D concentration domain at multiple time points."""
        fig, axs = plt.subplots(1, len(time_points), figsize=(15, 5))
        
        for i, t in enumerate(time_points):
            self.simulate_to_time(t)
            ax = axs[i] if len(time_points) > 1 else axs
            im = ax.imshow(np.flipud(self.Lattice.T), cmap='hot', extent=[0, 1, 1, 0], aspect='auto')
            ax.set_title(f't={t}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()

# Parameters and usage
D = 1
dx = 0.05
dt = 0.0001
time_points = [0.001, 0.01, 0.1, 1]

simulation = DiffusionSimulation(D=D, dx=dx, dt=dt)
simulation.compare_with_analytical(time_points)
simulation.visualize_2d_concentration(time_points)