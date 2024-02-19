import numpy as np
import matplotlib.pyplot as plt

# Variables
L = 1   # Length of the string
N = 200 # Number of intervals
c = 1   # Wave speed 
d_x = L / N
d_t = 0.001
x = np.linspace(0, L, N+1)
time_steps = 200

# Initial conditions
psi_current = np.sin(5 * np.pi * x) 
psi_next = np.zeros(N+1)
psi_prev = np.copy(psi_current)  

plt.figure(figsize=(10, 6))

for t in range(time_steps):
    for i in range(1, N):
        psi_next[i] = (2 * psi_current[i] - psi_prev[i] +
                       c**2 * (d_t**2 / d_x**2) *
                       (psi_current[i + 1] - 2 * psi_current[i] + psi_current[i - 1]))

    psi_prev = np.copy(psi_current)
    psi_current = np.copy(psi_next)
    
    if t in [0, 50, 100, 150, 199]:
        plt.plot(x, psi_current, label=f"t={t * d_t:.3f}")

plt.title("Wave Propagation - Initial Condition II: $\sin(5\pi x)$")
plt.xlabel("x")
plt.ylabel("$\Psi(x, t)$")
plt.legend()
plt.grid(True)
plt.savefig("Wave_propagation_for_ic_ii", dpi=300)
plt.show()
