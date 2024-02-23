import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Variables
L = 1
time_steps = 400
N = 400
d_x = L / N
d_t = 0.001
c = 1
x = np.linspace(0, L, N+1)


# Initial conditions
psi_current = np.where((x > 1/5) & (x < 2/5), np.sin(5 * np.pi * x), 0)
psi_next = np.zeros(N+1)
psi_prev = np.copy(psi_current)

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, psi_current, label="Wave at t=0")
ax.set_ylim(-1.5, 1.5)
ax.set_title("Wave Propagation - Initial Condition iii: $\sin(5\pi x)$")
ax.set_xlabel("x")
ax.set_ylabel("$\Psi(x, t)$")
plt.grid(True)

def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

def animate(t):
    global psi_current, psi_prev, psi_next
    for i in range(1, N):
        psi_next[i] = 2 * psi_current[i] - psi_prev[i] + c**2 * ((d_t)**2 / (d_x)**2) * (psi_current[i + 1] - 2 * psi_current[i] + psi_current[i -1])
    psi_prev = np.copy(psi_current)
    psi_current = np.copy(psi_next)

    line.set_ydata(psi_current)
    return line,


ani = animation.FuncAnimation(fig, animate, frames=time_steps, init_func=init, interval=5, blit=True)

# plt.show()

# ani.save('wave_propagation_initial_3_pi.mp4', writer='ffmpeg', fps=60)

