import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class WaveSimulation:
    def __init__(self, L=1, N=200, c=1, d_t=0.001):
        self.L = L
        self.N = N
        self.c = c
        self.d_t = d_t
        self.d_x = L / N
        self.x = np.linspace(0, L, N+1)


    def set_initial_condition(self, initial_condition):
        self.psi_current = initial_condition(self.x)
        self.psi_prev = np.copy(self.psi_current)
        self.psi_next = np.zeros(self.N+1)


    def simulate_step(self):
        for i in range(1, self.N):
            self.psi_next[i] = (2 * self.psi_current[i] - self.psi_prev[i] +
                                self.c**2 * (self.d_t**2 / self.d_x**2) *
                                (self.psi_current[i + 1] - 2 * self.psi_current[i] + self.psi_current[i - 1]))
        self.psi_prev, self.psi_current = self.psi_current, np.copy(self.psi_next)


    def simulate_to_time(self, target_time):
        steps = int(target_time / self.d_t)
        for _ in range(steps):
            self.simulate_step()


    def plot(self, plot_times, title="Wave Propagation"):
        plt.figure(figsize=(10, 6))
        initial_condition = self.psi_current

        for t in plot_times:
            self.set_initial_condition(lambda x: initial_condition)
            self.simulate_to_time(t)
            plt.plot(self.x, self.psi_current, label=f"t={t:.3f}s")

        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("$\Psi(x, t)$")
        plt.legend()
        plt.grid(True)
        plt.show()


class AnimatedWaveSimulation(WaveSimulation):
    def animate(self, frames, interval=5, save=False, filename="wave_animation.mp4"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim(-1.5, 1.5)
        line, = ax.plot(self.x, self.psi_current)
        ax.set_title("Wave Propagation Animation")
        ax.set_xlabel("x")
        ax.set_ylabel("$\Psi(x, t)$")
        plt.grid(True)

        def init():
            line.set_ydata([np.nan] * len(self.x))
            return line,

        def animate(i):
            self.simulate_step()
            line.set_ydata(self.psi_current)
            return line,

        ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, interval=interval, blit=True)

        if save:
            ani.save(filename, writer='ffmpeg', fps=60)
            plt.show()
        else:
            plt.show()


# wave_sim = WaveSimulation(L=1, N=200, c=1, d_t=0.001)
# initial_condition = lambda x: np.sin(2 * np.pi * x)
# wave_sim.set_initial_condition(initial_condition)
# plot_times = [0, 0.05, 0.1, 0.15, 0.199]
# wave_sim.plot(plot_times, title="Wave Propagation - $\sin(2\pi x)$")

animated_wave_sim = AnimatedWaveSimulation(L=1, N=200, c=1, d_t=0.001)
# initial_condition_1 = lambda x: np.where((x > 1/5) & (x < 2/5), np.sin(5 * np.pi * x), 0)
# animated_wave_sim.set_initial_condition(initial_condition_1)
# animated_wave_sim.animate(frames=400, interval=5, save=False, filename='wave_propagation.mp4')