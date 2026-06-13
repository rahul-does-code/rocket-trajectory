import matplotlib.pyplot as plt
import pandas as pd

def plot_trajectory(df, title="Rocket Trajectory"):
    df.plot(x = 'x', y = 'y', kind = "line", legend = False)
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title(title)
    plt.show()


def plot_euler_vs_rk4(df_euler, df_rk4, title="Euler vs RK4"):
    """
    Plots two trajectories on the same axes for comparison.
    """
    plt.plot(df_euler['x'], df_euler['y'], label = "Euler", color = "blue")
    plt.plot(df_rk4['x'], df_rk4['y'], label = "RK4", color = "red")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from simulate import simulate, simulate_rk4
    df_euler = simulate(45, 100, 10, 0.5, 0.1, dt=0.5)
    df_rk4 = simulate_rk4(45, 100, 10, 0.5, 0.1, dt=0.5)
    plot_euler_vs_rk4(df_euler, df_rk4)