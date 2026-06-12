import matplotlib.pyplot as plt
import pandas as pd

def plot_trajectory(df, title="Rocket Trajectory"):
    df.plot(x = 'x', y = 'y', kind = "line", legend = False)
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    from simulate import simulate
    df = simulate(45, 100, 10, 0.5, 0.1)
    print(df)
    print(len(df))
    plot_trajectory(df)