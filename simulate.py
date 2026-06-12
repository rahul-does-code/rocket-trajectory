import numpy as np
import pandas as pd
from physics import equations_of_motion

def simulate(launch_angle_deg, v0, mass, Cd, A, dt=0.1, max_time=300):
    """ Runs a single rocket trajectory simulation using Euler integration."""
    theta = np.radians(launch_angle_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    t = 0
    state = np.array([0, 0, vx, vy])
    results = []
    results.append([t, state[0], state[1],state[2],state[3]])

    while t <= max_time:
    
        state += equations_of_motion(state, t, Cd, A, mass)
        if state[1] <= 0:
            break
        t += dt
        results.append([t, state[0], state[1],state[2],state[3]])
       
    results = np.array(results)

    df = pd.DataFrame(results, columns = ["t", "x", "y", "vx", "vy"] )
    return df
