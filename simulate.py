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
    
        state += equations_of_motion(state, t, Cd, A, mass) * dt
        if state[1] <= 0:
            break
        t += dt
        results.append([t, state[0], state[1],state[2],state[3]])
       
    results = np.array(results)

    df = pd.DataFrame(results, columns = ["t", "x", "y", "vx", "vy"] )
    return df

def simulate_rk4(launch_angle_deg, v0, mass, Cd, A, dt=0.1, max_time=300):
   
    theta = np.radians(launch_angle_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    t = 0
    state = np.array([0, 0, vx, vy])
    results = []
    results.append([t, state[0], state[1],state[2],state[3]])

    
    while t <= max_time:
        
        k1 = equations_of_motion(state, t, Cd, A, mass)
        k2 = equations_of_motion(state + k1*dt/2, t + dt/2, Cd, A, mass)
        k3 = equations_of_motion(state + k2*dt/2, t + dt/2, Cd, A, mass)
        k4 = equations_of_motion(state + k3*dt, t + dt, Cd, A, mass)
        state = state + (k1 + 2*k2 + 2*k3 + k4) * dt/6
        if state[1] <= 0:
            break
        t += dt
        results.append([t, state[0], state[1],state[2],state[3]])
       
    results = np.array(results)

    df = pd.DataFrame(results, columns = ["t", "x", "y", "vx", "vy"] )
    return df