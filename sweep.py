import numpy as np
import pandas as pd
from simulate import simulate_rk4

def sweep_angles(angles, v0, mass, Cd, A, dt=0.1):
    """
    Sweeps launch angles and records max altitude and landing distance.
    
    Parameters:
        angles: array of launch angles in degrees
        v0, mass, Cd, A, dt: passed directly to simulate_rk4
    
    Returns:
        DataFrame with columns [angle, max_altitude, landing_x]
    """
    sweeper = []

    for value in angles:
        rk4 = simulate_rk4(value, v0, mass, Cd, A, dt)  
        sweeper.append({
            'angle': value, 
            'max_altitude': rk4['y'].max(),
            'landing_x': rk4['x'].iloc[-1]})

    swept = pd.DataFrame(sweeper)
    return swept
