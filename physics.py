import numpy as np

# Constants
G = 9.81        # m/s^2
RHO_0 = 1.225   # kg/m^3, sea-level air density
H_SCALE = 8500  # m, atmosphere scale height

def air_density(altitude):
    """Exponential atmosphere model. Returns density in kg/m^3. /
    Equations taken from NASA's Earth Atmospheric Model /
    https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html"""
    
    if altitude < 0:
        raise ValueError("Altitude must be at least 0!")
    if altitude >= 25000:
        T = -131.21 + (.00299 * altitude)
        p_pressure = 2.488 * (((T + 273.1)/216.6) ** (-11.388))
    elif 11000 < altitude < 25000:
        T = -56.46
        p_pressure = 22.65 * (np.exp(1.73 - (0.000157 * altitude)))
    else:
        T = 15.04 - (0.00649 * altitude)
        p_pressure = 101.29 * (((T + 273.1)/288.08) ** 5.256)

    p_density = p_pressure / (0.289 * (T + 273.1))

    return p_density

def drag_force(velocity, altitude, Cd, A, mass):
    """
    Returns drag deceleration (m/s^2) opposing velocity direction.
    velocity: np.array([vx, vy])
    """
    rho = air_density(altitude)
    v_mag = np.linalg.norm(velocity)

    if v_mag == 0:
        drag = 0
        return np.zeros(2)
    
    unit_vector = velocity/v_mag
    drag = 0.5 * rho * Cd * A * (v_mag ** 2)
    deceleration = drag / mass
    return deceleration * unit_vector * -1

def equations_of_motion(state, t, Cd, A, mass):
    
    x, y, vx, vy = state
    velocity = np.array([vx, vy])
    drag_x, drag_y = drag_force(velocity, y, Cd, A, mass)
    ax = drag_x
    ay = drag_y - G
    dstate_dt = np.array([vx, vy, ax, ay])

    return dstate_dt



