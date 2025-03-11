import numpy as np
import pandas as pd
from modules import coord_transform

""" 
Creating Isotropic RA,DEC .

First we generate a cube with  edge lenth (-1,1) with uniform points inside it
Calculate each points distance from origin, then check if it lies insode a unit sphere
Now normalize the poin
"""

def uniform_random_points_on_sphere(num_points):
    points = []

    while len(points) < num_points:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        w = np.sqrt(x**2 + y**2 + z**2)             # Calculate the norm (distance from origin)
        
        if w <= 1:      # Check if the point is inside the unit sphere (w <= 1)    
            x_normalized = x / w
            y_normalized = y / w
            z_normalized = z / w        # Normalize the point to lie on the surface of the unit sphere
            
            points.append([x_normalized, y_normalized, z_normalized])
    
    return np.array(points[0])

# For creating trigger time better to use time in seconds in a year so that its easir to calculate the satellite positions considering the time period.

""" creating random trigger time for GRBs
"""
def trigger_time(total_events): 
    random_times = np.random.uniform(0, 365.25, total_events) 

    # Convert random times to full timestamps within the year
    start = pd.Timestamp('2024-01-01')
    event_timestamps = start + pd.to_timedelta(random_times, unit='D')

    # Format the timestamps to "YYYY-MM-DD HH:MM:SS.sss" 
    formatted_events = event_timestamps.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3] 
    return formatted_events


def generate_grb(catalog):
    cartesian_points = uniform_random_points_on_sphere(1)
    radec_points = coord_transform.c2r(cartesian_points[0],cartesian_points[1],cartesian_points[2])
    ra = radec_points[0]
    dec = radec_points[1]
    t90 = np.random.choice(catalog['t90     '])
    flux = np.random.choice(catalog['flnc_band_phtfluxb'])
    #print("RA:", ra, "Dec:", dec, "T90:", t90, "Flux:", flux)
    return cartesian_points, ra, dec, t90, flux


# SATELLITE POSITION AND POINTING POSITION SIMULATION


def sat_positions(num_sat):
    earth_radius = 6371     # in km
    loe = 510   # in km
    orbit_radius =  earth_radius + loe      # in km
    dec_angles = np.radians(np.linspace(-90, 90, num_sat))  # the satellites are equally spaced on one side of the polar orbit
    ra_angles = np.radians(180)
    
    x_position = orbit_radius * np.cos(ra_angles)*np.cos(dec_angles)
    y_position = orbit_radius * np.sin(ra_angles)*np.cos(dec_angles)
    z_position = orbit_radius * np.sin(dec_angles)

    sat_pos = np.vstack([x_position, y_position, z_position]).T 
    sat_pos[np.abs(sat_pos) < 1e-10] = 0
    sat_pos = np.where(sat_pos != 0, np.round(sat_pos, 1), 0)
    return sat_pos


def sat_pointing_positions(offset_angle): 
    """
    This function returns the pointing position of the satellites in cartesian coordinates.
    All satellites points at an 'offset_angle(in degrres)' distance from the the centre (c_ra,c_dec) as in like in an equilateral triangle,
    with one satellite remaining to point at the centre(c_ra,c_dec).
    
    """
    c_ra, c_dec = 180, 0 #  The coordinates of the centre of pointing

    pointing_positions = [
        (c_ra, c_dec), #Centre
        (c_ra + offset_angle, c_dec + offset_angle), #Upper right arm
        (c_ra - offset_angle, c_dec + offset_angle), #Upper left arm
        (c_ra, c_dec - np.sqrt(2)*offset_angle)       #Lower center arm
    ]
    
    pointing = np.array([coord_transform.r2c(ra, dec) for ra, dec in pointing_positions])
    pointing = np.where(np.abs(pointing) < 1e-15, 0, pointing) # Replace values that are less than 1e-15 with 0 

    return np.array(pointing) # pointing is in cartesian coordinates


