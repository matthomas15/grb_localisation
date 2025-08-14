import numpy as np
import pandas as pd
from modules import coord_transform

""" 
Creating Isotropic RA,DEC .

First we generate a cube with  edge lenth (-1,1) with uniform points inside it
Calculate each points distance from origin, then check if it lies insode a unit sphere
Now normalize the poin
"""

def uniform_random_points_on_sphere(num_points,rng):
    points = []

    while len(points) < num_points:
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        z = rng.uniform(-1, 1)
        w = np.sqrt(x**2 + y**2 + z**2)             # Calculate the norm (distance from origin)
        
        if w <= 1:      # Check if the point is inside the unit sphere (w <= 1)    
            x_normalized = x / w
            y_normalized = y / w
            z_normalized = z / w        # Normalize the point to lie on the surface of the unit sphere
            
            points.append([x_normalized, y_normalized, z_normalized])
    
    return np.array(points[0])

# For creating trigger time better to use time in seconds in a year so that its easir to calculate the satellite positions considering the time period.

# """ creating random trigger time for GRBs
# """
# def trigger_time(total_events): 
#     random_times = np.random.uniform(0, 365.25, total_events) 

#     # Convert random times to full timestamps within the year
#     start = pd.Timestamp('2024-01-01')
#     event_timestamps = start + pd.to_timedelta(random_times, unit='D')

#     # Format the timestamps to "YYYY-MM-DD HH:MM:SS.sss" 
#     formatted_events = event_timestamps.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3] 
#     return formatted_events


def generate_grb(catalog,rng):
    cartesian_points = uniform_random_points_on_sphere(1,rng)
    radec_points = coord_transform.c2r(cartesian_points[0],cartesian_points[1],cartesian_points[2])
    ra = radec_points[0]
    dec = radec_points[1]
    t90 = rng.choice(catalog['t90     '])
    flux = rng.choice(catalog['flnc_band_phtfluxb'])
    #print("RA:", ra, "Dec:", dec, "T90:", t90, "Flux:", flux)
    return cartesian_points, ra, dec, t90, flux



