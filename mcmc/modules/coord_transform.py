import numpy as np

def r2c(ra, dec):         # ra,dec inputs are in degree
    """
    first we convert RA, DEC that is in degrees to radians
     """
    ra = np.radians(ra)
    dec = np.radians(dec)      

    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    # return np.vstack((x, y, z)).T
    return np.array((x,y,z)).T
    



def c2r(x, y, z):        
    """
    convert a single cartesian coordinates(x, y, z) into (RA,DEC)
    x,y,z needs to be normalised

    """

    w = np.sqrt(x**2 + y**2+ z**2)
    x = x/w
    y = y/w
    z = z/w

    ra = np.degrees(np.arctan2(y,x))
    if ra < 0:
            ra += 360
    dec = np.degrees(np.arcsin(z/w))
    #dec = np.degrees(np.arcsin(z))
    return np.array((ra,dec)) 


def eci_to_latlon(eci_positions, t_seconds):
    """
    Convert ECI positions (in km) to latitude and longitude (in degrees),
    accounting for Earth's rotation at time t_seconds.
    """
    earth_rot_rate = 2 * np.pi / 86164  # rad/s, sidereal rotation
    theta = earth_rot_rate * t_seconds

    lat_lon_list = []

    for pos in eci_positions:
        x_eci, y_eci, z_eci = pos

        # Rotate ECI -> ECEF
        x_ecef = np.cos(theta) * x_eci + np.sin(theta) * y_eci
        y_ecef = -np.sin(theta) * x_eci + np.cos(theta) * y_eci
        z_ecef = z_eci

        # Compute lat/lon
        lon = np.arctan2(y_ecef, x_ecef)
        r_xy = np.sqrt(x_ecef**2 + y_ecef**2)
        lat = np.arctan2(z_ecef, r_xy)

        lat_deg = np.degrees(lat)
        lon_deg = (np.degrees(lon) + 360) % 360  # wrap to [0, 360)
        if lon_deg > 180:
            lon_deg -= 360  # optional: wrap to [-180, 180]

        lat_lon_list.append((lat_deg, lon_deg))

    return lat_lon_list
