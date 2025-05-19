import numpy as np


def get_satellite_positions(
    t_seconds, num_sats,angular_spacing_deg, altitude_km=510, period_minutes=96, group_rotation_deg=0):

    R_earth = 6371.0  # km
    R_orbit = R_earth + altitude_km
    T = period_minutes * 60
    omega = 2 * np.pi / T  # rad/s

    spacing_rad = np.radians(angular_spacing_deg)
    group_rotation_rad = np.radians(group_rotation_deg)

    phase_offsets = spacing_rad * np.arange(num_sats)
    thetas = group_rotation_rad + phase_offsets + omega * t_seconds

    x = R_orbit * np.cos(thetas)
    y = np.zeros(num_sats)
    z = R_orbit * np.sin(thetas)

    return np.vstack((x, y, z)).T

# Random time in one orbit
t_random = np.random.uniform(0, 96 * 60)


def spherical_offsets(ra0_deg, dec0_deg, n_sats, angular_offset_deg):
    ra0 = np.radians(ra0_deg)
    dec0 = np.radians(dec0_deg)
    r = np.radians(angular_offset_deg)

    # Central unit vector
    v0 = np.array([
        np.cos(dec0) * np.cos(ra0),
        np.cos(dec0) * np.sin(ra0),
        np.sin(dec0)
    ])

    temp = np.array([1, 0, 0]) if abs(v0[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(temp, v0)
    u /= np.linalg.norm(u)
    v = np.cross(v0, u)

    ras, decs = [], []
    for i in range(n_sats):
        phi = 2 * np.pi * i / n_sats
        offset_vector = (np.cos(r) * v0 + np.sin(r) * (np.cos(phi) * u + np.sin(phi) * v))
        x, y, z = offset_vector
        dec = np.arcsin(z)
        ra = np.arctan2(y, x) % (2 * np.pi)
        ras.append(np.degrees(ra))
        decs.append(np.degrees(dec))

    return list(zip(ras, decs))



def get_pointing_radec(sat_positions,num_sats, angular_offset_deg):
    # Get satellite positions
    
    sat_positions /= np.linalg.norm(sat_positions, axis=1)[:, None]  # Normalize

    # Fixed Sun direction (-Y axis)
    sun_vector = np.array([0, -1, 0])

    # Compute axis perpendicular to Sun vector in the XY plane (Z axis cross Sun)
    z_axis = np.array([0.0,0.0,1.0])
    pointing_ax = np.cross(sun_vector, z_axis)
    pointing_ax /= np.linalg.norm(pointing_ax)

    # Get base RA in radians
    ra0 = np.arctan2(pointing_ax[1], pointing_ax[0]) % (2 * np.pi)
    ra1 = (ra0 + np.pi) % (2 * np.pi)

    group_ra = [ra0, ra1]

    # Determine satellite group memberships
    dotProd = np.dot(sat_positions, pointing_ax)
    groupMembership = np.array([0 if val > 0 else 1 for val in dotProd])

    # Offsets for each group
    n_0 = np.sum(groupMembership == 0)
    n_1 = np.sum(groupMembership == 1)

    group_offsets = [[], []]
    if n_0 > 0:
        group_offsets[0] = spherical_offsets(np.degrees(group_ra[0]), 0, n_0, angular_offset_deg)
    if n_1 > 0:
        group_offsets[1] = spherical_offsets(np.degrees(group_ra[1]), 0, n_1, angular_offset_deg)

    # Sort satellites by height above XY plane
    height_idx_array = [[xyz[2], groupMembership[idx], idx] for idx, xyz in enumerate(sat_positions)]
    height_idx_array.sort(reverse=True)

    pointings = np.zeros((num_sats, 2))  # (RA, Dec) per satellite
    offset_indices = [0, 0]

    for _, group_id, sat_idx in height_idx_array:
        group_idx = offset_indices[group_id]
        ra_dec = group_offsets[group_id][group_idx]
        pointings[sat_idx] = ra_dec
        offset_indices[group_id] += 1

    return pointings, [(np.degrees(group_ra[0]), 0), (np.degrees(group_ra[1]), 0)]


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
    
