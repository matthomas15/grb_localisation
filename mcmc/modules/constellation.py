import numpy as np

# here raan_rad is the group rotation
# def rotation_matrix(raan_rad, incl_rad):
#     R_rot =  np.array([
#         [np.cos(raan_rad), -np.sin(raan_rad), 0],
#         [np.sin(raan_rad),  np.cos(raan_rad), 0],
#         [0,                0,                1]
#     ])
#     R_incl = np.array([
#         [1, 0, 0],
#         [0, np.cos(incl_rad), -np.sin(incl_rad)],
#         [0, np.sin(incl_rad),  np.cos(incl_rad)]
#     ])
#     return R_rot @ R_incl




def get_satellite_positions(
    t_seconds, num_sats,angular_spacing_deg, inclination_deg, group_rotation_deg, altitude_km=510, period_minutes=96):

    R_earth = 6371.0  # km
    R_orbit = R_earth + altitude_km
    T = period_minutes * 60
    omega = 2 * np.pi / T  # rad/s


    spacing_rad = np.radians(angular_spacing_deg)
    group_rotation_rad = np.radians(group_rotation_deg)
    inclination_rad = np.radians(inclination_deg)

    phase_offsets = spacing_rad * np.arange(num_sats)
    thetas = group_rotation_rad + phase_offsets + omega * t_seconds

    # orbit in X-Y plane, equitorial orbit
    x = R_orbit * np.cos(thetas)
    y = R_orbit * np.sin(thetas)
    z = np.zeros(num_sats)

    #rotate abot x- axis to set inclination of orbit
    y_inc = y* np.cos(inclination_rad)
    z_inc = y* np.sin(inclination_rad)
    
    return np.vstack((x, y_inc, z_inc)).T

    # # Stack into position vectors
    # base_positions = np.vstack((x, y, z)).T

    # # Apply 3D orbital plane rotation
    # R = rotation_matrix(group_rotation_rad, inclination_rad)
    # rotated_positions = base_positions @ R.T

    # return rotated_positions

def generate_full_constellation(t_seconds):
    sats_per_plane = 8
    angular_spacing = 360 / sats_per_plane

    planes = [
        {"incl": 0, "rot": 0},       # Equatorial
        {"incl": 100, "rot": 0},      # Polar
        {"incl": 50, "rot": 0},      # Diagonal 1 (+45)
        {"incl": 150, "rot": 0},     # Diagonal 2 (-45)
    ]

    all_positions = []
    for plane in planes:
        positions = get_satellite_positions(
            t_seconds=t_seconds,
            num_sats=sats_per_plane,
            angular_spacing_deg=angular_spacing,
            inclination_deg=plane["incl"],
            group_rotation_deg=plane["rot"]
        )
        all_positions.append(positions)

    return np.vstack(all_positions)

# def spherical_offsets(ra0_deg, dec0_deg, n_sats, angular_offset_deg):
#     ra0 = np.radians(ra0_deg)
#     dec0 = np.radians(dec0_deg)
#     r = np.radians(angular_offset_deg)

#     # Central unit vector on celestial equator
#     v0 = np.array([
#         np.cos(dec0) * np.cos(ra0),
#         np.cos(dec0) * np.sin(ra0),
#         np.sin(dec0)
#     ])

#     u = np.array([0,0,1]) # z- axis
#     v = np.cross(u, v0)
#     v /= np.linalg.norm(v)
#     u = np.cross(v0, v)  # ensure right-handed frame


#     ras, decs = [], []
#     ras.append(ra0_deg)
#     decs.append(dec0_deg)

#     for i in range(n_sats-1):
#         phi = 2 * np.pi * i / (n_sats-1)
#         offset_vector = (np.cos(r) * v0 + np.sin(r) * (np.cos(phi) * u + np.sin(phi) * v))
#         x, y, z = offset_vector
#         dec = np.arcsin(z)
#         ra = np.arctan2(y, x) % (2 * np.pi)
#         ras.append(np.degrees(ra))
#         decs.append(np.degrees(dec))

#     return list(zip(ras, decs))

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

        # Local tangent basis
        temp = np.array([1, 0, 0]) if abs(v0[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(temp, v0)
        u /= np.linalg.norm(u)
        v = np.cross(v0, u)

        ras, decs = [], []

        for i in range(n_sats):
            phi = 2 * np.pi * i / n_sats
            offset_vector = (np.cos(r) * v0 +
                         np.sin(r) * (np.cos(phi) * u + np.sin(phi) * v))

            x, y, z = offset_vector
            dec = np.arcsin(z)
            ra = np.arctan2(y, x)
            ras.append(np.degrees(ra) % 360)
            decs.append(np.degrees(dec))

        return list(zip(ras, decs))



def get_pointing_radec(sat_positions, num_sats, angular_offset_deg):
    """
    Assigns RA/Dec pointing directions to satellites based on their X position.
    Satellites with positive X point around RA = 0°, negative X around RA = 180°.
    
    Parameters:
        sat_positions: np.ndarray of shape (N, 3), satellite ECI positions
        num_sats: int, total number of satellites
        angular_offset_deg: float, max angular offset in degrees
    
    Returns:
        pointings: np.ndarray of shape (N, 2), each row is (RA, Dec) in degrees
        group_radecs: list of central RA/Dec tuples for each group
    """

    sat_positions = np.array(sat_positions)

    # Group assignments based on X coordinate
    groupMembership = np.array([0 if sat[0] > 0 else 1 for sat in sat_positions])

    # Define central RA (in degrees) for each group
    group_ra_deg = [0, 180]

    # Count satellites in each group
    n_0 = np.sum(groupMembership == 0)
    n_1 = np.sum(groupMembership == 1)

    # Generate spherical offsets around each group's central RA
    group_offsets = [[], []]
    if n_0 > 0:
        group_offsets[0] = spherical_offsets(group_ra_deg[0], 0, n_0, angular_offset_deg) # n_0-1 if centre included
    if n_1 > 0:
        group_offsets[1] = spherical_offsets(group_ra_deg[1], 0, n_1, angular_offset_deg) # n_1-1 if centre is included

    # Sort satellites by height (Z), within groups, for deterministic assignment
    # height_idx_array = [[sat[2], groupMembership[idx], idx] for idx, sat in enumerate(sat_positions)]
    # height_idx_array.sort(reverse=True)
    height_idx_array = [[0, groupMembership[idx], idx] for idx in range(num_sats)]

    pointings = np.zeros((num_sats, 2))  # (RA, Dec) in degrees
    offset_indices = [0, 0]

    # central_assigned = [False, False]

    # for _, group_id, sat_idx in height_idx_array:
    #     if not central_assigned[group_id]:
    #         # First satellite in group: assign center
    #         pointings[sat_idx] = (group_ra_deg[group_id], 0)
    #         central_assigned[group_id] = True
    #     else:
    #         # Remaining satellites: assign offset directions
    #         ra_dec = group_offsets[group_id][offset_indices[group_id]]
    #         pointings[sat_idx] = ra_dec
    #         offset_indices[group_id] += 1

    for _, group_id, sat_idx in height_idx_array:
        group_idx = offset_indices[group_id]
        ra_dec = group_offsets[group_id][group_idx]
        pointings[sat_idx] = ra_dec
        offset_indices[group_id] += 1

    return pointings, [(group_ra_deg[0], 0), (group_ra_deg[1], 0)]



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
    



    ### Copy of the
