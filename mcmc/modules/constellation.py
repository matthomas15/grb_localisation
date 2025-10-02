import numpy as np

R_earth = 6371.0 #km
altitude_leo = 550 #km


def orbital_period(R_earth, altitude_leo):
    """
    Computes the orbital period of a satellite in a circular Earth orbit.
    
    Parameters:
    -----------
    altitude_km : float
        Altitude of the satellite above Earth's surface in kilometers.

    Returns:
    --------
    period_sec : float
        Orbital period in seconds.
    
    """
    G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
    M = 5.972e24     # mass of Earth (kg)

    r = (R_earth + altitude_leo)* 1e3  # total orbital radius in meters
    T = 2 * np.pi * np.sqrt(r**3 / (G * M))  # in seconds
    return T # seconds

orbit_period = orbital_period(R_earth, altitude_leo)

def get_satellite_positions(t_seconds, num_sats, angular_spacing_deg, inclination_deg, group_rotation_deg):
    """
    Generates posiitons of satellites in a
    """

    R_orbit = R_earth + altitude_leo
    T = orbit_period
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


def generate_full_constellation(sats_per_plane, t_seconds, planes):
    angular_spacing = 360 / sats_per_plane

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

def zenith_pointing(sat_positions):
    """
    Compute pointing directions (RA, Dec) toward zenith from each satellite position.
    
    Parameters:
        sat_positions: np.ndarray of shape (N, 3), satellite positions in ECI frame.
    
    Returns:
        pointings: np.ndarray of shape (N, 3), where each row is (X,Y,Z).
    """
    directions = sat_positions / np.linalg.norm(sat_positions, axis=1)[:, None]  # Normalize
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    return np.column_stack((x,y,z))

    # ra = np.degrees(np.arctan2(y, x)) % 360
    # dec = np.degrees(np.arcsin(z))
    # return np.column_stack((ra, dec))
    

def spherical_offsets(ra0_deg, dec0_deg, n_sats, offset_deg):
        ra0 = np.radians(ra0_deg)
        dec0 = np.radians(dec0_deg)
        r = np.radians(offset_deg)

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



def get_pointing_radec(sat_positions, num_sats, offset_deg):
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
        group_offsets[0] = spherical_offsets(group_ra_deg[0], 0, n_0-1, offset_deg) # n_0-1 if centre included
    if n_1 > 0:
        group_offsets[1] = spherical_offsets(group_ra_deg[1], 0, n_1-1, offset_deg) # n_1-1 if centre is included

    # Sort satellites by height (Z), within groups, for deterministic assignment
    # height_idx_array = [[sat[2], groupMembership[idx], idx] for idx, sat in enumerate(sat_positions)]
    # height_idx_array.sort(reverse=True)
    height_idx_array = [[0, groupMembership[idx], idx] for idx in range(num_sats)]

    pointings = np.zeros((num_sats, 2))  # (RA, Dec) in degrees
    offset_indices = [0, 0]

    central_assigned = [False, False]

    for _, group_id, sat_idx in height_idx_array:
        if not central_assigned[group_id]:
            # First satellite in group: assign center
            pointings[sat_idx] = (group_ra_deg[group_id], 0)
            central_assigned[group_id] = True
        else:
            # Remaining satellites: assign offset directions
            ra_dec = group_offsets[group_id][offset_indices[group_id]]
            pointings[sat_idx] = ra_dec
            offset_indices[group_id] += 1

    # for _, group_id, sat_idx in height_idx_array:
    #     group_idx = offset_indices[group_id]
    #     ra_dec = group_offsets[group_id][group_idx]
    #     pointings[sat_idx] = ra_dec
    #     offset_indices[group_id] += 1

    return pointings, [(group_ra_deg[0], 0), (group_ra_deg[1], 0)]
