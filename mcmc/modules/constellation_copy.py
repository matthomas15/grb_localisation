import numpy as np
from scipy.optimize import minimize
import os
from pathlib import Path

from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import ecliptic_frame
from skyfield.iokit import parse_tle_file



class HermesConstellation:
    def __init__(self, pointing_offset_deg, num_sat, oldest_TLE_time_days = 7.0):
        self.satellites = []
        self.ts = load.timescale()
        self.time = self.ts.now() # initialise as current time
        self.oldest_TLE_time_days = oldest_TLE_time_days

        Path("tle_data").mkdir(exist_ok=True)
        self.generate_fake_tles()
        names = [f"HERMES_{i}" for i in range(num_sat)]

        for name in names:
            fname = f'tle_data/{name}'
            with load.open(fname) as f:
                satellites = list(parse_tle_file(f, self.ts))
            for sat in satellites:
                self.satellites.append(sat)
        
        self.planets = load('de421.bsp')
        self.earth = self.planets['Earth']
        self.sun = self.planets['Sun']

        self.pointing_offset_deg = pointing_offset_deg
        self.pointing_offset = np.radians(self.pointing_offset_deg)

        self.pointing_strategy = PointingStrategy(self.pointing_offset)

    def generate_fake_tles(self, base_dir='tle_data'):#,tle_epoch =25144.84575769):
        os.makedirs(base_dir, exist_ok=True)
        ts = load.timescale()
        now = ts.now()
        year, day_of_year, fraction = now.utc_datetime().timetuple().tm_year, now.utc_datetime().timetuple().tm_yday, now.ut1 % 1
        tle_epoch = f"{year % 100:02d}{day_of_year:03d}{fraction:.8f}"  # YYDDD.DDDDDDDD
        # now = self.ts.now()
        # jd_epoch = now.tt
        # epoch_days = jd_epoch - 2451545.0  # Convert to TLE epoch format (days since J2000)
        # tle_epoch = f"{int(epoch_days):05d}.{(epoch_days % 1):08.8f}"[:14]


        
        inclinations = [0.0]#, 90.0, 45.0, 135.0]  # equatorial, polar, diagonal1, diagonal2
        # num_sats_per_group = 8
        #mean_anomalies = [i * 45.0 for i in range(num_sats_per_group)]
        mean_anomalies = [0.1, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 305.0]

        sat_idx = 0
        # for group_idx, inc in enumerate(inclinations):
        #     for i, mean_anom in enumerate(mean_anomalies):
        for inc in inclinations:
            for mean_anom in mean_anomalies:
                sat_number = 58460 + sat_idx
                sat_name = f'HERMES_{sat_idx}'
                fname = os.path.join(base_dir, sat_name)

                line1 = f'1 {sat_number}U 23185G   {tle_epoch}  .00036500  00000+0  53223-3 0  999{sat_idx % 10}\n'
                line2 = f'2 {sat_number}  {inc:8.4f} 000.0000 0000000 000.0000 {mean_anom:8.4f} 15.37081430 79686\n'

                with open(fname, 'w') as file:
                    file.writelines(f'{sat_name}\n')
                    file.writelines(line1)
                    file.writelines(line2)

                sat_idx += 1

    def __getitem__(self, idx):
        return self.satellites[idx]

    def get_current_time(self):
        return self.ts.now()
    
    def set_time(self, time = None, random = False):
        '''
        Sets the internal time variable

        Raises an exception if you try to set a time
        which is too far away from the epoch of your
        constellation's TLEs
        '''

        if time is None:
            if random:
                self.time = self.ts.now() + np.random.uniform(0, 13.99)
                return
            else:        
                raise Exception("You must enter a valid datetime object for the constellation or set random = True.")
        self.time = time

        distance_from_epochs = np.array([ np.abs(sat.epoch - self.time) for sat in self.satellites ])
        
        if max(distance_from_epochs) > 14:

            print('     Warning: You are setting a time outside the valid range\n     of your constellation\'s TLEs!')

            print('\n     You should update the TLEs for:')

            for diff, sat in zip(distance_from_epochs, self.satellites):
                if diff > 14:
                    print(f'       - {sat.name}')
    
    def set_random_time(self):
        '''
        Generates a random time for the GRB to occur

        Chooses a time within the next week

        Raises an exception if you try to calcul
        '''

        maxTime = 13.5
        self.time = self.time + np.random.uniform(0, maxTime)

    def add_satellite_from_tle(self, name, line1, line2):
        self.satellites.append( EarthSatellite(line1, line2, name, self.ts) )
    
    def get_xyz(self):
        '''
        Return cartesian position of all spacecraft in km
        '''
        cart = []

        for sat in self.satellites:
            cart.append(sat.at(self.time).frame_xyz(ecliptic_frame).km)

        return np.array(cart)
    
    def get_latlon(self):
        lat_lon_list = []
        for sat in self.satellites:
            geocentric = sat.at(self.time)
            lat, lon = wgs84.latlon_of(geocentric)#
            lat_lon_list.append((lat.degrees, lon.degrees))
        return lat_lon_list
   
    
    def get_pointing_radec(self):
        '''
        Return cartesian pointing vectors for each spacecraft

        Method: 
            Compute satellite ra relative to the Sun, and point at 90
            degree offset

        Compute pointings by taking dot product of pointing group with sat vector
        and only taking cases where Earth isn't in the way

        THEN compute distance from each sat to the Sun to order the pointings in RA
        (sats closer to the Sun should be pointing closer to the Sun than others
        in their group to avoid Earth pointing as much as possible)
        '''

        num_satellites = len(self.satellites)


        # Satellites point at right angles to the Sun, at declination 0

        vect = self.earth.at(self.time).observe(self.sun).frame_xyz(ecliptic_frame).km
        sun_vector = vect / np.linalg.norm(vect)

        z = np.array([0, 0, 1])
        pointing_ax = np.cross(sun_vector, z) # Perpendicular to Sun, at dec 0

        ra0 = np.arctan2( pointing_ax[1],  pointing_ax[0] )
        if ra0 < 0:
            ra0 = (2 * np.pi) - ra0
        ra0 = ra0 % (2 * np.pi)

        ra1 = (ra0 + np.pi) % (2 * np.pi)

        group_ra = [ra0, ra1]

        # Compute pointing groups by taking dot product between
        # satellite position and pointing direction

        sat_positions = self.get_xyz()
        sat_positions = sat_positions / np.linalg.norm(sat_positions)
        
        dotProd = np.dot( sat_positions, pointing_ax )
        groupMembership = np.array([ 0 if el > 0 else 1 for el in dotProd ]).astype('int')

        #Addition by Haritha
        n_0 = np.sum(groupMembership == 0)
        n_1 = np.sum(groupMembership == 1)

        group_offsets = [None, None]
        if n_0 > 0:
            group_offsets[0] = self.pointing_strategy.spherical_offsets(group_ra[0], 0, n_0) # Changed here
        if n_1 > 0:
            group_offsets[1] = self.pointing_strategy.spherical_offsets(group_ra[1], 0, n_1) # Changed here        

        # group_offsets = [self.pointing_strategy.get_offsets( sum( groupMembership == 0 ) ),
        #                  self.pointing_strategy.get_offsets( sum( groupMembership == 1 ) )
        #                 ]

        # Sort satellites by their height above the xy plane, and
        # keep track of where they belong in the pointing array

        height_idx_array = []

        for idx, xyz in enumerate(sat_positions):
            height_idx_array.append([xyz[2], groupMembership[idx], idx])

        height_idx_array = sorted(height_idx_array)[::-1]

        pointings = np.zeros([num_satellites, 2]) # xyz pointing for each satellite

        offset_indices = [0, 0]



        for i in range(num_satellites):

            sat_order_idx = height_idx_array[i][2]
            group_id = height_idx_array[i][1]
            group_idx = offset_indices[group_id]

            ra = group_ra[ group_id ]
            offset = group_offsets[ group_id ][ group_idx ]
            
            new_ra = ra + offset[0]
            if new_ra < 0:
                new_ra = 2 * np.pi - new_ra
            elif new_ra > (2 * np.pi):
                new_ra = new_ra % (2 * np.pi)

            new_dec = offset[1]

            pointings[ sat_order_idx ] = [ new_ra, new_dec ]

            
            offset_indices[group_id] = offset_indices[group_id] + 1

        # print(f"Group Membership: {groupMembership}")
        # print(f"Group Offsets: {group_offsets}")

        return pointings, [ [group_ra[0], 0], [group_ra[1], 0] ]
    

class PointingStrategy:
    def __init__(self, angular_offset):
        self.angular_offset = angular_offset


    @staticmethod
    def add_coordinates(radec1, radec2):

        res = radec1 + radec2

        if res[0] < 0:
            res[0] = res[0] + (2 * np.pi)
        if res[0] > 0:
            res[0] = res[0] % (2 * np.pi)

        return res

    @staticmethod
    def ang_sep(ra1, dec1, ra2, dec2): #Return angular separation of two points on the sphere in radians
        return np.arccos( np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2) )

    def spherical_offsets(self,ra0_deg, dec0_deg, n_sats):
        ra0 = np.radians(ra0_deg)
        dec0 = np.radians(dec0_deg)
        r = np.radians(self.angular_offset)

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
    


# def get_satellite_positions(
#     t_seconds, num_sats,angular_spacing_deg, inclination_deg, group_rotation_deg, altitude_km=510, period_minutes=96):

#     R_earth = 6371.0  # km
#     R_orbit = R_earth + altitude_km
#     T = period_minutes * 60
#     omega = 2 * np.pi / T  # rad/s


#     spacing_rad = np.radians(angular_spacing_deg)
#     group_rotation_rad = np.radians(group_rotation_deg)
#     inclination_rad = np.radians(inclination_deg)

#     phase_offsets = spacing_rad * np.arange(num_sats)
#     thetas = group_rotation_rad + phase_offsets + omega * t_seconds

#     # orbit in X-Y plane, equitorial orbit
#     x = R_orbit * np.cos(thetas)
#     y = R_orbit * np.sin(thetas)
#     z = np.zeros(num_sats)

#     #rotate abot x- axis to set inclination of orbit
#     y_inc = y* np.cos(inclination_rad)
#     z_inc = y* np.sin(inclination_rad)
    
#     return np.vstack((x, y_inc, z_inc)).T

# def generate_full_constellation(t_seconds):
#     sats_per_plane = 8
#     angular_spacing = 360 / sats_per_plane

#     planes = [
#         {"incl": 0, "rot": 0},       # Equatorial
#         {"incl": 100, "rot": 0},      # Polar
#         {"incl": 50, "rot": 0},      # Diagonal 1 (+45)
#         {"incl": 150, "rot": 0},     # Diagonal 2 (-45)
#     ]

#     all_positions = []
#     for plane in planes:
#         positions = get_satellite_positions(
#             t_seconds=t_seconds,
#             num_sats=sats_per_plane,
#             angular_spacing_deg=angular_spacing,
#             inclination_deg=plane["incl"],
#             group_rotation_deg=plane["rot"]
#         )
#         all_positions.append(positions)

#     return np.vstack(all_positions)

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

# def spherical_offsets(ra0_deg, dec0_deg, n_sats, angular_offset_deg):
#         ra0 = np.radians(ra0_deg)
#         dec0 = np.radians(dec0_deg)
#         r = np.radians(angular_offset_deg)

#         # Central unit vector
#         v0 = np.array([
#         np.cos(dec0) * np.cos(ra0),
#         np.cos(dec0) * np.sin(ra0),
#         np.sin(dec0)
#         ])

#         # Local tangent basis
#         temp = np.array([1, 0, 0]) if abs(v0[0]) < 0.9 else np.array([0, 1, 0])
#         u = np.cross(temp, v0)
#         u /= np.linalg.norm(u)
#         v = np.cross(v0, u)

#         ras, decs = [], []

#         for i in range(n_sats):
#             phi = 2 * np.pi * i / n_sats
#             offset_vector = (np.cos(r) * v0 +
#                          np.sin(r) * (np.cos(phi) * u + np.sin(phi) * v))

#             x, y, z = offset_vector
#             dec = np.arcsin(z)
#             ra = np.arctan2(y, x)
#             ras.append(np.degrees(ra) % 360)
#             decs.append(np.degrees(dec))

#         return list(zip(ras, decs))

# def get_pointing_radec(sat_positions, num_sats, angular_offset_deg):
#     # Get satellite positions
    
#     sat_positions /= np.linalg.norm(sat_positions, axis=1)[:, None]  # Normalize

#     # Fixed Sun direction (-Y axis)
#     sun_vector = np.array([0, -1, 0])

#     # Compute axis perpendicular to Sun vector in the XY plane (Z axis cross Sun)
#     z_axis = np.array([0.0,0.0,1.0])
#     pointing_ax = np.cross(sun_vector, z_axis)
#     pointing_ax /= np.linalg.norm(pointing_ax)

#     # Get base RA in radians
#     ra0 = np.arctan2(pointing_ax[1], pointing_ax[0]) % (2 * np.pi)
#     ra1 = (ra0 + np.pi) % (2 * np.pi)

#     group_ra = [ra0, ra1]

#     # Determine satellite group memberships
#     dotProd = np.dot(sat_positions, pointing_ax)
#     groupMembership = np.array([0 if val > 0 else 1 for val in dotProd])

#     # Offsets for each group
#     n_0 = np.sum(groupMembership == 0)
#     n_1 = np.sum(groupMembership == 1)

#     group_offsets = [[], []]
#     if n_0 > 0:
#         group_offsets[0] = spherical_offsets(np.degrees(group_ra[0]), 0, n_0, angular_offset_deg)
#     if n_1 > 0:
#         group_offsets[1] = spherical_offsets(np.degrees(group_ra[1]), 0, n_1, angular_offset_deg)

#     # Sort satellites by height above XY plane
#     height_idx_array = [[xyz[2], groupMembership[idx], idx] for idx, xyz in enumerate(sat_positions)]
#     height_idx_array.sort(reverse=True)

#     pointings = np.zeros((num_sats, 2))  # (RA, Dec) per satellite
#     offset_indices = [0, 0]

#     for _, group_id, sat_idx in height_idx_array:
#         group_idx = offset_indices[group_id]
#         ra_dec = group_offsets[group_id][group_idx]
#         pointings[sat_idx] = ra_dec
#         offset_indices[group_id] += 1

#     return pointings, [(np.degrees(group_ra[0]), 0), (np.degrees(group_ra[1]), 0)]


# def eci_to_latlon(eci_positions, t_seconds):
#     """
#     Convert ECI positions (in km) to latitude and longitude (in degrees),
#     accounting for Earth's rotation at time t_seconds.
#     """
#     earth_rot_rate = 2 * np.pi / 86164  # rad/s, sidereal rotation
#     theta = earth_rot_rate * t_seconds

#     lat_lon_list = []

#     for pos in eci_positions:
#         x_eci, y_eci, z_eci = pos

#         # Rotate ECI -> ECEF
#         x_ecef = np.cos(theta) * x_eci + np.sin(theta) * y_eci
#         y_ecef = -np.sin(theta) * x_eci + np.cos(theta) * y_eci
#         z_ecef = z_eci

#         # Compute lat/lon
#         lon = np.arctan2(y_ecef, x_ecef)
#         r_xy = np.sqrt(x_ecef**2 + y_ecef**2)
#         lat = np.arctan2(z_ecef, r_xy)

#         lat_deg = np.degrees(lat)
#         lon_deg = (np.degrees(lon) + 360) % 360  # wrap to [0, 360)
#         if lon_deg > 180:
#             lon_deg -= 360  # optional: wrap to [-180, 180]

#         lat_lon_list.append((lat_deg, lon_deg))

#     return lat_lon_list
    
