import numpy as np
from scipy.optimize import minimize
import os
from pathlib import Path

from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import ecliptic_frame
from skyfield.iokit import parse_tle_file



class HermesConstellation:
    def __init__(self, pointing_offset_deg, oldest_TLE_time_days = 7.0, fake_hermes = False, arg_perigees=None):
        self.satellites = []
        self.ts = load.timescale()
        self.time = self.ts.now() # initialise as current time
        self.oldest_TLE_time_days = oldest_TLE_time_days

        # Load the constellation from up to date TLE files
        # TODO Add the names of HERMES spacecraft once we have them
        names = ['SPIRIT', 'HERMES_1', 'HERMES_2', 'HERMES_3']
        Path("tle_data").mkdir(exist_ok=True)


        if fake_hermes:
            if arg_perigees is None:
                arg_perigees = [str(value + 1e-10)[:8] for value in [210, 220, 230]]
            #arg_perigees = [str(value + 1e-10)[:8] for value in [1, 60, 120, 180, 240, 300]]
            for hermes_idx in range(1,4):
                fname = f'tle_data/HERMES_{hermes_idx}'
                with open(fname, 'w') as file:
            # Use predefined argument of perigee values
                    arg_perigee = arg_perigees[hermes_idx - 1]
                    raan = '110.0000'  # Close to Spirit's plane but slightly different
            
                    line1 = f'1 5846{hermes_idx-1}U 23185G   25127.87562329  .00036500  00000+0  53223-3 0  9998\n'
                    line2 = f'2 5846{hermes_idx-1}  97.3898 {raan} 0008987  {arg_perigee} 157.6763 15.37081430 79686'
            
                    file.writelines(f'HERMES_{hermes_idx-1}\n')
                    file.writelines(line1)
                    file.writelines(line2)
        
        else:
            raise Exception("We don't have TLEs for HERMES yet, so make sure to set fake_hermes = True")

        # print('> Loading Satellite data from celestrak.org')
        for name in names:
            
            url = 'https://celestrak.org/NORAD/elements/gp.php?NAME=' + name + '&FORMAT=TLE'
            fname = f'tle_data/{name}'
            if not os.path.isfile(fname) or load.days_old(fname) >= self.oldest_TLE_time_days:
                sat = load.tle(url, reload=True, filename = fname)  

            with load.open(fname) as f:
                satellites = list(parse_tle_file(f, self.ts))

            for sat in satellites:
                self.satellites.append(sat)
                
        # print('> Loading planet data')
        self.planets = load('de421.bsp')
        self.earth = self.planets['Earth']
        self.sun = self.planets['Sun']

        self.pointing_offset_deg = pointing_offset_deg
        self.pointing_offset = np.radians(self.pointing_offset_deg)

        self.pointing_strategy = PointingStrategy(self.pointing_offset)

        # print('\n-- Done.\n')

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
    
