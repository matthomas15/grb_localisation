from skyfield.api import EarthSatellite, load, wgs84
import numpy as np
from scipy.optimize import minimize
from skyfield.framelib import ecliptic_frame




class SatelliteConstellation:
    def __init__(self, pointing_offset_deg):
        self.satellites = []
        self.ts = load.timescale()
        self.time = self.ts.now() # initialise as current time

        self.planets = load('de421.bsp')
        self.earth = self.planets['Earth']
        self.sun = self.planets['Sun']

        self.pointing_offset_deg = pointing_offset_deg
        self.pointing_offset = np.radians(self.pointing_offset_deg)

        self.pointing_strategy = PointingStrategy(self.pointing_offset)

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
                self.time = self.time + np.random.uniform(0, 13.99)
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

        group_offsets = [self.pointing_strategy.get_offsets( sum( groupMembership == 0 ) ),
                         self.pointing_strategy.get_offsets( sum( groupMembership == 1 ) )
                        ]

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


        return pointings, [ [group_ra[0], 0], [group_ra[1], 0] ]
    

class PointingStrategy:
    def __init__(self, angular_offset):

        self.angular_offset = angular_offset
        # Solve for offset magnitude for triangle strategy
        #   find offset where an sep between top and bottom
        #   right vertex equals self.angular_offset
        self.triangle_offset_magnitude = 0
        if self.angular_offset != 0:
            i = np.arange(0, 2 * self.angular_offset, 0.0001)
            top_ra, top_dec = np.array([0, self.angular_offset/2])
            right_top_sep = np.abs(self.ang_sep(top_ra, top_dec, i, -i ) - self.angular_offset)
            self.triangle_offset_magnitude = i[np.argmin(right_top_sep)]

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

    def get_offsets(self, group_membership):
        '''
        Pointing offsets for some central pointing based on
        the number of satellites in a pointing group

        Offsets are arranged from high dec to low dec, left
        to right.
        '''
        if group_membership == 1:
            return [np.array([0,0])]

        if group_membership == 2:
            return np.array( [ [-self.angular_offset/2, 0], 
                                [self.angular_offset/2, 0]
                              ])
        
        if group_membership == 3:
            return np.array([[0, self.triangle_offset_magnitude], 
                             [-self.triangle_offset_magnitude, -self.triangle_offset_magnitude],
                             [self.triangle_offset_magnitude, -self.triangle_offset_magnitude]
                            ])
        
        if group_membership == 4:
            return np.array( [ [0, self.angular_offset/2], 
                                [-self.angular_offset/2, 0],
                                [self.angular_offset/2, 0],
                                [0, -self.angular_offset/2], 
                              ])
        
        if group_membership == 5:
            return np.array( [ [-self.angular_offset/2, self.angular_offset/2],
                                [self.angular_offset/2, self.angular_offset/2], 
                                [0, 0],
                                [-self.angular_offset/2, -self.angular_offset/2], 
                                [self.angular_offset/2, -self.angular_offset/2],
                              ])

        raise Exception(f"No pointing strategy implemented for a group of {group_membership} satellites.")