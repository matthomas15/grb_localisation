import numpy as np
from modules import simulation
from modules import coord_transform
from scipy.special import gammaln


combined_bkgd = np.load('C:\\Users\\Haritha\\repo\\grb_localisation\\mcmc\\particle_background.npy')
def flux_limit_long(Area):
    Flux = 0.116*np.sqrt(Area/125) # 0.116 is flux limit of GBM detectoer of area 125 cm2 
    return Flux

def flux_limit_short(Area):
    Flux = 0.465*np.sqrt(Area/125) # 0.465 is flux limit of GBM detectoer of area 125 cm2 
    return Flux

R_earth =6371 # km
altitude_leo =550 # km

def occult_angle(R_earth,altitude_leo):
    """
    Its the half angle of the cone view thats blocked by earth as detector's position vector in Earth's LEO to earth's centre and radius of earth is considered. 
    The occult angle limit of nearly 67 is calculated considering the orbit height and the relative size of earth at that height.
    Any GRB that is within this angle is obstructed by earth.

    """
    occult_angle = np.rad2deg(np.arcsin(R_earth/(R_earth + altitude_leo)))
    return occult_angle

occ_angle= occult_angle(R_earth,altitude_leo)

def time_delay(r1, r2, d_grb):
    """
    Returns the delay in arrival time of grb signals between two satellites

    Input: 
        r1,r2 are position vectors of the two satellites
        d_grb is the position of the grb in cartesian coordinate

    Output: time delay in seconds    

    """
    c = 3e5 #  speed of light in km/s

    r_diff = r1 - r2
    t_delay = np.dot(r_diff, d_grb)/c
    return t_delay

def cos_angle_btw_vec(v1, v2):
    """
    Vectorized cosine of angle between a single vector v1 (shape: (3,))
    and multiple vectors v2 (shape: (N, 3)).
    
    Returns:
        cos_theta: array of shape (N,)
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    dot_prod = np.dot(v2, v1)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2, axis=1)

    cos_theta = dot_prod / (norm_v1 * norm_v2)
    return cos_theta



# def cos_angle_btw_vec(v1, v2):
#     """ 
#     Parameters: 
#     vectors [x1, y1, z1], [x2, y2, z2]    
    
#     Returns:
#        Cosine Angle between these vectors in radian

#     """
#     cos_theta = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) #  in radians
#     #theta = np.degrees(np.arccos(cos_theta)) 
#     return cos_theta

def angle_btw_vec(v1, v2):
    """
    Vectorized angle (in degrees) between a single vector v1 (shape: (3,))
    and multiple vectors v2 (shape: (N, 3)).
    
    Returns:
        angles: array of shape (N,)
    """
    cos_theta = cos_angle_btw_vec(v1, v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevents NaNs due to rounding errors
    theta = np.degrees(np.arccos(cos_theta))
    return theta
# def angle_btw_vec(v1, v2):
#     """ 
#     Parameters: 
#     vectors [x1, y1, z1], [x2, y2, z2]    
    
#     Returns:
#         Angle between these vectors in degree
#     """
#     cos_theta = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) #  in radians
#     theta = np.degrees(np.arccos(cos_theta)) 
#     return theta



# def get_particle_background_value(lat, lon):
#     """
#     Returns the particle background value at a given (lat, lon) index.
    
#     """

#     if np.isnan(lat) or np.isnan(lon):
#         raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")
#     lat = int( np.round( lat ) )
#     lon = int( np.round( lon ) )

#     bkgd = combined_bkgd[ lat + 90 - 1 ][ lon + 180 - 1 ]

#     return bkgd

# def get_nonzero_background_indices(positions):
#     """
#     Returns the list of indices where positions have nonzero particle background.
    
#     """

#     nonzero_indices = []
#     for i, (lat, lon) in enumerate(positions):
#         bkgd_value = get_particle_background_value(lat, lon)
#         if bkgd_value > 0:
#             nonzero_indices.append(i)  # Store the index
#     return nonzero_indices

def get_nonzero_background_indices(lat_lons):
    """
    Returns the indices of positions where particle background is non-zero.
    
    Parameters:
        lat_lons (np.ndarray): N x 2 array of [lat, lon] pairs
        combined_bkgd (np.ndarray): 2D array with shape (180, 360)
    
    Returns:
        np.ndarray: Array of indices where the background value > 0
    """
    lat_lons = np.array(lat_lons)

    # Round and convert lat/lon to integer indices
    indices = np.round(lat_lons).astype(int) + [90, 180]  # Shape: (N, 2)
    
    # Check for valid bounds (lat: 0-179, lon: 0-359)
    valid_mask = (
        (indices[:, 0] >= 0) & (indices[:, 0] < combined_bkgd.shape[0]) &
        (indices[:, 1] >= 0) & (indices[:, 1] < combined_bkgd.shape[1])
    )

    valid_indices = indices[valid_mask]
    original_indices = np.arange(len(lat_lons))[valid_mask]

    # Extract background values using advanced indexing
    bkgd_values = combined_bkgd[valid_indices[:, 0], valid_indices[:, 1]]

    # Find which are non-zero
    nonzero_mask = bkgd_values > 0

    return original_indices[nonzero_mask].tolist()

# def simulate_satellite_det(grb_vec, flux_avg, sat_pos, sat_pointing, flux_limit, lat_lon):

#     """
#     This function returns the arrival times on each satellite;t_obs and the flux observed by each satllite;f_obs.
#     The observed flux is set to zero if the satellite is in earth occulted region
#     or if the flux is below the instrument flux limit (for long and short grb this limit is different) 

#     """
#     sat2earth_vec = [-sat for sat in sat_pos]
#     occult = [angle_btw_vec(grb_vec, se_vec) for se_vec in sat2earth_vec]
#     costheta = np.array([cos_angle_btw_vec(grb_vec,sp) for sp in sat_pointing])
#     bkgd_index = get_nonzero_background_indices(lat_lon)
#     f_obs = flux_avg * costheta
#     t_obs = time_delay( np.array([0, 0, 0]),sat_pos, grb_vec) 
#     for i in range(len(f_obs)):
#         if  occult[i] < occ_angle or f_obs[i] < flux_limit or i in bkgd_index:#  
#             f_obs[i] = 0
#     return f_obs, t_obs

def simulate_satellite_det(grb_vec, flux_avg, sat_pos, sat_pointing, flux_limit, lat_lon):
    """
    Simulates satellite detections of a GRB.

    Parameters:
        grb_vec (3,): GRB direction vector
        flux_avg (float): Average flux
        sat_pos (N, 3): Satellite positions
        sat_pointing (N, 3): Satellite pointing directions
        flux_limit (float): Flux detection threshold
        lat_lon (N, 2): Lat/lon positions of satellites

    Returns:
        f_obs (N,): Observed flux per satellite
        t_obs (N,): Observed time delays per satellite
    """
    sat_pos = np.asarray(sat_pos)
    sat_pointing = np.asarray(sat_pointing)

    sat2earth_vec = -sat_pos

    occult = angle_btw_vec(grb_vec, sat2earth_vec)
    costheta = cos_angle_btw_vec(grb_vec, sat_pointing)

    bkgd_index = get_nonzero_background_indices(lat_lon)

    f_obs = flux_avg * costheta

    t_obs = time_delay(np.array([0, 0, 0]), sat_pos, grb_vec)

    mask = (occult >= occ_angle) & (f_obs >= flux_limit)
    mask[list(bkgd_index)] = False  # Set f_obs = 0 for background-blocked indices , mask is like setting 1 if true and 0 if false

    f_obs = f_obs * mask  # Zero out undetected values

    return f_obs, t_obs




def sigma_cc(t90, f_obs, Area):
    
    """
    Gives the sigma value of signal cross-correlation  from flux and sigma_CC relation
    Input:
        t90: Time interval in which 90 percent of  the photons arrive , This is to check if its a short or long grb

    """
    f_obs = np.asarray(f_obs)  
    sigma = np.zeros_like(f_obs, dtype=float) 

    nonzero_mask = f_obs > 0  

    if t90 > 2:  # Long GRB case
        sigma[nonzero_mask] = 10**(-0.88 * np.log10(f_obs[nonzero_mask]*np.sqrt(Area/50)) - 3) # 50cm2 is the actual area of HERMES detector 
        
    else:  # Short GRB case
        sigma[nonzero_mask] = 10**(-1.02 * np.log10(f_obs[nonzero_mask]*np.sqrt(Area/50)) - 1.33) 
        
    return sigma




# def compute_pairwise_noise(f_obs, t_90, rng):
#     """
    
#     """
#     num_sat = len(f_obs)
#     noise_matrix = np.zeros((num_sat, num_sat))
#     sigma_final = np.zeros((num_sat,num_sat))
#     sigma = sigma_cc(t_90, f_obs)
#     for i in range(num_sat - 1):
#         for j in range(i + 1, num_sat):
#             if f_obs[i] > 0 and f_obs[j] > 0:
#                 sigma_final[i, j] = np.max([sigma[i], sigma[j]])
#                 noise_matrix[i, j] = rng.normal(0, sigma_final[i,j])
#     return noise_matrix, sigma_final

# def log_likelihood_td(t_obs, t_pred, f_obs, t_90, noise_matrix, sigma_final):

#     num_sat = len(f_obs)
#     ll_sum = []

#     for i in range(num_sat - 1):
#         for j in range(i + 1, num_sat):
#             if f_obs[i] > 0 and f_obs[j] > 0:
#                 delta_t_obs = t_obs[i] - t_obs[j]
#                 delta_t_pred = t_pred[i] - t_pred[j]
#                 measured_delay = delta_t_obs + noise_matrix[i,j]
#                 gaussian_ll = -0.5 * ((measured_delay - delta_t_pred) ** 2) / (sigma_final[i,j] ** 2)
#                 ll_sum.append(gaussian_ll)
#     return np.sum(ll_sum)

def compute_pairwise_noise(f_obs, t_90, Area, rng):
    """
    Computes pairwise noise and sigma values only for satellites with f_obs > 0
    """
    valid_indices = np.where(f_obs > 0)[0]
    num_valid = len(valid_indices)

    noise_matrix = np.zeros((num_valid, num_valid))
    sigma_final = np.zeros((num_valid, num_valid))
    
    sigma = sigma_cc(t_90, f_obs, Area)  # Full sigma array

    for i in range(num_valid - 1):
        for j in range(i + 1, num_valid):
            si, sj = valid_indices[i], valid_indices[j]
            sigma_final[i, j] = max(sigma[si], sigma[sj])
            noise_matrix[i, j] = rng.normal(0, sigma_final[i, j])

    return noise_matrix, sigma_final


def log_likelihood_td(t_obs, t_pred, f_obs, noise_matrix, sigma_final):
    """
    Computes the log-likelihood using only valid satellites.
    
    valid_indices: array of indices with f_obs > 0
    """
    valid_indices = np.where(f_obs > 0)[0]
    t_obs_valid = t_obs[valid_indices]
    t_pred_valid = t_pred[valid_indices]
    num_valid = len(valid_indices)

    ll_sum = []

    for i in range(num_valid - 1):
        for j in range(i + 1, num_valid):
            delta_t_obs = t_obs_valid[i] - t_obs_valid[j]
            delta_t_pred = t_pred_valid[i] - t_pred_valid[j]
            measured_delay = delta_t_obs + noise_matrix[i, j]
            gaussian_ll = -0.5 * ((measured_delay - delta_t_pred) ** 2) / (sigma_final[i, j] ** 2)
            ll_sum.append(gaussian_ll)

    return np.sum(ll_sum)







def log_likelihood_flux(Ph_obs, f_pred, t90, Area): 

    """
    This is a poisson function. Ph_obs is pre-calculated using  observed flux(f_obs), t90 and area of the detector.

    """
    Ph_guess = np.maximum(f_pred * t90 * Area, 1e-10)  # Ph_guess cannot be zero, as log(Ph_guess) will be undefined
    poisson_ll = Ph_obs * np.log(Ph_guess) - Ph_guess - gammaln(Ph_obs + 1) # log of poisson function
    return  np.sum(poisson_ll)



def log_likelihood(theta, t_obs, f_obs, t_90,noise_matrix,sigma_final, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset, lat_lon):
    """
    We simulate the guess parameters in the similar way that we have simulated the satellite detetction for true values
    we will be using the guess direction and guess flux for the grb instead.

    """
    
    # t_obs_list= t_obs.tolist()
    # print(f"t_obs: {t_obs_list}")
    # theta_list=theta.tolist()
    # print(f"theta:{theta_list}")
    # f_obs_list= f_obs.tolist()
    # print(f"f_obs: {f_obs_list}")
    # t_90_list=t_90.tolist()
    # print(f"t_90:{t_90_list}")
    # noise_matrix_list=noise_matrix.tolist()
    # print(f"noise_matrix:{noise_matrix_list}")
    # Ph_obs_list= Ph_obs.tolist()
    # print(f"Ph_obs: {Ph_obs_list}")
    # # f_pred_list=f_pred.tolist()
    # # print(f"f_pred:{f_pred_list}")
    # # lat_lon_list=lat_lon.tolist()
    # print(f"lat_lon:{lat_lon}")
    # t_90_list= t_90.tolist()
    # print(f"t_90:{t_90_list}")
    # Area=Area.tolist()
    # print(f"Area:{Area}")
    # sat_pos=sat_pos.tolist()
    # print(f"sat_pos:{sat_pos}")
    # sat_pointing=sat_pointing.tolist()
    # print(f"sat_pointing:{sat_pointing}")
    # exit()


    if offset == 0:
        ra_guess, dec_guess = theta
        d_guess = coord_transform.r2c(ra_guess, dec_guess)
        t_pred  = time_delay( np.array([0, 0, 0]),sat_pos, d_guess) 
        total = log_likelihood_td(t_obs, t_pred, f_obs, t_90,noise_matrix,sigma_final)
    else:
        ra_guess, dec_guess, f_guess = theta
        d_guess = coord_transform.r2c(ra_guess, dec_guess)
        f_pred, t_pred = simulate_satellite_det(d_guess, f_guess, sat_pos, sat_pointing, flux_limit, lat_lon) 
        total =  log_likelihood_flux(Ph_obs, f_pred, t_90, Area)    + log_likelihood_td(t_obs, t_pred, f_obs, noise_matrix,sigma_final ) #
    return total


def flux_lower_bound(type,Area):
    if type == 'long':
        flux_limit = flux_limit_long(Area) 
    elif type == 'short':
        flux_limit =flux_limit_short(Area) 
    return flux_limit
    

def flux_higher_bound_walkers(flux_limit, Area):
    if flux_limit == flux_limit_long(Area): 
        flux_high = 2
    elif flux_limit ==  flux_limit_short(Area): 
        flux_high = 5
    return flux_high

def log_prior(theta, ra, flux_limit, offset):
    """
    The region in which the MCMC will search around to converge into the true value. 
    We look at half the sky.

TODO: The prior can be modified to remove the occulted regions. THINK!!? For each satellite WE CALCULATE THE OCCULTED REGION WHICH IS COMPUTATIONALLY EXPENSIVE? (FOR THE NUMBER OF SAMPLLES* NUMBER OF SATELLITES)

    """
    if offset == 0:
        ra_guess, dec_guess = theta
        if ra - 45 <= ra_guess <= ra + 45  and -90 <= dec_guess<= 90 : #  for l_grb (0.463, 30) and s_grb(1.861, 30)
            return 0.0
        return -np.inf
    
    else:
        ra_guess, dec_guess,f_guess = theta
    #if ra - 45 <= ra_guess <= ra + 45  and -90 <= dec_guess<= 90 and flux_limit <= f_guess <= 30: #  for l_grb (0.463, 30) and s_grb(1.861, 30)
    if 0 <= ra_guess <= 360  and -90 <= dec_guess<= 90 and flux_limit <= f_guess <= 30:# flux_limit instead of zero
        return 0.0
    return -np.inf



def log_probability(theta, ra, t_obs, f_obs, t_90,noise_matrix,sigma_final, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset, lat_lon):
    lp = log_prior(theta, ra, flux_limit, offset)
    if not np.isfinite(lp): 
        return -np.inf
    log_probability = lp + log_likelihood(theta, t_obs, f_obs, t_90, noise_matrix, sigma_final, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset,lat_lon )
    return log_probability 

labels = ["ra", "dec", "flux"] # this is used in plot_chains, cornerplot and show_results3