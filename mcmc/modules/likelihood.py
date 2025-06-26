import numpy as np
from modules import simulation
from modules import coord_transform
from scipy.special import gammaln



def time_delay(r1, r2, d_grb):
    """
    Returns the delay in arrival time of grb signals between two satellites

    Input: 
        r1,r2 are position vectors of the two satellites
        d_grb is the position of the grb in cartesian coordinate
    """
    c = 3e5 #  speed of light in km/s

    r_diff = r1 - r2
    t_delay = np.dot(r_diff, d_grb)/c
    return t_delay

def cos_angle_btw_vec(v1, v2):
    """ 
    Parameters: 
    vectors [x1, y1, z1], [x2, y2, z2]    
    
    Returns:
       Cosine Angle between these vectors in radian

    """
    cos_theta = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) #  in radians
    #theta = np.degrees(np.arccos(cos_theta)) 
    return cos_theta


def angle_btw_vec(v1, v2):
    """ 
    Parameters: 
    vectors [x1, y1, z1], [x2, y2, z2]    
    
    Returns:
        Angle between these vectors in degree
    
    """
    cos_theta = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) #  in radians
    theta = np.degrees(np.arccos(cos_theta)) 
    return theta

combined_bkgd = np.load('C:\\Users\\Haritha\\repo\\grb_localisation\\mcmc\\particle_background.npy')
def get_particle_background_value(lat, lon):
    """Returns the particle background value at a given (lat, lon) index."""
    if np.isnan(lat) or np.isnan(lon):
        raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")
    lat = int( np.round( lat ) )
    lon = int( np.round( lon ) )

    bkgd = combined_bkgd[ lat + 90 - 1 ][ lon + 180 - 1 ]

    return bkgd

def get_nonzero_background_indices(positions):
    """Returns the list of indices where positions have nonzero particle background."""

    nonzero_indices = []
    for i, (lat, lon) in enumerate(positions):
        bkgd_value = get_particle_background_value(lat, lon)
        if bkgd_value > 0:
            nonzero_indices.append(i)  # Store the index
    return nonzero_indices

def simulate_satellite_det(grb_vec, flux_avg, sat_pos, sat_pointing, flux_limit, lat_lon):
    """
    This function returns the arrival times on each satellite - t_obs and the flux observed by each satllite - f_obs.
    The observed flux is set to zero if the satellite is in earth occulted region
    or if the flux is below the instrument flux limit (for long and short grb this limit is different. WHY???) 
    
    The occult angle limit of 67 is calculated considering the orbit height and the relative size of earth at that height.
    """
    sat2earth_vec = [-sat for sat in sat_pos]
    occult_angle = [angle_btw_vec(grb_vec, se_vec) for se_vec in sat2earth_vec]
    costheta = np.array([cos_angle_btw_vec(grb_vec,sp) for sp in sat_pointing])
    bkgd_index = get_nonzero_background_indices(lat_lon)
    f_obs = flux_avg * costheta
    t_obs = time_delay( np.array([0, 0, 0]),sat_pos, grb_vec) 
    for i in range(len(f_obs)):
        if occult_angle[i] < 67  or f_obs[i] < flux_limit or i in bkgd_index :
            f_obs[i] = 0
    return f_obs, t_obs



def sigma_cc(t90, f_obs):
    
    """
    Gives the sigma value of signal cross-correlation  from flux and sigma_CC relation
    Input:
        t90: Time interval in which 90 percent of  the photons arrive , This is to check if its a short or long grb
       """
    f_obs = np.asarray(f_obs)  # Convert to numpy array for vectorized operations
    sigma = np.zeros_like(f_obs, dtype=float)  # Initialize array with zeros

    nonzero_mask = f_obs > 0  # Boolean mask for nonzero values

    if t90 > 2:  # Long GRB case
        sigma[nonzero_mask] = 10**(-0.88 * np.log10(f_obs[nonzero_mask]) - 3)
        #sigma[nonzero_mask] = 10**(-0.22 * np.log10(f_obs[nonzero_mask]) + 0.41)
        
    else:  # Short GRB case
        sigma[nonzero_mask] = 10**(-1.02 * np.log10(f_obs[nonzero_mask]) - 1.33)
        #sigma[nonzero_mask] = 10**(-0.14 * np.log10(f_obs[nonzero_mask]) - 1.08)
        

    return sigma

def compute_pairwise_noise(f_obs, t_90, rng):
    num_sat = len(f_obs)
    noise_matrix = np.zeros((num_sat, num_sat))
    sigma = sigma_cc(t_90, f_obs)
    for i in range(num_sat - 1):
        for j in range(i + 1, num_sat):
            if f_obs[i] > 0 and f_obs[j] > 0:
                sigma_final = np.max([sigma[i], sigma[j]])
                noise_matrix[i, j] = rng.normal(0, sigma_final)
    return noise_matrix

def log_likelihood_td(t_obs, t_pred, f_obs, t_90, noise_matrix):
    num_sat = len(f_obs)
    ll_sum = []
    sigma = sigma_cc(t_90, f_obs)

    for i in range(num_sat - 1):
        for j in range(i + 1, num_sat):
            if f_obs[i] > 0 and f_obs[j] > 0:
                delta_t_obs = t_obs[i] - t_obs[j]
                delta_t_pred = t_pred[i] - t_pred[j]
                sigma_final = np.max([sigma[i], sigma[j]])
                
                measured_delay = delta_t_obs + noise_matrix[i, j]
                gaussian_ll = -0.5 * ((measured_delay - delta_t_pred) ** 2) / (sigma_final ** 2)
                ll_sum.append(gaussian_ll)
    return np.sum(ll_sum)



# def log_likelihood_td(t_obs, t_pred, f_obs, t_90, rng ):
#     """
#     Log -likelihood function for timedelay is a guassian. Here the measured time delay  is the sum of true time dealy 
#     and the additional noise from signal cross-correlation. Note that we have not added Instrument noise yet 

#     """
#     num_sat= len(f_obs) 
#     ll_sum = []
     
#     for i in range(num_sat-1):
#         for j in range(i+1, num_sat):
#             if f_obs[i] > 0 and f_obs[j] > 0:
#                 delta_t_obs = t_obs[i]- t_obs[j]
#                 delta_t_pred = t_pred[i]- t_pred[j]

#                 sigma =  sigma_cc(t_90, f_obs) 
#                 sigma_final = np.max([sigma[i], sigma[j]])

#                 N_sigma_cc = rng.normal(loc=0, scale = sigma_final)
#                 measured_delay = delta_t_obs # + N_sigma_cc
                
#                 gaussian_ll = -0.5*((measured_delay-delta_t_pred)**2)/(sigma_final**2)

#                 ll_sum.append(gaussian_ll)

#     return np.sum(ll_sum)

# def log_likelihood_td(t_obs, t_pred, f_obs, t_90):
#     """
#     Log-likelihood function using only the 5 satellite pairs with the highest |delta_t_obs| 
#     (which likely correspond to the largest baselines).
#     """

#     num_sat = len(f_obs)
#     pair_data = []

#     sigma = sigma_cc(t_90, f_obs)  # compute once outside loops

#     for i in range(num_sat - 1):
#         for j in range(i + 1, num_sat):
#             if f_obs[i] > 0 and f_obs[j] > 0:
#                 delta_t_obs = t_obs[i] - t_obs[j]
#                 delta_t_pred = t_pred[i] - t_pred[j]
#                 sigma_final = np.max([sigma[i], sigma[j]])

#                 pair_data.append((abs(delta_t_obs), delta_t_obs, delta_t_pred, sigma_final))

#     # Sort by |delta_t_obs| in descending order
#     top_pairs = sorted(pair_data, key=lambda x: x[0], reverse=True)[:4]

#     ll_sum = []
#     for _, delta_t_obs, delta_t_pred, sigma_final in top_pairs:
#         N_sigma_cc = np.random.normal(loc=0, scale=sigma_final)
#         measured_delay = delta_t_obs + N_sigma_cc
#         gaussian_ll = -0.5 * ((measured_delay - delta_t_pred) ** 2) / (sigma_final ** 2)
#         ll_sum.append(gaussian_ll)
#     print(np.max(N_sigma_cc))
    

#     return np.sum(ll_sum)


def log_likelihood_flux(Ph_obs, f_pred, t90, Area): 
    """
    This is a poisson function. Ph_obs is pre-calculated using  observed flux(f_obs), t90 and area of the detector.

    """
    Ph_guess = np.maximum(f_pred * t90 * Area, 1e-10)  # Ph_guess cannot be zero, as log(Ph_guess) will be undefined
    poisson_ll = Ph_obs * np.log(Ph_guess) - Ph_guess - gammaln(Ph_obs + 1) # log of poisson function
    return  np.sum(poisson_ll)



def log_likelihood(theta, t_obs, f_obs, t_90,noise_matrix, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset, lat_lon):
    """
    We simulate the guess parameters in the similar way that we have simulated the satellite detetction for true values
    we will be using the guess direction and guess flux for the grb instead.
    """
   
    if offset == 0:
        ra_guess, dec_guess = theta
        d_guess = coord_transform.r2c(ra_guess, dec_guess)
        t_pred  = time_delay( np.array([0, 0, 0]),sat_pos, d_guess) 
        total = log_likelihood_td(t_obs, t_pred, f_obs, t_90,noise_matrix)
    else:
        ra_guess, dec_guess, f_guess = theta
        d_guess = coord_transform.r2c(ra_guess, dec_guess)
        f_pred, t_pred = simulate_satellite_det(d_guess, f_guess, sat_pos, sat_pointing, flux_limit, lat_lon) 
        total =  log_likelihood_flux(Ph_obs, f_pred, t_90, Area)   + log_likelihood_td(t_obs, t_pred, f_obs, t_90, noise_matrix ) 
    
    return total

def flux_lower_bound(type):
    if type == 'long':
        flux_limit = 0.463
    elif type == 'short':
        flux_limit = 1.861
    return flux_limit

def flux_higher_bound_walkers(flux_limit):

    if flux_limit == 0.463:
        flux_high = 2
    elif flux_limit == 1.861:
        flux_high = 5
    return flux_high

# This cant be applied to higher number of satellites!!
def sky_search(f_obs):
   
    # f_obs[1] == f_obs[2]  # time dealy case half sky??
    if f_obs[1] == 0 or f_obs[2] == 0:
        prior_low, prior_high = 90, 270
    elif f_obs[1] > f_obs[2]:
        prior_low, prior_high = 180, 270
    else:
        prior_low, prior_high = 90, 180

    return prior_low, prior_high

def log_prior(theta, ra, flux_limit, offset):
    """
    The region in which the MCMC will search around to converge into the true value. 
    We look at half the sky.
TODO: The prior can be modified to remove the occulted regions. THINK!!?
    """
    if offset == 0:
        ra_guess, dec_guess = theta
        if ra - 45 <= ra_guess <= ra + 45  and -90 <= dec_guess<= 90 : #  for l_grb (0.463, 30) and s_grb(1.861, 30)
            return 0.0
        return -np.inf
    
    else:
        ra_guess, dec_guess,f_guess = theta
    #if ra - 45 <= ra_guess <= ra + 45  and -90 <= dec_guess<= 90 and flux_limit <= f_guess <= 30: #  for l_grb (0.463, 30) and s_grb(1.861, 30)
    if 0 <= ra_guess <= 360  and -90 <= dec_guess<= 90 and flux_limit <= f_guess <= 30:
        return 0.0
    return -np.inf



def log_probability(theta, ra, t_obs, f_obs, t_90,noise_matrix, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset, lat_lon):
    lp = log_prior(theta, ra, flux_limit, offset)
    if not np.isfinite(lp): 
        return -np.inf
    log_probability = lp + log_likelihood(theta, t_obs, f_obs, t_90, noise_matrix, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset,lat_lon )
    return log_probability 

labels = ["ra", "dec", "flux"] # this is used in plot_chains, cornerplot and show_results3