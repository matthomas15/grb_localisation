import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import gammaln
from math import comb
from scipy.stats import chi2
from scipy.optimize import root

def s2c(theta, phi, r = 1):

    return np.array([ r * np.sin(theta) * np.cos(phi),
                        r * np.sin(theta) * np.sin(phi),
                        r * np.cos(theta)
    ])


def ang_sep(ra1, dec1, ra2, dec2): #Return angular separation of two points on the sphere in radians
    return np.arccos( np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2) )

def chi2_root(x, confidence, df):
    return confidence - chi2.cdf(x, df)

def get_chi2_confidence_limit(confidence, df):
    return root(chi2_root, 1, args = (confidence, df)).x[0]

def logLike_timeDifference(sat_coords, d):
    '''
    Compute log likelihood for the observed inter-arrival times
    on a sky grid defined by the global 'grid' array.

    Log likelihood is calculated using a gaussian probability
    of measuring inter-arrival time dT given simulated inter-arrival
    time from coordinates d_guess with variance from sanna+2020


    TODO: update this with accurate sigma_cc according to flux
    '''

    # Define constants

    nSats, nDim = sat_coords.shape
    speedOfLight = 3e5 #km/s
    
    sat_coords_normalised = sat_coords / speedOfLight

    numPairs = comb(nSats, 2)

    dTau = np.zeros(numPairs)
    dx = np.zeros([numPairs, nDim])

    t_i = - np.dot(sat_coords_normalised, d) # speed of light in km/s

    sigma_cc = 3e-3


    count = 0
    for i in range(nSats-1):
        for j in range(i+1, nSats):
            dTau[count] = t_i[j] - t_i[i] + sigma_cc * np.random.randn(1).item() # random noise on measurement
            dx[count] = sat_coords_normalised[j] - sat_coords_normalised[i]
            count += 1

    guess_dts = - np.dot( dx, coords ) # already normalised

    out = np.square( guess_dts.T - dTau ) / sigma_cc**2

    out = np.sum(out, axis = 1)

    return out

def logLike_fluxRatio(p, d, grb_flux):
    '''
    Compute log likelihood for the observed inter-arrival times
    on a sky grid defined by the global 'grid' array.

    Log likelihood is calculated using a gaussian probability
    of measuring flux ratio dF, which is a ratio distribution, 
    which can be approximated as a gaussian with mean:

    delta_i = sigma_i / mu_i = 1/np.sqrt(mu_i) (for large count Poisson realisations)
    delta_j = sigma_j / mu_j = 1/np.sqrt(mu_j) 

    mu = beta = mi/mj
    std = sqrt( beta**2 * (delta_1**2 + delta_2**2))


    '''

    nSats, nDim = p.shape
    _, nGridPoints = coords.shape


    # simulate 'measurement' of flux differences
    measured_f = np.random.poisson( grb_flux * np.dot(p, d) )
    
    # flux ratio approximation works better if we divide by the larger number
    measured_f, newOrder = np.array( list( sorted(zip(measured_f, range(len(measured_f)))) ) ).T

    numPairs = comb(nSats, 2)
    measured_df = np.zeros(numPairs) # measured flux differences
    guess_df = np.zeros([numPairs, nGridPoints])
    allPointings = np.dot(p, coords)[newOrder] # reorder the pointing guesses to align with flux sorting


    approx_vars = np.zeros(numPairs)

    count = 0
    for i in range(nSats-1):
        for j in range(i+1, nSats):
            measured_df[count] = measured_f[i] / measured_f[j]
            guess_df[count] = np.divide(allPointings[i], allPointings[j])

            # compute approximate variance for each ratio,
            #   using counts as proxy for mean
            m1 = measured_f[i]
            s1 = np.sqrt(m1)
            m2 = measured_f[j]
            s2 = np.sqrt(m2)
            beta = m1/m2
            delta_1 = s1/m1
            delta_2 = s2/m2

            sigma_squared =  beta**2 * (delta_1**2 + delta_2**2)
            approx_vars[count] = sigma_squared
            count += 1
    
    chisq = np.square( guess_df.T - measured_df) / approx_vars

    out = np.sum(chisq, axis = 1)
    
    return out

def estimate_localisation(sat_xyz, sat_pointings, grb_vector, grb_flux, area_element):
    '''
    Compute the localisation region for a set of satellites
    
    >>> IT IS ASSUMED ALL SATELLITES PASSED TO THIS FUNCTION DETECT THE GRB <<<

    If the above is not true, FOV logic will break down
    '''

    Ctime = logLike_timeDifference(sat_xyz, grb_vector)
    Cflux = logLike_fluxRatio(sat_pointings, grb_vector, grb_flux)
    combined_chisq = Ctime + Cflux

    # account for FOV of each satellite (presume t)
    
    FOV_limit = np.cos( np.radians( 90 ) ) # 90 degree FOV limitation

    cosTheta = np.dot(sat_pointings, coords)
    cosTheta[cosTheta < FOV_limit] = np.inf
    cosTheta = np.sum(cosTheta, axis = 0)
    combined_chisq[ np.isinf(cosTheta) ] = np.inf

    # Account for Earth occultation

    ''' TODO '''

    # Compute results

    combined_chisq = combined_chisq - np.amin(combined_chisq)
    loc68 = np.sum(combined_chisq < 2.3) * area_element

    return loc68, combined_chisq

