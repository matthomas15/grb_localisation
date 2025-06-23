
import numpy as np
import matplotlib.pyplot as plt
from modules import likelihood
import emcee
import corner
import os

from IPython.display import display, Math
from emcee.moves import StretchMove

labels = ["ra", "dec", "flux"]

def mcmc_sampler(ra, dec, steps, nwalk, move, t_obs, f_obs, t_90, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset, lat_lon, rng):
    """
    We use the mcmc sampler for our localisation.
    """
    flux_high = likelihood.flux_higher_bound_walkers(flux_limit)

    if offset == 0:
        pos = np.array([rng.uniform(ra-5, ra + 5, nwalk),
                rng.uniform(dec-5 ,dec + 5, nwalk)]).T
        nwalkers, ndim = pos.shape
    else:
        pos = np.array([rng.uniform(ra-5, ra + 5, nwalk),
                rng.uniform(dec-5 ,dec + 5, nwalk), 
                rng.uniform(flux_limit + 0.1, flux_high, nwalk)]).T  # For long grb (0.5 , 2)  shortgrb(2, 5)
        nwalkers, ndim = pos.shape

        # The sampler we are using
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, likelihood.log_probability, args=( ra, t_obs, f_obs, t_90, rng, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset,lat_lon),
        moves=StretchMove(a = move) # Changes this if necessary (default value = 2)
        )
    sampler.run_mcmc(pos,steps, progress=True)
    return sampler




def plot_chains(sampler, ra, dec, flux_avg,offset, save_path, chain_show= False, chain_save= False):
    """
    Plots the MCMC chains.
    """
    if offset==0:
        ndim = 2
    else:
        ndim = 3
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    true_values = np.array([ra, dec, flux_avg])
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.axhline(true_values[i], color="blue", linestyle="--", label=f"True {labels[i]}")
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.tight_layout()
    
    if chain_save:  
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        plt.savefig(os.path.join(save_path, "mcmc_chains.png") )
    if not chain_show:
        plt.close(fig)
        #plt.show()
    
    


def get_flat_samples(sampler, discard):
    """ returns the MCMC samples diiscarding initial samples where the walkers are still searching for a  minima"""
    flat_samples = sampler.get_chain(discard= discard, flat=True)
    # can include the extra parameter (thin = 15) that shows every point after 15 steps
    return flat_samples

def corner_plot(flat_samples, ra, dec, flux_avg, offset, save_path, corner_show = False, corner_save = False):
    if offset == 0:
        fig = corner.corner(flat_samples, labels = ["ra", "dec"], truths=[ra, dec] )
    else:
        fig = corner.corner(flat_samples, labels = ["ra", "dec", "flux"], truths= [ra, dec, flux_avg])
    
    if corner_save:
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        plt.savefig(os.path.join(save_path, "corner_plot.png"))
    if not corner_show:
        plt.close(fig)
        

def show_result(flat_samples, offset):
    if offset==0:
        ndim = 2
    else:
        ndim = 3
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = r"\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))



def run_mcmc(ra, dec, flux_avg, t_90, rng, t_obs, f_obs, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset,lat_lon, steps, nwalk, move, discard, save_path, corner_show= False, corner_save = False, chain_show= False, chain_save = False):
    """
    This function runs MCMC, plot chains, corner plots and also finds  the 68% credible interval region.
    """
    # initial_position = optimize_ll(ra, dec, flux_avg, t_obs, f_obs, t_90, Ph_obs, sat_pointing, flux_limit)
    # sampler = mcmc_sampler(initial_position,steps, nwalk, move, t_obs, f_obs, t_90, Ph_obs, sat_pointing, flux_limit)
    sampler = mcmc_sampler(ra, dec, steps, nwalk, move,  t_obs, f_obs, t_90, Ph_obs, Area, sat_pos, sat_pointing, flux_limit, offset,lat_lon, rng)
    flat_samples = get_flat_samples(sampler, discard)
    corner_plot(flat_samples, ra, dec, flux_avg, offset, save_path, corner_show, corner_save )
    plot_chains(sampler, ra, dec, flux_avg, offset,  save_path, chain_show, chain_save)
    
    result = show_result(flat_samples, offset)

    return flat_samples