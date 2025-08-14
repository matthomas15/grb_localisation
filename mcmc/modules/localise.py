import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree

# Finding credible interval
def fibonacci_sphere(num_points):
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z =  np.sin(phi) * np.cos(theta),  np.sin(phi) * np.sin(theta), np.cos(phi)
    return np.array((x,y,z)).T
    
def euclidean_distance(ra1, ra2, dec1, dec2):
    """
    input: ra, dec in degree
    output: In degree?
    """
    return np.sqrt((ra1-ra2)**2 + (dec1-dec2)**2)


def angular_distance(ra1, ra2, dec1, dec2):
    # Convert to radians
    ra1, ra2 = np.radians(ra1), np.radians(ra2)
    dec1, dec2 = np.radians(dec1), np.radians(dec2)
    
    # Spherical law of cosines
    cos_angle = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    # Clip for numerical safety
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))


num_vertices = 41253
def grid(num_vertices):
    mesh = fibonacci_sphere(num_vertices)
    grid_ra = np.degrees(np.arctan2(mesh[:, 1], mesh[:, 0])) + 180
    grid_dec = np.degrees(np.arcsin(mesh[:, 2]))
    return grid_ra, grid_dec

grid_ra, grid_dec = grid(num_vertices)

def ci_68_and_sigma(flat_samples, true_value, type, save_path, localisation_show, localisation_save):
    """
    Computes the 68% credible area and checks sigma containment for the true value.
    """
    ra_mcmc = flat_samples[:, 0]
    ra_mcmc = (ra_mcmc % 360)
    dec_mcmc = flat_samples[:, 1]
    limit = len(flat_samples)

    nearest_grid_indices = np.zeros(len(flat_samples), dtype=int)
    grid_occupancy = np.zeros_like(grid_ra)
    for i, (ra, dec) in enumerate(zip(ra_mcmc, dec_mcmc)):
        distances = angular_distance(ra, grid_ra, dec, grid_dec)
        closest_grid_point_index = np.argmin(distances)
        nearest_grid_indices[i] = closest_grid_point_index
        grid_occupancy[closest_grid_point_index] = grid_occupancy[closest_grid_point_index] + 1

    sigmas = np.zeros_like(grid_occupancy)

    trials = len(flat_samples)
    total_prob_consumed = 0
    total = trials
    normalised_grid_occupancy = grid_occupancy / total

    for j in range(trials):

        largest_value_index = np.argmax(normalised_grid_occupancy)
        prob = normalised_grid_occupancy[largest_value_index]
        normalised_grid_occupancy[largest_value_index] = 0
        total_prob_consumed += prob

        if total_prob_consumed < 0.68:
            sigmas[largest_value_index] = 1
        elif (total_prob_consumed > 0.68) and (total_prob_consumed < 0.95):
            sigmas[largest_value_index] = 2
        else:
            sigmas[largest_value_index] = 3

   

    area_per_grid_point = 1
    area_68 = np.sum(sigmas == 1) * area_per_grid_point
    #area_68 = np.sum(sigmas == 1) * area_per_grid_point
    print(f"68% confidence region area: {area_68:.2f} deg²")

    true_index = np.argmin(angular_distance(true_value[0], grid_ra, true_value[1], grid_dec))

    # Check 
    if sigmas[true_index] == 1:
        containment = "1-sigma"
        print("True value lies within 1σ (68%) confidence region.")
    elif sigmas[true_index] == 2:
        containment = "2-sigma"
        print("True value is within 2σ (95%) but not 1σ.")
    else:
        containment = "3-sigma"
        print("True value is outside 3σ confidence region.")

    plt.figure(figsize=(8, 6))
    grid_point_size = 30
    plt.scatter(true_value[0], true_value[1], color ='red',marker ='*', s=100, edgecolor='white', linewidth=0.5, zorder=3, label ="True position" )
    plt.scatter(grid_ra, grid_dec, c=sigmas, cmap="viridis", s = grid_point_size, alpha = 0.9)

    if type == "long":
        plt.xlim(max(0,true_value[0]-50),  min(360, true_value[0] + 50))
        plt.ylim(true_value[1]-10, true_value[1]+10)
    elif type == "short":
        plt.xlim(true_value[0]-100, true_value[0]+100)
        plt.ylim(true_value[1]-40, true_value[1]+40)


    plt.colorbar(label="Confidence level")
    plt.xlabel("Right Ascension (RA)")
    plt.ylabel("Declination (Dec)")
    plt.title("MCMC Sample Confidence levels")
    plt.legend(loc='upper right')


    if  localisation_save:
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        plt.savefig(os.path.join(save_path, "localization_plot.png"))
    if localisation_show:
        plt.show()
    else:
        plt.close()

    return area_68, containment, area_per_grid_point


# Using kdtree----------------------------------------------------------------------------------------------

def radec_to_unitvec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z))

def ci_68_and_sigma_2(flat_samples, true_value, type, save_path, localisation_show, localisation_save,
                      initial_num_vertices=41253, area_threshold=4.0, max_attempts=2):
    """
    Computes the 68% credible area and checks sigma containment for the true value.
    If area_68 < area_threshold, increases grid resolution and retries (up to max_attempts).
    """
    for attempt in range(max_attempts):
        num_vertices = initial_num_vertices * (10 ** attempt)
        print(f"\n[Attempt {attempt + 1}] Using {num_vertices} grid points.")

        # Generating Grid Points
        grid_ra, grid_dec = grid(num_vertices)

        # Convert RA/Dec to unit vectors for KDTree matching
        grid_vecs = radec_to_unitvec(grid_ra, grid_dec)
        ra_mcmc = flat_samples[:, 0] % 360
        dec_mcmc = flat_samples[:, 1]
        sample_vecs = radec_to_unitvec(ra_mcmc, dec_mcmc)

        # KDTree to assign each sample to nearest grid point
        tree = cKDTree(grid_vecs)
        _, nearest_indices = tree.query(sample_vecs, k=1)

        # Occupancy count per grid point
        grid_occupancy = np.bincount(nearest_indices, minlength=len(grid_ra))
        total = len(flat_samples)
        # Assign confidence levels based on cumulative probability
        sorted_indices = np.argsort(grid_occupancy)[::-1]
        sorted_probs = grid_occupancy[sorted_indices] / total
        
        sigmas = np.zeros_like(grid_occupancy, dtype=int)

        cumulative = 0.0
        for idx, prob in zip(sorted_indices, sorted_probs):
            prev_cumulative = cumulative
            cumulative += prob
            if prev_cumulative < 0.68:
                sigmas[idx] = 1
            elif prev_cumulative < 0.95:
                sigmas[idx] = 2
            else:
                sigmas[idx] = 3


        # Compute area per grid point
        sphere_area_in_degsq = 41253  # ≈ 41252.96 deg²
        area_per_grid_point = sphere_area_in_degsq / num_vertices
        area_68 = np.sum(sigmas == 1) * area_per_grid_point

        print(f"68% confidence region area: {area_68:.2f} deg² with resolution {num_vertices} vertices")

        # Break loop if area large enough or last attempt
        if area_68 >= area_threshold or attempt == max_attempts - 1:
            break

    # Evaluate containment of true position
    true_vec = radec_to_unitvec(np.array([true_value[0]]), np.array([true_value[1]]))
    _, true_index = tree.query(true_vec, k=1)

    if sigmas[true_index] == 1:
        containment = "1-sigma"
        print("True value lies within 1σ (68%) confidence region.")
    elif sigmas[true_index] == 2:
        containment = "2-sigma"
        print("True value is within 2σ (95%) but not 1σ.")
    else:
        containment = "3-sigma"
        print("True value is outside 3σ confidence region.")
        
    print("Grid cells in 1-sigma:", np.sum(sigmas == 1))
    print("Grid cells in 2-sigma:", np.sum(sigmas == 2))
    print("Grid cells in 3-sigma:", np.sum(sigmas == 3))
    print(f"Number of MCMC samples: {total}")
    print(f"Grid cells with >=1 sample: {np.sum(grid_occupancy > 0)}")
    print(f"Samples in most occupied grid cell: {np.max(grid_occupancy)}")
    print(f"68% region includes {np.sum(sigmas == 1)} grid cells.")
    print(f"95% region includes {np.sum(sigmas <= 2)} grid cells.")
    print(f"Area of 68% region: {np.sum(sigmas == 1) * area_per_grid_point:.2f} deg²")

    # -------- Plotting --------
    plt.figure(figsize=(8, 6))
    
    plt.scatter(true_value[0], true_value[1], color='red', marker='*', s=50,
                edgecolor='white', linewidth=0.5, zorder=3, label="True position")
    plt.scatter(grid_ra, grid_dec, c=sigmas, cmap="viridis", s=30, alpha=0.9)

    if type == "long":
        plt.xlim(max(0, true_value[0] - 10), min(360, true_value[0] + 10))
        plt.ylim(max(-90,true_value[1] - 10), min(true_value[1] + 10, 90))
    elif type == "short":
        plt.xlim(max(0,true_value[0] - 50),min(360, true_value[0] + 50))
        plt.ylim(max(-90,true_value[1] - 40), min(true_value[1] + 40, 90))

    plt.colorbar(label="Confidence level")
    plt.xlabel("Right Ascension (RA)")
    plt.ylabel("Declination (Dec)")
    plt.title("MCMC Sample Confidence Levels")
    plt.legend(loc='upper right')

    if localisation_save:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "localization_plot.png"))

    if localisation_show:
        plt.show()
    else:
        plt.close()

    return area_68, containment, area_per_grid_point


