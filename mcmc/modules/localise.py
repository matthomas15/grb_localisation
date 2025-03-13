import numpy as np
import matplotlib.pyplot as plt
import os


# Finding credible interval
def fibonacci_sphere(num_points):
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z =  np.sin(phi) * np.cos(theta),  np.sin(phi) * np.sin(theta), np.cos(phi)
    return np.array((x,y,z)).T
    
def euclidean_distance(ra1, ra2, dec1, dec2):
    return np.sqrt((ra1-ra2)**2 + (dec1-dec2)**2)



def ci_68_and_sigma(flat_samples, true_value, type, save_path, localisation_show, localisation_save):
    """
    Computes the 68% credible area and checks sigma containment for the true value.
    """
    
    std_dev_ra = np.std(flat_samples[:,0])

    if type=="short":
        num_vertices = int(41253/(std_dev_ra/2))
    elif type=="long":
        if std_dev_ra > 3:
            num_vertices = int(41253/(std_dev_ra/2))
        else:
            num_vertices =41253
        
    # Generate grid from Fibonacci sphere'
    mesh = fibonacci_sphere(num_vertices)
    grid_ra = np.degrees(np.arctan2(mesh[:, 1], mesh[:, 0])) + 180
    grid_dec = np.degrees(np.arcsin(mesh[:, 2]))

    num_grid_points = len(grid_ra) # WHY IS THIS LEN(GRID_RA)
    ra_mcmc = flat_samples[:, 0]
    dec_mcmc = flat_samples[:, 1]
    limit = len(flat_samples)

    nearest_grid_indices = np.zeros(len(flat_samples), dtype=int)
    grid_occupancy = np.zeros_like(grid_ra)
    for i, (ra, dec) in enumerate(zip(ra_mcmc, dec_mcmc)):
        distances = euclidean_distance(ra, grid_ra, dec, grid_dec)
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

    area_per_grid_point = 41253 / num_vertices
    area_68 = np.sum(sigmas == 1) * area_per_grid_point
    print(f"68% confidence region area: {area_68:.2f} deg²")

    true_index = np.argmin(euclidean_distance(true_value[0], grid_ra, true_value[1], grid_dec))

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
    if std_dev_ra > 3:
        grid_point_size = 280
    else :
        grid_point_size = 80
    plt.scatter(true_value[0], true_value[1], color ='red',marker ='*', s=100, edgecolor='white', linewidth=0.5, zorder=3, label ="True position" )
    plt.scatter(grid_ra, grid_dec, c=sigmas, cmap="viridis", s = grid_point_size, alpha = 0.9)
    if type == "long":
        plt.xlim(true_value[0]-50, true_value[0]+50)
        plt.ylim(true_value[1]-10, true_value[1]+10)
    elif type == "short":
        plt.xlim(true_value[0]-100, true_value[0]+100)
        plt.ylim(true_value[1]-40, true_value[1]+40)


    plt.colorbar(label="Confidence level")
    plt.xlabel("Right Ascension (RA)")
    plt.ylabel("Declination (Dec)")
    plt.title("MCMC Sample Confidence levels")
    plt.legend()


    if  localisation_save:
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        plt.savefig(os.path.join(save_path, "localization_plot.png"))
    if localisation_show:
        plt.show()
    else:
        plt.close()

    return area_68, containment, area_per_grid_point, std_dev_ra