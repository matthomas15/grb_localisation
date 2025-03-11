import numpy as np
import matplotlib.pyplot as plt


from modules import coord_transform
import os

from scipy.spatial import KDTree

from scipy.stats import gaussian_kde
from matplotlib.patches import Patch





# Finding credible interval
def fibonacci_sphere(num_points):
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z =  np.sin(phi) * np.cos(theta),  np.sin(phi) * np.sin(theta), np.cos(phi)
    return np.array((x,y,z)).T
    
def euclidean_distance(ra1, ra2, dec1, dec2):
    return np.sqrt((ra1-ra2)**2 + (dec1-dec2)**2)

def ci_68_and_sigma(flat_samples, true_value,save_path, localisation_show, localisation_save):
    """
    Computes the 68% credible area and checks sigma containment for the true value.
    """
    num_vertices = 41253
    vertex_area = 1

    mesh = fibonacci_sphere(num_vertices)
    ra_mcmc = flat_samples[:, 0]
    dec_mcmc = flat_samples[:, 1]
    limit = len(flat_samples)

    data_cartesian = coord_transform.r2c(ra_mcmc, dec_mcmc)
    data_cartesian /= np.linalg.norm(data_cartesian, axis=1, keepdims=True)

    kdtree = KDTree(mesh)
    _, nearest_vertex_indices = kdtree.query(data_cartesian)

    vertex_counts = np.bincount(nearest_vertex_indices, minlength=len(mesh))
    sorted_counts = np.sort(vertex_counts)[::-1]
    cumulative_points = np.cumsum(sorted_counts)

    # Find the 68% CI area
    ci_index = np.searchsorted(cumulative_points, 0.68 * limit)
    credible_interval_68 = (ci_index + 1) * vertex_area

    # Check where the true value falls
    true_cartesian = coord_transform.r2c(true_value[0], true_value[1])
    _, true_index = kdtree.query(true_cartesian)
    true_density_rank = np.where(sorted_counts == vertex_counts[true_index])[0][0]

    sigma_1_index = np.searchsorted(cumulative_points, 0.68 * limit)
    sigma_2_index = np.searchsorted(cumulative_points, 0.95 * limit)
    sigma_3_index = np.searchsorted(cumulative_points, 0.997 * limit)

    # Determine sigma containment
    if true_density_rank <= sigma_1_index:
        containment = "1-sigma"
    elif true_density_rank <= sigma_2_index:
        containment = "2-sigma"
    elif true_density_rank <= sigma_3_index:
        containment = "3-sigma"
    else:
        containment = "Outside 3-sigma"
    
  

    density_rank = np.argsort(np.argsort(-vertex_counts))
    colors = np.full(len(mesh), "yellow")  # Default to outside 3-sigma
    colors[density_rank <= sigma_3_index] = "blue"
    colors[density_rank <= sigma_2_index] = "green"
    colors[density_rank <= sigma_1_index] = "red"

        

    plt.figure(figsize=(8, 6))
    plt.scatter(ra_mcmc, dec_mcmc, c=colors[nearest_vertex_indices], s=1, alpha=0.5)
        
    plt.scatter(true_value[0], true_value[1], color='black', marker='x', s=100, label='True Position')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='1-sigma'),
        Patch(facecolor='green', label='2-sigma'),
        Patch(facecolor='blue', label='3-sigma'),
        Patch(facecolor='yellow', label='Outside 3-sigma')
    ]


    # Labels and legend
    plt.xlabel("Right Ascension (RA)")
    plt.ylabel("Declination (Dec)")
    plt.title("MCMC Localization Contours using KDTree")
    plt.legend(handles=legend_elements, loc='upper right')
    
    if  localisation_save:
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        plt.savefig(os.path.join(save_path, "localization_plot.png"))
    if localisation_show:
        plt.show()

    return credible_interval_68, containment