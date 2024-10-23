import numpy as np


global skygridRoot
skygridRoot = 'sphericalGrids/'



def loadSkygrid(points_per_degree):
    '''
    Loads a set of pre-generated points on the surface of a sphere
    approximately unifiform grid on the sphere, generated using the fibonacci
    spiral algorithm.
    '''
    if points_per_degree not in [1, 4, 16]:

        # TODO: generate new skygrid if this is requested
        raise Exception( "Error: Must choose 1, 4 or 16 points per degree" )
    

    phi, theta = np.load(skygridRoot + f'/{points_per_degree}deg_uniform_skygrid.npy')

    return phi, theta



