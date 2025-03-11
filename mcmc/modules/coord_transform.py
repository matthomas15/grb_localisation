import numpy as np

def r2c(ra, dec):         # ra,dec inputs are in degree
    """
    first we convert RA, DEC that is in degrees to radians
     """
    ra = np.radians(ra)
    dec = np.radians(dec)      

    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    # return np.vstack((x, y, z)).T
    return np.array((x,y,z)).T
    



def c2r(x, y, z):        
    """
    convert a single cartesian coordinates(x, y, z) into (RA,DEC)
    x,y,z needs to be normalised
    
    """

    w = np.sqrt(x**2 + y**2+ z**2)
    x = x/w
    y = y/w
    z = z/w

    ra = np.degrees(np.arctan2(y,x))
    if ra < 0:
            ra += 360
    dec = np.degrees(np.arcsin(z/w))
    #dec = np.degrees(np.arcsin(z))
    return np.array((ra,dec)) 

