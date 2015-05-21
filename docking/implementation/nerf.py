def nerf(A,B,C,R,phi,theta,lbc=""):
    """
    An implementation of the Natural Extension Reference Frame (NeRF) algorithm
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.83.8235&rep=rep1&type=pdf

    Args:
      A,B,C(double[]): absolute coordinates of the three previous points
                       (ancestor, ancestor of the ancestor, ...)
      R(double):       distance to the ancestor
      phi(double):     bond angle of (B,C,D)
      theta(double):   dihedral angle of the planes given by (A,B,C) and (B,C,D)
      lbc(double,optional): distance of B and C

    Returns:
      D(double[]): returns absolute coordinates given the above mentioned internal
        coordinates and the absolute coordinates of the previous points
    """
    cosphi   = math.cos(phi)
    sinphi   = math.sin(phi)
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    D    = R*np.asarray([cosphi, sinphi*costheta, sinphi*sintheta])

    AB   = np.asarray(B)-np.asarray(A)
    BC   = np.asarray(C)-np.asarray(B)
    lbc  = lbc or la.norm(BC)
    bc   = BC/lbc
    n    = normalize(np.cross(AB,bc))
    nxbc = np.cross(n,bc) # already normalized by construction

    M = np.transpose([bc,nxbc,n])
    D = np.dot(M,D)+C

    return D
