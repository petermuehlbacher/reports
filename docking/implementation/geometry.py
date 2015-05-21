def normalize(v):
    return v/la.norm(v)

def v_angle(v1,v2):
    return np.arccos(np.dot(v1/la.norm(v1),v2.T/la.norm(v2)))

def angle(A,B,C):
    p = np.asarray(A)-np.asarray(B)
    q = np.asarray(C)-np.asarray(B)
    cos = np.dot(p,q)
    sin = la.norm(np.cross(p,q))
    return np.arctan2(sin,cos)

def dihedralAngle(A,B,C,D):
    AB = normalize(np.asarray(B)-np.asarray(A))
    BC = normalize(np.asarray(C)-np.asarray(B))
    CD = normalize(np.asarray(D)-np.asarray(C))
    ABxBC = np.cross(AB,BC)
    BCxCD = np.cross(BC,CD)
    if np.dot(np.cross(CD,AB),BC) > 0:
        return v_angle(ABxBC,BCxCD)
    else:
        return -v_angle(ABxBC,BCxCD)

def calcInternalCoords(p,coord):
    """
    Calculates internal coordinates

    Args:
      coord(double[]):  coordinates of the current (=last) node given as a list
      p(Node[]):        list of predecessors (e.g. [...,grandparent,parent])

    Returns:
      dist(double):     distance to the parent node (bond length)
      phi(double):      bond angle between this, this.parent and this.parent.parent
      theta(double):    dihedral angle between this, this.parent, this.parent.parent
                        and this.parent.parent.parent*

    *..."this.parent" is only pseudo syntax, it's not actually implemented
    """
    dist = phi = theta = None
    if p[-1]:
        dist = np.linalg.norm(np.asarray(p[-1].coord) - np.asarray(coord))
        if p[-2]:
            phi = angle(p[-2].coord,p[-1].coord,coord)
            if p[-3]:
                theta = dihedralAngle(p[-3].coord,p[-2].coord,p[-1].coord,coord)

    return [dist,phi,theta]
