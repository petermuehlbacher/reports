import numpy as np
import numpy.linalg as la
import Node

# define it globally so that `buildChildren()` has access
molecule = []

def buildMolecule(S,alpha,beta,node):
    """
    Given some node instance like described in the corresponding class this
    function returns a list of coordinates (e.g. [[S_1,S_2,S_3],[a,b,c],...])

    Args:
      S(double[]):   starting point, e.g. [2,3,4]
      alpha(double): angle the whole system gets rotated about around the Z-axis
      beta(double):  angle the whole system gets rotated about in the XZ-plane
      node(Node):    a Node instance storing the molecule's data in a
                     tree-like structure

    Returns:
      molecule(double[][]): a list of triples of coordinates
    """

    A = node.coord

    # initiate the molecule
    molecule = [A]

    # usually the first atom (an N) only has one descendant (a CA), but just to
    # keep the code flexible pretend there may be more.
    for ii in node.children:             # ii  are 2nd level (e.g. CA)
        B = ii.coord
        molecule.append(B)
        for iii in ii.children:          # iii are 3rd level (e.g. C or CB)
            C = iii.coord
            molecule.append(C)
            buildChildren(iii,A,B)

    for atom in molecule:
    #    Ra = [[],[],[]]
    #    Rb = [[],[],[]]
    #
    #    atom = np.dot(Ra,atom)
    #    atom = np.dot(Rb,atom)
        atom = atom + np.asarray(S)

    return molecule


def buildChildren(parent,A,B):
    """
    A recursive method to traverse a tree-like structure and apply the NeRF
    algorithm to every child. Also adds the calculated coordinates to the
    `molecule` list.

    Args:
      parent(Node):  an instance of the point one wants to calculate the
                     childrens' absolute coordinates of
      A,B(double[]): absolute coordinates of the two previous points
                     (ancestor and ancestor of the ancestor)
    """
    C = parent.coord
    for child in parent.children:
        child.coord = nerf(A,B,C,child.dist,child.theta,child.phi,parent.dist)
        molecule.append(child.coord)

        buildChildren(child,B,C)

def nerf(A,B,C,R,theta,phi,lbc=""):
    """
    An implementation of the Natural Extension Reference Frame (NeRF) algorithm
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.83.8235&rep=rep1&type=pdf

    Args:
      A,B,C(double[]): absolute coordinates of the three previous points
                       (ancestor, ancestor of the ancestor, ...)
      R(double):       distance to the ancestor
      theta(double):   dihedral angle of the planes given by (A,B,C) and (B,C,D)
      phi(double):     bond angle of (B,C,D)
      lbc(double,optional): distance of B and C

    Returns:
      D(double[]): returns absolute coordinates given the above mentioned internal
        coordinates and the absolute coordinates of the previous points
    """
    cosphi    = math.cos(phi)
    sinphi    = math.sin(phi)
    Rcostheta = math.cos(theta)*R
    Rsintheta = math.sin(theta)*R

    D    = np.asarray([Rcostheta, Rsintheta*cosphi, Rsintheta*sinphi])
    D    = R*D

    BC   = np.asarray(C)-np.asarray(B)
    lbc  = lbc or la.norm(BC)
    bc   = BC/lbc
    AB   = np.asarray(B)-np.asarray(A)
    N    = np.cross(AB,bc)
    n    = N/la.norm(N)
    nxbc = np.cross(n,bc)

    M = np.transpose([bc,nxbc,n])
    D = np.dot(M,D)+C

    return D


# initialize the first few atoms
# molecule = [[0,0,0], [lAB,0,0], [0,0,0]]
# molecule[2].x = -lBC*math.cos(alpha) + l1
# molecule[2].y = -lBC*math.sin(alpha)
