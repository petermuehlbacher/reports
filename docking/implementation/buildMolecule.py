import numpy as np
import numpy.linalg as la
import Node
import nerf
import geometry

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
    # TODO: implement rotation via Euler angles
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
