import numpy as np
import numpy.linalg as la
import lennardjonespotential
import buildMolecule
import sys
# need to increase maximal amount of recursions
# i.e. children.children.[..].children
# could save it another format, but then buildMolecule.py gets complicated
sys.setrecursionlimit(10000)

def load_atoms(pdb_file):
    """
    Reads a PDB file and converts the data to a tree-like structure.

    Args:
      pdb_file(String): path to the pdb file that should be parsed

    Returns:
      tree(Node):       tree-like structure representing the composition of
                        the protein that's given by pdb_file; ignores H atoms
    """
    f = open(pdb_file,'r')
    pdb = f.readlines()
    f.close()

    # initialize variables
    tree = lastC = lastCA = lastN = None
    branch = 0
    remoteness = "A"

    # loop through every atom record
    for l in pdb:
        # ignore hydrogen as it is not specified consistently across PDB files
        if l[0:6] == "ATOM  " and not l[76:77] == " H":
            # TODO:
            # check to see if this is a second conformation of the previous atom
            # if len(atoms) != 0:
            #    if atoms[-1][17:26] == l[17:26]:
            #        continue

            # atom name, e.g. " CG1"," CA ", " O  "
            atomname   = l[12:16]
            # remoteness indicator where A<B<G<D<E<Z<H
            remoteness = l[14]
            # digit designating the branch direction;
            # left blank if the sidechain is unbranched
            branch     = int(l[15].replace(" ","0"))
            x          = l[30:37]
            y          = l[38:45]
            z          = l[46:53]
            coord      = [float(x),float(y),float(z)]

            #print("atomname: "+atomname)

            # new residue
            if atomname == " N  ":

                # first N in this file
                if not lastC:
                    tree  = Node(atomname,coord,None,None,None,[])
                    lastN = tree

                # new residue to append to lastC
                else:
                    dist,phi,theta = calcInternalCoords([lastN,lastCA,lastC],coord)
                    lastC.children.append(Node(atomname,coord,dist,phi,theta,[]))
                    lastN = lastC.children[-1]

            elif atomname == " CA ":
                dist,phi,theta = calcInternalCoords([lastCA,lastC,lastN],coord)
                lastN.children.append(Node(atomname,coord,dist,phi,theta,[]))
                lastCA = lastN.children[-1]

                # reset variables concerning the rest R
                paths = [[lastC,lastN,lastCA]]
                lastadded = lastCA
                lastbranch = 0
                lastremoteness = "A"

            elif atomname == " C  ":
                dist,phi,theta = calcInternalCoords([lastC,lastN,lastCA],coord)
                lastCA.children.append(Node(atomname,coord,dist,phi,theta,[]))
                lastC = lastCA.children[-1]
            elif atomname == " O  ":
                dist,phi,theta = calcInternalCoords([lastN,lastCA,lastC],coord)
                lastC.children.append(Node(atomname,coord,dist,phi,theta,[]))

            # if it's none of the above cases it's gotta be some rest R
            else:

                # if branches have already been used, but for this atom none
                # has been given, it has to be the final one and appended to
                # the any of the last ones (wlog to the last one)
                # e.g. the CZ in (CB,CG,CD1,CD2,CE1,CE2,CZ)
                if len(paths)>1 and branch == 0:
                    dist,phi,theta = calcInternalCoords(paths[lastbranch],coord)
                    lastadded.children.append(Node(atomname,coord,dist,phi,theta,[]))

                # this branch already exists --> just append it
                # e.g. the CG in (CD,CG) or the NE1 in (CD1,CD2,NE1,...)
                elif len(paths) > branch:
                    dist,phi,theta = calcInternalCoords(paths[branch],coord)
                    paths[branch][-1].children.append(Node(atomname,coord,dist,phi,theta,[]))
                    # set the current path's last element to the children
                    # we just appended by appending it
                    paths[branch].append(paths[branch][-1].children[-1])
                    lastadded = paths[branch][-1]

                # first occurence of this branch
                else:

                    # if it is as far away as the last one, append it to the
                    # parent; i.e. it is just another branch of the same parent
                    # as the last one
                    # e.g. the CD2 in (CG,CD1,CD2)
                    if remoteness == lastremoteness:
                        # expand the list by one element
                        paths.append(None)
                        # take the last branch without the most recent element
                        # e.g. (in the example above): take path 1 and cut off
                        # the CD1 - cutting of only the last element always
                        # works as the remoteness is equal to the last node's one
                        paths[branch] = paths[lastbranch][0:-1]
                        dist,phi,theta = calcInternalCoords(paths[branch],coord)

                        # add it to the internal list
                        paths[branch].append(Node(atomname,coord,dist,phi,theta,[]))
                        # add it to the tree
                        paths[branch][-2].children.append(paths[branch][-1])
                        # update the lastadded internal variable
                        lastadded = paths[branch][-1]

                    # this is a new branch that is really extending and not just
                    # another neighbour
                    # e.g. the CD1 in (CG,CD1,CD2)
                    else:
                        # expand the list by one element
                        paths.append(None)
                        # copy the parent's path
                        paths[branch] = paths[lastbranch]
                        dist,phi,theta = calcInternalCoords(paths[branch],coord)
                        # add it to the tree
                        lastadded.children.append(Node(atomname,coord,dist,phi,theta,[]))
                        # append the recently added element to the internal list
                        paths[branch].append(lastadded.children[-1])
                        # update the lastadded internal variable
                        lastadded = paths[branch][-1]

            #print(tree)
            #print("---")

        lastbranch     = branch
        lastremoteness = remoteness

    return tree

class PdbSyntaxError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def angle(A,B,C):
    p = np.asarray(A)-np.asarray(B)
    q = np.asarray(C)-np.asarray(B)
    cos = np.dot(p,q)
    sin = la.norm(np.cross(p,q))
    return np.arctan2(sin,cos)

def dihedralAngle(A,B,C,D):
    r = np.asarray(C)-np.asarray(B)
    p = np.asarray(B)-np.asarray(A)
    q = np.asarray(D)-np.asarray(C)
    normpr = la.norm(p)*la.norm(r)
    cos = np.dot(p,r)/normpr
    sin = la.norm(np.cross(p,r))/normpr
    return np.arctan2(sin, cos)

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
