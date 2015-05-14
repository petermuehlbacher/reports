import lennardjonespotential
import buildMolecule

def load_atoms(pdb_file):
    f = open(pdb_file,'r')
    pdb = f.readlines()
    f.close()

    lastresseqnr = 0
    lastbranch = 0
    lastremoteness = "A"

    # loop through every atom record
    for l in pdb:
        # ignore hydrogen as it is not specified consistently across PDB files
        if l[0:6] == "ATOM  " and not l[76:77] == " H":
            # TODO:
            # check to see if this is a second conformation of the previous atom
            # if len(atoms) != 0:
            #    if atoms[-1][17:26] == l[17:26]:
            #        continue

            atomname   = l[12:15] # atom name         e.g. " CG1"," CA ", " O  "
            element    = l[12:13] # chemical element  e.g. " C","NA"

            remoteness = l[14:14] # remoteness indicator where A<B<G<D<E<Z<H
            branch     = l[15:15] # digit designating the branch direction;
                                  # left blank if the sidechain is unbranched
            resseqnr   = l[22:26] # residue sequence number
            x          = l[30:37]
            y          = l[38:45]
            z          = l[46:53]
            coord      = [float(x),float(y),float(z)]
            branch     = int(float(branch))

            if not resseqnr == lastresseqnr: # new residue
                if not lastC: # first N in this file
                    tree  = Node(coord,None,None,None,[])
                    lastN = tree
                else: # new residue to append to lastC
                    dist  = np.linalg.norm(lastC.coord - coord)
                    phi   = angle(lastN - lastC, lastCA - lastC)
                    theta = dihedralAngle(lastN.coord,lastCA.coord,lastC.coord,coord)
                    lastC.children[0] = Node(coord,dist,phi,theta,[])
                # if we are at a new residue, start by adding this N as a child
                # to the last C
            lastresseqnr = resseqnr

            if atomname == " CA ":
                dist  = np.linalg.norm(lastN.coord - coord)
                # if there already was a C then that's not the first residue
                if lastC:
                    phi    = angle(lastC.coord,lastN.coord,coord)
                    theta  = dihedralAngle(lastCA.coord,lastC.coord,lastN.coord,coord)
                    lastN.children.append(Node(coord,dist,phi,theta,[]))
                else:
                    lastN.children.append(Node(coord,dist,None,None,[]))
                lastCA = lastN.children[-1]

                # reset paths
                if lastC:
                    paths = [[lastC,lastN,lastCA]]
                else:
                    paths = [[lastN,lastCA]]
                lastadded = lastCA
            elif atomname == " C  ":
                dist  = np.linalg.norm(lastCA.coord - coord)
                phi   = angle(lastN.coord,lastCA.coord,coord)
                if lastC:
                    theta = dihedralAngle(lastC.coord,lastN.coord,lastCA.coord,coord)
                    lastCA.children.append(Node(coord,dist,phi,theta,[]))
                else:
                    lastCA.children.append(Node(coord,dist,phi,None,[]))
                lastC = lastCA.children[-1]
            elif atomname == " O  ":
                dist  = np.linalg.norm(lastC.coord - coord)
                phi   = angle(lastCA.coord,lastC.coord,coord)
                theta = dihedralAngle(lastN.coord,lastCA.coord,lastC.coord,coord)
                # first child should always be the backbone
                lastC.children[1] = Node(coord,dist,phi,theta,[])

            # if it's none of the above cases it's gotta be some rest
            else:

                # if branches have already been used, but for this atom none
                # has been given, it has to be the final one and appended to
                # the any of the last ones (wlog to the last one)
                # e.g. the CZ in (CB,CG,CD1,CD2,CE1,CE2,CZ)
                if paths[1] and branch == 0:
                    ####### CALCULATE STUFF #######
                    dist  = np.linalg.norm(lastadded.coord - coord)
                    lp    = paths[lastbranch]
                    phi   = angle(lp[-2].coord,lp[-1].coord,coord)
                    theta = dihedralAngle(lp[-3].coord,lp[-2].coord,lp[-1].coord,coord)
                    ###############################
                    lastadded.children.append(Node(coord,dist,phi,theta,[]))

                # this branch already exists --> just append it
                # e.g. the CG in (CD,CG) or the NE1 in (CD1,CD2,NE1,...)
                elif paths[branch]:
                    ####### CALCULATE STUFF #######
                    dist  = np.linalg.norm(lastadded.coord - coord)
                    p     = paths[branch]
                    phi   = angle(p[-2].coord,p[-1].coord,coord)
                    # if this is not the CG of the first atom,
                    # then there is a dihedral angle to calculate
                    if p[-3]:
                        theta = dihedralAngle(p[-3].coord,p[-2].coord,p[-1].coord,coord)
                    else:
                        theta = None
                    ###############################
                    paths[branch][-1].children.append(Node(coord,dist,phi,theta,[]))
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
                        # take the last branch without the most recent element
                        # e.g. (in the example above): take path 1 and cut off
                        # the CD1 - cutting of only the last element always
                        # works as the remoteness is equal to the last node's one
                        paths[branch] = paths[lastbranch][0:-2]

                        ####### CALCULATE STUFF #######
                        p     = paths[branch]
                        dist  = np.linalg.norm(p[-1].coord - coord)
                        phi   = angle(p[-2].coord,p[-1].coord,coord)
                        theta = dihedralAngle(p[-3].coord,p[-2].coord,p[-1].coord,coord)
                        ###############################

                        # add it to the internal list
                        paths[branch].append(Node(coord,dist,phi,theta,[]))
                        # add it to the tree
                        paths[branch][-2].children.append(paths[branch][-1])
                        # update the lastadded internal variable
                        lastadded = paths[branch][-1]

                    # this is a new branch that is really extending and not just
                    # another neighbour
                    # e.g. the CD1 in (CG,CD1,CD2)
                    else:
                        # copy the parent's path
                        paths[branch] = paths[lastbranch]

                        ####### CALCULATE STUFF #######
                        dist  = np.linalg.norm(lastadded.coord - coord)
                        p     = paths[branch]
                        phi   = angle(p[-2].coord,p[-1].coord,coord)
                        theta = dihedralAngle(p[-3].coord,p[-2].coord,p[-1].coord,coord)
                        ###############################

                        # add it to the tree
                        lastadded.children.append(Node(coord,dist,phi,theta,[]))
                        # append the recently added element to the internal list
                        paths[branch].append(lastadded.children[-1])
                        # update the lastadded internal variable
                        lastadded = paths[branch][-1]

        lastbranch = branch
        lastremoteness = remoteness

    return tree

class PdbSyntaxError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def angle(A,B,C):
    p = A-B
    q = C-B
    cos = np.dot(p, q)
    sin = la.norm(np.cross(p, q))
    return np.arctan2(sin, cos)

def dihedralAngle(A,B,C,D):
    r = C-B
    p = B-A
    q = D-C
    normpr = la.norm(p)*la.norm(r)
    cos = np.dot(p,r)/normpr
    sin = la.norm(np.cross(p,r))/normpr
    return np.arctan2(sin, cos)
