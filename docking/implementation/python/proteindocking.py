
# coding: utf-8

# In[2]:

# math
import autograd.numpy as np
import autograd.numpy.linalg as la
import math
# handling files
import sys, os
# automatic differentiation
from autograd import value_and_grad, grad

# minimization
from scipy.optimize import minimize
# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# the automatic differentiation library uses a recursive
# data structure for recursive functions
sys.setrecursionlimit(100000)


### Auxiliary functions

# In[3]:

def norm_squared(v):
    return np.dot(v,v)

def cross(v1,v2):
    x1 = v1[0]
    x2 = v2[0]
    y1 = v1[1]
    y2 = v2[1]
    z1 = v1[2]
    z2 = v2[2]
    x = y1*z2-z1*y2
    y =-x1*z2+z1*x2
    z = x1*y2-y1*x2
    return np.array([x,y,z])

def normalize(v):
    return v/la.norm(v)

def v_angle(v1,v2):
    return np.arccos(np.dot(v1/la.norm(v1),v2.T/la.norm(v2)))

def angle(A,B,C):
    p = np.array(A)-np.array(B)
    q = np.array(C)-np.array(B)
    cos = np.dot(p,q)
    sin = la.norm(cross(p,q))
    angle = np.arctan2(sin,cos)
    return min(angle,math.pi-angle)

def dihedralAngle(A,B,C,D):
    AB = normalize(np.array(B)-np.array(A))
    BC = normalize(np.array(C)-np.array(B))
    CD = normalize(np.array(D)-np.array(C))
    ABxBC = cross(AB,BC)
    BCxCD = cross(BC,CD)
    if np.dot(cross(CD,AB),BC) > 0:
        return v_angle(ABxBC,BCxCD)
    else:
        return -v_angle(ABxBC,BCxCD)

def getABC(p,X,Y,Z):
    p3 = p[-3]
    p2 = p[-2]
    p1 = p[-1]
    if p3 is not None:
        A = np.array([X[p3],Y[p3],Z[p3]])
    else:
        A = None
    B = np.array([X[p2],Y[p2],Z[p2]])
    C = np.array([X[p1],Y[p1],Z[p1]])
    return [A,B,C]


## Absolute to Internal Coordinates

# In[4]:

def load_atoms(pdb_file):
    """
    Reads a PDB file and converts the data to a tree-like structure.

    Args:
      pdb_file(String): path to the pdb file that should be parsed

    Returns:
      
    """
    f = open(pdb_file,'r')
    pdb = f.readlines()
    f.close()

    # initialize variables
    tree = lastC = lastCA = lastN = None
    branch = 0
    remoteness = "A"
    
    counter = 0
    
    atomnames = []
    children  = []
    X         = []
    Y         = []
    Z         = []
    
    phis      = []
    thetas    = []
    dists     = []
    
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
            x          = float(l[30:38])
            y          = float(l[38:46])
            z          = float(l[46:54])
            
            atomnames.append(atomname)
            children.append([])
            X.append(x)
            Y.append(y)
            Z.append(z)
            
            D = np.array([x,y,z])

            # new residue
            if atomname == " N  ":

                # first N in this file
                if not lastC:
                    lastN = 0
                    A = B = C = None

                # new residue to append to lastC
                else:
                    A = np.array([X[lastN], Y[lastN], Z[lastN]])
                    B = np.array([X[lastCA],Y[lastCA],Z[lastCA]])
                    C = np.array([X[lastC], Y[lastC], Z[lastC]])

                    children[lastC].append(counter)
                    lastN = counter

            elif atomname == " CA ":
                # if this isn't the second atom in the entire protein
                if lastCA:
                    A = np.array([X[lastCA],Y[lastCA],Z[lastCA]])
                    B = np.array([X[lastC], Y[lastC], Z[lastC]])
                else:
                    A = B = None
                C = np.array([X[lastN], Y[lastN], Z[lastN]])

                children[lastN].append(counter)
                lastCA = counter

                # reset variables concerning the rest R
                paths = [[lastC,lastN,lastCA]]
                lastadded = lastCA
                lastbranch = 0
                lastremoteness = "A"

            elif atomname == " C  ":
                if lastC:
                    A = np.array([X[lastC], Y[lastC], Z[lastC]])
                else:
                    A = None
                B = np.array([X[lastN], Y[lastN], Z[lastN]])
                C = np.array([X[lastCA],Y[lastCA],Z[lastCA]])

                children[lastCA].append(counter)
                lastC = counter
            elif atomname == " O  ":
                A = np.array([X[lastN], Y[lastN], Z[lastN]])
                B = np.array([X[lastCA],Y[lastCA],Z[lastCA]])
                C = np.array([X[lastC], Y[lastC], Z[lastC]])
                
                children[lastC].append(counter)

            # if it's none of the above cases it's gotta be some rest R
            else:

                # if branches have already been used, but for this atom none
                # has been given, it has to be the final one and appended to
                # the any of the last ones (wlog to the last one)
                # e.g. the CZ in (CB,CG,CD1,CD2,CE1,CE2,CZ)
                if len(paths)>1 and branch == 0:
                    A,B,C = getABC(paths[lastbranch],X,Y,Z)
                    children[lastadded].append(counter)

                # this branch already exists --> just append it
                # e.g. the CG in (CD,CG) or the NE1 in (CD1,CD2,NE1,...)
                elif len(paths) > branch:
                    p = paths[branch]
                    A,B,C = getABC(p,X,Y,Z)
                    children[p[-1]].append(counter)
                    p.append(counter)
                    lastadded = counter

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
                        
                        p = paths[branch]
                        A,B,C = getABC(p,X,Y,Z)
                        
                        # add it to the internal list
                        p.append(counter)
                        # add it to the tree
                        children[p[-2]].append(counter)
                        # update the lastadded internal variable
                        lastadded = counter

                    # this is a new branch that is really extending and not just
                    # another neighbour
                    # e.g. the CD1 in (CG,CD1,CD2)
                    else:
                        # expand the list by one element
                        paths.append(None)
                        # copy the parent's path
                        paths[branch] = paths[lastbranch]
                        
                        p = paths[branch]
                        A,B,C = getABC(p,X,Y,Z)
                        
                        # add it to the tree
                        children[lastadded].append(counter)
                        # append the recently added element to the internal list
                        p.append(counter)
                        # update the lastadded internal variable
                        lastadded = counter

            
            dist = phi = theta = None
            if C is not None:
                dist = la.norm(C - D)
                if B is not None:
                    phi = angle(B,C,D)
                    if A is not None:
                        theta = dihedralAngle(A,B,C,D)
            thetas.append(theta)
            dists.append(dist)
            phis.append(phi)

            lastbranch     = branch
            lastremoteness = remoteness

            counter = counter+1

    return [atomnames,X,Y,Z,phis,thetas,dists,children]


# In[5]:

bigmolecule = load_atoms("test.txt")
molecule = load_atoms("test_small.txt")
atomnames,X,Y,Z,phis,thetas,dists,children = molecule


# In[6]:

def getVal(x):
    if isinstance(x, float):
        return x
    else:
        try:
            return x.value
        except:
            print(str(x)+"is neither float nor autograd object!")
            pass

def plotMolecule(X,Y,Z,children):
    Xs,Ys,Zs = [[],[],[]]
    Xe,Ye,Ze = [[],[],[]]

    for atom in range(len(X)):
        for child in children[atom]:
            Xs.append(getVal(X[atom]))
            Ys.append(getVal(Y[atom]))
            Zs.append(getVal(Z[atom]))

            Xe.append(getVal(X[child]))
            Ye.append(getVal(Y[child]))
            Ze.append(getVal(Z[child]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(Xs)):
        ax.plot([Xs[i], Xe[i]], [Ys[i],Ye[i]],zs=[Zs[i],Ze[i]])

    plt.show()
    
plotMolecule(X,Y,Z,children)


## Internal to Absolute Coordinates

### Auxiliary Functions

#### NeRF

# In[7]:

def nerf(A,B,C,R,phi,theta,lbc="",D=None):
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
      D(double[]): returns absolute coordinates given the above mentioned
        internal coordinates and the absolute coordinates of the previous points
    """
    
    cosphi   = math.cos(phi)
    sinphi   = math.sin(phi)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    D = R*np.array([cosphi, sinphi*costheta, sinphi*sintheta])
    
    AB   = B-A
    BC   = C-B
    lbc  = lbc or la.norm(BC)
    bc   = BC/lbc
    n    = normalize(cross(AB,bc))
    nxbc = cross(n,bc) # already normalized by construction
    
    M = np.transpose(np.array([bc,nxbc,n]))
    absD = np.dot(M,D)+C
    return absD


#### Recursive Algorithm to Construct Children

# In[8]:

def buildChildren(parent,A,B,atomList):
    """
    A recursive method to traverse a tree-like structure and apply the NeRF
    algorithm to every child. Also adds the calculated coordinates to the
    `atomList` list.

    Args:
      parent(Int):  an instance of the point one wants to calculate the
                     childrens' absolute coordinates of
      A,B(double[]): absolute coordinates of the two previous points
                     (ancestor and ancestor of the ancestor)
      atomList([]):  list of a molecule's atoms
      newThetas(double[]): new dihedral angles that should be used
      parentNodeChanged(Boolean): if the parent node's coordinates were
                     updated some calculations don't need to be done again
    """
    C = np.array([__X__[parent],__Y__[parent],__Z__[parent]])
    lbc = __dists__[parent]

    for child in __children__[parent]:
        theta = __thetas__[child]
        dist  = __dists__[child]
        phi   = __phis__[child]
        
        element = __atomnames__[child][1]
        
        coord = nerf(A,B,C,dist,phi,theta,lbc)
        __X__[child],__Y__[child],__Z__[child] = coord
        buildChildren(child,B,C,atomList)
        
        atomList.append([coord, element])


### Internal to Absolute Coordinates

# In[9]:

def buildMolecule(S,eAngles,atomnames,X,Y,Z,phis,thetas,dists,children):
    """
    Given some node instance like described above this
    function returns a list of coordinates (e.g. [[S_1,S_2,S_3],[a,b,c],...])

    Args:
      S(double[]):       starting point, e.g. [2,3,4]
      eAngles(double[]): Euler angles; the whole system gets rotated about
                         around the X-axis by eAngles[0], around the Y-axis
                         by eAngles[1] and around the Z-axis by eAngles[2]

    Returns:
      atomList(double[][]): a list of triples of coordinates
    """
    # super ugly temporary hack as I don't want to need to let `buildChildren()`
    # pass all those variables to itself all the time
    global __atomnames__, __X__, __Y__, __Z__, __phis__, __thetas__, __dists__, __children__
    __atomnames__ = atomnames
    __X__ = X
    __Y__ = Y
    __Z__ = Z
    __phis__ = phis
    __thetas__ = thetas
    __dists__ = dists
    __children__ = children
    
    atomList = []
    A = np.array([X[0],Y[0],Z[0]])
    B = np.array([X[1],Y[1],Z[1]])

    # initiate the molecule
    atomList.append([A, "N"])
    atomList.append([B, "C"])
    # initiate rotation matrizes
    sinEA = np.sin(eAngles)
    cosEA = np.cos(eAngles)
    Rx = np.array([[1,0,0],[0,cosEA[0],sinEA[0]],[0,-sinEA[0],cosEA[0]]])
    Ry = np.array([[cosEA[1],0,sinEA[1]],[0,1,0],[-sinEA[1],0,cosEA[1]]])
    Rz = np.array([[cosEA[2],sinEA[2],0],[-sinEA[2],cosEA[2],0],[0,0,1]])
    
    # usually the first atom (an N) only has one descendant (a CA),
    # but just to keep the code flexible pretend there may be more.
    for i in children[1]: # i are 3rd level (e.g. C or CB)
        C = np.array([X[i],Y[i],Z[i]])
        atomList.append([C, "C"])
        buildChildren(i,A,B,atomList)

    for atom in atomList:
        atom[0] = np.dot(Rx,atom[0])
        atom[0] = np.dot(Ry,atom[0])
        atom[0] = np.dot(Rz,atom[0])
        atom[0] = atom[0] + S
    
    return atomList


# In[10]:

# list for indexing purposes
elements = ["C", "N", "O"]

# parameters for different materials for the LJ potential
# just initiate it with random values
A = [
    [5,5,5,5], # C-C, C-N, C-O, N-rest
    [5,5,5,5], # N-C, N-N, N-O, C-rest
    [5,5,5,5], # O-C, O-N, O-O, O-rest
    [5,5,5,5]  # rest-C, rest-N, rest-O, rest-rest
]

# parameters for different materials for the LJ potential
# just initiate it with random values
B = [
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
]

LJargs = [A, B]


# In[11]:

def LJ(r2,el1="",el2=""):
    """
    Lennard-Jones potential for a given distance r^2
    
    Args:
      r2(double): Squared distance between two particles
                  whose LJ-potential should be calculated
      el1,el2(String): chemical element
    
    Returns:
      U(double):  Potential as a function of r2
    """
    if el1 in elements:
        el1_index = elements.index(el1)
    else:
        el1_index = len(elements)
    if el2 in elements:
        el2_index = elements.index(el2)
    else:
        el2_index = len(elements)
    
    A = LJargs[0][el1_index][el2_index]
    B = LJargs[1][el1_index][el2_index]
    
    ir2  = 1./r2 # (r^2)^-1
    ir6  = ir2*ir2*ir2 # (r^6)^-1

    # add +10 so that the potential is strictly positive
    #return (A*ir6 - B)*ir6+10
    return A*((B*B/r2)**3-1)**2 # = A*(B^12/r^12 - 2B^6/r^6 + 1) --> A~eps,B~rm^2,+1-> LJ>0

def VLJ(fixedMol,flexMol):
    """
    Calculates Lennard-Jones potential of the particles of two given systems,
    where the internal potential of fixedMol is neglected as it is constant.
    
    Args:
      fixedMol(double[][]): List of absolute coordinate triples for one molecule
                            that's assumed to be fixed.
      flexMol(double[][]):  List of absolute coordinate triples for one molecule
                            whose internal coordinates, as well as its point of
                            reference and global rotations are assumed to be 
                            variable.
    
    Returns:
      U(double):            Potential for the entire system, minus the potential 
                            of fixedMol alone (which is constant anyways)
    """
    flexMolPot = 0
    intermolecularPot = 0
    
    n = len(flexMol)
    for i in range(n):
        for j in range(i+1,n):
            a = flexMol[i]
            b = flexMol[j]
            flexMolPot = flexMolPot + LJ(norm_squared(a[0]-b[0]),a[1],b[1])
    
    for a in fixedMol:
        for b in flexMol:
            intermolecularPot = intermolecularPot + LJ(norm_squared(a[0] - b[0]),a[1],b[1])
    
    return flexMolPot + intermolecularPot


# In[12]:

def objective(params,anglesToUpdate,flexMol,fixedMol=[]):
    """
    Wrapper function for the `VLJ()` function, which is to be optimized.
    
    Args:
      params(double[]): First three values are the absolute position of 
                        flexMol's reference point (=its PDB file's first
                        atom), the following values are dihedral angles
                        of points specified by `anglesToUpdate`.
      anglesToUpdate(double[]): See `params`.
      flexMol: List as returned by `load_atoms()`.
      fixedMol(double[][]): Just as in `VLJ()`.
    
    Returns:
      U(double): Result of `VLJ()` given the parameters as in `params`.
    """
    thetas = flexMol[5]
    S = np.array(params[0:3])
    eAngles = params[3:6]
    newThetas = params[6:]
    for i in range(len(anglesToUpdate)):
        thetas[anglesToUpdate[i]] = newThetas[i]
    flexMol = buildMolecule(S,eAngles,*flexMol)
    return VLJ(fixedMol, flexMol)


##### Comparison to the central difference quotient

# In[13]:

def getAnglesToUpdate(fullThetaList):
    """
    Given some list of dihedral angles some of which may be None
    (see `load_atoms()`) this function sorts out all those that do not
    have a valid value (i.e. aren't numbers) and also returns a list
    of indices of the valid ones.
    
    For example:
       getAnglesToUpdate([None, None, 1.3,2.1])=[[1.3,2.1],[2,3]]
    
    Args:
      fullThetaList(double[]): List of dihedral angles as returned by
                               `load_atoms()`.
    
    Returns:
      thetas,anglesToUpdate(double[][]): `thetas` is `fullThetaList`
      with all the None-valued values removed, while `anglesToUpdate`
      is the list of indices (in `fullThetaList`) of the `thetas`.
    """
    thetas = []
    anglesToUpdate = []

    for i in range(len(fullThetaList)):
        theta = fullThetaList[i]
        if theta is not None:
            thetas.append(theta)
            anglesToUpdate.append(i)
    
    return [thetas,anglesToUpdate]


# In[19]:

t, atu = getAnglesToUpdate(bigmolecule[5])
t_p = t[:]
t_m = t[:]

t_p[0] = getVal(t_p[0])+0.0001
t_m[0] = getVal(t_m[0])-0.0001
params = np.array([0.0]*6+t)
params_p = np.array([0.0]*6+t_p)
params_m = np.array([0.0]*6+t_m)

vg = value_and_grad(objective)
g  = grad(objective)


# In[20]:

res_grad = g(params,atu,bigmolecule)
print "the potential is invariant under global rotations/translations,"
print "but rounding errors cause the gradient to be of magnitude e-12."
print res_grad[:6]
print "–––"
print "central difference quotient of the potential's first dihedral angle:"
print((objective(params_p,atu,bigmolecule)-objective(params_m,atu,bigmolecule))/0.0002)
print "gradient without the global position/rotation arguments"
print "(first entry corresponds to the difference quotient above):"
print res_grad[6:]


# In[22]:

print la.norm(res_grad)


##### Import training data and store them in `molecules[]`

# In[15]:

path = './training_data'
files = os.listdir(path)
txt_files = [i for i in files if i.endswith('.txt')]
molecules = []
for file_name in txt_files:
    mol = load_atoms("./training_data/"+file_name)
    molecules.append(mol)
    plotMolecule(mol[1],mol[2],mol[3],mol[-1])


# In[16]:

A_00 = 1. # ~= A_{C-C}

def vT(params):
    """
    The sum over the norm of gradients of the objective function of
    the molecules in the training set, given some parameters for the
    Lennard-Jones potential.
    This sum should be minimal if the parameters are chosen optimally
    under the assumption that the molecules in the training data are
    in a metastable state.
    
    Note that this function alters the global variable `LJargs`.
    This shortcoming should get fixed in a future version.
    
    Args:
      params(double[]): "Vectorified" version of the Lennard-Jones
      parameter matrices, with the first element (`A_00`) missing as
      it is being hold fixed as a reference value, or in other words:
      All the other values are expressed in terms of multiples of `A_00`.
      
      Alternatively, if `params` only has one single element all
      the elements of `A` are set to `A_00` (which is specified
      globally) and all the elements of `B` are set to this parameter.
    """
    output = 0
    
    global LJargs
    numOfElements = len(LJargs[0])
    numOfParams   = numOfElements*(numOfElements+1)-1 # A_00 is fixed!
    k = 0
    for i in range(numOfElements):
        for j in range(numOfElements-i):
            if len(params) == 2:
                A_ij = params[0]
                B_ij = params[1]
            elif len(params)==1:
                A_ij = A_00
                B_ij = params[0]
            else:
                if i==0 and j==0:
                    A_ij = A_00
                else:
                    A_ij = params[k]
                B_ij = params[(numOfParams-1)/2+k] # A_00 is fixed!
            k = k+1
            LJargs[0][i][j] = A_ij
            LJargs[0][j][i] = A_ij
            LJargs[1][i][j] = B_ij
            LJargs[1][j][i] = B_ij
    
    for molecule in molecules:
        thetas, anglesToUpdate = getAnglesToUpdate(molecule[5])
        
        oParams = [0.0]*6+thetas
        val,grad = vg(np.array(oParams),anglesToUpdate,molecule)
        output = output + norm_squared(grad)/val**2.
        #print grad[6:].value
    
    return output

vgT = value_and_grad(vT)
gT  = grad(vT)


# In[18]:

import datetime
print datetime.datetime.now().time() # temporary to estimate the needed time

initValue = .01

Xvalues = np.arange(initValue, 2., .1)

potX  = []
bestB = initValue
bestVal = vT(np.array([initValue]))
for i in Xvalues:
    val = vT(np.array([i]))
    potX.append(math.log10(getVal(val)))
    print datetime.datetime.now().time() # temporary to estimate the needed time
    print "B="+ str(i) + " -> vT(1.0,B)=" + str(math.log10(getVal(val)))
    if val < bestVal:
        bestVal = val
        bestB = i


# In[19]:

plt.clf()
plt.plot(Xvalues,potX)
plt.xlabel('B')
plt.ylabel('$log_{10}(\sum_{mol} V_{1,B}(mol))$')
plt.title('$V_{A,B}=Ar^{-12}-Br^{-6}+C$')
plt.show()


# In[20]:

# number of elements
k = len(LJargs[0])
A = A_00 #x0[0]
B = bestB
x0 = [A]*(k*(k+1)/2-1) + [B]*(k*(k+1)/2) # A_00 is constant!
bnds = ((.1,5.0),)*(k*(k+1)-1)

# nit:10, fun:2.7e-05
# returns some pretty random values for unused variables (like A_{N-N})
res = minimize(vT, x0, method='TNC', bounds=bnds, jac=gT ) #1tol=10**(-9.0),

# nit:18, fun:3.0e-05
#res = minimize(vT, x0, method='SLSQP', tol=10**(-9.),bounds=bnds )

# nit:10, fun:3.0e-05
# leaves most values unchanged or only slightly changed
#res = minimize(vT, x0, method='L-BFGS-B', tol=10**(-9.), bounds=bnds )

print res


# In[ ]:

A,B = [A_00,bestB]
eps   = B*B/(4*A)
sigma = (A/B)**(1/6.)
rm    = 2**(1/6.)*sigma
print(rm,eps)


# In[ ]:

r = np.arange(sigma, 3., 0.01)
plt.clf()
plt.plot(r, (A/r**12. - B/r**6.))
plt.xlabel('r')
plt.ylabel('LJ(r)')
plt.title('Lennard-Jones potential for C-C')
plt.show()


# In[ ]:

newThetas, anglesToUpdate = getAnglesToUpdate(thetas)

oParams = [0.0]*6+newThetas
v,g = vg(np.array(oParams),anglesToUpdate,molecule)
print v
print g[6:]

