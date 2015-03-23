import numpy

# size of the potential well
eps  = 1
# distance at which the potential reaches its minimum
rm   = 1
rm6  = rm**6
rm12 = rm6**2

# O(1)
def LJ(r2):
    """
    Lennard-Jones potential for a given distance r^2
    """
    ir2  = 1.0/r2 # (r^2)^-1
    ir6  = ir2**3 # (r^6)^-1
    ir12 = ir6**2 # (r^12)^-1

    return eps*(rm12*ir12 - 2*rm6*ir6)

# O(n^2)
def VLJ(X,Y,Z):
    """
    Lennard-Jones potential of an entire system.
    X...x coordinates of 1,...,n points
    Y...y coordinates of 1,...,n points
    Z...z coordinates of 1,...,n points
    """
    energy = 0.0

    # double loop over all particles
    for i in range(len(X)):
        for j in range(len(X)):
            if i>j:
                rij2 = (X[i]-X[j])**2 + (Y[i]-Y[j])**2 + (Z[i]-Z[j])**2
                # summation of the Lennard-Jones energy
                energy += LJ(rij2)

    return energy

def gradVLJ(X1,Y1,Z1,X2,Y2,Z2,alpha,beta,gamma):
    """
    Gradient of the Lennard-Jones potential.
    Returns \nabla U(q) where q describes the current system in the state space
    (i.e. q=(X,Y,Z,alpha,beta,gamma,phi1,phi2,...), where (X,Y,Z) is the point
    of reference for the system), (alpha,beta,gamma) are the rotations of the
    entire system around the x,y,z axis (respectively) and (phi1,phi2,...) are
    torsion angles. (Latter is still under consideration.)

    Ci...c coordinate of 1,...,n points of system i (A=X/Y/Z, i=1/2)
    Typically system 1 will store the points of the protein that's being docked
    to, while system 2 will store the points of the protein that's free to move
    around.
    """
    # not going for the array here, as I'd like to keep the possibility of
    # having a dynamic number of parameters (e.g. torsion angles)
    gradient = [0,0,0,0,0,0]

    for i in range(len(X)):
        for j in range(len(X)):
            if i>j:
                rij2 = (X[i]-X[j])**2 + (Y[i]-Y[j])**2 + (Z[i]-Z[j])**2
                irij2 = 1.0 / rij2
                irij6 = irij2*irij2*irij2
                irij12 = irij6*irij6

                # dU/dx
                gradient[0] += 1 # yet to be implemented
                #
                #
                #
                
