class Point( object ):
    def __init__( self, x, y, z, data ):
        self.x, self.y, self.z = x, y, z
        self.data = data
    def distFrom( self, x, y, z )
        return math.sqrt( (self.x-x)**2 + (self.y-y)**2 + (self.z-z)**2 )

def buildMolecule(a,b,c,distances,bondAngles,dihedralAngles):
    # initiate the first 3 atoms along the x-axis and the xy-plane
    l1 = distances[0]
    l2 = distances[1]
    alpha = bondAngles[0]
    molecule = [Point(0,0,0), Point(l1,0,0), Point(0,0,0)]
    molecule[2].x = -l2*math.cos(alpha) + l1
    molecule[2].y = -l2*math.sin(alpha)
