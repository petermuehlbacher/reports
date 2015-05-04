class Node( object ):
    """
    Stores an atom's coordinates (internal and absolute), as well as its
    children, defining a tree like structure.

    Sample data structure:

        Node([0,0,0],None,None,None,[          <-- ~N
            Node([4,0,0],4,None,None,[         <-- ~CA
                Node([4,3,0],3,π/2,None,[      <-- ~CB
                    Node(None,3,π/4,π/3,[      <-- ~CG
                        Node(…),               <-- ~R
                        Node(…)])]),
                Node([4,0,2],2,π/2,None,[      <-- ~C
                    Node(None,3,π,π/2,[]),     <-- ~O
                    Node(None,3,π,0,[          <-- ~N
                        Node(…),               <-- ~CA > R
                        Node(…)]               <-- ~C>{O,N}
        ])…))

    Note that the top three levels already have given coordinates, so that the
    construction, given all the angles and distances is unique and works without
    further assumptions. Of course their internal coordinates have to be chosen
    so that they do not conflict with the prescribed absolute coordinates (x,y,z).
    """
    def __init__( self, coord, dist, phi, theta, children ):
        """
        Args:
          coord(int[]):     coordinates of the node given as a list (e.g. [0,0,0])
                            (usually that's what we're calculating from the
                            other arguments, also called "internal coordinates")
          dist(double):     distance to the parent node (bond length)
          phi(double):      bond angle between this, this.parent and this.parent.parent
          theta(double):    dihedral angle between this, this.parent, this.parent.parent
                            and this.parent.parent.parent*
          children(Node[]): list of other node elements

        *..."this.parent" is only pseudo syntax, it's not actually implemented
        """
        self.coord                      = coord
        self.dist, self.phi, self.theta = dist, phi, theta
        self.children                   = children
