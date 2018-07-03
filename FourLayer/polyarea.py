import sys
import numpy as np

def simpoly(x,y):
    """
    A function that calculates the area of a 2-D simple polygon (no matter concave or convex)
    Must name the vertices in sequence (i.e., clockwise or counterclockwise)
    Inputs must be float type
    Formula used: http://en.wikipedia.org/wiki/Polygon#Area_and_centroid
    Definition of "simply polygon": http://en.wikipedia.org/wiki/Simple_polygon
    Input x: x-axis coordinates of vertex array
          y: y-axis coordinates of vertex array
    Output: polygon area
    """

    ind_arr = np.arange(len(x))-1  # for indexing convenience
    s = 0
    for ii in ind_arr:
      s = s + (x[ii]*y[ii+1] - x[ii+1]*y[ii])

    return abs(s)*0.5

if __name__ == "__main__":
    usage_str = 'Usage:\npython polyarea.py c 1 2 4 8 3 5 \npython polyarea.py f coord.txt\npython polyarea.py help'

    if len(sys.argv) == 1:
        print "Error: argument input needed"
        print usage_str
        exit()

    if sys.argv[1].lower() == 'c':
        # Recognize the inputs as the coordinates 
        if len(sys.argv) <= 7:
            print "Error: at least three 2-D points needed for a valid polygon"
            print "Exiting..."
            exit()
        elif np.mod(len(sys.argv[2:]), 2) != 0:
            print "Error: the number of input arguments should be even"
            print "Exiting..."
            exit()
        else:
            print "This polygon has", (len(sys.argv)-2)/2, "vertices"
        
        a = np.zeros(len(sys.argv)-2)  # the default a.dtype.name is float64

        ind = 0
        for coord in sys.argv[2:]:
            a[ind]=float(eval(coord))  # convert the input arguments to "float" type
            ind = ind+1

        b = a.reshape(-1, 2).copy()
        x = b[:,0].copy()   # get x coords
        y = b[:,1].copy()   # get y coords

        print "The area of this polygon is", simpoly(x,y)

    elif sys.argv[1].lower() == 'f':
        # Get the input from a file
        # in which the 1st column is x coordinates, and the 2nd column is y coordinates
        d = np.loadtxt(sys.argv[2])
        x = d[:,0]
        y = d[:,1]
        print "The area of this polygon is", simpoly(x,y)

    elif sys.argv[1].lower() == 'help':
        print usage_str
        exit()

    else:
        print "Error need input arg either c (command line) or f (file)"
        print "Exiting"
        exit()
