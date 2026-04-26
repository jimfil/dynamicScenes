import laspy
import numpy as np

def readpoints(filename, year = 2016, test = False) :    
    if test:
        las = laspy.read("data/"+ str(year) + "/test/" + filename)
    else:
        las = laspy.read("data/"+ str(year) + "/val/" + filename)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    # # remove points with classification 2 (ground)
    # points = points[las.classification != 2]
    return points

def savepoints(filename,points):
    np.savetxt("data/"+filename,points, delimiter=",")


if __name__ == "__main__":
    points = readpoints("2_5D4KVPDO.laz")
    savepoints("output_scene2.csv", points)