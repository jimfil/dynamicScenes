import laspy
import numpy as np

def readpoints(filepath):    
    las = laspy.read(filepath)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    return points

def savepoints(filepath, points):
    np.savetxt(filepath, points, delimiter=",")


if __name__ == "__main__":
    points = readpoints("2_5D4KVPDO.laz")
    savepoints("output_scene2.csv", points)