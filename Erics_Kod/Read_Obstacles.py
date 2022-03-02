import numpy as np
from matplotlib import pyplot as plt
import csv

circ_obst_coords = []
rect_obst_coords = []

circ_obst_radius = []
rect_obst_width_height = []

with open('Obstacles', 'r') as filestream:
    next(filestream) # Skip first row
    for line in filestream: # Read every row
        if line is not "\n":
            currentline = line.split(',')
            if('None' not in currentline[:3]):
                circ_obst_coords.append( [float(currentline[0]), float(currentline[1])] )
                circ_obst_radius.append(float(currentline[2]))
            if('None'not in currentline[3:]):
                rect_obst_coords.append( [float(currentline[3]), float(currentline[4])] )
                rect_obst_width_height.append( [float(currentline[5]), float(currentline[6])] )


print(circ_obst_coords)
print(circ_obst_radius)

print(rect_obst_coords)
print(rect_obst_width_height)





