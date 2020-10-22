import numpy as np
import Flood_Pybind as flood
import Mat_Pybind as Mat
a = np.array([[9.0,9,9,9,9,9,9,9,9],[9,0,0,0,9,0,0,0,9],[9,0,0,9,0,0,0,0,9],[9,9,9,0,0,0,9,9,9]])
print(a)
a = flood.floodFill(a,[2,2],3,4)
print("")
print(a)