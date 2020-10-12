import numpy as np
import Flood_Pybind as flood
import Mat_Pybind as Mat
a = np.array([[9.0,9,9,9,9,9,9,9,9],[9,0,0,0,9,0,0,0,9],[9,0,0,9,0,0,0,0,9],[9,9,9,0,0,0,9,9,9]])
print(a)
m = Mat.buildMat_double(a)
m.print()
flood.floodFill(m,(2,2),5,4)
m.print()