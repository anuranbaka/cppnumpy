import numpy as np
import Mat_Debug
import Flood_Pybind as flood
import Mat_Pybind as Mat
a = np.array([[9.0,9,9,9,9,9,9,9,9],[9,0,0,0,9,0,0,0,9],[9,0,0,9,0,0,0,0,9],[9,9,9,0,0,0,9,9,9]])
print(a)
m = Mat.buildMat_double(a)
m.print()
print("")
flood.floodFill(m,(2,2),3,4)
flood.floodFill(m,(2,7),7,4)
m.print()