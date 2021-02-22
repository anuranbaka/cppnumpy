import numpy as np
import matDebug as md
map1 = np.array([1,2,3,4,5,6,7,8,9])
map2 = md.copyTest(map1[::2])
print(map1)
print(map2)
