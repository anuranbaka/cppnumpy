import numpy as np
import FloodPybind as flood
import matPybind as Mat
def fuzzyFill1(a, b):
    if(a >= b - 1 and a <= b + 1):
        return True
    return False
def fuzzyFill3(a, b):
    if(a >= b - 3 and a <= b + 3):
        return True
    return False
map1 = np.array([[9,9,9,9,9,9,9,9,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,9,0,0,0,0,9]
                ,[9,9,9,0,0,0,9,9,9]
                ,[9,0,0,0,0,9,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,9,9,9,9,9,9,9,9]])
print("Map1 before floodFill:")
print(map1)
print("Filling top-left with 5 (connectivity 8):")
flood.floodFill(map1,[2,2],5,8)
print(map1)
print("Filling center with 7 (connectivity 4):")
flood.floodFill(map1,[4,4],7,4)
print(map1)