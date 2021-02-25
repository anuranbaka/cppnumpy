import numpy as np
import floodPybind as flood
map1 = np.array([[9,9,9,9,9,9,9,9,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,9,0,0,0,0,9]
                ,[9,9,9,0,0,0,9,9,9]
                ,[9,0,0,0,0,9,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,9,9,9,9,9,9,9,9]])
map2 = np.array([[9,9,9,9,9,9,9,9,9]
                ,[9,0,0,0,0,0,0,0,9]
                ,[9,0,0,0,0,0,0,0,9]
                ,[9,0,5,5,5,0,0,0,9]
                ,[9,0,5,1,5,0,0,0,9]
                ,[9,0,5,5,5,0,5,5,9]
                ,[9,0,5,5,5,0,5,5,9]
                ,[9,0,0,0,0,5,0,0,9]
                ,[9,0,0,0,9,9,9,9,9]])
print("Map1 before floodFill:")
print(map1)
print("Filling top-left with 5 (connectivity 8):")
flood.floodFill(map1,[2,2],5,8)
print(map1)
print("Filling center with 7 (connectivity 4):")
flood.floodFill(map1,[4,4],7,4)
print(map1)
print("Map2 before floodFill:")
print(map2)
print("Filling with 7s from [3,2] (connectivity 4):")
flood.floodFill(map2,[3,2],7,4)
print(map2)
print("Filling with 4s from [5,6] (connectivity 8):")
flood.floodFill(map2,[5,6],4,8)
print(map2)
print("slicing out every other row, then filling 7s with 2s")
flood.floodFill(map2[::2],[3,2],2,4)
print(map2)