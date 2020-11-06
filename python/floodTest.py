import numpy as np
import floodPybind as flood
map = np.array([[9,9,9,9,9,9,9,9,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,9,0,0,0,0,9]
                ,[9,9,9,0,0,0,9,9,9]
                ,[9,0,0,0,0,9,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,9,9,9,9,9,9,9,9]])
print("Map1 before floodFill:")
print(map)
print("Filling top-left with 5 (connectivity 8):")
flood.floodFill(map,[2,2],5,8)
print(map)
print("Filling center with 7 (connectivity 4):")
flood.floodFill(map,[4,4],7,4)
print(map)