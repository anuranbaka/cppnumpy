import numpy as np
import Flood_Pybind as flood
import Mat_Pybind as Mat
map1 = np.array([[9.0,9,9,9,9,9,9,9,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,9,0,0,0,0,9]
                ,[9,9,9,0,0,0,9,9,9]
                ,[9,0,0,0,0,9,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,0,0,0,9,0,0,0,9]
                ,[9,9,9,9,9,9,9,9,9]])
#map2 = np.array([[9.0,9,9,9,9,9,9,9,9]
#                ,[9,0,0,0,9,1,1,1,9]
#                ,[9,0,0,0,2,0,0,1,9]
#                ,[9,0,0,2,0,0,0,1,9]
#                ,[9,9,2,0,0,0,2,9,9]
#                ,[9,1,0,0,0,2,0,0,9]
#                ,[9,1,0,0,2,0,0,0,9]
#                ,[9,1,1,1,9,0,0,0,9]
#                ,[9,9,9,9,9,9,9,9,9]])
#map3 = np.array([[0.0,0,0,0,9,0,0,0,0]
#                ,[0,0,0,0,9,1,1,1,0]
#                ,[0,0,0,0,9,0,0,1,0]
#                ,[0,0,0,0,9,0,0,1,0]
#                ,[9,9,9,9,9,9,9,9,9]
#                ,[0,1,0,0,9,0,0,0,0]
#                ,[0,1,0,0,9,0,0,0,0]
#                ,[0,1,1,1,9,0,0,0,0]
#                ,[0,0,0,0,9,0,0,0,0]])
print("Map1 before floodFill:")
print(map1)
print("Filling top-left with 5 (connectivity 8):")
flood.floodFill(map1,[2,2],5,8)
print(map1)
print("Filling center with 7 (connectivity 4):")
flood.floodFill(map1,[4,4],7,4)
print(map1)
#print("Map2 before flood fill:")
#print(map2)
#print("Map2 Custom filling center with 5 (threshold 1):")
#flood.floodFillCustom(map2,[4,4],5,fuzzyFill1,4)
#print(map2)
#print("Map2 reset, then Custom filling center with 5 (threshold 3):")
#flood.floodFill(map1,[4,4],0,4)
#flood.floodFill(map2,[4,4],5,fuzzyFill3,4)
#print(map2)
#print("Map3 before flood fill:")
#print(map3)
#print("Map3 Custom filling lower left corner with 5 (threshold 1):")
#flood.floodFill(map3,[7,2],5,fuzzyFill1,4)
#print(map3)
#print("Map3 fill with 5 on row 7 starting from column 7:")
#submat = map3.roi[7,:]
#flood.floodFill(submat,[0,6],5,4)
#print(map3)