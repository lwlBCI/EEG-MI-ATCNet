
import numpy as np
a = [1, 2, 3] ; b = [4, 5, 6]; c = [7, 8, 9]

# 每一页 / 每一个二维数组内容不同：
w3 = np.array( [ [a,b], [a,c], [b,c], [a,a] ] )
print(w3)




# print("values",values)

sw_concat=w3[:,-1,:]
print(sw_concat)