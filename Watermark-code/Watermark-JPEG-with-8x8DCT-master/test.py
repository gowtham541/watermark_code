import cv2 as cv
import numpy as np
from scipy.fftpack import fft, dct
x,y= 0,0
for i in range (0,1000):
    for j in range (0,1000):
        for i in range(3):
            for j in range(3,6):
                if x == 230:
                    break
                else:
                    # print(x,y)
                    # part8x8[i,j] +=beta*dct_bsrc[x,y]
                    y+=1
                    if y == 230:
                        y=0	
                        x+=1
# matrix = np.random.randint(0,255,size=(8,8))
matrix = np.array([[154,123,123,123,123,123,123,136],[192,180,136,154,154,154,136,110],[254,198,154,154,180,154,123,123],[239,180,136,180,180,166,123,123],[180,154,136,167,166,149,136,136],[128,136,123,136,154,180,198,154],[123,105,110,149,136,136,180,166],[110,136,123,123,123,136,154,136]])
matrix_32 = matrix.astype('float32')/255.0
matrix_dct = cv.dct(matrix_32)*255.0
print('matrix = ' , matrix)
# print('matrix32 = ',matrix_32)
print('matrix_dct32 = ',matrix_dct)
print('matrix_idct32= ', cv.idct(matrix_dct))
matrix_dct_scipy = dct(matrix)
# print('matrix_ori = ',matrix)
print('matrix_dct_scipy = ',matrix_dct_scipy)
