import cv2 as cv  
import numpy as np  
import math
import walk
# def extract(src,dst):
# 	wmrgb=cv.imread(src)
# 	wmyuv=cv.cvtColor(wmrgb,cv.COLOR_RGB2YUV) 
# 	wmf=wmyuv.astype('float32')
# 	part8x8rownum=int(wmf.shape[0]/8)
# 	part8x8colnum=int(wmf.shape[1]/8)
# 	r=5
# 	fingernum=230
# 	extractxydict=walk.findpoint((3,4,-1),r)
# 	finishfinger=np.zeros([fingernum,fingernum,3],np.uint8)
# 	i,j=0,0
# 	count=0
# 	flag=0
# 	for parti in range(part8x8rownum):
# 		for partj in range(part8x8colnum):
# 			part8x8=cv.dct(wmf[8*parti:8*parti+8,8*partj:8*partj+8,0])
# 			if (part8x8.shape[0]<8)|(part8x8.shape[1]<8):
# 				continue
# 			for t in range(r):
# 				if (i==fingernum):
# 					break
# 				rx,ry=extractxydict[t]
# 				#观察r1和r2的大小关系,得出水印像素点黑白情况
# 				r1=part8x8[rx,ry]
# 				r2=part8x8[7-rx,7-ry]#r1的中心对称格子
# 				if r1>r2:
# 					finishfinger[i,j]=0#黑
# 				else:
# 					if r1<r2:
# 						finishfinger[i,j]=255#白
# 				j+=1
# 				if (j==fingernum):
# 					j=0
# 					i=i+1
# 				count+=1
# 	print('finishfinger=  ' ,finishfinger)
# 	# print("countextract=",count)
# 	cv.imwrite(dst,(finishfinger),[int(cv.IMWRITE_JPEG_QUALITY),100])
def extract(src,dst,host_image):
	host=cv.imread(host_image)  
	#Lưu ảnh gốc từ RGB sang YUV
	hostyuv=cv.cvtColor(host,cv.COLOR_RGB2YUV)  
	#Chuyển đổi giá trị ảnh sang float 32
	hostf=hostyuv.astype('float32')
	
	# print('host= ',hostf.shape)
	wmrgb=cv.imread(src)
	wmyuv=cv.cvtColor(wmrgb,cv.COLOR_RGB2YUV) 
	wmf=wmyuv.astype('float32')
	# print('wm=  ',wmf.shape)
	part8x8rownum=int(wmf.shape[0]/8)
	part8x8colnum=int(wmf.shape[1]/8)
	fingernum=230
	# extractxydict=walk.findpoint((3,4,-1),r)
	finishfinger=np.zeros([fingernum,fingernum],np.float32)
	x,y=0,0
	count=0
	flag=0
	beta = 0.01
	for parti in range(part8x8rownum):
		for partj in range(part8x8colnum):
			part8x8=cv.dct(wmf[8*parti:8*parti+8,8*partj:8*partj+8,0])
			# print('part8x8= ',part8x8)
			host_part8x8= cv.dct(hostf[8*parti:8*parti+8,8*partj:8*partj+8,0])
			if (part8x8.shape[0]<8)|(part8x8.shape[1]<8):
				continue
			for i in range(3):
				for j in range(3,6):
					if x == 230:
						break
					else:
						# print('part8x8[i,j] = ',part8x8[i,j])
						finishfinger[x,y] = (part8x8[i,j]- host_part8x8[i,j])/beta
						# print('part8x8= ',part8x8[i,j])
						# print('host_part8x8 = ',host_part8x8[i,j])
						# print(finishfinger[x,y])
						y+=1
						if y == 230:
							y=0	
							x+=1
	 # float conversion/scale
	# dct_bsrc = cv.dct(imf)
	# finishfinger = cv.idct(imf)*255.0
	finishfinger = (cv.idct(finishfinger))
	for i in range(finishfinger.shape[0]):
		for j in range(finishfinger.shape[1]):
			if abs(finishfinger[i,j])>128:
				finishfinger[i,j]= abs(finishfinger[i,j])
			else:
				finishfinger[i,j] = 0
	# finishfinger= abs(finishfinger)
	# finishfinger = np.uint8(dct_bsrc)*255.0 
	# print('finishfinger= ',finishfinger )
	# print('finishfinger=  ',finishfinger.shape)
	# print("countextract=",count)
	cv.imwrite(dst,(finishfinger),[int(cv.IMWRITE_JPEG_QUALITY),100])
def main():
	for x in range(6):
		host_image = 'host.jpg'
		wmname="finishwm"+str(x)
		exname=str(x)+"extractfinger"
		# extract(wmname+".jpg",exname+".jpg")# thuật toán của tác giả
		extract(wmname+".jpg",exname+".jpg",host_image)
		img=cv.imread(exname+".jpg")
		cv.namedWindow(exname,0)	
		k=240
		cv.resizeWindow(exname,k,int(k*img.shape[0]/img.shape[1]))
		cv.imshow(exname,img)
		cv.waitKey(0)
if __name__ == '__main__':
	main()