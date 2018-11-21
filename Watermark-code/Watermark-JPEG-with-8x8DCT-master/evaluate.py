import cv2 as cv  
import numpy as np  
import math
import walk
def psnr(src,dst):
	im1=cv.imread(src)
	im2=cv.imread(dst) 
	#均方误差
	mse = np.mean((im2 - im1)**2)
	sqrtmse=math.sqrt(mse)
	#psnr
	psnr=20*math.log10(255/sqrtmse)
	return psnr
def main():
	src='fingerprint.jpg'
	src_img ='host.jpg'
	for x in range(6):
		name=str(x)+"extractfinger"+".jpg"
		print(x,"PSNR_watermark=",'%.3f'%psnr(src,name))
		name_img = "finishwm"+str(x)+".jpg"
		print(x,"PSNR_image=",'%.3f'%psnr(src_img,name_img))
if __name__ == '__mai__':
	main()