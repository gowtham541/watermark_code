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
	src='embedfinger.jpg'
	src_img ='host.jpg'
	for x in range(6):
		print("-----Image Quality = "+str(100-2*x)+'%-----')
		name=str(x)+"extractfinger"+".jpg"
		print(str(x)+". PSNR_watermark_image_after_extract = ",'%.3f'%psnr(src,name))
		name_img = "finishwm"+str(x)+".jpg"
		print(str(x),". PSNR_host_image_after_extract=",'%.3f'%psnr(src_img,name_img))
if __name__ == '__main__':
	main()