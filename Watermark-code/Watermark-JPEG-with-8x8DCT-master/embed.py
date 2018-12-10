import cv2 as cv  
import numpy as np
import math
import walk	
# def embed(srcs,host_image):
# 	#xử lý ảnh mờ thủy vân thành đen trắng
# 	src=cv.imread(srcs)    
# 	src=cv.bitwise_not(src)  
# 	#chuyển ảnh thành rgb thành gray
# 	graysrc=cv.cvtColor(src,cv.COLOR_BGR2GRAY)  
# 	# cv.imshow('graysrc',graysrc)
# 	# Lọc trung bình
# 	medianblurimg=cv.medianBlur(graysrc,3)
# 	# ngưỡng màu nếu vượt quá 70 thành 0
# 	ret,bsrc = cv.threshold(graysrc,70,255,1)  
# 	cv.imshow('source',bsrc)
# 	cv.imwrite('embedfinger.jpg',bsrc,
# 		[int(cv.IMWRITE_JPEG_QUALITY),100])

# 	#Lưu trữ ảnh
# 	host=cv.imread(host_image)  
# 	#Lưu ảnh gốc từ RGB sang YUV
# 	hostyuv=cv.cvtColor(host,cv.COLOR_RGB2YUV)  
# 	#Chuyển đổi giá trị ảnh sang float 32
# 	hostf=hostyuv.astype('float32')

# 	#Quá trình nhúng thủy vân
# 	#Hoàn thành nhúng
# 	finishwm=hostf
# 	#mục tiêu hoàn thành phân chia khối 8x8
# 	#Nếu wmblocks=hostf,sau đó sửa đổi wmblocks sẽ sửa hostf,ở đây có 2 con trỏ, các biến trỏ đến hình ảnh là con trỏ
# 	wmblocks=np.zeros([hostf.shape[0],hostf.shape[1],3],np.float32)
# 	wmblocks[:,:,:]=hostf[:,:,:]
# 	# số hàng cột của ma trận được tạo thành khối 8x8
# 	part8x8rownum=int(host.shape[0]/8)
# 	part8x8colnum=int(host.shape[1]/8)
# 	print('host size = ',host.shape)
# 	# Tổng số điểm ảnh thủy vân
# 	fingernum=bsrc.shape[0]*bsrc.shape[1]
# 	print('fingernum=', fingernum)
# 	# r là số lượng điểm ảnh thủy vân được lưu trữ trong mỗi khối 8x8
# 	r=math.ceil(fingernum/(part8x8rownum*part8x8colnum))
# 	# print("r=",r)
# 	# Trong một lưới đơn vị gồm các khối 8 x 8, một cặp lưới đơn vị đối xứng với tâm của chúng
# 	# Mối quan hệ kích thước của mỗi cặp (cái trước lớn hơn cái sau, cái trước nhỏ hơn cái sau) được sử dụng để ghi lại màu đen và trắng của các pixel vân tay sẽ được lưu trữ
# 	#Lấy lưới từ giữa sang phía trên bên phải để chuẩn bị cho cặp lưới
# 	xydict=walk.findpoint((3,4,-1),r)
# 	print('xydict=  ',xydict)
# 	# Trình tạo pixel vân tay
# 	fpgij=walk.fpg(bsrc)
# 	print('fpgij= ',fpgij)
# 	# Di chuyển các khối 8x8
# 	count=0
# 	flag=0
# 	for parti in range(part8x8rownum):
# 		if (flag):
# 			break
# 		for partj in range(part8x8colnum):
# 			if (flag):
# 				break
# 			#chia ảnh thành các khối 8x8 rồi DCT
# 			part8x8=cv.dct(hostf[8*parti:8*parti+8,8*partj:8*partj+8,0])
# 			# Không xem xét các khối không có kích thước 8x8
# 			if (part8x8.shape[0]<8)|(part8x8.shape[1]<8):
# 				continue
# 			# r pixel dấu vân tay trên mỗi khối 8x8dct
# 			for t in range(r):
# 				if (flag):
# 					break
# 				# Nhận các điểm pixel dấu vân tay sẽ được lưu tại thời điểm này thông qua trình tạo
# 				i,j=next(fpgij)
# 				if (i==-1&j==-1):
# 					flag=1
# 				# Các tọa độ mạng được sử dụng cho các pixel vân tay
# 				rx,ry=xydict[t]
# 				# Sửa đổi mối quan hệ kích thước giữa r1 và r2 để phản ánh điều kiện đen trắng của pixel hình mờ
# 				r1=part8x8[rx,ry]
# 				r2=part8x8[7-rx,7-ry]#Lưới đối xứng trung tâm của r1
# 				detat=abs(r1-r2)
# 				p=float(detat+0.1)# Hệ số độ sâu nhúng
# 				if bsrc[i,j]==0:# 0 màu đen, thân vân tay, được ghi với r1> r2
# 					if(r1<=r2):# khi r1<=r2
# 						part8x8[rx,ry]+=p
# 				else:# 255, được ghi bằng r1 <r2
# 					if(r1>=r2):
# 						part8x8[7-rx,7-ry]+=p
# 				if not flag:
# 					count+=1
# 			# Sau khi lưu trữ r pixel vân tay, thực hiện nghịch đảo DCT trên khối 8x8 này.
# 			finishwm[8*parti:8*parti+8,8*partj:8*partj+8,0]=cv.idct(part8x8)
# 			wmblocks[8*parti:8*parti+8,8*partj:8*partj+8,0]=finishwm[8*parti:8*parti+8,8*partj:8*partj+8,0]
# 			# & Để kết nối với dấu ngoặc đơn, cung cấp các dòng 8x8
# 			if (wmblocks.shape[0]>8*parti+7)&(wmblocks.shape[1]>8*partj+7):
# 				wmblocks[8*parti:8*parti+8,8*partj+7,0]=100
# 				wmblocks[8*parti+7,8*partj:8*partj+8,0]=100
# 	wmrgb=cv.cvtColor(finishwm.astype('uint8'),cv.COLOR_YUV2RGB) 	
# 	#Vẽ đường
# 	cv.imshow('wmblocks',cv.cvtColor(wmblocks.astype('uint8'),cv.COLOR_YUV2RGB) )
# 	print('finishwm = ',finishwm)
# 	for x in range(6):
# 		name="finishwm"+str(x)
# 		filename=name+".jpg"
# 		cv.imwrite(filename,wmrgb,[int(cv.IMWRITE_JPEG_QUALITY),100-x])
# 		img=cv.imread(filename)
# 		cv.namedWindow(name,0)	
# 		k=480
# 		cv.resizeWindow(name,k,int(k*img.shape[0]/img.shape[1]));
# 		cv.imshow(name,img)
# 		cv.waitKey(0)
	# print("countembed=",count)
def embed(srcs,host_image):
	#xử lý ảnh mờ thủy vân thành đen trắng
	src=cv.imread(srcs)    
	src=cv.bitwise_not(src)  
	#chuyển ảnh thành rgb thành gray
	graysrc=cv.cvtColor(src,cv.COLOR_BGR2GRAY)  
	# cv.imshow('graysrc',graysrc)
	# ngưỡng màu nếu vượt quá 70 thành 0
	ret,bsrc = cv.threshold(graysrc,70,255,1)  
	cv.imshow('source',bsrc)
	cv.imwrite('embedfinger.jpg',bsrc,
		[int(cv.IMWRITE_JPEG_QUALITY),100])

	#Lưu trữ ảnh
	host=cv.imread(host_image)  
	#Lưu ảnh gốc từ RGB sang YUV
	hostyuv=cv.cvtColor(host,cv.COLOR_RGB2YUV)  
	#Chuyển đổi giá trị ảnh sang float 32
	hostf=hostyuv.astype('float32')

	#Quá trình nhúng thủy vân
	#Hoàn thành nhúng
	finishwm=hostf
	#mục tiêu hoàn thành phân chia khối 8x8
	#Nếu wmblocks=hostf,sau đó sửa đổi wmblocks sẽ sửa hostf,ở đây có 2 con trỏ, các biến trỏ đến hình ảnh là con trỏ
	wmblocks=np.zeros([hostf.shape[0],hostf.shape[1],3],np.float32)
	wmblocks[:,:,:]=hostf[:,:,:]
	# số hàng cột của ma trận được tạo thành khối 8x8
	part8x8rownum=int(host.shape[0]/8)
	part8x8colnum=int(host.shape[1]/8)
	print('host size = ',host.shape)
	# Tổng số điểm ảnh thủy vân
	fingernum=bsrc.shape[0]*bsrc.shape[1]
	print('fingernum=', fingernum)
	imf = np.float32(bsrc)/255.0  # float conversion/scale
	dct_bsrc = cv.dct(imf)
	idct_bsrc = cv.idct(dct_bsrc)*255.0
	# print('bsrc = ',bsrc)
	# print('dct =' ,dct_bsrc)
	# print('idct = ',idct_bsrc)
	x,y = 0,0
	beta = 0.01
	for parti in range(part8x8rownum):
		for partj in range(part8x8colnum):
			part8x8=cv.dct(hostf[8*parti:8*parti+8,8*partj:8*partj+8,0])
			if (part8x8.shape[0]<8)|(part8x8.shape[1]<8):
				continue
			#-------- bắt đầu thực hiện nhúng Watermark-----------------#
			for i in range(3):
				for j in range(3,6):
					if x == 230:
						break
					else:
						part8x8[i,j] +=beta*dct_bsrc[x,y]
						y+=1
						if y == dct_bsrc.shape[1]:
							y=0	
							x+=1			
			finishwm[8*parti:8*parti+8,8*partj:8*partj+8,0]=cv.idct(part8x8)
			wmblocks[8*parti:8*parti+8,8*partj:8*partj+8,0]=finishwm[8*parti:8*parti+8,8*partj:8*partj+8,0]
			# Sau khi lưu trữ r pixel vân tay, thực hiện nghịch đảo DCT trên khối 8x8 này.
			finishwm[8*parti:8*parti+8,8*partj:8*partj+8,0]=cv.idct(part8x8)
			wmblocks[8*parti:8*parti+8,8*partj:8*partj+8,0]=finishwm[8*parti:8*parti+8,8*partj:8*partj+8,0]
			# & Để kết nối với dấu ngoặc đơn, cung cấp các dòng 8x8
			if (wmblocks.shape[0]>8*parti+7)&(wmblocks.shape[1]>8*partj+7):
				wmblocks[8*parti:8*parti+8,8*partj+7,0]=100
				wmblocks[8*parti+7,8*partj:8*partj+8,0]=100
	wmrgb=cv.cvtColor(finishwm.astype('uint8'),cv.COLOR_YUV2RGB) 	
	cv.imshow('wmblocks',cv.cvtColor(wmblocks.astype('uint8'),cv.COLOR_YUV2RGB) )
	# print('finishwm = ',finishwm)
	for x in range(6):
		name="finishwm"+str(x)
		filename=name+".jpg"
		cv.imwrite(filename,wmrgb,[int(cv.IMWRITE_JPEG_QUALITY),100-2*x])
		img=cv.imread(filename)
		cv.namedWindow(name,0)	
		k=480
		cv.resizeWindow(name,k,int(k*img.shape[0]/img.shape[1]));
		cv.imshow(name,img)
		cv.waitKey(0)

from gooey import Gooey, GooeyParser
import argparse
@Gooey(program_name='THỦY VÂN SỐ DỰA TRÊN KỸ THUẬT DCT',)
def main():
	parser = GooeyParser(description= 'Thủy vân ảnh số dựa trên kỹ thuật DCT - Nguyễn Văn Trung - D9DTVT')
	parser.add_argument('-f', '--watermark-image',help= 'Thủy vân cần nhúng', default='fingerprint.jpg',widget='FileChooser')
	parser.add_argument('-b', '--host-image',help='Ảnh gốc', default='host.jpg',widget='FileChooser')
	args = parser.parse_args()
	watermark = args.watermark_image
	host_image = args.host_image
	embed(watermark,host_image)
if __name__ == '__main__':
	main()
