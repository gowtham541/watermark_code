import cv2 as cv  
import numpy as np
import math
import walk	
def embed(srcs,host_image):
	#xử lý ảnh mờ thủy vân thành đen trắng
	src=cv.imread(srcs)    
	src=cv.bitwise_not(src)  
	#chuyển ảnh thành rgb thành gray
	graysrc=cv.cvtColor(src,cv.COLOR_BGR2GRAY)  
	# cv.imshow('graysrc',graysrc)
	# Lọc trung bình
	medianblurimg=cv.medianBlur(graysrc,3)
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
	# Tổng số điểm ảnh thủy vân
	fingernum=bsrc.shape[0]*bsrc.shape[1]
	# r là số lượng điểm ảnh thủy vân được lưu trữ trong mỗi khối 8x8
	r=math.ceil(fingernum/(part8x8rownum*part8x8colnum))
	# print("r=",r)
	#在8x8块的单位格子,分别与其中心的对称的单位格子,成一对
	#每一对的大小关系(前者比后者大,前者比后者小)用来记录要存的指纹像素点的黑与白
	#从中间往右上方走格子,为产生格子对做准备
	xydict=walk.findpoint((3,4,-1),r)
	#指纹像素点生成器
	fpgij=walk.fpg(bsrc)
	#遍历8x8块
	count=0
	flag=0
	for parti in range(part8x8rownum):
		if (flag):
			break
		for partj in range(part8x8colnum):
			if (flag):
				break
			#chia ảnh thành các khối 8x8 rồi DCT
			part8x8=cv.dct(hostf[8*parti:8*parti+8,8*partj:8*partj+8,0])
			#不考虑不够8x8大小的块
			if (part8x8.shape[0]<8)|(part8x8.shape[1]<8):
				continue
			#每个8x8dct块存r个指纹像素点
			for t in range(r):
				if (flag):
					break
				#通过生成器得到此刻要存的指纹像素点
				i,j=next(fpgij)
				if (i==-1&j==-1):
					flag=1
				#指纹像素点要用的格子坐标
				rx,ry=xydict[t]
				#修改r1和r2的大小关系,来反映水印像素点黑白情况
				r1=part8x8[rx,ry]
				r2=part8x8[7-rx,7-ry]#r1的中心对称格子
				detat=abs(r1-r2)
				p=float(detat+0.1)#嵌入深度
				if bsrc[i,j]==0:#0黑的,指纹主体,用r1>r2来记录
					if(r1<=r2):#一定要让r1大于r2
						part8x8[rx,ry]+=p
				else:#255白的,用r1<r2来记录
					if(r1>=r2):
						part8x8[7-rx,7-ry]+=p
				if not flag:
					count+=1
			#存完r个指纹像素点后,对此8x8块进行逆DCT
			finishwm[8*parti:8*parti+8,8*partj:8*partj+8,0]=cv.idct(part8x8)
			wmblocks[8*parti:8*parti+8,8*partj:8*partj+8,0]=finishwm[8*parti:8*parti+8,8*partj:8*partj+8,0]
			#&要用括号连接,给8x8划线
			if (wmblocks.shape[0]>8*parti+7)&(wmblocks.shape[1]>8*partj+7):
				wmblocks[8*parti:8*parti+8,8*partj+7,0]=100
				wmblocks[8*parti+7,8*partj:8*partj+8,0]=100
	wmrgb=cv.cvtColor(finishwm.astype('uint8'),cv.COLOR_YUV2RGB) 	
	#划线图
	cv.imshow('wmblocks',cv.cvtColor(wmblocks.astype('uint8'),cv.COLOR_YUV2RGB) )
	
	for x in range(6):
		name="finishwm"+str(x)
		filename=name+".jpg"
		cv.imwrite(filename,wmrgb,[int(cv.IMWRITE_JPEG_QUALITY),100-x])
		img=cv.imread(filename)
		cv.namedWindow(name,0)	
		k=480
		cv.resizeWindow(name,k,int(k*img.shape[0]/img.shape[1]));
		cv.imshow(name,img)

	# print("countembed=",count)
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
