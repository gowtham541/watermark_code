import numpy as np
import pywt
import os
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct 
import math
import cv2
current_path = str(os.path.dirname(__file__))  

image = 'mis1.jpg'   
watermark = 'qrcode.png' 

def convert_image(image_name, size):
    img = Image.open(image_name).resize((size, size), 1)
    img = img.convert('L')
    img.save('./dataset/' + image_name)

    # image_array = np.array(img)
    # image_array = np.float32(image_array) 
    # image_array /= 255 
    image_array = np.array(img.getdata(), dtype=np.float).reshape((size, size))
    #print (image_array[0][0])               #qrcode white color = 1.0
    #print (image_array[10][10])             #qrcode black color = 0.0  

    return image_array
    
def embed_watermark(watermark_array, orig_image):
    watermark_array_size = watermark_array[0].__len__()
    watermark_flat = watermark_array.ravel()
    ind = 0

    for x in range (0, orig_image.__len__(), 8):
        for y in range (0, orig_image.__len__(), 8):
            if ind < watermark_flat.__len__():
                subdct = orig_image[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[ind]
                orig_image[x:x+8, y:y+8] = subdct
                ind += 1 
    print((orig_image[0:2,0:2]))

    return orig_image
      


def apply_dct(image_array):
    size = image_array[0].__len__()
    all_subdct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct

    return all_subdct


def inverse_dct(all_subdct):
    size = all_subdct[0].__len__()
    all_subidct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct

    return all_subidct


def get_watermark(dct_watermarked_coeff, watermark_size):
    # watermark = [[0 for x in range(watermark_size)] for y in range(watermark_size)] 

    subwatermarks = []

    for x in range (0, dct_watermarked_coeff.__len__(), 8):
        for y in range (0, dct_watermarked_coeff.__len__(), 8):
            coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])

    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

    return watermark


def recover_watermark(image_array):


    dct_watermarked_coeff = apply_dct(image_array)
    watermark_array = get_watermark(dct_watermarked_coeff,256)

    # watermark_array *= 255;
    watermark_array =  np.uint8(watermark_array)

#Save result
    img = Image.fromarray(watermark_array)
    img.save('./result/recovered_watermark.jpg')


def print_image_from_array(image_array, name):
    # image_array *= 255;
    # image_array =  np.uint8(image_array)
    image_array_copy = image_array.clip(0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)
    img.save('./result/' + name)




def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    print('MSE = ',mse)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



image_array = convert_image(image, 2048)
watermark_array = convert_image(watermark, 256)

dct_array = apply_dct(image_array)
print_image_from_array(dct_array, 'LL_after_DCT.jpg')

dct_array = embed_watermark(watermark_array, dct_array)
print_image_from_array(dct_array, 'LL_after_embeding.jpg')

coeffs_image = inverse_dct(dct_array)
print_image_from_array(coeffs_image, 'LL_after_IDCT.jpg')


# reconstruction
image_array_H=coeffs_image
print_image_from_array(image_array_H, 'image_with_watermark.jpg')

# recover images
recover_watermark(image_array = image_array_H)

#############----------RESULT------------###########################


#----------Original pic vs WM pic-------------------#######
# original_pic = cv2.imread("mis1.jpg")
# contrast_pic= cv2.imread('image_with_watermark.jpg')
# d = psnr(original_pic,contrast_pic)
# #----------Original WM vs Recovered WM---------------------------------------######-
# foo = Image.open('recovered_watermark.jpg')
# foo = foo.resize((128,128),Image.ANTIALIAS)
# foo.save('recovered_watermark_foo.jpg',quality=95)
# original_wm = cv2.imread('qrcode.png')
# contrast_wm= cv2.imread('recovered_watermark_foo.jpg')
# d1 = psnr(original_wm,contrast_wm)
# #--------------------------------------------
# print(d,d1)

