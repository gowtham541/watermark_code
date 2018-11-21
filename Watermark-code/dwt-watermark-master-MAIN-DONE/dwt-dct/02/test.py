import nghich as n 
import numpy as np
from PIL import Image
image = 'mis1.jpg'   
watermark = 'qrcode.png' 
image_name = image
size = 256
image_array = n.convert_image(image, 2048)
watermark_array = n.convert_image(watermark, 256)
img = Image.open(image_name).resize((256, 256), 1)
img = img.convert('L')
img.save('./test/' + image_name)
image_array = np.array(img.getdata(), dtype=np.float).reshape((size, size))
image_array = np.array(img.getdata(), dtype=np.float)
watermark_array_size = watermark_array[0].__len__()
watermark_flat = watermark_array.ravel()
print(watermark_flat)
print(watermark_array)
# print(np.size(image_array))
# print(image_array)
# print(img)
# print(np.shape(watermark_array))
# print(np.shape(image_array))