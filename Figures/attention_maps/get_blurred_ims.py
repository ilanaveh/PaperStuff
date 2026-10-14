"""
22/09/26
For the attention-maps figure, get blurred versions of the example face images.
"""
from PIL import Image, ImageFilter
import os


img_pth = '/home/projects/bagon/ilanaveh/data/RAF-DB/basic/Image/aligned_224'
save_pth = 'blurred_face_ims'
blur_sigma = 8
# image_ids = ['0189', '0244', '1807', '1756', '1503', '1643',]
# image_ids = ['1174', '1809', '1365',]
image_ids = ['2234', '0551', '2146', '1059', '0316', '1281', '1145', '0793', '0441', '1413', '0293', '0347']

for id in image_ids:
    img = Image.open(os.path.join(img_pth, f'test_{id}.jpg'))
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    img.save(os.path.join(save_pth, f'test_{id}_blur{blur_sigma}.jpg'))

