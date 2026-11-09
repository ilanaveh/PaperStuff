"""
22/09/26
For the attention-maps figure, get blurred versions of the example face images.
"""
import sys
from pathlib import Path

# Support running this script directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Config import path_config as paths

from PIL import Image, ImageFilter
import os


img_pth = paths.RAFDB_IMAGES_DIR
save_pth = paths.BLURRED_FACES_DIR
blur_sigma = 8
# image_ids = ['0189', '0244', '1807', '1756', '1503', '1643',]
# image_ids = ['1174', '1809', '1365',]
# image_ids = ['2234', '0551', '2146', '1059', '0316', '1281', '1145', '0793', '0441', '1413', '0293', '0347']
# image_ids = ['2934']
image_ids = ['2991', '2988', '2886', '2850', '2793', '2788', ]
for id in image_ids:
    img = Image.open(os.path.join(img_pth, f'test_{id}.jpg'))
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    img.save(os.path.join(save_pth, f'test_{id}_blur{blur_sigma}.jpg'))

