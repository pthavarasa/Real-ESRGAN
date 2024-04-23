import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

model_name = 'RealESRGAN_x4plus'
tile = 0
tile_pad = 10
pre_pad = 0
fp32 = False
gpu_id = None
output = 'results'
path = 'inputs'

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
model_path = os.path.join('weights', model_name + '.pth')
if not os.path.isfile(model_path):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    for url in file_url:
        # model_path will be updated
        model_path = load_file_from_url(
            url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

dni_weight = None
upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)
os.makedirs(output, exist_ok=True)

img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if len(img.shape) == 3 and img.shape[2] == 4:
    img_mode = 'RGBA'
else:
    img_mode = None

output, _ = upsampler.enhance(img, outscale=4)


extension = f".{path.split('.')[-1]}"
if img_mode == 'RGBA':  # RGBA images should be saved in png format
    extension = 'png'
save_path = os.path.join(output, f'_out.{extension}')
cv2.imwrite(save_path, output)