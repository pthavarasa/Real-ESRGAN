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
output2 = '/content/Real-ESRGAN/results/'
path = '/content/Real-ESRGAN/upload/'


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

def upscale_image(from_path, to_path):
    os.makedirs(to_path, exist_ok=True)
    img = cv2.imread(from_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    output, _ = upsampler.enhance(img, outscale=4)


    imgname, extension = os.path.splitext(os.path.basename(from_path))
    if img_mode == 'RGBA':  # RGBA images should be saved in png format
        extension = '.png'
    save_path = os.path.join(to_path, f'{imgname}{extension}')
    cv2.imwrite(save_path, output)

from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="/content/Real-ESRGAN/results/"), name="results")

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        iii = os.path.join("/content/Real-ESRGAN/upload/", file.filename)
        with open(iii, 'wb') as f:
            f.write(contents)
        upscale_image(iii, output2)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")