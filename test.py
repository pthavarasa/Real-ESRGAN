import requests
import os

def upscale_from_server(orininal_image_path, upscale_image_path):
    url = 'https://ngrok-free.app/upload'
    file_name = os.path.basename(orininal_image_path)
    file = {'file': open(orininal_image_path, 'rb')}
    resp = requests.post(url=url, files=file)
    print(resp.json())

    img_data = requests.get("https://ngrok-free.app/static/"+file_name).content
    with open(upscale_image_path, 'wb') as handler:
        handler.write(img_data)

    url = 'https://ngrok-free.app/delete'
    resp = requests.post(url=url, json ={"filename":file_name})
    print(resp.json())

if __name__ == "__main__":
    upscale_from_server(r"A_closeup_view_of_a_red_and.png", r"upscale1.png")