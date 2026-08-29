from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import struct
import requests
from fastapi.responses import JSONResponse
from datetime import datetime

app = FastAPI()

class Label():
    label=0

global_label = Label()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def hello(hello: str = "Hello world!!!"):
    return hello


gps_data = {}


@app.post("/gps")
async def gps(request: Request):
    body = await request.body()
    gps_id = int.from_bytes(body[:4], byteorder='big', signed=False)
    gps_data[gps_id] = body[4:].decode('ascii')


# @app.post("/predict")
# async def predict(request: Request):
#     body = await request.body()
#     target_url = "https://u275926-8c03-69ce0551.westb.seetacloud.com:8443/predict"
#
#     response = requests.post(target_url, data=body)
#     print(response)
#
#     return Response(status_code=200)


@app.post("/image")
async def predict(request: Request):

  body = await request.body()

  # image_id = int.from_bytes(body[:16],byteorder='big', signed=False)
  image_data = body[16:]

  nparr = np.frombuffer(image_data, np.uint8)
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   # gps_par=gps_data[image_id]
  print("2222222")



  path = r"E:\python_projects\img_receive"
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  file_name = f"{global_label.label}.png"
  # 生成完整的文件路径
  path = os.path.join(path, file_name)
  cv2.imwrite(path, image)

  global_label.label += 1
  # print(gps_par)

  return Response(status_code=200)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
