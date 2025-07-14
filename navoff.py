import asyncio
import websockets
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import traceback
import json

# --- Device and Model Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO('yolov8l.pt')
midas = torch.hub.load("intel-isl/MiDaS", 'DPT_Large')
midas.to(DEVICE)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = midas_transforms.dpt_transform

if DEVICE == 'cuda':
    yolo_model.half()
    midas.half()

DANGER_DEPTH_VALUE = 1.2  # meters

# --- Pipeline Functions ---
def process_full_pipeline(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img_rgb).to(DEVICE)
    if DEVICE == 'cuda':
        input_batch = input_batch.half()
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map[np.isnan(depth_map) | np.isinf(depth_map)] = 0
    yolo_results = yolo_model(frame, verbose=False, imgsz=320)[0]
    frame_width = frame.shape[1]
    return get_zone_based_navigation(depth_map, yolo_results, frame_width)

def get_zone_based_navigation(depth_map, yolo_results, frame_width):
    zones = ['extreme left', 'left', 'center', 'right', 'extreme right']
    zone_indices = [0, frame_width//5, 2*frame_width//5, 3*frame_width//5, 4*frame_width//5, frame_width]
    zone_status = {}
    zone_objects = {z: [] for z in zones}

    for i in range(5):
        zone = depth_map[:, zone_indices[i]:zone_indices[i+1]]
        avg_depth = np.mean(zone)
        zone_status[zones[i]] = 'blocked' if avg_depth < DANGER_DEPTH_VALUE else 'clear'

    for box in yolo_results.boxes:
        x_center = int(box.xywh[0][0].item())
        obj_class = int(box.cls[0])
        obj_name = yolo_model.names[obj_class] if obj_class < len(yolo_model.names) else "object"
        for i in range(5):
            if zone_indices[i] <= x_center < zone_indices[i+1]:
                zone_objects[zones[i]].append(obj_name)
                break

    if zone_objects['center']:
        obj = zone_objects['center'][0]
        if zone_status['center'] == 'blocked':
            return f"{obj} ahead, proceed with caution."
        else:
            return f"{obj} ahead, you can walk straight, but proceed with caution."
    if zone_status['center'] == 'blocked':
        return "Obstacle ahead, proceed with caution."
    if zone_objects['left']:
        obj = zone_objects['left'][0]
        if zone_status['left'] == 'blocked':
            return f"{obj} on your left, proceed with caution."
        else:
            return f"{obj} on your left, you can go straight, but proceed with caution."
    if zone_objects['right']:
        obj = zone_objects['right'][0]
        if zone_status['right'] == 'blocked':
            return f"{obj} on your right, proceed with caution."
        else:
            return f"{obj} on your right, you can go straight, but proceed with caution."
    if zone_status['center'] == 'clear':
        return "You can walk straight."
    elif zone_status['left'] == 'clear':
        return "You can go left."
    elif zone_status['right'] == 'clear':
        return "You can go right."
    elif zone_status['extreme left'] == 'clear':
        return "You can go extreme left."
    elif zone_status['extreme right'] == 'clear':
        return "You can go extreme right."
    else:
        return "No clear path detected, proceed with caution."

async def process_image_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image data"}
    try:
        advice = process_full_pipeline(frame)
        return {"advice": advice}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# --- WebSocket Handler ---
async def handler(websocket):
    print("Client connected.")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                result = await process_image_bytes(message)
                await websocket.send(json.dumps(result))
            else:
                await websocket.send(json.dumps({"error": "Send image as bytes"}))
    except Exception as e:
        print(f"[WebSocket Error]: {e}")
    finally:
        print("Client disconnected.")

# --- Server Startup ---
async def main():
    print("Starting server on ws://0.0.0.0:8765 ...")
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=8*1024*1024):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
