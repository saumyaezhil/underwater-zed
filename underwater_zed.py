import pyzed.sl as sl
import cv2
import os
import time
import numpy as np

REFRACTIVE_SCALE = 1.33
MAX_DEPTH = 5000
MIN_DEPTH = 200

zed = sl.Camera()

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.camera_fps = 30
init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init.depth_stabilization = 1
init.coordinate_units = sl.UNIT.MILLIMETER
init.enable_right_side_measure = True # must be int

timestamp = time.strftime("%Y%m%d_%H%M%S")
record_path = f"underwater_record_{timestamp}.svo"

record_params = sl.RecordingParameters(
    record_path,
    sl.SVO_COMPRESSION_MODE.LOSSLESS
)

status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED")
    exit(1)

# Enable recording
err = zed.enable_recording(record_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Recording failed:", err)
    zed.close()
    exit(1)

print(f"Recording started → {record_path}")
print("Press CTRL + Q to stop recording")

runtime = sl.RuntimeParameters()
runtime.confidence_threshold = 70
runtime.texture_confidence_threshold = 100

image = sl.Mat()
depth = sl.Mat()

def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    return img

def clean_depth(d):
    d = d * REFRACTIVE_SCALE
    d[d > MAX_DEPTH] = 0
    d[d < MIN_DEPTH] = 0
    return d

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        img = enhance(image.get_data())
        dmap = clean_depth(depth.get_data())

        vis = cv2.normalize(dmap, None, 0, 255, cv2.NORM_MINMAX)
        vis = vis.astype(np.uint8)

        cv2.imshow("RGB", img)
        cv2.imshow("Depth", vis)

        key = cv2.waitKey(1)

        # CTRL + Q
        if key == 17:
            print("Stopping recording...")
            break

zed.disable_recording()
zed.close()
cv2.destroyAllWindows()

print("Recording saved successfully.")

