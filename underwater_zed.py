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
init.depth_stabilization = True
init.coordinate_units = sl.UNIT.MILLIMETER
init.enable_right_side_measure = True

runtime = sl.RuntimeParameters()
runtime.confidence_threshold = 70
runtime.texture_confidence_threshold = 100

status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    exit()

image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()

saving = False
base = "underwater_zed_" + time.strftime("%Y%m%d_%H%M%S")
os.makedirs(base, exist_ok=True)

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

count = 0

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        img = image.get_data()
        dmap = depth.get_data()

        img = enhance(img)
        dmap = clean_depth(dmap)

        vis = dmap.copy()
        vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
        vis = vis.astype(np.uint8)

        cv2.imshow("RGB", img)
        cv2.imshow("Depth", vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            saving = not saving

        if key == ord('q'):
            break

        if saving:
            cv2.imwrite(f"{base}/rgb_{count}.png", img)
            np.save(f"{base}/depth_{count}.npy", dmap)
            np.save(f"{base}/pc_{count}.npy", point_cloud.get_data())
            count += 1

zed.close()
cv2.destroyAllWindows()
