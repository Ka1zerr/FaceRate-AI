import asyncio
from main import _extract_landmarks
import cv2

image = cv2.imread('uploads/dummy.jpg') # Need an actual image here if possible, but let's just make one
if image is None:
    # Just find any jpg in the folder
    import glob
    files = glob.glob('*.jpg')
    if files:
        image = cv2.imread(files[0])
    
if image is not None:
    res = _extract_landmarks(image)
    if res:
        pts, pts_3d, box, width, height = res
        from main import NOSE_BOTTOM, UPPER_LIP_VERMILION, LOWER_LIP_VERMILION, CHIN
        phil = abs(pts_3d[NOSE_BOTTOM][1] - pts_3d[UPPER_LIP_VERMILION][1])
        chin = abs(pts_3d[LOWER_LIP_VERMILION][1] - pts_3d[CHIN][1])
        print(f"Philtrum: {phil}, Chin: {chin}, Ratio: {chin/phil}")
