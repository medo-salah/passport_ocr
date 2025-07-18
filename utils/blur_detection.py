import cv2
import numpy as np

def is_blurry(image, threshold=100):
    img_array = np.array(image)
    # Check if image is already grayscale
    if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm
