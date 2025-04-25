from typing import Dict
import numpy as np

def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool = False) -> Dict:
    # init hist dict, assume 8-bit uint image
    hist = {i: 0 for i in range(256)}  
    
    # count occurrences of each intensity value
    height, width = img_array.shape
    for y in range(height):
        for x in range(width):
            intensity = int(img_array[y, x])
            hist[intensity] += 1
    
    # normalize (optional)
    if return_normalized:
        total_pixels = height * width
        for key in hist:
            hist[key] = hist[key] / total_pixels
    
    return hist

def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict) -> np.ndarray:
    # init new image
    modified_image = np.zeros_like(img_array)
    
    # apply transform
    height, width = img_array.shape
    for y in range(height):
        for x in range(width):
            intensity = int(img_array[y, x])
            modified_image[y, x] = modification_transform.get(intensity, intensity)
    
    return modified_image
