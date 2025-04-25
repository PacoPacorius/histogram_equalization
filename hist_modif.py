import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import cv2  # For loading images in the demo
from hist_utils import  (
    calculate_hist_of_img,
    apply_hist_modification_transform
    )


def _greedy_histogram_matching_bin_filling(input_hist: Dict, ref_hist: Dict) -> Dict:
    # Create a list of input intensity values sorted by intensity
    input_intensities = []
    for intensity in range(256):
        count = int(round(input_hist.get(intensity, 0)))
        if count > 0:
            # Add the intensity value 'count' times
            input_intensities.extend([intensity] * count)
    
    # Create a list of reference intensity values sorted by intensity
    ref_intensities = []
    for intensity in range(256):
        count = int(round(ref_hist.get(intensity, 0)))
        if count > 0:
            # Add the intensity value 'count' times
            ref_intensities.extend([intensity] * count)
    
    # Make sure lists are the same length by duplicating or truncating
    if len(input_intensities) > len(ref_intensities):
        # If there are more input pixels, duplicate reference pixels
        ratio = len(input_intensities) / len(ref_intensities)
        ref_intensities = [ref_intensities[int(i/ratio) % len(ref_intensities)] for i in range(len(input_intensities))]
    elif len(input_intensities) < len(ref_intensities):
        # If there are more reference pixels, sample from reference
        ratio = len(ref_intensities) / len(input_intensities)
        ref_intensities = [ref_intensities[int(i*ratio)] for i in range(len(input_intensities))]
    
    # Sort both lists - input ascending and reference as-is
   # input_intensities.sort()
    
    # Create the transformation mapping
    transform = {}
    
    # Map each input intensity to a reference intensity
    for i, input_intensity in enumerate(input_intensities):
        transform[input_intensity] = ref_intensities[i]
    
    
    return transform

def _non_greedy_histogram_matching_bin_filling(input_hist: Dict, ref_hist: Dict) -> Dict:
    # Get the basic mapping from greedy approach
    transform = _greedy_histogram_matching_bin_filling(input_hist, ref_hist)
    
    # Ensure monotonicity (preserve ordering)
    last_value = 0
    for i in range(256):
        current_value = transform.get(i, i)
        if current_value < last_value:
            transform[i] = last_value
        else:
            last_value = current_value
    
    return transform

def _post_disturbance_histogram_matching_bin_filling(input_hist: Dict, ref_hist: Dict) -> Dict:
    # Total number of pixels
    total_pixels = sum(input_hist.values())

    # Step 1: Create an equalized histogram (uniform distribution)
    uniform_target = total_pixels / 256
    uniform_hist = {i: uniform_target for i in range(256)}

    # Step 2: Create equalization transform using bin filling
    equalized_transform = _greedy_histogram_matching_bin_filling(input_hist, uniform_hist)

    # Step 3: Create transform from uniform to reference histogram
    uniform_to_ref_transform = _greedy_histogram_matching_bin_filling(uniform_hist, ref_hist)

    # Step 4: Combine the two transforms
    combined_transform = {}
    for i in range(256):
        equalized_val = equalized_transform.get(i, i)
        combined_transform[i] = uniform_to_ref_transform.get(equalized_val, equalized_val)

    return combined_transform

def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:
    # Calculate the histogram of the input image
    input_hist = calculate_hist_of_img(img_array)
    
    # Create the transformation mapping based on the mode
    if mode == "greedy":
        mod_transform = _greedy_histogram_matching_bin_filling(input_hist, hist_ref)
    elif mode == "non-greedy":
        mod_transform = _non_greedy_histogram_matching_bin_filling(input_hist, hist_ref)
    elif mode == "post-disturbance":
        mod_transform = _post_disturbance_histogram_matching_bin_filling(input_hist, hist_ref)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Apply the transformation
    return apply_hist_modification_transform(img_array, mod_transform)

def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    # Create a uniform histogram as the reference
    img_size = img_array.shape[0] * img_array.shape[1]
    
    uniform_target = img_size / 256
    uniform_hist = {i: uniform_target for i in range(256)}
    
    # Call function
    return perform_hist_modification(img_array, uniform_hist, mode)

def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    # Calculate the reference histogram
    ref_hist = calculate_hist_of_img(img_array_ref)
    
    # Call function with ref image histogram
    return perform_hist_modification(img_array, ref_hist, mode)

