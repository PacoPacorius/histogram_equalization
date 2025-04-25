import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import cv2  # For loading images in the demo

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
            modified_image[y, x] = modification_transform.get(intensity)
    
    return modified_image

def _calculate_cdf(hist: Dict) -> Dict:
    """
    Calculate the cumulative distribution function of a histogram.
    """
    cdf = {}
    cumulative = 0
    total = sum(hist.values())
    
    for i in range(256):
        cumulative += hist.get(i, 0)
        cdf[i] = cumulative / total if total > 0 else 0
    
    return cdf


def _greedy_histogram_matching_bin_filling(input_hist: Dict, ref_hist: Dict) -> Dict:
    """
    Greedy algorithm for histogram matching using bin filling approach.
    
    Parameters:
    input_hist: Input histogram
    ref_hist: Reference histogram
    
    Returns:
    Transform dictionary mapping input intensities to reference intensities
    """
    # Create a mapping for each intensity level
    transform = {}
    
    # Step 1: Sort input intensity levels by value
    sorted_input_intensities = sorted(range(256), key=lambda x: x)
    
    # Step 2: Get counts for each intensity level
    input_counts = [input_hist.get(i, 0) for i in range(256)]
    ref_counts = [ref_hist.get(i, 0) for i in range(256)]
    
    # Step 3: Create cumulative histograms
    input_cumul = np.cumsum(input_counts)
    ref_cumul = np.cumsum(ref_counts)
    
    # Normalize the cumulative histograms
    input_total = input_cumul[-1]
    ref_total = ref_cumul[-1]
    
    if input_total == 0 or ref_total == 0:
        # Handle edge case of empty histograms
        return {i: i for i in range(256)}
    
    input_cumul_norm = input_cumul / input_total
    ref_cumul_norm = ref_cumul / ref_total
    
    # Step 4: Map each input intensity to reference intensity
    # based on closest cumulative histogram value
    for i in range(256):
        target_value = input_cumul_norm[i]
        closest_idx = np.argmin(np.abs(ref_cumul_norm - target_value))
        transform[i] = closest_idx
    
    return transform

def _non_greedy_histogram_matching_bin_filling(input_hist: Dict, ref_hist: Dict) -> Dict:
    """
    Non-greedy algorithm for histogram matching using bin filling approach.
    Preserves the monotonicity of the mapping.
    
    Parameters:
    input_hist: Input histogram
    ref_hist: Reference histogram
    
    Returns:
    Transform dictionary mapping input intensities to reference intensities
    """
    # Get the basic mapping from greedy approach
    transform = _greedy_histogram_matching_bin_filling(input_hist, ref_hist)
    
    # Ensure monotonicity (preserve ordering)
    last_value = 0
    for i in range(256):
        if transform[i] < last_value:
            transform[i] = last_value
        else:
            last_value = transform[i]
    
    return transform

def _post_disturbance_histogram_matching_bin_filling(input_hist: Dict, ref_hist: Dict) -> Dict:
    """
    Post-disturbance algorithm for histogram matching using bin filling approach.
    First equalizes the input histogram, then applies bin filling to match the reference.
    
    Parameters:
    input_hist: Input histogram
    ref_hist: Reference histogram
    
    Returns:
    Transform dictionary mapping input intensities to reference intensities
    """
    # Step 1: Create an equalized histogram (uniform distribution)
    total_pixels = sum(input_hist.values())
    uniform_target = total_pixels / 256
    uniform_hist = {i: int(uniform_target) for i in range(256)}
    
    # Make sure we account for rounding errors
    remaining = total_pixels - sum(uniform_hist.values())
    if remaining > 0:
        uniform_hist[128] += int(remaining)  # Add remainder to middle value
    
    # Step 2: Create equalization transform using cumulative distribution method
    input_cumul = np.cumsum([input_hist.get(i, 0) for i in range(256)])
    input_cumul_norm = input_cumul / input_cumul[-1] if input_cumul[-1] > 0 else np.zeros(256)
    
    equalized_transform = {}
    for i in range(256):
        equalized_transform[i] = int(np.round(input_cumul_norm[i] * 255))
    
    # Step 3: Create transform from uniform to reference histogram
    uniform_to_ref_transform = _greedy_histogram_matching_bin_filling(uniform_hist, ref_hist)
    
    # Step 4: Combine the two transforms
    combined_transform = {}
    for i in range(256):
        equalized_val = equalized_transform[i]
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
    
    # Apply the transformation
    return apply_hist_modification_transform(img_array, mod_transform)

def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    # we only need to call perform_hist_modification the right way
    # we need a uniform distribution
    # create a uniform histogram as the reference
    img_size = img_array.shape[0] * img_array.shape[1]
    
    uniform_target = img_size / 256
    uniform_hist = {i: uniform_target for i in range(256)}
    
    # call function
    return perform_hist_modification(img_array, uniform_hist, mode)

def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    ref_hist = calculate_hist_of_img(img_array_ref)
    
    # call function with ref image histogram
    return perform_hist_modification(img_array, ref_hist, mode)

def display_images_and_histograms(images_dict, figsize=(15, 10)):
    num_images = len(images_dict)
    fig, axes = plt.subplots(num_images, 2, figsize=figsize)
    
    for i, (title, img) in enumerate(images_dict.items()):
        # Display image
        if num_images == 1:
            ax_img = axes[0]
            ax_hist = axes[1]
        else:
            ax_img = axes[i, 0]
            ax_hist = axes[i, 1]
        
            ax_img.imshow(img, cmap='gray')
        ax_img.set_title(title)
        ax_img.axis('off')
        
        # Display histogram
        hist = calculate_hist_of_img(img)
        ax_hist.bar(hist.keys(), hist.values(), color='blue', alpha=0.7)
        ax_hist.set_title(f"Histogram of {title}")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
