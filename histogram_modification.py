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
    # Get total pixel count from input histogram
    total_pixels = sum(input_hist.values())
    
    # Step 1: Create sorted list of input intensities
    input_intensities = []
    for intensity, count in input_hist.items():
        input_intensities.extend([intensity] * count)
    input_intensities.sort()
    
    # Step 2: Calculate target bin sizes from reference histogram
    target_counts = {}
    for level, count in ref_hist.items():
        target_counts[level] = count
    
    # Step 3: Create the transform mapping by filling reference bins
    transform = {}
    pixel_index = 0
    
    for ref_intensity in range(256):
        target_bin_size = target_counts.get(ref_intensity, 0)
        
        # Fill this bin with pixels
        pixels_to_assign = min(target_bin_size, total_pixels - pixel_index)
        
        # Assign all pixels in this range to the current reference intensity
        for i in range(pixel_index, pixel_index + pixels_to_assign):
            if i < len(input_intensities):
                input_intensity = input_intensities[i]
                transform[input_intensity] = ref_intensity
        
        pixel_index += pixels_to_assign
        
        # If we've assigned all pixels, we're done
        if pixel_index >= len(input_intensities):
            break
    
    # Make sure all input intensities have a mapping
    for i in range(256):
        if i not in transform:
            # Find closest assigned intensity
            closest = min(transform.keys(), key=lambda x: abs(x - i))
            transform[i] = transform[closest]
    
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
    for i in range(1, 256):
        if i in transform and i-1 in transform and transform[i] < transform[i-1]:
            transform[i] = transform[i-1]
    
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
        if i in equalized_transform:
            equalized_val = equalized_transform[i]
            combined_transform[i] = uniform_to_ref_transform.get(equalized_val, equalized_val)
        else:
            combined_transform[i] = i
    
    return combined_transform

def _greedy_histogram_matching(input_hist: Dict, ref_hist: Dict) -> Dict:
    input_cdf = _calculate_cdf(input_hist)
    ref_cdf = _calculate_cdf(ref_hist)
    
    # Create the transform mapping
    transform = {}
    for i in range(256):
        # Find the value in reference CDF that is closest to input CDF at level i
        input_cdf_val = input_cdf[i]
        min_diff = float('inf')
        best_match = i
        
        for j in range(256):
            diff = abs(ref_cdf[j] - input_cdf_val)
            if diff < min_diff:
                min_diff = diff
                best_match = j
        
        transform[i] = best_match
    
    return transform

def _non_greedy_histogram_matching(input_hist: Dict, ref_hist: Dict) -> Dict:
    """
    Non-greedy algorithm for histogram matching.
    Preserves the relative ordering of intensity values.
    """
    # Calculate cumulative distribution functions
    input_cdf = _calculate_cdf(input_hist)
    ref_cdf = _calculate_cdf(ref_hist)
    
    # Create lookup table for mapping
    transform = {}
    for i in range(256):
        # Find j such that ref_cdf[j] is closest to input_cdf[i]
        min_diff = float('inf')
        j_closest = i
        
        for j in range(256):
            if abs(input_cdf[i] - ref_cdf[j]) < min_diff:
                min_diff = abs(input_cdf[i] - ref_cdf[j])
                j_closest = j
        
        transform[i] = j_closest
    
    # Ensure monotonicity (preserve ordering)
    for i in range(1, 256):
        if transform[i] < transform[i-1]:
            transform[i] = transform[i-1]
    
    return transform

def _post_disturbance_histogram_matching(input_hist: Dict, ref_hist: Dict) -> Dict:
    """
    Post-disturbance algorithm for histogram matching.
    First equalizes the input histogram, then applies a transform to match the reference.
    """
    # First, equalize the input histogram (create a uniform distribution)
    input_cdf = _calculate_cdf(input_hist)
    equalized_transform = {i: int(round(input_cdf[i] * 255)) for i in range(256)}
    
    # Create a uniform histogram (ideally what we'd get after equalization)
    uniform_hist = {i: 1 for i in range(256)}
    
    # Now create a transform from uniform to reference
    uniform_to_ref_transform = _greedy_histogram_matching(uniform_hist, ref_hist)
    
    # Combine the two transforms
    combined_transform = {}
    for i in range(256):
        # First apply equalization, then map to reference
        equalized_val = equalized_transform[i]
        combined_transform[i] = uniform_to_ref_transform[equalized_val]
    
    return combined_transform

def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:
    """
    Modify an image's histogram to match a reference histogram.
    
    Parameters:
    img_array: Input image array
    hist_ref: Reference histogram to match
    mode: Modification mode ("greedy", "non-greedy", or "post-disturbance")
    
    Returns:
    Modified image array
    """
    # Calculate the histogram of the input image
    input_hist = calculate_hist_of_img(img_array)
    
    # Create the transformation mapping based on the mode
    if mode == "greedy":
        mod_transform = _greedy_histogram_matching(input_hist, hist_ref)
    elif mode == "non-greedy":
        mod_transform = _non_greedy_histogram_matching(input_hist, hist_ref)
    elif mode == "post-disturbance":
        mod_transform = _post_disturbance_histogram_matching(input_hist, hist_ref)
    else:
        raise ValueError("Mode must be one of 'greedy', 'non-greedy', or 'post-disturbance'")
    
    # Apply the transformation
    return apply_hist_modification_transform(img_array, mod_transform)

def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    """
    Perform histogram equalization on an image.
    
    Parameters:
    img_array: Input image array
    mode: Equalization mode ("greedy", "non-greedy", or "post-disturbance")
    
    Returns:
    Equalized image array
    """
    # For histogram equalization, we want a uniform distribution
    # Create a uniform histogram as the reference
    img_size = img_array.shape[0] * img_array.shape[1]
    if len(img_array.shape) > 2:
        img_size *= img_array.shape[2]
    
    uniform_target = img_size / 256
    uniform_hist = {i: uniform_target for i in range(256)}
    
    # Use perform_hist_modification to convert to uniform histogram
    return perform_hist_modification(img_array, uniform_hist, mode)

def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    """
    Match the histogram of an image to a reference image.
    
    Parameters:
    img_array: Input image array
    img_array_ref: Reference image array
    mode: Matching mode ("greedy", "non-greedy", or "post-disturbance")
    
    Returns:
    Processed image with histogram matched to reference
    """
    # Calculate the histogram of the reference image
    ref_hist = calculate_hist_of_img(img_array_ref)
    
    # Use perform_hist_modification to match the histogram
    return perform_hist_modification(img_array, ref_hist, mode)

def display_images_and_histograms(images_dict, figsize=(15, 10)):
    """
    Display multiple images and their histograms.
    
    Parameters:
    images_dict: Dictionary of {title: image_array}
    figsize: Figure size as (width, height)
    """
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
        
        if len(img.shape) > 2:
            ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
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
