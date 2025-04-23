import numpy as np
import cv2
import matplotlib.pyplot as plt
from histogram_modification import (
    calculate_hist_of_img,
    perform_hist_eq,
    perform_hist_matching,
    display_images_and_histograms
)

def main():
    # Load two images
    try:
        # Try to load images - use grayscale for simplicity
        img1 = cv2.imread('input_img.jpg', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('ref_img.jpg', cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            # If images can't be loaded, create synthetic test images
            print("Could not load images. Creating synthetic test images instead.")
            img1 = create_synthetic_image1()
            img2 = create_synthetic_image2()
    except Exception as e:
        print(f"Error loading images: {e}")
        print("Creating synthetic test images instead.")
        img1 = create_synthetic_image1()
        img2 = create_synthetic_image2()
    
    # Display original images
    display_images_and_histograms({
        "Original Image 1": img1,
        "Original Image 2": img2
    })
    
    # Part 1: Histogram Equalization
    print("Performing histogram equalization...")
    
    # Perform histogram equalization using all three modes
    eq_greedy = perform_hist_eq(img1, "greedy")
    eq_non_greedy = perform_hist_eq(img1, "non-greedy")
    eq_post_disturbance = perform_hist_eq(img1, "post-disturbance")
    
    # Display equalized images
    display_images_and_histograms({
        "Original Image": img1,
        "Equalized (Greedy)": eq_greedy,
        "Equalized (Non-Greedy)": eq_non_greedy,
        "Equalized (Post-Disturbance)": eq_post_disturbance
    }, figsize=(15, 12))
    
    # Part 2: Histogram Matching
    print("Performing histogram matching...")
    
    # Perform histogram matching using all three modes
    match_greedy = perform_hist_matching(img1, img2, "greedy")
    match_non_greedy = perform_hist_matching(img1, img2, "non-greedy")
    match_post_disturbance = perform_hist_matching(img1, img2, "post-disturbance")
    
    # Display matched images
    display_images_and_histograms({
        "Source Image": img1,
        "Reference Image": img2,
        "Matched (Greedy)": match_greedy,
        "Matched (Non-Greedy)": match_non_greedy,
        "Matched (Post-Disturbance)": match_post_disturbance
    }, figsize=(15, 15))

def create_synthetic_image1(size=(256, 256)):
    """Create a synthetic image with a bimodal histogram"""
    img = np.zeros(size, dtype=np.uint8)
    
    # Create a gradient from dark to light
    for i in range(size[0]):
        img[i, :] = int(i * 256 / size[0])
    
    # Add some noise
    noise = np.random.normal(0, 10, size).astype('int')
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def create_synthetic_image2(size=(256, 256)):
    """Create a synthetic image with a different histogram distribution"""
    img = np.zeros(size, dtype=np.uint8)
    
    # Create a circular pattern
    center_x, center_y = size[0] // 2, size[1] // 2
    for i in range(size[0]):
        for j in range(size[1]):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            img[i, j] = int(np.clip(255 - dist, 0, 255))
    
    # Add some noise
    noise = np.random.normal(0, 15, size).astype('int')
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

if __name__ == "__main__":
    main()
