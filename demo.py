import numpy as np
import cv2
import matplotlib.pyplot as plt
from histogram_modification import (
    calculate_hist_of_img,
    perform_hist_eq,
    perform_hist_matching,
    display_images_and_histograms
)

# load images in grayscale
img1 = cv2.imread('input_img.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('ref_img.jpg', cv2.IMREAD_GRAYSCALE)


# display original images
display_images_and_histograms({
    "Original Image 1": img1,
    "Original Image 2": img2
})

# Part 1: Histogram Equalization
print("Performing histogram equalization...")

# perform histogram equalization using all three modes
eq_greedy = perform_hist_eq(img1, "greedy")
eq_non_greedy = perform_hist_eq(img1, "non-greedy")
eq_post_disturbance = perform_hist_eq(img1, "post-disturbance")

# display equalized images
display_images_and_histograms({
    "Original Image": img1,
    "Equalized (Greedy)": eq_greedy,
    "Equalized (Non-Greedy)": eq_non_greedy,
    "Equalized (Post-Disturbance)": eq_post_disturbance
}, figsize=(15, 12))

# Part 2: Histogram Matching
print("Performing histogram matching...")

# perform histogram matching using all three modes
match_greedy = perform_hist_matching(img1, img2, "greedy")
match_non_greedy = perform_hist_matching(img1, img2, "non-greedy")
match_post_disturbance = perform_hist_matching(img1, img2, "post-disturbance")

# display matched images
display_images_and_histograms({
    "Source Image": img1,
    "Reference Image": img2,
    "Matched (Greedy)": match_greedy,
    "Matched (Non-Greedy)": match_non_greedy,
    "Matched (Post-Disturbance)": match_post_disturbance
}, figsize=(15, 15))

