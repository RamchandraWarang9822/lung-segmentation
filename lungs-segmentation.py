import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

# Path to the directory containing lung images
input_dir = './Images/'
output_dir = './Output/'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of all files in the input directory
image_files = [file for file in os.listdir(input_dir) if file.endswith('.nii.gz')]

# Process each lung image
for file_name in image_files:
    # Load the CT scan data
    ct_img = nib.load(os.path.join(input_dir, file_name))
    ct_data = ct_img.get_fdata()

    # Apply Otsu's thresholding to segment lungs
    threshold_value = threshold_otsu(ct_data)
    lung_mask = ct_data > threshold_value

    # Label connected components
    labeled_mask = label(lung_mask)

    # Get properties of connected components
    regions = regionprops(labeled_mask)

    # Find the largest connected component (lung area)
    lung_area = None
    max_area = 0
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            lung_area = region

    # Create a binary mask for the largest connected component (lung)
    lung_mask = np.zeros_like(labeled_mask)
    lung_mask[labeled_mask == lung_area.label] = 1

    # Overlay lung contours on the original image and save
    plt.figure(figsize=(8, 8))
    plt.imshow(ct_data, cmap='gray')
    plt.contour(lung_mask, colors='red', linewidths=2, levels=[0.5])
    plt.axis('off')

    # Save the image with lung contours
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'lung_contours_{file_name.replace(".nii.gz", ".png")}'))
    plt.close()
