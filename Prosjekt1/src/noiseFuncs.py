import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter

from src import *

def get_expected_mean_noise(exposure):
    # Directory containing the dark images
    directory = 'Data/Average/Noise/'

    # Initialize lists to store exposure times and noise values
    exposure_times = []
    noise_values = []

    # List all .txt files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # Extract the exposure time from the filename
            # Example filename: Dark_Image_E100_avg.txt
            exposure_time_str = filename.split('_')[2].replace('E', '').replace('avg.txt', '')
            exposure_time = int(exposure_time_str)
            exposure_times.append(exposure_time)
            
            # Read the image data from the file
            file_path = os.path.join(directory, filename)
            image_data = np.loadtxt(file_path, delimiter=',')
            
            # Calculate the mean noise (mean pixel value)
            mean_noise = np.mean(image_data)
            noise_values.append(mean_noise)

    # Convert lists to numpy arrays for plotting
    exposure_times = np.array(exposure_times)
    noise_values = np.array(noise_values)

    # Sort the data by exposure times
    sorted_indices = np.argsort(exposure_times)
    exposure_times = exposure_times[sorted_indices]
    noise_values = noise_values[sorted_indices]

    polynomial_coefficients = np.polyfit(exposure_times, noise_values, 2)
    polynomial_function = np.poly1d(polynomial_coefficients)
    
    # Estimate the expected noise for the given exposure time
    expected_noise = polynomial_function(exposure)

    return expected_noise

def dark_image_noise(exposure):
    directory = 'Data/Average/Noise/'
    file = 'Dark_Image_E' + str(exposure) + '_avg.txt'
    im = np.loadtxt(directory + file, delimiter=',')
    return im

def adjust_extreme_pixels(image, neighbor_threshold_factor=2):
    """
    Adjust pixels that are significantly higher or lower than their neighboring pixels to the local median.

    Args:
        image (np.ndarray): The input image.
        neighbor_threshold_factor (float): The factor by which a pixel must differ from the average of its neighbors to be considered overexposed or underexposed.

    Returns:
        np.ndarray: The image with overexposed or underexposed pixels set to the local median.
    """
    # Compute the local median values using a median filter
    local_median = median_filter(image, size=4)
    
    # Identify overexposed and underexposed pixels by comparing to local median values
    overexposed_mask = image > local_median * neighbor_threshold_factor
    underexposed_mask = image < local_median / neighbor_threshold_factor
    
    # Set overexposed and underexposed pixels to the local median
    image[overexposed_mask | underexposed_mask] = local_median[overexposed_mask | underexposed_mask]
    
    return image




def remove_noise(image, exposure, neighbor_threshold_factor=2):
    """
    Remove noise from an image by subtracting the dark image (noise) and ensuring no values fall below 0.
    Additionally, set pixels that are significantly higher or lower than their neighboring pixels to the local median.

    Args:
        image (np.ndarray): The input image from which noise is to be removed.
        exposure (float): The exposure time used to capture the image.
        neighbor_threshold_factor (float): The factor by which a pixel must differ from the average of its neighbors to be considered overexposed or underexposed.

    Returns:
        np.ndarray: The denoised image with no values below 0 and overexposed or underexposed pixels set to the local median.
    """
    try:
        # Try to subtract the dark image noise
        denoised_image = image - dark_image_noise(exposure)
    except Exception as e:
        print(f"Error occurred: {e}. Using expected mean noise instead.")
        # If an error occurs, use the expected mean noise
        denoised_image = image - get_expected_mean_noise(exposure)
    
    # Ensure no values fall below 0
    denoised_image = np.maximum(denoised_image, 0)
    
    # Adjust overexposed and underexposed pixels
    denoised_image = adjust_extreme_pixels(denoised_image, neighbor_threshold_factor)


    return denoised_image