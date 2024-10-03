import numpy as np
import scipy.signal as signal
import pandas as pd
from typing import Tuple, List

def read_bip_file(file_path, width=1936, height=1216):
    with open(file_path, 'rb') as file:
        data = file.read()
    
    # Konverter binærdata til numpy array med 8-bit unsigned integers
    raw_data = np.frombuffer(data, dtype=np.uint16)
    
    # Reshape array til 2D bilde
    image = raw_data.reshape((height, width))
    return image

def calculate_average_image(file_paths, output_path, width=1936, height=1216):
    """
    Calculates the average image from a list of .bip file paths.
    """
    images = [read_bip_file(file_path, width, height) for file_path in file_paths]
    average_image = np.mean(images, axis=0)

    image_data = np.ceil(average_image).astype(int)
    
    np.savetxt(output_path, image_data, delimiter=',', fmt='%d')


def find_consecutive_range_means(arr):
    """
    Finds the mean of consecutive ranges in an array of integers.

    Parameters:
    - arr: numpy array of integers.

    Returns:
    - means: list of floats, each representing the mean of a consecutive range.
    """
    means = []
    start = arr[0]
    current_range = [start]
    
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1] + 1:
            means.append(np.mean(current_range))
            current_range = [arr[i]]
        else:
            current_range.append(arr[i])
    
    # Append the mean of the last range
    means.append(np.mean(current_range))
    
    return means

def calculate_rms(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

def theoretical_fwhm(order=1,slit_width = (25e-6), focal_length = (30e-3), groove_spacing = (1 / (600 * 1e3)), alpha = 0):
    """
    Compute the theoretical FWHM using the slit width, the grating groove spacing,
    and the focal length of the collimator lens

    Parameters:
    slit_width (float): Slit width in meters.
    focal_length (float): Focal length of the middle objective in meters.
    groove_spacing (float): Grating groove spacing in meters.

    Returns:
    float: Theoretical FWHM in meters.
    """
    
    fwhm = (groove_spacing * np.cos(alpha))/(order * focal_length) * slit_width
    return fwhm

def find_peaks_and_fwhm(wavelengths: np.ndarray, intensities: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Locate all peaks and find the FWHM values at these peaks.
    
    Parameters:
    - wavelengths (np.ndarray): Array of wavelength values.
    - intensities (np.ndarray): Array of intensity values corresponding to the wavelengths.
    
    Returns:
    - peaks_wavelengths (list): List of wavelengths where peaks are located.
    - fwhm_values (list): List of FWHM values for each peak.
    """
    
    # Detect peaks
    peaks_indices, _ = signal.find_peaks(intensities, height=0.1*max(intensities), distance=10)
    
    # Find FWHM for each peak
    results_half = signal.peak_widths(intensities, peaks_indices, rel_height=0.5)
    fwhm_pixel_widths = results_half[0]
    
    # Convert pixel widths to wavelength widths
    fwhm_values = []
    peaks_wavelengths = []
    for i, peak_index in enumerate(peaks_indices):
        half_width = fwhm_pixel_widths[i] / 2
        min_w_pos = int(peak_index - half_width)
        max_w_pos = int(peak_index + half_width)
        
        # Ensure indices are within bounds
        min_w_pos = max(min_w_pos, 0)
        max_w_pos = min(max_w_pos, len(wavelengths) - 1)
        
        # Calculate FWHM in wavelength units
        fwhm = wavelengths[max_w_pos] - wavelengths[min_w_pos]
        fwhm_values.append(fwhm)
        peaks_wavelengths.append(wavelengths[peak_index])
    
    return peaks_wavelengths, fwhm_values