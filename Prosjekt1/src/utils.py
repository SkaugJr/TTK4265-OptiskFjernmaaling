import numpy as np
import scipy as sp
from typing import List, Tuple

from src import *

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


def calc_fwhm(spectra: List[np.ndarray], wavelengths: List[np.ndarray], lines: List[int]) -> Tuple[List[List[float]], List[int]]:
    """
    Calculate the full width at half maximum (FWHM) for each peak in the given lines of the spectra.
    
    Parameters:
    - spectra (list): A list of numpy arrays representing the calibrated spectra.
    - wavelengths (list): A list of numpy arrays representing the corresponding wavelengths.
    - lines (list): A list of line indices to calculate the FWHM.
    
    Returns:
    - all_fwhm_list (list): A list of lists, where each inner list contains the FWHM values for each peak in a line.
    - skip_list (list): A list of line indices that were skipped due to an incorrect number of peaks.
    """
    
    # Open data
    all_fwhm_list: List[List[float]] = []
    skip_list: List[int] = []
    for line in lines:
        peaks_width_nm: List[float] = []
        for spectrum, wavelength in zip(spectra, wavelengths):
            smooth_line = spectrum[line]  # Consider smoothing the line.
        
            # Detect position of peaks
            max_val = max(smooth_line)
            peak_height = 0.1 * max_val  # Minimum value of peak
            distance = 18  # Minimum distance between peaks
            peaks_pos, peaks_height_dict = sp.signal.find_peaks(smooth_line, height=peak_height, distance=distance)    
            peaks_height = peaks_height_dict['peak_heights']
            
            # Find width at half maximum           
            results_half = sp.signal.peak_widths(smooth_line, peaks_pos, rel_height=0.5)  # At half maximum
            peaks_width = results_half[0]
            
            # Convert width from pixel to nm
            num_peaks = len(peaks_height)
            for peak in range(num_peaks):
                
                # Find wavelength pos of half width on each side
                half_width = peaks_width[peak] / 2
                min_w_pos = int(peaks_pos[peak] - half_width)
                max_w_pos = int(peaks_pos[peak] + half_width)
                
                # Ensure indices are within bounds
                min_w_pos = max(min_w_pos, 0)
                max_w_pos = min(max_w_pos, len(wavelength) - 1)
                
                # Convert half width pos to nm 
                min_w = wavelength[min_w_pos]
                max_w = wavelength[max_w_pos]
                
                # Width in nm
                width_nm = max_w - min_w
                peaks_width_nm.append(float(width_nm))
            
        if len(peaks_width_nm) == 15:  # Change if needed
            all_fwhm_list.append(peaks_width_nm)
        else:
            skip_list.append(line)
    
    return all_fwhm_list, skip_list