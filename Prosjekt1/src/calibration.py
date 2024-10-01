import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import re

from src import *

def calibrate_wavelengths(ImageWidth, Spectral_lines, Pixel_positions, Degree, Output_file):
    """
    Calibrate pixel positions to wavelengths.

    Parameters:
    - ImageWidth: int, number of pixels.
    - Spectral_lines: list of float, spectral lines or known wavelengths in nm.
    - Pixel_positions: list of int, pixel positions corresponding to the known spectral lines.
    - Degree: int, degree of the polynomial fit (1 for linear).
    - Output_file: str, path to the output CSV file.

    Returns:
    - df: pandas DataFrame, containing pixel positions and calibrated wavelengths.
    """
    if len(Spectral_lines) != len(Pixel_positions):
        raise ValueError("The number of wavelengths must match the number of pixel positions.")
    if len(Spectral_lines) < 2:
        raise ValueError("At least two wavelengths and pixel positions are required for calibration.")

    # Fit a polynomial function to the known wavelengths at the pixel positions
    wavelength_fit = np.polyfit(Pixel_positions, Spectral_lines, Degree)
    wavelength_calibration = np.poly1d(wavelength_fit)

    # Calibrate the pixel positions to wavelengths
    calibrated_wavelengths = wavelength_calibration(np.arange(ImageWidth))

    # Create a DataFrame
    df = pd.DataFrame({
        'Pixel Position': np.arange(len(calibrated_wavelengths)),
        'Calibrated Wavelength (nm)': calibrated_wavelengths
    })

    # Save the DataFrame to a CSV file
    df.to_csv(Output_file, index=False)
    return df


def pix_to_wavelength(spectrogram, csv_file='Data/Calibrated/calibrated_wavelengths.csv', wavelength_min=None, wavelength_max=None):
    """
    Crop the spectrogram to a specific wavelength range.

    Args:
        spectrogram (np.ndarray): The spectrogram data.
        csv_file (str): Path to the CSV file containing the calibrated wavelengths.
        wavelength_min (float): Minimum wavelength to crop to.
        wavelength_max (float): Maximum wavelength to crop to.
    """
    # Read the CSV file to get the calibrated wavelengths
    df = pd.read_csv(csv_file)
    wavelengths = df.iloc[:, 1].values

    # Set the default wavelength range if not provided
    if wavelength_min is None:
        wavelength_min = 400
    if wavelength_max is None:
        wavelength_max = 800

    # Crop the wavelengths and the corresponding image data
    valid_indices = (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
    cropped_wavelengths = wavelengths[valid_indices]
    cropped_spectrogram = spectrogram[:, valid_indices]

    return cropped_spectrogram, cropped_wavelengths

def expected_irradiance(wavelengths=None):
    """
    Calculate the expected irradiance based on the calibration certificate data.

    Args:
        wavelengths (np.ndarray, optional): The wavelengths at which to interpolate the expected irradiance.
                                             If None, the function returns B_expected and W_L.

    Returns:
        If wavelengths is None:
            tuple: (B_expected, W_L)
        If wavelengths is provided:
            np.ndarray: The interpolated expected irradiance at the given wavelengths.
    """
    # Certificate data
    W_L = np.loadtxt('Data/Calibrated/calibrationCertificate200W.txt', usecols=0)  # Wavelengths in nm
    B_0 = np.loadtxt('Data/Calibrated/calibrationCertificate200W.txt', usecols=1)  # Irradiance in mW/m^2/nm
    # sigma = np.loadtxt('Data/Calibrated/calibrationCertificate200W.txt', usecols=2)  # Uncertainty in %

    R = 0.92
    r_0 = 0.5  # from source to lambertian surface
    alpha = 0  # angle between source and normal to lambert
    p = 0.98  # diffusion coefficient of lambertian surface
    B_expected = B_0 * (r_0 / R) ** 2 * p * np.cos(alpha)  # Irradiance in mW/m^2/nm

    if wavelengths is None:
        return B_expected, W_L
    else:
        interp_func = interp1d(W_L, B_expected, kind='linear', fill_value="extrapolate")
        B_interpolated = interp_func(wavelengths)
        return B_interpolated

def scaling_factor(exposure_time_ms):
    """
    Calculate the scaling factor for converting counts to irradiance.

    Args:
        exposure_time_ms (float): The exposure time in milliseconds.
    """
    # Load the scaling factor calibrated for 1 ms exposure time
    scaling_factor = np.loadtxt('Data/Calibrated/Scaling_Factor.txt', delimiter=",") # Scaling factor in mW/m^2/nm/count

    # Adjust the scaling factor for the given exposure time
    adjusted_scaling_factor = scaling_factor  / exposure_time_ms 

    return adjusted_scaling_factor

def counts_to_irradiance(spectrogram, exposure_time_ms):
    """
    Convert counts to irradiance using a scaling factor.
    
    Parameters:
    spectrogram (numpy.ndarray): The spectrogram data in counts.
    exposure_time_ms (float): The exposure time in milliseconds.
    
    Returns:
    numpy.ndarray: The spectrogram data in irradiance (mW/m^2/nm).
    """

    # Convert counts to irradiance
    new_spec = scaling_factor(exposure_time_ms) * spectrogram 

    return new_spec

def full_calibration(image, exposure_time_ms):
    """
    Perform full calibration of the spectrogram.

    Args:
        image (np.ndarray): The spectrogram data.
        exposure_time_ms (float): The exposure time in milliseconds.

    Returns:
        np.ndarray: The calibrated spectrogram data.
    """

    image = noiseFuncs.remove_noise(image, exposure_time_ms)

    # Crop the spectrogram to a specific wavelength range
    cropped_spectrogram, cropped_wavelengths = pix_to_wavelength(image)

    # Convert counts to irradiance
    calibrated_spectrogram = counts_to_irradiance(cropped_spectrogram, exposure_time_ms)

    return calibrated_spectrogram, cropped_wavelengths

def bip_to_full_calibration(filepath):
    """
    Perform full calibration of the spectrogram.

    Args:
        image (np.ndarray): The spectrogram data.
        exposure_time_ms (float): The exposure time in milliseconds.

    Returns:
        np.ndarray: The calibrated spectrogram data.
    """
    match = re.search(r'_e(\d+\.\d+)_', filepath)
    if match:
        exposure_time = float(match.group(1))
    else:
        raise ValueError("Exposure time not found in filename")
    image = utils.read_bip_file(filepath)

    spec, wave = full_calibration(image, exposure_time)

    visualization.plot_visible_spectrum_cmap(spec, wave)

    return spec, wave