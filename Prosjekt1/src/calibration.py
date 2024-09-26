import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def pix_to_wavelength(spectrogram, csv_file='../Data/Calibrated/calibrated_wavelengths.csv', wavelength_min=None, wavelength_max=None):
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

def expected_irradiance(wavelengths):
    #Certificate data
    W_L = np.loadtxt('../Data/Calibrated/calibrationCertificate200W.txt', usecols=0) # Wavelengths in nm
    B_0 = np.loadtxt('../Data/Calibrated/calibrationCertificate200W.txt', usecols=1) # Irradiance in mW/m^2/nm
    # sigma = np.loadtxt('../Data/Calibrated/calibrationCertificate200W.txt', usecols=2) # Uncertainty in %

    R = 0.92
    r_0 = 0.5 # from source to lambertian surface
    alpha = 0 # angle between source and normal to lambert
    p = 0.98 # diffusion coefficient of lambertian surface
    B_expected = B_0*(r_0/R)**2*p*np.cos(alpha) # photons/cm^2/s/Å
    
    interp_func = interp1d(W_L, B_expected, kind='linear', fill_value="extrapolate")
    B = interp_func(wavelengths)

    return B

def scaling_factor(exposure_time_ms):
    """
    Calculate the scaling factor for converting counts to irradiance.

    Args:
        exposure_time_ms (float): The exposure time in milliseconds.
    """
    # Load the scaling factor calibrated for 1 ms exposure time
    scaling_factor = np.loadtxt('../Data/Calibrated/Scaling_Factor.txt', delimiter=",") # Scaling factor in mW/m^2/nm/count

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