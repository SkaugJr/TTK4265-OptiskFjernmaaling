import numpy as np

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

    image_data = average_image
    
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

def theoretical_fwhm(slit_width = (25e-6), focal_length = (16e-3), groove_spacing = (1 / (600 * 1e3)), order=1, lin_coeff=1, alpha = 0):
    """
    Compute the theoretical FWHM using the slit width, the grating groove spacing,
    and the focal length of the middle objective.

    Parameters:
    slit_width (float): Slit width in meters.
    focal_length (float): Focal length of the middle objective in meters.
    groove_spacing (float): Grating groove spacing in meters.

    Returns:
    float: Theoretical FWHM in meters.
    """
    alpha = 0 # Angle of incidence to grating

    # fwhm = (slit_width * focal_length) / (order * groove_spacing)
    fwhm = (lin_coeff * np.cos(alpha))/(order * focal_length) * slit_width
    return fwhm

def calculate_fwhm(wavelengths, intensities):
    """
    Calculate the full width at half maximum (FWHM) of a peak.

    Parameters:
    - wavelengths: numpy array of wavelengths.
    - intensities: numpy array of intensities.

    Returns:
    - fwhm: float, the full width at half maximum.
    """
    # Find the maximum intensity and its index
    max_intensity = np.max(intensities)
    max_index = np.argmax(intensities)

    # Find the half maximum intensity
    half_max_intensity = max_intensity / 2

    # Find the indices of the intensities closest to the half maximum intensity
    left_index = np.argmin(np.abs(intensities[:max_index] - half_max_intensity))
    right_index = np.argmin(np.abs(intensities[max_index:] - half_max_intensity)) + max_index

    # Find the wavelengths at the half maximum intensities
    left_wavelength = wavelengths[left_index]
    right_wavelength = wavelengths[right_index]

    # Calculate the FWHM
    fwhm = right_wavelength - left_wavelength

    return fwhm