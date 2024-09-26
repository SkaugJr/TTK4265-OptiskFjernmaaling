import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_visible_spectrum_cmap(spectrogram, wavelengths, pixel_range=(366, 766)):
    """
    Plot the spectrogram with the visible spectrum on the wavelength axis, limited to a specific pixel height range.

    Args:
        spectrogram (np.ndarray): The spectrogram data.
        wavelengths (np.ndarray): The calibrated wavelengths.
        pixel_range (tuple): The range of pixel heights to plot (start, end).
    """
    # Create the visible spectrum colormap
    colors = [
        (0.0, 0.0, 0.5),  # 380 nm (violet)
        (0.0, 0.0, 1.0),  # 450 nm (blue)
        (0.0, 1.0, 1.0),  # 495 nm (cyan)
        (0.0, 1.0, 0.0),  # 570 nm (green)
        (1.0, 1.0, 0.0),  # 590 nm (yellow)
        (1.0, 0.5, 0.0),  # 620 nm (orange)
        (1.0, 0.0, 0.0),  # 700 nm (red)
        (0.5, 0.0, 0.0)   # 800 nm (deep red)
    ]

    # Create the colormap
    visible_spectrum_cmap = LinearSegmentedColormap.from_list('visible_spectrum', colors, N=256)
    
    # Slice the spectrogram to the specified pixel range
    start_pixel, end_pixel = pixel_range
    spectrogram_slice = spectrogram[start_pixel:end_pixel, :]
    
    # Plot the spectrogram with grayscale intensity
    plt.figure(figsize=(20, 14))
    extent = [wavelengths.min(), wavelengths.max(), start_pixel, end_pixel]
    plt.imshow(spectrogram_slice, cmap='gray', aspect='auto', extent=extent)
    plt.colorbar(label=r'Intensity [$\frac{mW}{m^2 nm}$]')
    
    # Create a secondary x-axis with the visible spectrum colormap
    ax = plt.gca()
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=visible_spectrum_cmap), ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_ticks([])

    plt.xlabel('Wavelength (nm)')
    plt.xticks(np.arange(400, 801, 20))
    plt.ylabel('Pixel Position')
    plt.yticks(np.arange(start_pixel, end_pixel, 50))
    plt.title('Calibrated Spectrogram with Visible Spectrum')
    plt.show()