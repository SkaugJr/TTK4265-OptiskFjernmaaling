import numpy as np

def read_bip_file(file_path, width=1936, height=1216):
    with open(file_path, 'rb') as file:
        data = file.read()
    
    # Konverter binærdata til numpy array med 8-bit unsigned integers
    raw_data = np.frombuffer(data, dtype=np.uint8)
    
    # Kombiner 8-bit verdier til 16-bit verdier
    total_pixels = width * height
    pixel_data = np.zeros(total_pixels, dtype=np.uint16)
    
    for i in range(total_pixels):
        byte_index = i * 2
        pixel_data[i] = (raw_data[byte_index] << 8) | raw_data[byte_index + 1]
    
    # Reshape array til 2D bilde
    image = pixel_data.reshape((height, width))
    return image

def read_bip_file_8bit(file_path, width=1936, height=1216):
    """
    Read a .bip file with 8-bit values and return the image data as a numpy array.

    Args:
        file_path (str): The path to the .bip file.
        width (int, optional): The width of the image. Defaults to 1936.
        height (int, optional): The height of the image. Defaults to 1216.

    Returns:
        np.ndarray: The image data.
    """
    with open(file_path, 'rb') as file:
        data = file.read()
    
    # Convert binary data to numpy array with 8-bit unsigned integers
    pixel_data = np.frombuffer(data, dtype=np.uint)
    
    # Reshape array to 2D image
    image = pixel_data.reshape((height, width))
    return image

def calculate_average_image(file_paths, output_path, width, height):
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