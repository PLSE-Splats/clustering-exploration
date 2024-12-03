import numpy as np
from PIL import Image

def save_array_to_image(array: np.ndarray, name: str) -> Image:
    """Save a numpy array to an image in the output directory and display it.
    
    Args:
        array: The array to save. Expected to be a h x w x c float array.
        name: The name of the file to save the image to.
    Returns:
        The saved image.
    """
    
    # Clamp the array to [0, 1].
    clamped_array = np.clip(array, 0, 1)
    
    # Convert h x w x c float array to uint8 8-bit array.
    array_int = (clamped_array * 255).astype(np.uint8)
    
    # Create an image from the array.
    image = Image.fromarray(array_int)
    
    # Save the image.
    image.save(f'output/{name}.png')
    
    # Return the image.
    return image