from numpy import clip, uint8, ndarray
from PIL import Image
from os import makedirs
from os.path import join, dirname
from constants import OUTPUT_DIR


def save_array_to_image(array:ndarray, name: str) -> Image:
    """Save a numpy array to an image in the output directory and display it.

    Args:
        array: The array to save. Expected to be a h x w x c float array.
        name: The name of the file to save the image to.
    Returns:
        The saved image.
    """

    # Clamp the array to [0, 1].
    clamped_array =clip(array, 0, 1)

    # Convert h x w x c float array to uint8 8-bit array.
    array_int = (clamped_array * 255).astype(uint8)

    # Create an image from the array.
    image = Image.fromarray(array_int)

    # Save the image.
    output_path = join(OUTPUT_DIR, f"{name}.png")
    makedirs(dirname(output_path), exist_ok=True)
    image.save(output_path)

    # Return the image.
    return image
