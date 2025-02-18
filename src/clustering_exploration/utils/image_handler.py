from __future__ import annotations

from os import makedirs
from os.path import dirname, join

import numpy as np
from joblib import Parallel, delayed
from numpy import array, clip, ndarray, uint8, zeros
from PIL import Image
from tqdm.auto import tqdm

from clustering_exploration.utils.constants import IMAGE_HEIGHT, IMAGE_WIDTH, MINIMUM_TRANSMITTANCE, OUTPUT_DIR


def save_array_to_image(pixel_array: ndarray, name: str) -> Image:
    """Save a numpy array to an image in the output directory and display it.

    Args:
        pixel_array: The array to save. Expected to be a [ H x [ W x [ R, G, B ] ] ] float array.
        name: The name of the file to save the image to.
    Returns:
        The saved image.
    """

    # Clamp the array to [0, 1].
    clamped_array = clip(pixel_array, 0, 1)

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


def alpha_compose_splats(splats: list[float]) -> ndarray[float]:
    """Alpha compose an ordered array of splats.

    Splats are provided in an ordered array of the form [ K x [ A, R, G, B ] ].

    Args:
        splats: Ordered array of splats.

    Returns:
        Final color of the splats (R, G, B).
    """
    # Define the transmittance and pixel color.
    transmittance = 1.0
    final_color = zeros(3)

    # Loop through each splat.
    for alpha, *color in splats:
        # Skip transparent cluster.
        if not alpha:
            continue

        # Exit once the transmittance is basically zero.
        if transmittance <= MINIMUM_TRANSMITTANCE:
            break

        # Compute the pixel color.
        final_color += alpha * array(color) * transmittance

        # Update the transmittance.
        transmittance *= 1 - min(1, alpha)

    # Return the computed color.
    return final_color


def compute_image_from_clusters(clustered_splats: list, output_image_name: str) -> Image:
    """Compute the image from the clustered splats and save the result.

    Args:
        clustered_splats: The clustered splats. Expecting [ H x W x [ K x [ A, R, G, B ] ] ].
        output_image_name: The name of the output image.
    Returns:
        The computed image.
    """
    # Alpha compose.
    composed_image = Parallel(n_jobs=-1)(
        delayed(alpha_compose_splats)(pixel_clusters) for pixel_clusters in tqdm(clustered_splats)
    )
    

    # Save the image.
    return save_array_to_image(np.array(composed_image).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3), output_image_name)
