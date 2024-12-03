import argparse

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compare_images(image1_path, image2_path):
    """
    Compare two images using SSIM and PSNR metrics.

    Parameters:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.

    Returns:
        dict: A dictionary containing SSIM and PSNR values.
    """
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1.shape != img2.shape:
        raise ValueError("The two images must have the same dimensions for comparison.")

    # Convert images to grayscale for SSIM calculation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    ssim_value = ssim(gray1, gray2)

    # Compute PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr_value = float("inf")  # Perfect match
    else:
        max_pixel = 255.0
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))

    return {"SSIM": ssim_value, "PSNR": psnr_value}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two images using SSIM and PSNR metrics."
    )
    parser.add_argument("image1", type=str, help="Path to the first image.")
    parser.add_argument("image2", type=str, help="Path to the second image.")

    args = parser.parse_args()

    # Compare images and print results
    try:
        metrics = compare_images(args.image1, args.image2)
        print(f"SSIM: {metrics['SSIM']}")
        print(f"PSNR: {metrics['PSNR']} dB")
    except Exception as e:
        print(f"Error: {e}")
