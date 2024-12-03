import argparse
import json
import os
import Image
import numpy as np
import polars as pl
import cluster_algo as ca
import matplotlib.pyplot as plt
import metric_utils as mu


def load_config(config_path):
    """
    Load a JSON configuration file.

    Parameters:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def evaluate_algo(config):

    print("Evaluating algorithm...")
    print(f"Config: {config}")

    dataset_path = config["dataset_path"]

    # Get clustering algorithm
    algo_name = config["algo_name"]
    cluster_func = getattr(ca, algo_name)
    parameters = config["parameters"]

    # iterate directories in dataset_path
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            process_dataset(item_path, cluster_func, parameters)


def read_csv(dataset_path):
    df_temp = pl.read_csv(dataset_path, n_rows=1)
    column_names = df_temp.columns

    schema_overrides = {name: pl.Float32 for name in column_names if name != "pixelNum"}
    schema_overrides["pixelNum"] = pl.Int64  # Set 'pixelNum' as Int64

    # Load the CSV with the specified schema overrides
    df = pl.read_csv(dataset_path, schema_overrides=schema_overrides)
    return df


def process_dataset(item_path, cluster_func, parameters):
    gt_img = os.path.join(item_path, "gt.png")
    dataset = os.path.join(item_path, "alpha_vals.csv")
    df = read_csv(dataset)

    image = Image.open(gt_img)
    width, height = image.size

    # Get the CSV data from Polars DataFrame
    splats = df.to_numpy().astype(np.float32)

    # Extract splat data (all Gaussian data)
    splat_data = splats[:, 7:].reshape((width * height, (splats.shape[1] - 7) // 5, 5))
    parameters["splats_to_cluster"] = splat_data

    clustered_rgb = cluster_func(**parameters)
    computed_image = ca.compute_image_from_clusters(clustered_rgb)
    plt.figure(figsize=(10, 8))
    plt.axis("off")  # Hide axes for better visualization

    # Save the image
    plt.imshow(np.array(computed_image).reshape((height, width, 3)))
    plt.savefig("computed_image.png")

    # Create Json file save psnr and ssim with gt
    psnr = mu.compute_psnr(gt_img, "computed_image.png")
    ssim = mu.compute_ssim(gt_img, "computed_image.png")

    with open("results.json", "w") as f:
        json.dump({"psnr": psnr, "ssim": ssim}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalute algorithm using from dataset for 3DGS."
    )
    parser.add_argument("config", type=str, help="Path to the test_config.")

    args = parser.parse_args()

    # Compare images and print results
    try:
        config = load_config(args.config)
        evaluate_algo(config)
    except Exception as e:
        print(f"Error: {e}")
