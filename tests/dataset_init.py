import argparse
import json
import os
import shutil
import subprocess


def load_config(config_path):
    """
    Load a JSON configuration file.

    Parameters:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    with open(config_path) as file:
        config = json.load(file)
    return config


def generate_dataset(config):
    """
    Generate a dataset for 3DGS.

    Parameters:
        config (dict): A dictionary containing the configuration settings.
    """
    print("Generating dataset...")
    print(f"Config: {config}")

    targets = config["dataset_targets"]
    model_path = config["model_path"]
    colmap_path = config["colmap_path"]
    dataset_path = config["dataset_path"]
    views = config["views"]
    gs_path = config["gaussian_splatting_path"]

    for target in targets:
        print(f"Generating dataset for target: {target}")
        for view in views:
            print(f"Generating dataset for view: {view}")
            # Add dataset generation logic here
            target_model_path = os.path.join(model_path, target)
            target_colmap_path = os.path.join(colmap_path, target)
            target_dataset_path = os.path.join(dataset_path, target + "_" + str(view))

            # if dataset path not exists, create it
            os.makedirs(target_dataset_path, exist_ok=True)
            if os.path.exists(
                os.path.join(target_dataset_path, "collected_splats.csv")
            ):
                print(f"view:{view} Dataset already exists, skipping...")
                continue
            render_file_path = os.path.join(gs_path, "render_single_view.py")
            render_view_cmd = [
                "python3",
                render_file_path,
                "--view_index",
                str(view),
                "-m",
                target_model_path,
                "-s",
                target_colmap_path,
                "--skip_test",
            ]

            result = subprocess.run(
                render_view_cmd,
                capture_output=True,
                text=True, check=False,  # Ensures output is in string format, not bytes
            )
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

            # if there is error exit
            if result.returncode != 0:
                raise RuntimeError(f"Error during rendering view {view}")

            # Save the csv, gt, render image to the dataset path
            os.makedirs(target_dataset_path, exist_ok=True)

            # mv the csv file to the dataset path
            csv_file = os.path.join(gs_path, "collected_splats.csv")
            shutil.copy(
                csv_file, os.path.join(target_dataset_path, "collected_splats.csv")
            )

            # mv the gt file to the dataset path
            gt_img = os.path.join(
                target_model_path, "train", "ours_30000", "gt", f"{view:05d}.png"
            )

            render_img = os.path.join(
                target_model_path, "train", "ours_30000", "renders", f"{view:05d}.png"
            )

            shutil.copy(gt_img, os.path.join(target_dataset_path, "gt.png"))
            shutil.copy(render_img, os.path.join(target_dataset_path, "render.png"))

    # Add dataset generation logic here

    print("Dataset generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for 3DGS.")
    parser.add_argument("--config", type=str, help="Path to the test_config.")

    args = parser.parse_args()

    # Compare images and print results
    try:
        config = load_config(args.config)
        generate_dataset(config)
    except Exception as e:
        print(f"Error: {e}")
