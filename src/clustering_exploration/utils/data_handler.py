from os import makedirs
from os.path import dirname, exists, join

from numpy import load, ndarray, save
from polars import Float32, Schema, UInt8, UInt32, all, scan_csv

from clustering_exploration.utils.constants import CACHE_DIR, DATA_DIR, IMAGE_HEIGHT, IMAGE_WIDTH


def define_schema() -> Schema:
    """Define Polars schema to read CSV dataset.

    Returns:
        Polars schema for clustering data pulled from 3DGS.
    """

    # Define the column names.
    column_names = [
        f"gaussian_{i}_{part}" for i in range(500) for part in ["alpha", "depth", "color_r", "color_g", "color_b"]
    ]
    column_names = [
        "sample_index",
        "out_color_r",
        "out_color_g",
        "out_color_b",
        "background_r",
        "background_g",
        "background_b",
        *column_names,
    ]

    # Define schema.
    schema_dict = {name: Float32 for name in column_names}
    schema_dict["sample_index"] = UInt32
    schema_dict["background_r"] = UInt8
    schema_dict["background_g"] = UInt8
    schema_dict["background_b"] = UInt8
    return Schema(schema_dict)


def load_splats(data_name: str) -> ndarray:
    """Loads splats given a dataset name.

    Will try to load from cache if possible, otherwise will look for the CSV in the `data` directory and create a cache after.

    Args:
        data_name: Name of the dataset to load.
    Returns:
        3D Numpy array of splats in (h x w) x number of splats x (alpha, depth, R, G, B).
    """
    # Define the cache file path.
    cache_path = join(CACHE_DIR, f"{data_name}.npy")

    # Check if the cache exists.
    if exists(cache_path):
        return load(cache_path)

    # Load the data from CSV.
    data_path = join(DATA_DIR, f"{data_name}.csv")
    raw_data = scan_csv(data_path, schema=define_schema())

    # Extract the splats.
    raw_splats = (
        raw_data.select(
            all().exclude(
                "sample_index",
                "out_color_r",
                "out_color_g",
                "out_color_b",
                "background_r",
                "background_g",
                "background_b",
            )
        )
        .collect()
        .to_numpy()
    )
    splats = raw_splats.reshape((IMAGE_HEIGHT * IMAGE_WIDTH, raw_splats.shape[1] // 5, 5))

    # Save the cache.
    makedirs(dirname(cache_path), exist_ok=True)
    save(cache_path, splats)

    # Return the splats.
    return splats
