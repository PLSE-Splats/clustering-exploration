from abc import ABC, abstractmethod

from joblib import Parallel, delayed
from numpy import array, empty, ndarray, prod, sum, zeros
from numpy.random import default_rng
from tqdm.auto import tqdm


class AlgorithmBase(ABC):
    """Base class for clustering algorithms."""

    def __init__(self, splats: ndarray, number_of_clusters: int):
        """Initialize the algorithm.

        Args:
            splats: Splats for all pixels. Shape: [ H x W x [ number of splats x [ A, D, R, G, B ] ] ].
            number_of_clusters: Number of clusters to create.
        """
        self.splats = splats
        self.number_of_clusters = number_of_clusters

    @abstractmethod
    def pixel_cluster(self, splats: ndarray) -> ndarray:
        """Cluster the splats for pixel.

        Args:
            splats: Splats for a single pixel. Shape: [ number of splats x [ A, D, R, G, B ] ].

        Returns:
            Clustered splats for the pixel. Shape: [ number of clusters x [ A, R, G, B ] ].
        """

    @staticmethod
    def _commutative_combine(splat_clusters: list) -> ndarray:
        """Commutatively combine clustered splats into a single splat.

        Args:
            splat_clusters: List of clustered splats. Shape: [ number of clusters x [ number of splats x [ A, R, G, B ] ] ].
        """
        pixel_output = empty((len(splat_clusters), 4))
        for index, cluster_list in enumerate(splat_clusters):
            # Skip empty clusters.
            if not cluster_list:
                pixel_output[index] = zeros(4)
                continue

            # Convert to numpy array.
            cluster = array(cluster_list)

            # Compute the output alpha and color.
            pixel_output[index, 0] = 1 - prod(1 - cluster[:, 0])
            alpha_sum = sum(cluster[:, 0])
            if alpha_sum:
                pixel_output[index, 1:] = sum(cluster[:, 0].reshape(-1, 1) * cluster[:, 1:], axis=0) / alpha_sum
            else:
                pixel_output[index, 1:] = zeros(3)

        # Return the clustered pixel.
        return pixel_output

    def compute(self) -> list:
        """Compute clustering for all pixels.

        Will shuffle the splats in each pixel before clustering.

        Returns:
            Clustered splats for all pixels. Shape: [ H x W x [ number of clusters x [ A, R, G, B ] ] ].
        """
        print("Shuffling splats...")
        default_rng().shuffle(self.splats, axis=1)

        print("Clustering splats...")
        return Parallel(n_jobs=-1)(delayed(self.pixel_cluster)(splats) for splats in tqdm(self.splats))
