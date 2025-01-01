from abc import ABC, abstractmethod

from joblib import Parallel, delayed
from numpy import ndarray
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
