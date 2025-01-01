from numpy import concatenate, max, min, ndarray

from clustering_exploration.algorithms.algorithm_base import AlgorithmBase


class BinnedAlgorithm(AlgorithmBase):
    """Equally sized binning clustering algorithm."""

    def __init__(self, splats: ndarray, number_of_clusters: int):
        """Initialize the algorithm.

        Args:
            splats: Splats for all pixels. Shape: [ H x W x [ number of splats x [ A, D, R, G, B ] ] ].
            number_of_clusters: Number of clusters to create.
        """
        super().__init__(splats, number_of_clusters)

        # Compute the minimum and maximum depth of the image.
        depth_values = splats[:, :, 1]
        depth_values = depth_values[depth_values > 0]
        self.min_depth = min(depth_values)
        self.max_depth = max(depth_values)

        print(f"Min depth: {self.min_depth}, Max depth: {self.max_depth}")
        print(f"Each bin will be {(self.max_depth - self.min_depth) / self.number_of_clusters} units wide.")

    def pixel_cluster(self, splats: ndarray) -> ndarray:
        # Initialize clustering 2D list: cluster -> [[alpha, *color], ...].
        pixel_clustering = [[] for _ in range(self.number_of_clusters)]

        # Loop through each splat.
        for splat in splats:
            # Skip transparent splats.
            if splat[0] == 0:
                continue

            # Compute the bin index.
            bin_index = int(
                (splat[1] - self.min_depth) / (self.max_depth - self.min_depth) * (self.number_of_clusters - 1)
            )

            # Add the splat to the bin.
            pixel_clustering[bin_index].append(concatenate(([splat[0]], splat[2:])))

        # Commutative combination of the splats in each cluster (alpha, color).
        return self._commutative_combine(pixel_clustering)
