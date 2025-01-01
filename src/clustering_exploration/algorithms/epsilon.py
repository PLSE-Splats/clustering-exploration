from numpy import array, concatenate, empty, ndarray, prod, sum, zeros

from clustering_exploration.algorithms.algorithm_base import AlgorithmBase


class EpsilonAlgorithm(AlgorithmBase):
    """Epsilon spatial clustering algorithm."""

    def __init__(self, splats: ndarray, epsilon: float):
        """Initialize the algorithm.

        Args:
            splats: Splats for all pixels. Shape: [ H x W x [ number of splats x [ A, D, R, G, B ] ] ].
            epsilon: The epsilon value for clustering.
        """
        super().__init__(splats, 0)
        self.epsilon = epsilon

    def pixel_cluster(self, splats: ndarray) -> ndarray:
        # Define clustering dictionary: depth -> [(alpha, color)].
        pixel_clustering = {}

        # Loop through each splat.
        for splat in splats:
            splat_alpha, splat_depth, *splat_color = splat
            splat_color = array(splat_color)
            # Skip transparent splats.
            if splat_alpha == 0:
                continue

            combined_splat_info = concatenate(([splat_alpha], splat_color))

            # Case 1: No clusters.
            if not pixel_clustering:
                pixel_clustering[splat_depth] = [combined_splat_info]
                continue

            # Case 2: Cluster is within epsilon distance.
            closest_depth = min(pixel_clustering.keys(), key=lambda depth: abs(depth - splat_depth))
            if abs(splat_depth - closest_depth) <= self.epsilon:
                pixel_clustering[closest_depth].append(combined_splat_info)
            # Case 3: case 2 fails.
            else:
                pixel_clustering[splat_depth] = [combined_splat_info]

        # Sort the clusters by depth.
        pixel_clustering = dict(sorted(pixel_clustering.items()))

        # Commutative combination of the splats in each cluster (alpha, color).
        pixel_output = empty((len(pixel_clustering), 4))
        for index, cluster_list in enumerate(pixel_clustering.values()):
            cluster = array(cluster_list)
            pixel_output[index, 0] = 1 - prod(1 - cluster[:, 0])
            alpha_sum = sum(cluster[:, 0])
            if alpha_sum:
                pixel_output[index, 1:] = sum(cluster[:, 0].reshape(-1, 1) * cluster[:, 1:], axis=0) / alpha_sum
            else:
                pixel_output[index, 1:] = zeros(3)

        # Return the clustered pixel.
        return pixel_output
