from os import environ

from numpy import argsort, array, empty, median, ndarray, prod, sum, zeros
from sklearn.cluster import KMeans

from clustering_exploration.algorithms.algorithm_base import AlgorithmBase


class KMeansAlgorithm(AlgorithmBase):
    """Offline K-Means clustering algorithm."""

    def __init__(self, splats: ndarray, num_clusters: int):
        """Initialize the K-Means algorithm.

        Args:
            splats: The splats to cluster.
            num_clusters: The number of clusters to create.
        """
        super().__init__(splats, num_clusters)
        environ["OMP_NUM_THREADS"] = "1"

    def pixel_cluster(self, splats: ndarray) -> ndarray:
        # Get depth values.
        depths = splats[:, 1]

        # Run K-Means clustering.
        kmeans = KMeans(n_clusters=self.number_of_clusters).fit(depths.reshape(-1, 1))

        # Initialize clustering 2D list: cluster -> [[alpha, depth, *color], ...].
        pixel_clustering = [[] for _ in range(self.number_of_clusters)]

        # Loop through each splat and place it in the appropriate cluster.
        for splat_index, cluster_index in enumerate(kmeans.labels_):
            pixel_clustering[cluster_index].append(splats[splat_index])

        # Commutatively combine cluster values [[median depth, alpha, color], ...].
        depth_clusters = empty((self.number_of_clusters, 5))
        for cluster_index, cluster_splats in enumerate(pixel_clustering):
            # Convert to numpy array.
            cluster_splats_array = array(cluster_splats)

            # Compute fields.
            depth_clusters[cluster_index, 0] = median(cluster_splats_array[:, 1])
            depth_clusters[cluster_index, 1] = 1 - prod(1 - cluster_splats_array[:, 0])
            alpha_sum = sum(cluster_splats_array[:, 0])
            if alpha_sum:
                depth_clusters[cluster_index, 2:] = (
                    sum(cluster_splats_array[:, 0].reshape(-1, 1) * cluster_splats_array[:, 2:], axis=0) / alpha_sum
                )
            else:
                depth_clusters[cluster_index, 2:] = zeros(3)

        return depth_clusters[argsort(depth_clusters[:, 0])][:, 1:]
