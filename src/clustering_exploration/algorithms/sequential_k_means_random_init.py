from numpy import abs, argmin, argsort, array, ndarray, zeros
from random import randint

from clustering_exploration.algorithms.algorithm_base import AlgorithmBase

# Cluster field indices.
DEPTH = 0
SPLAT_COUNT = 1
ALPHA_SUM = 2
TRANSMITTANCE = 3
PREMULTIPLIED_COLOR = 4


class SequentialKMeansRandomInitAlgorithm(AlgorithmBase):
    """Offline K-Means clustering algorithm."""

    def __init__(self, splats: ndarray, number_of_clusters: int):
        super().__init__(splats, number_of_clusters)

    def cluster_splat(self, clusters: ndarray, target_cluster_index: int, splat: ndarray) -> None:
        # Extract splat information.
        alpha, depth, *color = splat

        # Skip zero alpha or depth.
        if not alpha or not depth:
            return

        # Update cluster information.
        clusters[target_cluster_index, SPLAT_COUNT] += 1
        clusters[target_cluster_index, ALPHA_SUM] += alpha
        clusters[target_cluster_index, TRANSMITTANCE] *= 1 - alpha
        clusters[target_cluster_index, PREMULTIPLIED_COLOR:] += alpha * array(color)

        # Update cluster mean.
        current_mean = clusters[target_cluster_index, DEPTH]
        clusters[target_cluster_index, DEPTH] = current_mean + (depth - current_mean) / clusters[
            target_cluster_index, SPLAT_COUNT]

    def pixel_cluster(self, splats: ndarray) -> ndarray:
        # For each pixel, [ K x [ mean, number, alpha_sum, transmittance, premultiplied_r, premultiplied_g, premultiplied_b ] ]
        # After clustering, (1 - transmittance) gives final cluster alpha, and (pre_multiplied_color / alpha_sum) gives final cluster color
        clusters = zeros((self.number_of_clusters, 7))
        clusters[:, TRANSMITTANCE] = 1

        # Remember which splat index was used for initialization
        splat_index_used_for_init = set()

        # Find initial guesses by getting a random splat.
        for cluster_index in range(self.number_of_clusters):
            # Pick a random splat index.
            random_splat_index = randint(0, splats.shape[0] - 1)

            # Ensure that splat wasn't picked before.
            while random_splat_index in splat_index_used_for_init:
                random_splat_index = randint(0, splats.shape[0] - 1)

            # Get the random splat.
            random_splat = splats[random_splat_index]

            # TODO: ensure empty splat was not chosen.

            # Initialize cluster.
            self.cluster_splat(clusters, cluster_index, random_splat)

            # Record the usage of the splat index.
            splat_index_used_for_init.add(random_splat_index)

        # Cluster the remaining splats.
        for splat_index, splat in enumerate(splats):
            # Skip splats used for initialization.
            if splat_index in splat_index_used_for_init:
                continue

            # Compute cluster index.
            splat_depth = splat[DEPTH]
            target_cluster_index = argmin(abs(clusters[:, DEPTH] - splat_depth))

            # Update cluster information.
            self.cluster_splat(clusters, target_cluster_index, array(splat))

        # Compute final transmittance and color.
        for cluster_index, cluster in enumerate(clusters):
            clusters[cluster_index, TRANSMITTANCE] = 1 - cluster[TRANSMITTANCE]
            clusters[cluster_index, PREMULTIPLIED_COLOR:] /= cluster[ALPHA_SUM]

        # Sort clusters and return.
        return clusters[argsort(clusters[:, DEPTH])][:, TRANSMITTANCE:]
