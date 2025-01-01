from numpy import argmin, argsort, array, ndarray, zeros

from clustering_exploration.algorithms.algorithm_base import AlgorithmBase


class SequentialKMeansAlgorithm(AlgorithmBase):
    """Offline K-Means clustering algorithm."""

    # Cluster field indices.
    MEAN = 0
    SPLAT_COUNT = 1
    ALPHA_SUM = 2
    TRANSMITTANCE = 3
    PREMULTIPLIED_COLOR = 4

    def __init__(self, splats: ndarray, number_of_clusters: int):
        super().__init__(splats, number_of_clusters)

        # Flag for if initial guesses have been made.
        self.initial_guesses_found = False

    def pixel_cluster(self, splats: ndarray) -> ndarray:
        # For each pixel, [ K x [ mean, number, alpha_sum, transmittance, premultiplied_r, premultiplied_g, premultiplied_b ] ]
        # After clustering, (1 - transmittance) gives final cluster alpha, and (pre_multiplied_color / alpha_sum) gives final cluster color
        clusters = zeros((self.number_of_clusters, 7))
        clusters[:, self.TRANSMITTANCE] = 1

        for alpha, depth, *color in splats:
            # Skip zero alpha or depth.
            if not alpha or not depth:
                continue

            # Compute cluster index.
            target_cluster_index = (
                argmin(abs(clusters[:, self.MEAN] - depth))
                if self.initial_guesses_found
                else self._initial_guess_index(clusters, depth)
            )

            # Update cluster information.
            clusters[target_cluster_index, self.SPLAT_COUNT] += 1
            clusters[target_cluster_index, self.ALPHA_SUM] += alpha
            clusters[target_cluster_index, self.TRANSMITTANCE] *= 1 - alpha
            clusters[target_cluster_index, self.PREMULTIPLIED_COLOR :] += alpha * array(color)

            # Update cluster mean.
            current_mean = clusters[target_cluster_index, self.MEAN]
            clusters[target_cluster_index, self.MEAN] = (
                current_mean + (depth - current_mean) / clusters[target_cluster_index, self.SPLAT_COUNT]
            )

        # Compute final transmittance and color.
        clusters[:, self.TRANSMITTANCE] = 1 - clusters[:, self.TRANSMITTANCE]
        clusters[:, self.PREMULTIPLIED_COLOR :] /= clusters[:, self.ALPHA_SUM].reshape(-1, 1)

        # Sort clusters and return.
        return clusters[argsort(clusters[:, self.MEAN])][:, self.TRANSMITTANCE :]

    def _initial_guess_index(self, clusters, depth) -> int:
        """Compute the initial guess index for the given depth.

        Assumes an initial guess is needed and possible. Will update the all guesses found flag if reached the end of the clusters.

        Args:
            clusters: The current clusters.
            depth: The depth to use as a guess.

        Returns:
            The index of the cluster to use this depth as an initial guess.
        """
        for cluster_index, cluster in enumerate(clusters):
            # If we made it to the last index, then we will make all initial guesses.
            if cluster_index == self.number_of_clusters - 1:
                self.initial_guesses_found = True

            # Use the cluster if it's exactly the same or if it's the next empty one.
            if cluster[self.MEAN] == depth or not cluster[self.MEAN]:
                return cluster_index

        # Should never reach here.
        self.initial_guesses_found = True
        return 0
