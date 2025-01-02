from numpy import abs, argmin, argsort, array, ndarray, zeros

from clustering_exploration.algorithms.algorithm_base import AlgorithmBase

# Cluster field indices.
DEPTH = 0
SPLAT_COUNT = 1
ALPHA_SUM = 2
TRANSMITTANCE = 3
PREMULTIPLIED_COLOR = 4

class SequentialKMeansAlgorithm(AlgorithmBase):
    """Offline K-Means clustering algorithm."""


    def __init__(self, splats: ndarray, number_of_clusters: int):
        super().__init__(splats, number_of_clusters)

        # Flag for if initial guesses have been made.
        self.initial_guesses_found = False
        
    def pixel_cluster(self, splats: ndarray) -> ndarray:
        # For each pixel, [ K x [ mean, number, alpha_sum, transmittance, premultiplied_r, premultiplied_g, premultiplied_b ] ]
        # After clustering, (1 - transmittance) gives final cluster alpha, and (pre_multiplied_color / alpha_sum) gives final cluster color
        clusters = zeros((self.number_of_clusters, 7))
        clusters[:, TRANSMITTANCE] = 1

        initial_guesses_found = False

        for alpha, depth, *color in splats:
            # Skip zero alpha or depth.
            if not alpha or not depth:
                continue

            # Compute cluster index.
            target_cluster_index = argmin(abs(clusters[:, DEPTH] - depth))

            # If there are still initial guesses to be made.
            if not initial_guesses_found:
                # Loop through each non-zero cluster and check for an exact match.
                for cluster_index, cluster in enumerate(clusters):
                    # If we made it to the last index, then we will make all initial guesses.
                    if cluster_index == self.number_of_clusters - 1:
                        initial_guesses_found = True

                    # Use the cluster if it's exactly the same.
                    if cluster[DEPTH] == depth:
                        target_cluster_index = cluster_index
                        break

                    # Use next empty cluster.
                    if not cluster[self.number_of_clusters]:
                        target_cluster_index = cluster_index
                        break

            # Update cluster information.
            clusters[target_cluster_index, SPLAT_COUNT] += 1
            clusters[target_cluster_index, ALPHA_SUM] += alpha
            clusters[target_cluster_index, TRANSMITTANCE] *= 1 - alpha
            clusters[target_cluster_index, PREMULTIPLIED_COLOR:] += alpha * array(color)

            # Update cluster mean.
            current_mean = clusters[target_cluster_index, DEPTH]
            clusters[target_cluster_index, DEPTH] = current_mean + (depth - current_mean) / clusters[target_cluster_index, SPLAT_COUNT]

        # Compute final transmittance and color.
        for cluster_index, cluster in enumerate(clusters):
            clusters[cluster_index, TRANSMITTANCE] = 1 - cluster[TRANSMITTANCE]
            clusters[cluster_index, PREMULTIPLIED_COLOR:] /= cluster[ALPHA_SUM]

        # Sort clusters and return.
        return clusters[argsort(clusters[:, DEPTH])][:, TRANSMITTANCE:]
