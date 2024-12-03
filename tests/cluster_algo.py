import numpy as np
import random
from joblib import Parallel, delayed
from tqdm import tqdm


def orig_color_accum(depth, alpha, color, final_color, final_alpha):
    final_color = (
        final_color[0] + final_alpha * alpha * color[0],
        final_color[1] + final_alpha * alpha * color[1],
        final_color[2] + final_alpha * alpha * color[2],
    )
    final_alpha *= 1 - alpha
    return final_alpha, final_color


def commutative_accum(depth, alpha, color, final_color, final_alpha):
    final_color = (
        final_color[0] + alpha * color[0],
        final_color[1] + alpha * color[1],
        final_color[2] + alpha * color[2],
    )
    final_alpha *= 1 - alpha
    return final_alpha, final_color


# Add alpha-correction
def alpha_correction(alpha, samples):
    alpha = 1 - (1 - alpha) ** samples
    return alpha


def stochastic_oit(depth, alpha, color, final_color):
    random_value = random.random()
    if random_value < alpha:
        final_color = (
            final_color[0] + color * color[0],
            final_color[1] + color * color[1],
            final_color[2] + color * color[2],
        )
    return final_color


def stochastic_oit_correction(depth, alpha, color, cluster_color, samples):
    random_value = random.random()
    alpha = alpha_correction(alpha, samples)
    if random_value < alpha:
        cluster_color = (
            cluster_color[0] + alpha * color[0],
            cluster_color[1] + alpha * color[1],
            cluster_color[2] + alpha * color[2],
        )

    return cluster_color


def depth_weight(depth):
    return 1.0 / (depth + 1.0)


def weighted_oit_np(depth, alpha, color, final_color, alpha_sum):
    weight = depth_weight(depth)

    # Accumulate weighted color
    final_color += (alpha * color) * weight

    # Accumulate weighted alpha
    alpha_sum += alpha * weight

    return alpha_sum, final_color


def weighted_oit(depth, alpha, color, final_color, alpha_sum):
    weight = depth_weight(depth)

    final_color = (
        final_color[0] + alpha * color[0] * weight,
        final_color[1] + alpha * color[1] * weight,
        final_color[2] + alpha * color[2] * weight,
    )

    alpha_sum += alpha * weight
    return alpha_sum, final_color


def cluster_splats(splats_to_cluster, epsilon):
    """Cluster splats that are within an epsilon distance of each other."""

    def cluster_pixel(pixel_splats):
        """Compute clustering on a single pixel."""

        # Define clustering dictionary: depth -> [(depth, alpha, color)].
        pixel_clustering = {}

        # Loop through each splat.
        for splat in pixel_splats:
            splat_alpha, splat_depth, *splat_color = splat
            splat_color = np.array(splat_color)
            # Skip transparent splats.
            if splat_alpha == 0:
                continue

            # Case 1: No clusters.
            if not pixel_clustering:
                pixel_clustering[splat_depth] = [
                    (splat_depth, splat_alpha, splat_color)
                ]
                continue

            # Case 2: Cluster is within epsilon distance.
            closest_depth = min(
                pixel_clustering.keys(), key=lambda depth: abs(depth - splat_depth)
            )
            if abs(splat_depth - closest_depth) <= epsilon:
                pixel_clustering[closest_depth].append(
                    (splat_depth, splat_alpha, splat_color)
                )
            # Case 3: case 2 fails.
            else:
                pixel_clustering[splat_depth] = [
                    (splat_depth, splat_alpha, splat_color)
                ]

        # Sort the clusters by depth.
        pixel_clustering = dict(sorted(pixel_clustering.items()))

        # Alpha compose the splats within a clusters.
        pixel_output = np.zeros((len(pixel_clustering), 4))
        for index, cluster in enumerate(pixel_clustering.values()):
            transmittance = 1.0
            alpha_sum = 0.0
            cluster_color = np.zeros(3)

            # Compose.
            for splat_depth, splat_alpha, splat_color in cluster:
                # cluster_color += splat_alpha * splat_color * transmittance
                alpha_sum, cluster_color = weighted_oit_np(
                    splat_depth, splat_alpha, splat_color, cluster_color, alpha_sum
                )
                transmittance *= 1 - splat_alpha

            # Normalize by total alpha.
            cluster_alpha = np.clip(1 - transmittance, 0, 1)
            pixel_output[index][0] = cluster_alpha
            pixel_output[index][1:] = (
                np.clip(cluster_color / alpha_sum, 0, 1)
                if cluster_alpha > 0
                else np.zeros(3)
            )

        # Return the clustered pixel.
        return pixel_output

    # Parallelize the clustering process.
    return Parallel(n_jobs=-1)(
        delayed(cluster_pixel)(pixel_splats) for pixel_splats in tqdm(splats_to_cluster)
    )


def compute_image_from_clusters(clustered_pixels):
    """Compute the final pixel color by alpha compositing the clusters."""

    def alpha_compose_pixel(pixel_clusters):
        # Define the transmittance and pixel color for the first cluster.
        transmittance = 1.0
        pixel_color = (
            np.zeros(3)
            if not pixel_clusters[0, 0]
            else pixel_clusters[0, 1:] * pixel_clusters[0, 0]
        )

        # Loop through remaining clusters.
        for k in range(1, len(pixel_clusters)):
            # Skip transparent cluster.
            if not pixel_clusters[k, 0]:
                continue

            # Exit once the transmittance is basically zero.
            if transmittance <= 0.001:
                break

            # Compute the transmittance.
            transmittance *= 1 - min(1, pixel_clusters[k - 1, 0])

            # Compute the pixel color.
            pixel_color += pixel_clusters[k, 0] * pixel_clusters[k, 1:] * transmittance

        # Return the computed pixel color.
        return pixel_color

    # Compute the color of each pixel.
    return Parallel(n_jobs=-1)(
        delayed(alpha_compose_pixel)(pixel_clusters)
        for pixel_clusters in tqdm(clustered_pixels)
    )
