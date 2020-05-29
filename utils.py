import cv2
import numpy as np
import tensorflow as tf
import scipy.io as scio
import scipy
from scipy.ndimage.filters import gaussian_filter


def gaussian_kernel_2d(kernel_size=3, sigma=0.):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))


def truncation_normal_distribution(standard_variance):
    return tf.truncated_normal_initializer(0.0, standard_variance)

def structural_similarity_index_metric(feature, labels):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    weight = gaussian_kernel_2d(11, 1.5)
    weight = tf.constant(weight)
    weight = tf.reshape(weight, [11, 11, 1, 1])
    weight = tf.cast(weight, tf.float32)
    mean_f = tf.nn.conv2d(feature, weight, [1, 1, 1, 1], padding="SAME")
    mean_y = tf.nn.conv2d(labels, weight, [1, 1, 1, 1], padding="SAME")
    mean_f_mean_y = tf.multiply(mean_f, mean_y)
    square_mean_f = tf.multiply(mean_f, mean_f)
    square_mean_y = tf.multiply(mean_y, mean_y)
    variance_f = tf.nn.conv2d(tf.multiply(feature, feature), weight, [1, 1, 1, 1], padding="SAME") - square_mean_f
    variance_y = tf.nn.conv2d(tf.multiply(labels, labels), weight, [1, 1, 1, 1], padding="SAME") - square_mean_y
    variance_fy = tf.nn.conv2d(tf.multiply(feature, labels), weight, [1, 1, 1, 1], padding="SAME") - mean_f_mean_y
    ssim = ((2*mean_f_mean_y + c1)*(2*variance_fy + c2)) / \
           ((square_mean_f + square_mean_y + c1)*(variance_f + variance_y + c2))
    return 1 - tf.reduce_mean(ssim, reduction_indices=[1, 2, 3])


def get_density_map_gaussian(N, M, points, adaptive_kernel=False, fixed_value=15):
    density_map = np.zeros([N, M], dtype=np.float32)
    h, w = density_map.shape[:2]
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    if adaptive_kernel:
        # referred from https://github.com/vlad3996/computing-density-maps/blob/master/make_ShanghaiTech.ipynb
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances = tree.query(points, k=4)[0]

    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[1]), min(w-1, p[0])
        if num_gt > 1:
            if adaptive_kernel:
                sigma = int(np.sum(distances[idx][1:4]) // 3 * 0.3)
            else:
                sigma = fixed_value
        else:
            sigma = fixed_value  # np.average([h, w]) / 2. / 2.
        sigma = max(1, sigma)

        gaussian_radius = sigma * 3
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < 0 or p[0] < 0:
            continue
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(density_map.shape[0], p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(density_map.shape[1], p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    return density_map