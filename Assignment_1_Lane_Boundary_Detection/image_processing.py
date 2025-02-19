import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolve(image: np.ndarray, filter: np.ndarray):
    i_h, i_w = image.shape
    f_h, f_w = filter.shape

    h_pad = f_h // 2  # assuming odd sized kernel
    w_pad = f_w // 2  # assuming odd sized kernel

    padded_image = np.zeros((i_h + 2 * h_pad, i_w + 2 * w_pad))
    padded_image[h_pad : h_pad + i_h, w_pad : w_pad + i_w] = image

    output_image = np.zeros_like(image)

    kernel = np.flipud(np.fliplr(filter))

    for i in range(i_h):
        for j in range(i_w):
            region = padded_image[i : i + f_h, j : j + f_w]
            output_image[i, j] = np.sum(region * kernel)

    return output_image


def guassian_kernel(size, sigma=1.0):
    k = (size - 1) // 2
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))

    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    return gaussian


def bgr_to_gray(image: np.ndarray):

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    gray = 0.114 * b + 0.587 * g + 0.299 * r

    return gray.astype(np.uint8)


def get_trig_table(num_angles, theta, rho):
    rho_inv = 1.0 / rho
    angles = np.arange(num_angles) * theta

    sin_table = np.float32(np.sin(angles) * rho_inv)
    cos_table = np.float32(np.cos(angles) * rho_inv)

    return cos_table, sin_table


def adaptive_thresholds(image: np.ndarray, percentile):
    all_thresholds = []

    for color in range(3):
        channel_values = image[:, :, color].flatten()
        all_thresholds.append(np.percentile(channel_values, percentile))
    return all_thresholds


def generate_heatmap(image: np.ndarray, channel_index):
    channel = image[:, :, channel_index].astype(np.float32)

    min_val, max_val = np.min(channel), np.max(channel)
    normalized = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    return heatmap, normalized


def display_heatmaps(image: np.ndarray):
    heatmap_b, norm_b = generate_heatmap(image, 0)
    heatmap_g, norm_g = generate_heatmap(image, 1)
    heatmap_r, norm_r = generate_heatmap(image, 2)

    fig, axes = plt.subplots(2, 2, figsize=(7, 10))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    for ax, heatmap, title in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        [heatmap_b, heatmap_g, heatmap_r],
        ["Blue Channel Heatmap", "Green Channel Heatmap", "Red Channel Heatmap"],
    ):

        im = ax.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity (0-255)")

    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(7, 10))

    b_thres, g_thres, r_thres = adaptive_thresholds(image, percentile=90)

    mask_b = np.where(image[:, :, 0] > b_thres, image[:, :, 0], 0)
    mask_g = np.where(image[:, :, 1] > g_thres, image[:, :, 1], 0)
    mask_r = np.where(image[:, :, 2] > r_thres, image[:, :, 2], 0)
    mask_combined = np.maximum.reduce([mask_b, mask_g, mask_r])

    axes[0, 0].imshow(mask_b, cmap="Blues")
    axes[0, 0].set_title(f"Blue > {b_thres}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask_g, cmap="Greens")
    axes[0, 1].set_title(f"Green > {g_thres}")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(mask_r, cmap="Reds")
    axes[1, 0].set_title(f"Red > {r_thres}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mask_combined, cmap="gray")
    axes[1, 1].set_title("Combined Mask")
    axes[1, 1].axis("off")

    fig.tight_layout()
    plt.show()


def histogram(image: np.ndarray):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    plt.figure(figsize=(8, 6))
    plt.plot(hist, color="black")
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    plt.show()

    return hist


def color_histogram(image: np.ndarray):
    colors = ["b", "g", "r"]
    channel_names = ["Blue", "Green", "Red"]

    plt.figure(figsize=(8, 6))

    for i, color in enumerate(colors):
        hist, bins = np.histogram(image[:, :, i].flatten(), bins=256, range=[0, 256])
        plt.plot(hist, color=color, label=f"{channel_names[i]} Channel")

    plt.title("Color Histogram (BGR)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    plt.legend()
    plt.show()
