import image_processing
import numpy as np
import cv2


def median_filter(image: np.ndarray, kernel_size):
    h, w, _ = image.shape
    pad = kernel_size // 2
    filtered_image = np.copy(image)

    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            for c in range(3):
                neighborhood = image[y - pad : y + pad + 1, x - pad : x + pad + 1, c]
                filtered_image[y, x, c] = np.median(neighborhood)

    return filtered_image


def mean_filter(image: np.ndarray, kernel_size=3):
    h, w, _ = image.shape
    pad = kernel_size // 2
    filtered_image = np.copy(image)

    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            for c in range(3):
                neighborhood = image[y - pad : y + pad + 1, x - pad : x + pad + 1, c]
                filtered_image[y, x, c] = np.mean(neighborhood)

    return filtered_image


def blur_intensity(image: np.ndarray, x, y):
    h, w, _ = image.shape

    kernel_size = 5
    half_k = kernel_size // 2

    x1, x2 = max(0, x - half_k), min(w - 1, x + half_k)
    y1, y2 = max(0, y - half_k), min(h - 1, y + half_k)

    neighborhood = image[y1 : y2 + 1, x1 : x2 + 1]
    std_dev = np.std(neighborhood, axis=(0, 1))

    if np.mean(std_dev) < 10:
        return 1.0
    elif np.mean(std_dev) < 20:
        return 0.7
    elif np.mean(std_dev) < 35:
        return 0.4
    else:
        return 0.2


def brick_image_blur(image: np.ndarray):
    h, w, _ = image.shape

    denoised_image = median_filter(image, kernel_size=5)
    denoised_image = mean_filter(denoised_image, kernel_size=3)

    strong_blur_kernel = image_processing.guassian_kernel(size=9, sigma=2)
    mild_blur_kernel = image_processing.guassian_kernel(size=3, sigma=1)

    strong_blur = np.stack(
        [
            image_processing.convolve(denoised_image[:, :, i], strong_blur_kernel)
            for i in range(denoised_image.shape[2])
        ],
        axis=2,
    )

    mild_blur = np.stack(
        [
            image_processing.convolve(denoised_image[:, :, i], mild_blur_kernel)
            for i in range(denoised_image.shape[2])
        ],
        axis=2,
    )

    output_image = image.copy()

    for y in range(h):
        for x in range(w):
            alpha = blur_intensity(denoised_image, x, y)
            output_image[y, x] = (
                alpha * strong_blur[y, x] + (1 - alpha) * mild_blur[y, x]
            ).astype(np.uint8)

    return output_image


def sobel_edge_detection(image: np.ndarray):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    pad = 1
    padded_image = np.pad(image, pad, mode="constant", constant_values=0)
    edge_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i : i + 3, j : j + 3]
            grad_x = np.sum(region * sobel_x)
            grad_y = np.sum(region * sobel_y)
            edge_image[i, j] = np.sqrt(grad_x**2 + grad_y**2)

    edge_image[:3, :] = 0
    edge_image[-3:, :] = 0
    edge_image[:, :3] = 0
    edge_image[:, -3:] = 0

    return np.clip(edge_image, 0, 255).astype(np.uint8)


def hough_lines_bricky(image: np.ndarray, threshold, line_gap, line_length, max_lines):
    h, w = image.shape
    rho = 1
    theta = np.pi / 180

    # No of different angles and rhos
    num_angles = int(round(np.pi / theta))
    num_rhos = int(round(((w + h) * 2 + 1) / rho))

    # Accumulator for each combination of angles and rhos
    accumulator = np.zeros((num_angles, num_rhos), dtype=np.int32)

    # Detecting and storing all the non-zero points
    non_zero_points = list(np.argwhere(image))
    masked_image = np.zeros_like(image)
    masked_image[image != 0] = 1
    masked_image = masked_image.astype(np.uint8)

    # Tables for storing sine and cosine values
    cos_table, sin_table = image_processing.get_trig_table(num_angles, theta, rho)

    # Random number generator
    rand_gen = np.random.RandomState()
    num_remaining_points = len(non_zero_points)
    lines = []
    shift = 16

    while num_remaining_points > 0 and len(lines) < max_lines:
        random_index = rand_gen.randint(num_remaining_points)
        random_point = non_zero_points[random_index]

        non_zero_points[random_index] = non_zero_points[num_remaining_points - 1]
        num_remaining_points -= 1

        i, j = random_point[0], random_point[1]

        # Check if point already used by another line
        if masked_image[i, j] == 0:
            continue

        best_angle_idx = 0
        max_votes = threshold - 1

        # Find most promising direction from the point
        for n in range(num_angles):
            r = int(round(j * cos_table[n] + i * sin_table[n]))
            r += (num_rhos - 1) // 2
            votes = accumulator[n, r] + 1
            accumulator[n, r] = votes
            if max_votes < votes:
                max_votes = votes
                best_angle_idx = n

        # Check if threshold cleared
        if threshold > max_votes:
            continue

        # Line
        a = -sin_table[best_angle_idx]
        b = cos_table[best_angle_idx]
        x_start = j
        y_start = i

        a_gt_b = abs(a) > abs(b)
        flag = 1

        if a_gt_b:
            if a > 0:
                dx_start = 1
            else:
                dx_start = -1

            dy_start = int(round(b * (1 << shift) / abs(a)))
            y_start = (y_start << shift) + (1 << (shift - 1))
        else:
            flag = 0
            if b > 0:
                dy_start = 1
            else:
                dy_start = -1

            dx_start = int(round(a * (1 << shift) / abs(b)))
            x_start = (x_start << shift) + (1 << (shift - 1))

        line_end = [(0, 0), (0, 0)]

        for k in range(2):
            gap = 0
            x, y = x_start, y_start
            dx, dy = dx_start, dy_start

            if k > 0:
                dx, dy = -dx, -dy

            while True:
                if flag:
                    j1 = x
                    i1 = y >> shift
                else:
                    j1 = x >> shift
                    i1 = y

                if j1 < 0 or j1 >= w or i1 < 0 or i1 >= h:
                    break

                if masked_image[i1, j1]:
                    gap = 0
                    line_end[k] = (j1, i1)
                else:
                    gap += 1
                    if gap > line_gap:
                        break

                x += dx
                y += dy

        x_len = abs(line_end[1][0] - line_end[0][0])
        y_len = abs(line_end[1][1] - line_end[0][1])

        if x_len >= line_length or y_len >= line_length:
            for k in range(2):
                x, y = x_start, y_start
                dx, dy = dx_start, dy_start

                if k > 0:
                    dx, dy = -dx, -dy

                while True:
                    if flag:
                        j1 = x
                        i1 = y >> shift
                    else:
                        j1 = x >> shift
                        i1 = y

                    if masked_image[i1, j1]:
                        masked_image[i1, j1] = 0
                        for n in range(num_angles):
                            r = int(round(j1 * cos_table[n] + i1 * sin_table[n]))
                            r += (num_rhos - 1) // 2
                            accumulator[n, r] -= 1

                    if i1 == line_end[k][1] and j1 == line_end[k][0]:
                        break

                    x += dx
                    y += dy

            lines.append(
                [line_end[0][0], line_end[0][1], line_end[1][0], line_end[1][1]]
            )

    return np.array(lines)


def perpendicular_distance(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    if x2 - x1 == 0 and x4 - x3 == 0:
        return abs(x1 - x3)

    if x2 - x1 == 0:
        m2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - m2 * x3
        return abs(m2 * x1 + b2 - y1) / np.sqrt(m2**2 + 1)

    if x4 - x3 == 0:
        m1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - m1 * x1
        return abs(m1 * x3 + b1 - y3) / np.sqrt(m1**2 + 1)

    m1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - m1 * x1

    m2 = (y4 - y3) / (x4 - x3)
    b2 = y3 - m2 * x3

    if abs(m1 - m2) < 1e-10:
        return abs(b1 - b2) / np.sqrt(m1**2 + 1)

    def point_to_line_distance(xp, yp, m, b):
        return abs(m * xp - yp + b) / np.sqrt(m**2 + 1)

    d1 = point_to_line_distance(x1, y1, m2, b2)
    d2 = point_to_line_distance(x2, y2, m2, b2)

    d3 = point_to_line_distance(x3, y3, m1, b1)
    d4 = point_to_line_distance(x4, y4, m1, b1)

    return min(d1, d2, d3, d4)


def merge_similar_lines(lines, slope_threshold=0.1, distance_threshold=20):
    if len(lines) == 0:
        return lines

    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            slope = float("inf")
        else:
            slope = (y2 - y1) / (x2 - x1)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_data.append(
            {"line": line, "slope": slope, "length": length, "merged": False}
        )

    merged_lines = []

    for i in range(len(line_data)):
        if line_data[i]["merged"]:
            continue

        current_group = [i]
        line_data[i]["merged"] = True

        for j in range(i + 1, len(line_data)):
            if line_data[j]["merged"]:
                continue

            slope_diff = abs(line_data[i]["slope"] - line_data[j]["slope"])
            if line_data[i]["slope"] == float("inf") and line_data[j]["slope"] == float(
                "inf"
            ):
                slope_diff = 0
            elif line_data[i]["slope"] == float("inf") or line_data[j][
                "slope"
            ] == float("inf"):
                slope_diff = float("inf")

            perp_distance = perpendicular_distance(
                line_data[i]["line"], line_data[j]["line"]
            )

            if slope_diff <= slope_threshold and perp_distance <= distance_threshold:
                current_group.append(j)
                line_data[j]["merged"] = True

        x_vals, y_vals = [], []
        for idx in current_group:
            x1, y1, x2, y2 = line_data[idx]["line"]
            x_vals.extend([x1, x2])
            y_vals.extend([y1, y2])

        x_min, x_max = min(x_vals), max(x_vals)

        longest_line_idx = max(current_group, key=lambda idx: line_data[idx]["length"])
        x1, y1, x2, y2 = line_data[longest_line_idx]["line"]

        if x2 - x1 == 0:  # Vertical line
            merged_lines.append([x1, min(y_vals), x1, max(y_vals)])
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            y_min = slope * x_min + intercept
            y_max = slope * x_max + intercept
            merged_lines.append([x_min, y_min, x_max, y_max])

    return np.array(merged_lines)


def filter_by_slope(lines, min_abs_slope=0.5):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line

        if x2 - x1 == 0:
            slope = float("inf")
        else:
            slope = abs((y2 - y1) / (x2 - x1))

        if slope > min_abs_slope:
            filtered_lines.append(line)

    return np.array(filtered_lines)


def overlay_lines_bricky(original_image: np.ndarray, edges: np.ndarray):
    num_lines = 30
    lines = hough_lines_bricky(
        edges,
        threshold=50,
        line_gap=20,
        line_length=10,
        max_lines=num_lines,
    )

    merged_lines = filter_by_slope(lines, min_abs_slope=0.3)

    merged = False
    while not merged:
        n = len(merged_lines)
        if n <= 5:
            break

        merged_lines = merge_similar_lines(
            merged_lines, slope_threshold=0.3, distance_threshold=40
        )

        if len(merged_lines) == n:
            merged = True

    merged = False
    while not merged:
        n = len(merged_lines)
        if n <= 5:
            break

        merged_lines = merge_similar_lines(
            merged_lines, slope_threshold=0.4, distance_threshold=50
        )

        if len(merged_lines) == n:
            merged = True

    def line_length(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # merged_lines = [
    #     line
    #     for line in merged_lines
    #     if line_length(line[0], line[1], line[2], line[3]) >= 100
    # ]
    
    merged_lines = list(merged_lines)

    merged_lines.sort(
        key=lambda line: line_length(line[0], line[1], line[2], line[3]), reverse=True
    )
    num_lines = min(7, len(merged_lines))
    merged_lines = merged_lines[:num_lines]

    merged_lines = [
        [int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in merged_lines
    ]

    print("Original lines:", len(lines))
    print("After merging:", len(merged_lines))

    output_image = original_image.copy()

    for x1, y1, x2, y2 in merged_lines:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    return output_image, merged_lines


def bricky(original_image: np.ndarray):
    image = original_image.copy()

    # image_processing.display_heatmaps(image)
    # image_processing.color_histogram(image)

    # cv2.imshow("Original Image", original_image)
    # cv2.waitKey(0)

    blurred_image = brick_image_blur(image)
    # cv2.imshow("Preprocessed image", blurred_image)
    # cv2.waitKey(0)

    b = blurred_image[:, :, 0]
    g = blurred_image[:, :, 1]
    r = blurred_image[:, :, 2]

    grayscale = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    # cv2.imshow("Grayscale", grayscale)
    # cv2.waitKey(0)

    edges = sobel_edge_detection(grayscale)
    # cv2.imshow("Sobel edges", edges)
    # cv2.waitKey(0)

    threshold = 75
    edges_binary = (edges > threshold).astype(np.uint8) * 255
    # cv2.imshow("Thresholded edges", edges_binary)
    # cv2.waitKey(0)

    lined_image, lines = overlay_lines_bricky(
        original_image=original_image, edges=edges_binary
    )
    # cv2.imshow("lines", lines)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    return lined_image, lines
