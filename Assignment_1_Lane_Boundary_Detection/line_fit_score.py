import numpy as np
from itertools import combinations


def get_line_fit_score(lines):
    intersections = []

    for (x1, y1, x2, y2), (x3, y3, x4, y4) in combinations(lines, 2):
        A1, B1, C1 = y2 - y1, x1 - x2, x2 * y1 - x1 * y2
        A2, B2, C2 = y4 - y3, x3 - x4, x4 * y3 - x3 * y4

        det = A1 * B2 - A2 * B1
        if det == 0:
            continue

        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        intersections.append((x, y))

    if len(intersections) == 0:
        return 0

    intersections = np.array(intersections)
    centroid = np.mean(intersections, axis=0)
    distances = np.linalg.norm(intersections - centroid, axis=1)

    return np.sum(distances)
