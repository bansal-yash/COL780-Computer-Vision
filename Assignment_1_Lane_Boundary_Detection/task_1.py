import os
import cv2
import grass
import brick
import numpy as np
from multiprocessing import Pool

# python3 main.py 1 input_dataset Task1_output_images


def segregate_image(image: np.ndarray):
    height, width, _ = image.shape

    if height < width:
        return "bricky"

    cropped_image = image[int(0.20 * height) :, :]

    lower_red = np.array([0, 0, 50])
    upper_red = np.array([50, 50, 255])

    lower_green = np.array([0, 50, 0])
    upper_green = np.array([50, 255, 50])

    red_mask = cv2.inRange(cropped_image, lower_red, upper_red)
    green_mask = cv2.inRange(cropped_image, lower_green, upper_green)

    red_ratio = np.sum(red_mask > 0) / red_mask.size
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    if green_ratio > red_ratio:
        return "grassy"
    else:
        return "bricky"


def process_image(image_name, input_folder, output_folder):

    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    output_path = os.path.join(output_folder, image_name)

    category = segregate_image(image=image)

    if category == "grassy":
        print(f"Processing white {image_name}...")
        lined_image, _ = grass.grassy(image)
        path_2 = os.path.join("outputs_grassy", image_name)

        cv2.imwrite(output_path, lined_image)
        cv2.imwrite(path_2, lined_image)
        print("done " + image_name)

    else:
        print(f"Processing brick {image_name}...")
        lined_image, _ = brick.bricky(image)
        path_2 = os.path.join("outputs_bricky", image_name)
        cv2.imwrite(output_path, lined_image)
        cv2.imwrite(path_2, lined_image)


def task1(input_folder, output_folder):
    # for image_name in os.listdir(input_folder):
    #     process_image(image_name, input_folder, output_folder)

    with Pool() as pool:
        pool.starmap(
            process_image,
            [
                (image_name, input_folder, output_folder)
                for image_name in os.listdir(input_folder)
            ],
        )
