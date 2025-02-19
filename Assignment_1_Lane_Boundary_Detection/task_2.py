import os
import cv2
import grass
from multiprocessing import Pool
from line_fit_score import get_line_fit_score

# python3 main.py 2 input_dataset grassy.csv


def process_image(image_name, input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    print(f"Processing {image_name}...")

    lined_image, lines = grass.grassy(image)
    path_2 = os.path.join("outputs_grassy", image_name)
    score = get_line_fit_score(lines)

    cv2.imwrite(path_2, lined_image)
    print(f"\n{image_name}: {score}\n")

    return (image_name, score)


def task2(input_folder, output_csv):
    results = []
    with Pool() as pool:
        results = pool.starmap(
            process_image,
            [
                (image_name, input_folder)
                for image_name in os.listdir(input_folder)
            ],
        )

    print(results)
    results.sort(key=lambda x: x[0])

    with open(output_csv, "w") as f:
        f.write("Image_Name,Line_Fit_Score\n")

        for image_name, score in results:
            f.write(f"{image_name},{score:.2f}\n")
