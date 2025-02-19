import os
import sys
import task_1
import task_2
import time

# python3 main.py 1 input_dataset Task1_output_images
# python3 main.py 2 input_dataset grassy.csv

if __name__ == "__main__":
    task = int(sys.argv[1])

    if task == 1:
        input_folder = sys.argv[2]
        output_folder = sys.argv[3]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not os.path.exists("outputs_grassy"):
            os.makedirs("outputs_grassy")

        if not os.path.exists("outputs_bricky"):
            os.makedirs("outputs_bricky")

        t1 = time.time()

        task_1.task1(input_folder, output_folder)

        t2 = time.time()
        print("time taken for all the images:- ", (t2 - t1) / 60, " mins")

    if task == 2:
        input_folder = sys.argv[2]
        output_csv = sys.argv[3]

        if not os.path.exists("outputs_grassy"):
            os.makedirs("outputs_grassy")

        t1 = time.time()

        task_2.task2(input_folder, output_csv)

        t2 = time.time()
        print("time taken for all the images:- ", (t2 - t1) / 60, " mins")
