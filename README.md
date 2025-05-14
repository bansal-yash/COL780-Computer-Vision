# COL780 – Computer Vision  
**Assignments from the Computer Vision course (COL780) at IIT Delhi, taught by Prof. Chetan Arora**

This repository contains assignments that apply classical and deep learning-based computer vision techniques to solve real-world image processing and classification problems.

---

## 📁 Assignments

### [Assignment 1: Lane Boundary Detection](./Assignment_1_Lane_Boundary_Detection)
- Detected lane boundaries in ground images using classical image processing techniques.
- Techniques used:
  - Blurring, adaptive thresholding, Canny edge detection, masking, and Hough transforms.
- Performed heatmap-based analysis to tune hyperparameters.
- Implemented an image segregation method to apply different edge detection algorithms based on image characteristics.

---

### [Assignment 2: PatchCamelyon Image Classification](./Assignment_2_PatchCamelyon_Classification)
- Fine-tuned pretrained ResNet and VGG models for classifying tumor vs. non-tumor tissue in PatchCamelyon dataset.
- Conducted ablation studies on:
  - Data augmentations, optimizers, learning rate schedulers, loss functions, number of layers, and input image sizes.
- Developed a custom model in PyTorch incorporating Inception and Residual blocks.
- Achieved **91% accuracy** on the PatchCamelyon test dataset.

---

### [Assignment 3: Object Detection on Foggy Cityscapes](./Assignment_3_FoggyCityscapes_Object_Detection)
- Evaluated inference performance of pretrained **Deformable DETR** and **Grounding DINO** models using Hugging Face Transformers.
- Performed:
  - Qualitative evaluations via visual overlays of predicted vs. ground truth boxes.
  - Quantitative evaluations using `pycocotools` for metrics like mAP, class-wise average precision, and performance under varying IoU/score thresholds.
- Trained the Deformable DETR model on the dataset to improve mAP.
- Explored soft prompt tuning to enhance Grounding DINO’s performance.

---

Each assignment combines theory with practical implementation using modern computer vision frameworks and tools. Topics covered range from classical image processing to state-of-the-art deep learning-based detection and classification.
