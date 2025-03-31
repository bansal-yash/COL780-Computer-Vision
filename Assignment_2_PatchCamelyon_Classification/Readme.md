# PatchCamelyon Image Classification  

## Yash Bansal (2022CS51133)  

**Dataset Reference:** [PatchCamelyon GitHub Repository](https://github.com/basveeling/pcam)  

This repository contains the code for the COL780 Assignment 2: PatchCamelyon (PCam) Image Classification. The assignment involves classifying histopathology images using deep learning models.  

## Installation  

Ensure you have Python installed (preferably Python 3.12). Then, install the required dependencies by running:  

```bash
pip install -r requirements.txt
```  

## Dataset and Model Setup  

### Test Images  
Place all test images in a folder named `test_images`.  

### Trained Models  
Store all trained models in a folder named `trained_models`.  

Alternatively, you can download the trained models from the following link:  
[Trained Models](https://csciitd-my.sharepoint.com/my?id=%2Fpersonal%2Fcs5221133%5Fiitd%5Fac%5Fin%2FDocuments%2FCodes%2FCOL780%2FAssignment%5F2%2Ftrained%5Fmodels&ga=1)  

## Running the Code  

### Part 1: VGG16-based Model  
To run inference using a VGG16-based model, execute:  
```bash
python3 prediction_1.py
```  

### Part 2: Custom Architecture  
To run inference using the custom deep learning model, execute:  
```bash
python3 prediction_2.py
```  

### Part 3: Competitive Section  
To run the best-performing model for the competitive section, execute:  
```bash
python3 prediction_3.py
```  

## Output  

The results will be saved in `results.txt`.  