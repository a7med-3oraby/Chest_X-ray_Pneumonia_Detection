# 🩺 Chest X-ray Pneumonia Detection using HOG + PCA + KNN

This project implements a classical machine learning approach to detect pneumonia from chest X-ray images using **Histogram of Oriented Gradients (HOG)** features, **PCA** for dimensionality reduction, **SMOTE** for balancing, and **K-Nearest Neighbors (KNN)** for classification.

## 📂 Dataset

The project uses the public [Chest X-ray Pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle, which contains labeled X-ray images for:

- `NORMAL` (healthy lungs)
- `PNEUMONIA` (infected lungs)

## 🧠 Pipeline Overview
**1. 🖼️Image Preprocessing** 

  - Grayscale conversion.

  - Resized to 64×64.

  - Normalized to range [0, 1]
    

**2. 📐Feature Extraction**

  - Extracted HOG features from each image using skimage.feature.hog
    

**3. 📉Dimensionality Reduction**

  - Used PCA to reduce HOG feature vectors to 100 components.
    
    
**4. ⚖️Class Balancing**

  - Use SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance in training data.
    

**5. 🤖Model Training**

  - Used KNN classifier (optimal K found via elbow method).


**6. 📊Evaluation**

  - Evaluated on test data using accuracy, classification report, and confusion matrix.
    

**7. 🔍Single Image Prediction**

  - Load and preprocess a new image

  - Extract HOG features

  - Transform using PCA

  - Predict class using the trained model

  - Display the image and prediction result


## 📈 Results
Classifier: KNN (k=3)

Final Accuracy: ~ 84 % (taking in consideration that we didn't use any deep laerning techniques , only a classical machine learning alogrithms, and also the Imbalance in the data.)

Evaluation Metrics: Precision, Recall, F1-score, Confusion Matrix.

## 📊 Visualizations
Class distribution before/after SMOTE

Elbow method to select optimal k

Confusion matrix on test set

Example image predictions (NORMAL / PNEUMONIA)
## 🛠️ Tools & Technologies

- **Python**
- **PIL (Python Imaging Library)** and **scikit-learn: KNeighborsClassifier, PCA, metrics**
- **scikit-image: hog** 
- **imbalanced-learn: SMOTE**



## 📌 Project Highlights
- Classical ML approach (no deep learning)

- Efficient pipeline for medical image classification

- Combines image preprocessing, feature engineering, and supervised learning
