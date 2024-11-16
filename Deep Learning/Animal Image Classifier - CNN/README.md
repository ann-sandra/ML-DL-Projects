# **Animal Image Classifier**

This project is an implementation of a Convolutional Neural Network (CNN) to classify images into two categories: **cats** and **dogs**. The model was trained on labeled datasets and achieved an impressive accuracy of **90%** after **25 epochs**. 

---

## **Project Overview**
The objective of this project is to classify input images of size **64x64x3** into one of two classes: **cats** or **dogs**. The implementation is based on a **Convolutional Neural Network (CNN)**, leveraging the hierarchical feature extraction capabilities of CNNs for image classification tasks.

---

## **Dataset Structure**
The dataset is organized as follows:

```
dataset/
│
├── training_set/
│   ├── cats/
│   ├── dogs/
│
└── test_set/
    ├── cats/
    ├── dogs/
```

- **Training Set:** Contains labeled images of cats and dogs used for training the model.
- **Test Set:** Contains labeled images used for evaluating the model's performance.

---

## **Model Architecture**
The CNN model consists of the following layers:

1. **Input Image Dimensions:** \( 64 \times 64 \times 3 \) (height, width, RGB channels)
2. **Convolutional Layer 1:** 
   - Filter size: \( 3 \times 3 \), Stride: 1
   - Output: \( 62 \times 62 \times n \)
3. **Max Pooling 1:** 
   - Filter size: \( 2 \times 2 \), Stride: 2
   - Output: \( 31 \times 31 \times n \)
4. **Convolutional Layer 2:** 
   - Filter size: \( 3 \times 3 \), Stride: 1
   - Output: \( 29 \times 29 \times m \)
5. **Max Pooling 2:** 
   - Filter size: \( 2 \times 2 \), Stride: 2
   - Output: \( 14 \times 14 \times m \)
6. **Flattening:** Converts the 3D matrix into a 1D vector for the dense layers.
7. **Fully Connected Layer (ANN):** Dense layer with 128 neurons and ReLU activation.
8. **Output Layer:** 
   - Number of neurons: 2 (for binary classification)
   - Activation Function: Softmax

---

## **Implementation Details**
- **Programming Language:** Python
- **Libraries Used:**
  - TensorFlow/Keras
- **Training Configuration:**
  - Epochs: 25
  - Optimizer: Adam
  - Loss Function: Binary Crossentropy
- **Data Preprocessing:**
  - Resizing images to \( 64 \times 64 \)
  - Normalizing pixel values to the range [0, 1]

---

## **Results and Performance**
- **Accuracy on Test Data:** ~90%
---

## **How to Use**
### **Prerequisites**
1. Python 3.7+
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Future Improvements**
1. Extend to more animal categories (multi-class classification).
2. Experiment with data augmentation to improve generalization.
3. Use transfer learning with pre-trained models like VGG16 or ResNet.

---
