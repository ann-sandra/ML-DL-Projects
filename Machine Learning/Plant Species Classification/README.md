# **Plant Species Classifier**

This project predicts the species of a plant from the Iris dataset into one of three classes: **Iris-setosa**, **Iris-versicolor**, or **Iris-virginica**. The classification is based on the features **sepal length**, **sepal width**, **petal length**, and **petal width**. Multiple classification models built with own algorithms are implemented and evaluated, including **Naive Bayes**, **K-Nearest Neighbors (KNN)**, and **Decision Tree**, with performance validated using **K-Fold Cross-Validation**.

---

## **Project Overview**
The Iris dataset is one of the most well-known datasets in machine learning and is often used for classification tasks. This project evaluates several models to determine the best-performing algorithm for classifying plant species. A custom implementation of **K-Nearest Neighbors (KNN)** is also developed to compare its performance with the Scikit-Learn implementation.

The primary goal is to create a robust model that achieves high accuracy while avoiding overfitting, achieved through **K-Fold Cross-Validation**.

---

## **Dataset Structure**
The dataset is organized as follows:

```
dataset/
├── Iris.csv
```

- **Columns Include:**
  - **Sepal Length (cm):** Length of the sepal.
  - **Sepal Width (cm):** Width of the sepal.
  - **Petal Length (cm):** Length of the petal.
  - **Petal Width (cm):** Width of the petal.
  - **Species:** Target variable with three classes: `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`.

---

## **Data Preprocessing**
The following steps were performed to prepare the data for training and testing:

1. **Feature Scaling:**
   - Normalized the numerical features (`sepal length`, `sepal width`, `petal length`, `petal width`) to a range between 0 and 1.

2. **Data Splitting:**
   - Split the dataset into **training** (67%) and **test** (33%) sets.

---

## **Model Building and Results**

### **1. Naive Bayes**
- **Method:** Gaussian Naive Bayes  
- **Validation:** K-Fold Cross-Validation with `n=10`  
- **Accuracy:** 97%  
- **Strength:** High accuracy with cross-validation ensures that the model is well-generalized.

---

### **2. K-Nearest Neighbors (KNN) - Scikit-Learn Implementation**
- **Configuration:**  
  - **Distance Metric:** Euclidean Distance  
  - **Number of Neighbors (k):** 5  
- **Validation:** K-Fold Cross-Validation with `n=10`  
- **Accuracy:** 83.3%  

---

### **3. K-Nearest Neighbors (KNN) - Custom Implementation**
A custom implementation of KNN was developed to better understand its mechanics. The custom KNN functions include:

- **`knn_predict`**: Predicts the class of a test point.
- **`find_nearest_neighbors`**: Finds the `k` nearest neighbors of a point based on Euclidean distance.
- **`majority_vote`**: Determines the predicted class based on the majority vote of the neighbors.

- **Validation:** K-Fold Cross-Validation with `n=10`  
- **Accuracy:** 84.6% (higher than the Scikit-Learn implementation).
<img src="KNN Iris.png" alt="KNN" width="500"/>  

---

### **4. Decision Tree**
- **Criteria:** Entropy  
- **Accuracy:** 90% on the test set.  
- **Strength:** Simple and interpretable model with good accuracy.

---

## **Implementation Details**
- **Programming Language:** Python  
- **Libraries Used:**
  - `NumPy`: For numerical computations.  
  - `Pandas`: For data manipulation.  
  - `Matplotlib`: For visualizing decision boundaries and predictions.  
  - `Scikit-Learn`: For implementing Naive Bayes, KNN, and Decision Tree models.

---

## **How to Use**
### **Prerequisites**
1. Python 3.7+
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
---

## **Future Improvements**
1. Incorporate additional features or datasets to expand the scope of classification tasks.  
2. Apply advanced models like Random Forest, Gradient Boosting, or Neural Networks for higher accuracy.  
3. Implement a web interface for user-friendly classification.

---

## **Acknowledgements**
- The dataset used in this project is sourced from the **Iris Dataset**, originally introduced by **Ronald A. Fisher** in 1936.  
- Special thanks to the machine learning community for resources and insights.


