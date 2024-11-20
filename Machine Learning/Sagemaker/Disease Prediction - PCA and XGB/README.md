# **Cardiovascular Disease Prediction**

This project aims to predict the presence or absence of cardiovascular disease in individuals based on health-related features. Using machine learning techniques, including XGBoost and Principal Component Analysis (PCA), we aim to develop a robust prediction model and evaluate its performance.

---

## **Project Overview**

Cardiovascular diseases (CVDs) are the leading cause of death worldwide, making early prediction critical for preventive healthcare. This project uses a dataset containing demographic, physical, and lifestyle attributes to predict CVD using machine learning models.  

---

## **Programming Language and Libraries**

- **Programming Language**: Python  
- **Libraries Used**:  
  - `NumPy`: Numerical computations.  
  - `Pandas`: Data manipulation.  
  - `Scikit-Learn`: Preprocessing, and classification models.  
  - `Matplotlib`: Visualization.  
  - `Seaborn`: Data visualization.  

---

## **Dataset Structure**

The dataset contains the following attributes:  

| **Feature**                 | **Description**                              | **Type**        |  
|-----------------------------|----------------------------------------------|-----------------|  
| `age`                       | Age in days                                 | Objective       |  
| `height`                    | Height in centimeters                       | Objective       |  
| `weight`                    | Weight in kilograms                         | Objective       |  
| `gender`                    | Gender (categorical code)                   | Objective       |  
| `ap_hi`                     | Systolic blood pressure                     | Examination     |  
| `ap_lo`                     | Diastolic blood pressure                    | Examination     |  
| `cholesterol`               | Cholesterol level (1: normal, 2: above normal, 3: well above normal) | Examination |  
| `gluc`                      | Glucose level (1: normal, 2: above normal, 3: well above normal) | Examination |  
| `smoke`                     | Smoking status (binary)                     | Subjective      |  
| `alco`                      | Alcohol intake (binary)                     | Subjective      |  
| `active`                    | Physical activity (binary)                  | Subjective      |  
| `cardio`                    | Target variable indicating CVD presence     | Target          |  

---

## **Exploratory Data Analysis (EDA)**

To understand patterns in the data, several visualizations were created using Seaborn, focusing on the following aspects:  

1. Distribution of age, height, and weight.  
2. Relationships between features such as blood pressure (`ap_hi`, `ap_lo`) and the presence of CVD.  
3. Analysis of categorical features (`cholesterol`, `gluc`) with respect to the target variable.  

---

## **Data Preprocessing**

1. **Missing Values**: Checked and handled missing or inconsistent data.  
2. **Feature Scaling**: Normalized numerical features for uniformity.  
3. **Train-Test Split**: Divided the dataset into training and test sets (80% training, 20% test).  

---

## **Model Building**

### **1. XGBoost Model**

An XGBoost classifier was trained on the dataset using the following hyperparameters:  
- **Depth**: 6  
- **Learning Rate**: 0.3  

**Performance Metrics on Test Set**:  
- **Precision**: 0.7902  
- **Recall**: 0.7299  
- **Accuracy**: 0.7672  

---

## **Dimensionality Reduction with PCA**

To reduce computational complexity, Principal Component Analysis (PCA) was applied using SageMaker. The original 11 features were reduced to 6 principal components, preserving the majority of the variance in the data.  

### **Model Training After PCA**

The XGBoost model was retrained on the transformed data using the same hyperparameters as before.  

**Performance Metrics on Test Set**:  
- **Precision**: 0.7902  
- **Recall**: 0.7299  
- **Accuracy**: 0.7672  

---

## **Results**

Both the original and PCA-transformed datasets yielded identical model performance, indicating the effectiveness of dimensionality reduction without compromising predictive power.  

### **Key Observations**:  
- Features such as `cholesterol`, `ap_hi`, and `ap_lo` significantly influence CVD prediction.  
- Dimensionality reduction effectively reduced computational complexity while maintaining model accuracy.  

---

## **How to Use**

### **Prerequisites**

1. Python 3.7+  
2. Install required libraries:  
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

---

## **Future Improvements**

1. **Hyperparameter Optimization**:  
   Use Grid Search or Bayesian Optimization to fine-tune model parameters for improved performance.  
2. **Additional Models**:  
   Experiment with deep learning models or ensemble techniques to enhance predictions.  
3. **Feature Engineering**:  
   Explore additional derived features and interaction terms for better insights.  
4. **Class Imbalance**:  
   Address any imbalance in the target variable using techniques such as SMOTE.  

