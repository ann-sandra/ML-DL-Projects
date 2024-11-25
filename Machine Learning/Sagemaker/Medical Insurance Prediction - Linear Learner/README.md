# **Medical Insurance Premium Prediction**

This project leverages **AWS SageMaker** and **S3** to predict the medical insurance premium for individuals based on their personal details. The objective is to estimate health insurance costs incurred by individuals using machine learning techniques such as **Linear Learner** and **Artificial Neural Networks (ANN)**.

---

## **Problem Statement**

The aim of the project is to predict the health insurance premium incurred by individuals based on the following features:
- **Age**: Age of the individual.
- **Sex**: Gender of the individual (Female, Male).
- **BMI**: Body Mass Index, ideally ranging between 18.5 and 24.9.
- **Children**: Number of dependents covered by health insurance.
- **Smoker**: Smoking habits (Yes/No).
- **Region**: Geographical area in the US (Northeast, Southeast, Southwest, Northwest).
- **Charges**: Individual medical costs billed by health insurance.

### **Dataset**
- **Source**: [Medical Insurance Dataset](https://www.kaggle.com/mirichoi0218/insurance)  
- **Structure**: The dataset contains 7 columns:
  - `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and `charges`.

---

## **Libraries Used**

- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `seaborn`, `matplotlib`
- **AWS Integration**: `boto3`, `sagemaker`

---

## **Data Preprocessing**

1. **Handling Missing Values**:  
   - Null values were replaced using **KNN Imputer**, which accommodates uneven data distributions.  

2. **Encoding Categorical Columns**:  
   - `sex` and `region` were encoded using **One-Hot Encoding**.

3. **Feature Scaling**:  
   - Numerical columns were scaled to a range of 0–1 using **Standard Scaler**.  
   - This ensures all features are on the same scale, which is crucial for both **Linear Learner** and **ANN** models.  

---

## **Exploratory Data Analysis (EDA)**

### **Key Insights**

1. **Age vs Charges**:  
   - As age increases, the insurance charges also tend to increase.  
   - There’s a noticeable positive correlation between `age` and `charges`.  

2. **Dataset Balance**:  
   - The dataset is well-balanced across categorical features such as `sex`, `region`, and `smoker`.  
   - No oversampling or undersampling was necessary.

3. **Outliers**:  
   - Outliers were detected in the `BMI` column (range: 15.3–50).  
   - These outliers were capped using the **90th percentile quantile**.  
   - No significant outliers were found in `age` and `children`.  

4. **Feature Distribution**:  
   - **Age**: Most data points are distributed between 20 and 40 years.  
   - **Children**: Majority of individuals have zero dependents.  

---

## **Model Building**
Data was split into **80% training** and **20% testing**.

### **Linear Learner (AWS SageMaker)**

1. **Data Preparation**:  
   - Data was converted to **RecordIO format** using AWS SageMaker's utilities.  
   - Training and testing datasets were uploaded to **AWS S3** for model training.

2. **Hyperparameters**:  
   - **Feature Dimensions**: 8  
   - **Batch Size**: 100  
   - **Epochs**: 100  
   - **Number of Models**: 32  
   - **Loss Function**: Absolute Loss  

3. **Training Process**:  
   - Trained the model using `ml.c4.xlarge` instance.  
   - Deployed the trained model endpoint using `ml.m4.xlarge`.  

4. **Evaluation Metrics**:
   - **Root Mean Squared Error (RMSE)**: 5090  
   - **Mean Squared Error (MSE)**: 25,952,588.07  
   - **Mean Absolute Error (MAE)**: 2824.19  
   - **R² Score**: 0.755  
   - **Adjusted R²**: 0.747  

---

### **Artificial Neural Network (ANN)**

1. **Architecture**:
   - **Input Layer**: 50 neurons, activation: ReLU  
   - **Hidden Layer 1**: 150 neurons, activation: ReLU, dropout: 0.5  
   - **Hidden Layer 2**: 150 neurons, activation: ReLU, dropout: 0.5  
   - **Hidden Layer 3**: 50 neurons, activation: Linear  
   - **Output Layer**: 1 neuron, activation: Linear  

2. **Hyperparameters**:
   - **Loss Function**: Mean Squared Error  
   - **Optimizer**: Adam  
   - **Epochs**: 100  
   - **Batch Size**: 20  
   - **Validation Split**: 0.2  

3. **Evaluation Metrics**:
   - **RMSE**: 5090.16  
   - **MSE**: 25,909,768.0  
   - **MAE**: 2854.26  
   - **R² Score**: 0.833  
   - **Adjusted R²**: 0.827  

4. **Observations**:
   - The model showed signs of **overfitting** after 10 epochs, as the validation loss fluctuated while training loss decreased consistently.  
   - To avoid overfitting, training should ideally stop at 10 epochs.

---

## **Results**

| **Metric**        | **Linear Learner** | **ANN**         |  
|--------------------|--------------------|-----------------|  
| RMSE              | 5090              | 5090.16         |  
| MSE               | 25,952,588.07     | 25,909,768.0    |  
| MAE               | 2824.19           | 2854.26         |  
| R² Score          | 0.755             | 0.833           |  
| Adjusted R²       | 0.747             | 0.827           |  

- While both models performed similarly in terms of error, **ANN** achieved higher R² and Adjusted R² values, indicating it explains the variance in data better than Linear Learner.

---

## **Future Scope**

1. **Hyperparameter Optimization**:
   - Implement advanced optimization techniques such as **Bayesian Optimization** or **GridSearchCV** to tune the models further.  

2. **Feature Engineering**:
   - Explore additional features such as medical history or lifestyle factors for improved predictions.  

3. **Model Comparison**:
   - Experiment with other models such as **XGBoost** or **Random Forest Regressor** to compare performance.  

4. **Regularization**:
   - Apply regularization techniques like L2 or L1 to further reduce overfitting in ANN.

---

## **How to Run**

1. **Set Up AWS SageMaker**:
   - Upload the dataset to an **S3 bucket**.
   - Configure the SageMaker environment with appropriate roles and permissions.

2. **Install required libraries:  **:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn

---

## **Conclusion**

This project demonstrates how to predict insurance premiums using machine learning techniques on AWS SageMaker. The **ANN model** outperformed the **Linear Learner** in explaining variance in the data, making it the better choice for this use case. Both models, however, showed potential for further improvement through hyperparameter tuning and feature enhancement.

