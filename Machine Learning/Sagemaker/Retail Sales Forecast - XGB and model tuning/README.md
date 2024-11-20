# **Retail Sales Forecast**

This project implements **XGBoost** and hyperparameter optimization using AWS SageMaker to forecast weekly retail sales for a particular department. By leveraging historical sales data, promotional markdowns, and store-specific features, the model helps make informed business decisions, optimize inventory, and improve retail processes.

---

## **Project Overview**

The objective of this project is to develop an accurate model to forecast weekly retail sales. The dataset includes information on stores, sales, promotional markdowns, and other economic indicators.  

**Key Features**:  
- Includes holiday effects (e.g., Super Bowl, Thanksgiving, Christmas).  
- Incorporates markdowns as promotional events to drive sales.  
- SageMaker is used for training and hyperparameter tuning.  

---

## **Dataset Structure**

The dataset consists of three files:  

1. **Features Dataset** (`Features_data_set.csv`):  
   Contains regional, store-level, and promotional data.  
   | Column        | Description                                 |  
   |---------------|---------------------------------------------|  
   | `Store`       | Store number.                              |  
   | `Date`        | Week.                                      |  
   | `Temperature` | Average temperature in the region.         |  
   | `Fuel_Price`  | Regional fuel cost.                        |  
   | `MarkDown1-5` | Promotional markdown data.                 |  
   | `CPI`         | Consumer Price Index.                      |  
   | `Unemployment`| Unemployment rate.                         |  
   | `IsHoliday`   | Indicates if the week is a holiday.        |  

2. **Sales Dataset** (`sales_data_set.csv`):  
   Historical sales data for each store and department.  
   | Column         | Description                                 |  
   |----------------|---------------------------------------------|  
   | `Store`        | Store number.                              |  
   | `Dept`         | Department number.                         |  
   | `Date`         | Week.                                      |  
   | `Weekly_Sales` | Sales for the given store and department.  |  
   | `IsHoliday`    | Indicates if the week is a holiday.        |  

3. **Stores Dataset** (`stores_data_set.csv`):  
   Metadata about each store.  
   | Column  | Description       |  
   |---------|-------------------|  
   | `Store` | Store number.     |  
   | `Type`  | Store type (A, B, C). |  
   | `Size`  | Store size (sq. ft.).|  

**Dataset Source**: [Kaggle - Retail Sales Forecast](https://www.kaggle.com/manjeetsingh/retaildataset)  

---

## **Programming Language and Libraries**

- **Programming Language**: Python  
- **Libraries Used**:  
  - `pandas`: Data manipulation and analysis.  
  - `numpy`: Numerical computations.  
  - `seaborn`: Data visualization.  
  - `matplotlib`: Plotting and visualizations.  
  - `zipfile`: Managing compressed files.  

---

## **Data Preprocessing**

### Steps:  

1. **Merging Datasets**:  
   - Combined `Features_data_set` and `sales_data_set` on `Store`, `Date`, and `IsHoliday`.  
   - Merged the result with `stores_data_set` on `Store`.  

2. **Handling Missing Values**:  
   - Imputed missing data using **KNN Imputer**.  

3. **Encoding**:  
   - Encoded categorical columns (`Type`, `Store`, `Dept`) using label encoding.  

4. **Feature Scaling**:  
   - Applied **StandardScaler** to normalize numerical features.  

5. **Data Visualization**:  
   - Explored correlations and insights between sales, markdowns, and holidays.  

---

## **Model Building**

### **1. XGBoost Regression**  

- **Hyperparameters**:  
   - **max_depth=5**: Limits tree depth to prevent overfitting while capturing important patterns.  
   - **learning_rate=0.1**: Controls how much each tree contributes to the final model, balancing speed and generalization.  
   - **n_estimators=100**: Specifies the number of trees in the model.  
   - **colsample_bytree=1**: Uses all features for building each tree.  
   - **reg_alpha=0** and **reg_lambda=1**: Apply regularization to avoid overfitting with moderate L2 regularization and no L1 penalty.  


- **Performance**:  
  - RMSE: **9779.869**  
  - R²: **0.819**  

### **2. XGBoost on SageMaker**  
- **Performance**:  
  - RMSE: **7492.593**  
  - R²: **0.894**  

### **3. Hyperparameter Optimization with SageMaker**  
- **SageMaker Optimized Hyperparameters**:
   - **max_depth=10**: Enables deeper trees to model complex relationships.  
   - **objective='reg:linear'**: Uses linear regression as the objective function for forecasting.  
   - **colsample_bytree=0.3**: Randomly selects 30% of features for each tree to reduce overfitting.  
   - **alpha=10**: Applies strong L1 regularization to improve model sparsity.  
   - **eta=0.1**: Maintains a low learning rate for smooth convergence.  
   - **num_round=100**: Trains the model over 100 boosting iterations for optimal accuracy.
      
- **Tuned Parameters**:  
  - Optimized for `max_depth`, `learning_rate`, `alpha`, and `colsample_bytree`.  

- **Performance**:  
  - RMSE: **4266.012**  
  - R²: **0.964**  

---

## **Results**

| **Model**            | **RMSE**   | **MSE**       | **MAE**   | **R²**   |  
|-----------------------|------------|---------------|-----------|----------|  
| XGBoost (Baseline)    | 9779.869   | 95645850.0    | 6435.3916 | 0.819    |  
| XGBoost (SageMaker)   | 7492.593   | 56138950.0    | 4353.634  | 0.894    |  
| Optimized XGBoost     | 4266.012   | 18198860.0    | 1811.6404 | 0.964    |  

---

## **How to Use**

### **Prerequisites**

1. Python 3.7+  
2. Install required libraries:  
   ```bash
   pip install pandas numpy seaborn matplotlib xgboost
   ```  

---

## **Future Improvements**

1. **Additional Features**:  
   - Incorporate external economic indicators (e.g., GDP, regional demographics).  

2. **Advanced Models**:  
   - Experiment with deep learning models like LSTMs for time-series forecasting.  

3. **Real-Time Forecasting**:  
   - Deploy the model to a cloud service for real-time sales predictions.  


