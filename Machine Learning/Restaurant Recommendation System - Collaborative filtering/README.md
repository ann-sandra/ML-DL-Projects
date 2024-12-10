# Restaurant Recommendation System

## Overview  
This project implements a **Restaurant Recommendation System** using **collaborative filtering** methods to predict ratings and recommend restaurants to users. Collaborative filtering is a widely used technique in recommendation engines for predicting user preferences based on similar users or items. The project explores both **memory-based** and **model-based** approaches to provide recommendations.

---

## Approaches Implemented  

### 1. **Memory-Based Filtering**  
Memory-based collaborative filtering uses raw data to calculate similarities and generate recommendations.  

#### a. **User-Based Filtering**  
- **Method**: Pearson Correlation  
- **Process**:  
  Identifies users who are similar to the target user based on their ratings and generates recommendations from their preferences.

- **Formula**:  

The similarity between two users \(u\) and \(v\) is calculated using the following formula:

sim(u, v) = 

    ∑(i ∈ Iuv) (r(u,i) - r̄u)(r(v,i) - r̄v)
    --------------------------------------
    √[∑(i ∈ Iuv) (r(u,i) - r̄u)²] * √[∑(i ∈ Iuv) (r(v,i) - r̄v)²]

Where:
- r(u,i): Rating given by user \(u\) for item \(i\).
- r̄u: Average rating of user \(u\).
- Iuv: Set of items rated by both users \(u\) and \(v\).


- **Functions**:  
  - `pearson(user1, user2, df)`: Calculates similarity between two users.  
  - `get_neighbours(user_id, df)`: Identifies users most similar to the target user.  
  - `recommend(user, df, n_users=2, n_recommendations=2)`: Generates recommendations for a user based on neighbors' weighted scores.  

- **Example**:  
  The neighbors of user `U1103` are:  
  - `('U1068', similarity = 0.79)`  
  - `('U1028', similarity = 0.297)`  

#### b. **Item-Based Filtering**  
- **Method**: Slope-One Recommendation  
- **Process**:  
  Predicts ratings by calculating the average difference between ratings of item pairs and adjusting based on user preferences.

- **Formula**:  
  For an item (i) and another item (j):

<div align="center"> 
    dev(i, j) = (Σ (r<sub>u,j</sub> - r<sub>u,i</sub>)) / |U<sub>ij</sub>|
</div>

Where:

- U<sub>ij</sub>: Set of users who rated both items (i) and (j).  
- r<sub>u,j</sub>: Rating given by user (u) for item (j).

Predicted rating for item (i):

<div align="center"> 
    r̂<sub>u,i</sub> = (Σ [(dev(i, j) + r<sub>u,j</sub>) * f(i, j)]) / Σ f(i, j)
</div>

Where f(i, j) is the frequency of co-occurrence between (i) and (j).
 

- **Functions**:  
  - `get_dev_fr(data)`: Computes average differences and frequencies between item pairs.  
  - `slopeone(user, data)`: Predicts ratings for unrated items using weighted differences.

---

### 2. **Model-Based Filtering**  
Model-based collaborative filtering involves building a model to uncover patterns in the data.  

#### a. **Method**: Alternating Least Squares (ALS)  
- **Process**:  
  Uses matrix factorization to decompose the user-item interaction matrix into two lower-dimensional matrices \( U \) (user matrix) and \( P \) (item matrix) that represent latent factors.

- **Formula**:  
  Given a rating matrix \( R \):  
  \[
  R \approx U \cdot P^T
  \]
  Where:  
  - \( U \): User-feature matrix  
  - \( P \): Item-feature matrix  

  The optimization problem is to minimize the following cost function:  
  \[
  J = \sum_{(u, i) \in R} (R_{ui} - U_u \cdot P_i^T)^2 + \lambda \left( \| U \|^2 + \| P \|^2 \right)
  \]
  Where:  
  - \( \lambda \): Regularization parameter to prevent overfitting  

---

## Data Source  
The dataset used for this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).  

### Dataset Structure  
The primary dataset file is `rating_final.csv`, which contains the following columns:  
- `userID`  
- `placeID`  
- `rating`  
- `food_rating`  
- `service_rating`  

**Note**: Ratings are given on a scale of 0 to 2. Zero ratings are replaced with a very small value to distinguish them from missing ratings. Only overall ratings are used in this analysis.  

---

## Libraries Used  
- **pandas**  
- **numpy**  

---

## Advantages & Disadvantages of Approaches  

### Memory-Based Filtering  
**Advantages**:  
- Simple implementation  
- Easy to explain results  
- Supports new user addition seamlessly  

**Disadvantages**:  
- Slow with large datasets (loads all data into memory)  
- Sparse data may lead to missing recommendations  

### Model-Based Filtering  
**Advantages**:  
- Performs well on sparse datasets  
- Reduces overfitting issues  

**Disadvantages**:  
- Possible information loss due to dimensionality reduction  
- Results can be harder to interpret  

---

## Citation  
Blanca Vargas-Govea, Juan Gabriel González-Serna, Rafael Ponce-Medellín. *Effects of relevant contextual features in the performance of a restaurant recommender system.* In RecSys’11: Workshop on Context Aware Recommender Systems (CARS-2011), Chicago, IL, USA, October 23, 2011.  

This project demonstrates the implementation of collaborative filtering methods, including **user-based filtering**, **item-based filtering**, and **ALS**, for building a robust restaurant recommendation system.
