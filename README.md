# Heart Disease Prediction using Machine Learning

## Project Overview
This project is an end-to-end implementation of a machine learning pipeline aimed at predicting whether a patient has heart disease or not based on clinical features. The project utilizes various machine learning models and techniques to achieve high accuracy, including Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest classifiers. The data for this project is sourced from the UCI Heart Disease dataset.

## Problem Definition
The goal of this project is to answer the following question:
> Can we predict whether a patient has heart disease based on their medical attributes?

## Dataset
The dataset contains 14 key clinical attributes related to patient health, such as age, sex, chest pain type, cholesterol levels, fasting blood sugar, and more. The target variable is binary, indicating whether or not the patient has heart disease.

- Data source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

## Key Features
- `age`: Age of the patient in years
- `sex`: Gender of the patient (0 = Female, 1 = Male)
- `cp`: Chest pain type (0-3)
- `chol`: Serum cholesterol level
- `thalach`: Maximum heart rate achieved
- `fbs`: Fasting blood sugar level
- `restecg`: Resting electrocardiographic results

## Models Used
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Random Forest Classifier

## Steps Involved
1. **Data Cleaning and Exploration (EDA)**: 
   - Analyzing missing values, data distributions, and correlations between features.
   - Visualizations to understand data patterns.

2. **Model Building**:
   - Building Logistic Regression, KNN, and Random Forest models.
   - Hyperparameter tuning with `GridSearchCV` and `RandomizedSearchCV`.
   - Cross-validation for model evaluation.

3. **Model Evaluation**:
   - Metrics used: Accuracy, Precision, Recall, F1-Score.
   - ROC Curve and AUC score to evaluate model performance.

4. **Model Comparison**:
   - Comparative analysis of model performance using bar plots and metrics.

## Results
- **Best Model**: Logistic Regression achieved the best accuracy (88.52%) on the test dataset.
- **Hyperparameter Tuning**: Tuning improved the Random Forest model's accuracy from 83% to 86%.

## Visualizations
- ROC Curve, Precision-Recall Curve, and Confusion Matrix for model performance.
- Feature importance visualizations to understand which features contribute most to the model's decision-making process.

## Tools and Libraries
- **Python**: `pandas`, `NumPy`, `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Model Tuning**: `GridSearchCV`, `RandomizedSearchCV`

## Conclusion
The Logistic Regression model was the most effective in predicting heart disease, though there is still room for improvement by integrating more complex models or feature engineering techniques.

## How to Run This Project
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
