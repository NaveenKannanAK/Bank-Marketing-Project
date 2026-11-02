# Portuguese Bank Marketing – Customer Subscription Prediction

## Project Overview
This project analyzes Portuguese bank marketing campaign data and develops machine learning models to predict whether a customer will subscribe to a term deposit.

The project aims to help the bank identify potential customers and improve marketing campaign effectiveness.

## Project Objectives
- Perform Exploratory Data Analysis (EDA).
- Identify important factors influencing customer subscriptions.
- Prepare and preprocess the dataset.
- Build and compare machine learning classification models.
- Optimize the classification threshold.
- Identify important predictive features.
- Save the final model for future deployment.

## Dataset
- Dataset: Portuguese Bank Marketing Dataset
- Records: 41,176
- Target variable: `y`
- Target classes: Yes / No

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

## Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (Linear SVM)

## Final Model
Random Forest was selected based on the highest F1-score in the final model comparison.

| Metric | Score |
|---|---:|
| Accuracy | 87.30% |
| Precision | 45.21% |
| Recall | 60.02% |
| F1-score | 51.57% |
| ROC-AUC | 80.86% |
| PR-AUC | 48.18% |

## Project Structure
```text
Bank-Marketing-Project/
├── dataset/
├── notebooks/
├── saved_models/
├── reports/
├── images/
├── README.md
└── requirements.txt
```

## How to Run
1. Clone or download this repository.
2. Install the required packages:
   `pip install -r requirements.txt`
3. Open the Jupyter Notebook in the `notebooks/` folder.
4. Run the notebook cells in order.

## Saved Model Files
- `best_random_forest_model.pkl`
- `scaler.pkl`
- `feature_columns.pkl`
- `classification_threshold.pkl`

## Conclusion
The project demonstrates how data analysis and machine learning can support targeted bank marketing campaigns by identifying customers who are more likely to subscribe to term deposits.