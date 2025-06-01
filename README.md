# Bank Customer Churn Prediction

This project is an end-to-end machine learning solution designed to predict customer churn in a banking context. The ability to identify customers likely to leave is essential for banks to retain clients and reduce potential revenue loss.

The project includes data preprocessing, model development, evaluation, and deployment via an interactive Streamlit web application.

## Dataset

The dataset used is from [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data), containing information on bank customers, their activities, and whether they churned or not.

## Web Application

A Streamlit web application was built to interactively predict customer churn. Users can input customer data and receive a prediction result.
Output: Whether the customer is likely to leave the bank or stay.

### <ins> Demo </ins>
A demo video showcasing the Streamlit web application.

https://github.com/user-attachments/assets/bad4add3-824e-483b-a54d-c161ff395efc


## Machine Learning Models Used

The following machine learning models were trained and compared:

- Logistic Regression
- Random Forest (Final Model)
- Support Vector Classifier (SVC)
- Gradient Boosting
- XGBoost


## Project Workflow
1. Data Inspection
    - Load and preview the dataset
    - Understand data structure and types
2. Data Cleaning
    - Handle missing values
    - Drop irrelevant columns (e.g. RowNumber, CustomerId, Surname)
3. Outlier Detection
    - Identify outliers using box plots
4. Feature Engineering
    - Encode categorical variables
    - Scale numerical features
5. Exploratory Data Analysis (EDA)
    - Visualize distributions, correlations, and churn patterns
6. Model Training
    - Train and evaluate models
    - Compare results using metrics, confusion matrix
7. Hyperparameter Tuning
    - Use GridSearchCV to improve model performance
8. Threshold Optimazation
    - Find optimal classification thresholds to maximize F1-score and recall
9. Feature Removal Analysis
    - Evaluate model performance with and without selected features
10. Model Selection
    - Choose best model based on performance metrics and business relevance
    - Final model: Random Forest using all features due to highest recall
11. Web Application Deployment
    - Developed and deployed a Streamlit web application
    - App includes dataset information, input fields, and displays prediction results



## Model Evaluation
Only the three top models are presented here, based on their performance after hyperparameter tuning and threshold Optimization.

| Model | Accuracy | Precision | Recall | F1-Score |
| -------- | -------- | -------- | -------- | -------- |
| Random Forest | 0.845 | 0.611 | 0.701 | 0.653 |
| Gradient Boosting | 0.867 | 0.790 | 0.491 | 0.606 |
| XGBoost | 0.864 | 0.774 | 0.489 | 0.599 |



The chart below compares model performance (Accuracy, Precision, Recall, and F1-Score) before and after feature removal and threshold optimization:

![Model Comparison - All vs Removed Features](https://github.com/user-attachments/assets/1e810294-a94c-433c-acc4-392a6c27c6e6)

### <ins> Key Insights </ins>
<ins>With reduced features:</ins>
- **Random Forest** had the highest recall (70.12%), detecting more churners. 
- **XGBoost** had stronger precision, resulting in fewer false positives.  

<ins> With all features: </ins> <br>
Similar results: **Random Forest** leads in recall (70.12%), **XGBoost** offers a better balance between precision and accuracy.  

### Final Decision
Given that recall is more important in a churn prediction context (we aim to identify as many at-risk customers as possible), Random Forest using all features was selected as the final model. Although XGBoost had competitive precision and accuracy, Random Forest better meets the business objective of minimizing false negatives (missed churners).


## Libraries Used:
### 🖥 <ins>Web Application</ins>
<div align="left">
  <img src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" alt="Python-Logo">
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit-Logo">
</div>

### 📈<ins>Data Analysis</ins>
<div align="left">
  <img src="https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas-Logo">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn-Logo">
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy-Logo">
</div>

### 📉 <ins>Data Visualization</ins>
<div align="left">
  <img src="https://img.shields.io/badge/Matplotlib-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" alt="matplotlib-Logo">
  <img src="https://img.shields.io/badge/Seaborn-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" alt="seaborn-Logo">
</div>



## Project Structure
```
.
├── Data
│   ├── Customer-Churn-Records.csv          # Raw dataset
│   └── Processed_Customer_Data.csv         # Cleaned and processed dataset
│
├── models
│   ├── Random_Forest_Classifier_model.pkl  # Final trained model
│   └── scaler.pkl                          # Preprocessing scaler
│
├── resources                               # Visual resources
│   └── images/*.png                        # Images used in the project
│
├── app.py                                  # Streamlit app for deployment
├── customer_churn.ipynb                    # Jupyter notebook with data and model analyses
├── model.py                                # Model training and evaluation script
└── requirements.txt                        # Required Python packages

```

## Installation Guide

 1. Clone the Repository
```bash
git clone https://github.com/yzcodeX/customer-churn-prediction.git
cd customer-churn-prediction
```
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Launch the Streamlit App
```bash
streamlit run app.py
```

if the above command doesn't work, try this instead:
```bash
python -m streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.


## Future Improvements
- Enhance the Streamlit app with a more user-friendly front-end.
- Creating an interactive dashboard using Tableau.

## License
This project is licensed under the  <a href="https://github.com/yzcodeX/customer-churn-prediction/blob/main/LICENSE"> MIT License.
