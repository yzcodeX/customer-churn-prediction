import os
import joblib
import pandas    as pd
import streamlit as st
from PIL         import Image

# load model and scaler
MODEL_PATH  = "./models/Random_Forest_Classifier_model.pkl"
SCALER_PATH = "./models/scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    raise FileNotFoundError(f"Model or Scaler not found in {MODEL_PATH} or {SCALER_PATH}.")

# function for data loading
def load_data():
    df = pd.read_csv("Data/Processed_Customer_Churn_Data.csv")
    return df

# function for data analysis (EdA)
def eda(df):
    st.subheader("Data preview")
    st.write(df.head())
    st.write("Number of Records: 10.000")
    st.divider()
    
    st.subheader("Data description")
    st.write(df.describe().T)
    st.divider()
   
    st.subheader("Customer exits by Card Type")
    image = Image.open("resources/piechart_Churn_by_Card_Type.png")
    st.image(image, caption = "percentage distribution of Card Types", use_container_width = True)
    st.divider()
    
    st.subheader("Customer exits by Gender and Geography")
    image = Image.open("resources/subplot_churn_by_gender_geography.png")
    st.image(image, caption = "Churn distribution by Gender and Geography", use_container_width = True)
    st.divider()
    
    st.subheader("Customer exits by Age and Tenure (years)")
    image = Image.open("resources/subplot_distribution_age_tenure.png")
    st.image(image, caption = "Churn distribution by Age and Tenure (years)", use_container_width = True)
    st.divider()
    
    st.subheader("Number of exit Customer by Member Activity")
    image = Image.open("resources/countplot_active_member.png")
    st.image(image, caption = "Customer Churn by Activity (0 = not active; 1 = active)", use_container_width = True)

# -------------------------------------
# function for categorical data
def preprocess_input(input_data, df):
    # create a DataFrame from user input
    input_df = pd.DataFrame([input_data])

    categorical_features = ['Geography', 'Card Type', 'Gender']
    
    # convert categorical features into dummy variables (One-Hot-Encoding)
    input_df = pd.get_dummies(input_df, columns=categorical_features)
    
    # all columns are present (including the dummy variables)
    all_columns = [col for col in df.columns if col not in ['Exited']]  # remove unnecessary column
    
    missing_columns = set(all_columns) - set(input_df.columns)
    for col in missing_columns:
        input_df[col] = 0  # add missing columns with 0 values

    # making sure columns are in the same order as in the training data set
    input_df = input_df[all_columns]
    input_df = input_df.astype(int)
    return input_df

# function for prediction
def predict_churn(input_data):
    threshold = 0.287 # optimal threshold for Random Forest
    
    input_df         = preprocess_input(input_data, df)
    input_scaled     = scaler.transform(input_df)
    prediction_proba = model.predict_proba(input_scaled)[:,1]
    prediction       = 1 if prediction_proba >= threshold else 0
    
    return prediction

# function that explains the features
def feature_explanation():
    st.write("CreditScore: Numerical value representing the customer's creditworthiness.")
    st.write("Age: Customer's age in years.")
    st.write("Tenure: Number of years the customer has been with the bank.")
    st.write("Balance: Amount of money the customer holds in their account.")
    st.write("NumOfProducts: Number of products the customer has with the bank.")
    st.write("HasCrCard: Indicates whether the customer owns a credit card (1 = Yes, 0 = No).")
    st.write("IsActiveMember: Indicates whether the customer is an active member (1 = Yes, 0 = No).")
    st.write("EstimatedSalary: Customer's estimated annual salary.")
    st.write("Satisfaction Score: Rating given by the customer regarding service satisfaction.")
    st.write("Points Earned: Loyalty points accumulated through credit card usage.")
    st.write("Gender: Customer's gender.")
    st.write("Geography: Customer's country of residence.")
    st.write("Card Type: Type of credit card the customer holds (e.g., Gold, Silver, Platinum).")

# -----------------------------------
# start streamlit app
st.title("Customer churn prediction with Random Forest Classifier")
df = load_data()

# navigation with slidebar
menu = st.sidebar.radio("Navigation", ["Home", "Data analysis", "Prediction"])

if menu == "Home":
    st.image("resources/churn_banner.png") 
    st.write("App realised by: yzCodeX")
    st.header("Feature explanation")
    feature_explanation()
    
elif menu == "Data analysis":
    eda(df)
elif menu == "Prediction":
    st.subheader("Churn prediction")
    
    input_data = {
        "CreditScore":       st.number_input("Credit Score", min_value = 300, max_value = 900, value = 650),
        "Age":               st.number_input("Age", min_value = 18, max_value = 100, value = 35),
        "Tenure":            st.slider("Customer Loyalty (years)", min_value = 0, max_value = 10, value = 5),
        "Balance":           st.number_input("Balance", min_value = 0, value = 50000),
        "NumOfProducts":     st.number_input("Number of Products", min_value = 1, max_value = 10, value = 2),
        "HasCrCard":         st.radio("Has Credit Card?", options = [0, 1], index = 0),
        "IsActiveMember":    st.radio("Is Active Member?", options = [0, 1], index = 1),
        "EstimatedSalary":   st.number_input("Estimated Salary", min_value = 0, value = 50000),
        "SatisfactionScore": st.slider("Satisfaction Score", min_value = 0, max_value = 10, value = 8),
        "PointEarned":       st.number_input("Points Earned", min_value = 0, value = 1000),
        "Gender":            st.selectbox("Gender", ["Female", "Male"]),
        "Geography":         st.selectbox("Geography", ["France", "Germany", "Spain"]),
        "Card Type":         st.radio("Card Type", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])
    }

    if st.button("Start Prediction"):
        prediction = predict_churn(input_data)

        if prediction == 1:
            st.warning("Customer is expected to: LEAVE.")
        else:
            st.balloons()
            st.success("Customer is expected to: STAY.")