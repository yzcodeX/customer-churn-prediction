import os
import joblib
import pandas                as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score
from sklearn.ensemble        import RandomForestClassifier


# load data
DATA_PATH = "Data/Processed_Customer_Churn_Data.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found in {DATA_PATH}.")

df = pd.read_csv(DATA_PATH)

# remove unnecessary columns
# df = df.drop(columns = ["RowNumber", "CustomerId", "Surname", "Complain"], axis = 1)

# # convert categorical variables (one-hot encoding)
# df = pd.get_dummies(df, columns=["Geography", "Gender", "Card Type"], drop_first = False)

# define features and target
X = df.drop(["Exited"], axis = 1)
y = df["Exited"]

# division into training and test data and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# scale the data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled  = sc.transform(X_test)

# creating XGBoost model with best Hyperparameters
model = RandomForestClassifier(n_estimators = 300, max_depth = 20, min_samples_split = 5, min_samples_leaf = 2, random_state = 1)
model.fit(X_train_scaled, y_train)
# y_pred = model_xgb.predict(X_test_scaled)

# path for saving model and scaler
MODEL_PATH  = "./models/Random_Forest_Classifier_model.pkl"
SCALER_PATH = "./models/scaler.pkl"

# create folder if not exist
os.makedirs(os.path.dirname(MODEL_PATH),  exist_ok = True) 
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok = True)

# save model
joblib.dump(model, MODEL_PATH, compress = 3) # compress: reduce size of pkl file (to upload it on GitHub)
joblib.dump(sc, SCALER_PATH)
print("Model was successfully saved: ",  MODEL_PATH)
print("Scaler was successfully saved: ", SCALER_PATH)