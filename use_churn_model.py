import pandas as pd
import pickle

# Import dataset
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/Y_test.csv")

# load saved_model
file_name = "saved_model/xgb_predict_churn_model.pkl"
xgb_model_loaded = pickle.load(open(file_name, "rb"))

# test
ind = 3
test = X_test.iloc[[ind]]

print(f"The person belongng to the {ind}th row has the following properties: \n{test}")
print(f"According to the ML saved_model, will this person Churn anytime soon? {xgb_model_loaded.predict(test)[0]}")
