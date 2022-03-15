import xgboost as xgb
import pandas as pd
import sklearn.model_selection as ms
from tabulate import tabulate
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Import dataset
df = pd.read_csv("data/telco_churn_data.csv")

# Remove any columns with only one unique value in its row (like is_person==yes)
for name in df.columns:
    name = str(name)
    if len(df[name].unique()) == 1:
        print("All the entries in this column are the same: ", df[name].unique())
        df.drop(labels=name, axis=1, inplace=True)

# Manually check and remove irrelevant (and thus confounding) columns
df.drop(labels="customerID", axis=1, inplace=True)

# Replace all empties with NaN and than with zero IN THE DATA (even though xgboost does this for you)
df = df.replace(r'^\s*$', 0.00, regex=True)

# Remove whitespace IN THE COLUMN TITLES (checked)
df.columns = df.columns.str.replace(' ', '_')

# Remove whitespace IN THE DATA (like payment{with card, without card} becomes with_card, etc.
df.replace(" ", "_", regex=True, inplace=True)

# Remove parenthesis IN THE DATA: ( and )
df["PaymentMethod"] = df["PaymentMethod"].str.replace(r"\(", "")
df["PaymentMethod"] = df["PaymentMethod"].str.replace(r"\)", "")

# Manually check which Float columns are wrongly specified as objects and change them to float
# if df["TotalCharges"].dtype == object:
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

# Show in tabulate form
print(tabulate(df[6668:6674], tablefmt="psql", showindex=True, headers=df.columns))

# ################ PREPARE DATA FOR ML ##################

# Create the X
X = df.drop("Churn", axis=1).copy()

# Check which columns are duplicate and manually remove them to prevent error
# duplicate_columns = df.columns[df.columns.duplicated()]
# print('duplicates ', duplicate_columns)

# Check which data types there are
# print(X_train.dtypes)

# Create the y
y = df["Churn"].copy()
# y = y.eq('Yes').mul(1)

# Do one hot encoding
X_encoded = pd.get_dummies(X, columns=["gender",
                                       "SeniorCitizen",
                                       "Partner",
                                       "Dependents",
                                       "PhoneService",
                                       "MultipleLines",
                                       "InternetService",
                                       "OnlineSecurity",
                                       "OnlineBackup",
                                       "DeviceProtection",
                                       "TechSupport",
                                       "StreamingTV",
                                       "StreamingMovies",
                                       "Contract",
                                       "PaymentMethod",
                                       "PaperlessBilling", ])

print("len of y", len(y))
print("len of x encoded", len(X_encoded))

# ################# SPLIT THE DATA ####################

# Split it in X_train, Y_train, X_val and Y_val
# Use stratify to maintain the 26% rate when slitting test and val data
X_train, X_test, y_train, y_test = ms.train_test_split(X_encoded, y, random_state=42, stratify=y)

# Save the x train and y train to csv to check if all is in order before building the saved_model
csv_X_train = pd.DataFrame(X_train)
csv_X_train.to_csv('data/X_train.csv', index=False)
csv_Y_train = pd.DataFrame(y_train)
csv_Y_train.to_csv('data/Y_train.csv', index=False)

# Save the test set
csv_X_test = pd.DataFrame(X_test)
csv_X_test.to_csv('data/X_test.csv', index=False)
csv_y_test = pd.DataFrame(y_test)
csv_y_test.to_csv('data/y_test.csv', index=False)

# Print out your training data in a table
# print(tabulate(X_train, tablefmt="psql"))

# Check that the churn rate is kept in the splitted data
# # print("Churn rate of actual data is ", sum(y) / len(y))
# print("Churn rate in training set is ", sum(y_train)/ len(y_train))
# print("Churn rate val set is ", sum(y_test)/ len(y_test))

# ############################# BUILD AND TRAIN THE MODEL ###################################

# Create a shell to create your saved_model in
clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
                            missing=-999.9,
                            seed=42,
                            # First run the GridSearchCV to find following optimal params:
                            gamma=0.1,
                            learning_rate=0.1,
                            max_depth=3,
                            reg_lambda=10.0,
                            scale_pos_weight=1,
                            subsample=0.8,
                            colsample_bytree=0.5
                            )

# Create the trees inside your shell
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric='aucpr',  # This evaluates how well the preds are made
            eval_set=[(X_test, y_test)]
            )

# ###################### Plotting a confusion matrix to see type I and type II errors #######

# # Plot a confusion matrix
# sklearn.metrics.plot_confusion_matrix(clf_xgb,
#                                       X_test,
#                                       y_test,
#                                       values_format='d',
#                                       display_labels=['did not leave', 'did leave'])
# plt.show()

# ############################## Optimize your parameters using Cross Validation ##########################
# - Note that it takes about 10 min to run
# - Uncomment to use
# ####

# # Fill in the parameters to try out and optimise:
# param_grid = {
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.01, 0.05, 0.10],
#     'gamma': [0.0, 0.25, 1.0],
#     'reg_lambda': [0.0, 1.0, 10.0],
#     'scale_pos_weight': [1, 3, 5]
# }
# # 3 different values already takes long, so just do this and repeat is optimal values are on the outside of a [1.2.3]
#
# # Fill in the optimal params:
# optimal_params = ms.GridSearchCV(
#     estimator=xgb.XGBClassifier(objective='binary:logistic',
#                                 subsample=0.9,              # Use only 90% of training data to speed this process up
#                                 colsample_bytree=0.5,       # Use only half of features per tree to speed up
#                                 seed=42),
#     param_grid=param_grid,                                  # We filled this in earlier
#     scoring='roc_auc',
#     verbose=2,                                              # set 0 to turn off, 2 to turn on
#     n_jobs=10,
#     cv=3                                                    # We do a 3fold of cross validation
# )
#
# optimal_params.fit(X_train,
#                    y_train,
#                    early_stopping_rounds=10,
#                    eval_metric='auc',
#                    eval_set=[(X_test, y_test)],
#                    verbose=False
#                    )
#
# print("The optimal parameter combination is: ", optimal_params.best_params_)

# ################################### SAVE THE MODEL ##################

try:
    # save
    file_name = "saved_model/xgb_predict_churn_model.pkl"
    pickle.dump(clf_xgb, open(file_name, "wb"))
except:
    print("Model could not save as pickle!")

# ################################### DRAW A TREE #######################
#
# Printing this out could help, e.g. give us a good starting point for optimising parameters
#
# ####

# Once more, repeat creating the shell and training it
clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
                            seed=42,
                            # First run the GridSearchCV to find following optimal params:
                            gamma=0.25,
                            learning_rate=0.1,
                            max_depth=4,
                            reg_lambda=10,
                            scale_pos_weight=2,
                            subsample=0.8,
                            colsample_bytree=0.5,
                            n_estimators=1)
clf_xgb.fit(X_train, y_train)

# Set params to make the tree pretty
node_params = {'shape': 'box',
               'style': 'filled, rounded',
               'fillcolor': 'blue',
               'fontcolor': 'white'}
leaf_params = {'shape': 'box',
               'style': 'filled',
               'fillcolor': 'green',
               'fontcolor': 'white'}

# Draw the tree
dot = xgb.to_graphviz(clf_xgb,
                      num_trees=0,
                      size="10,10",
                      condition_node_params=node_params,
                      leaf_node_params=leaf_params
                      )

# Show (True/False) and save the tree in a folder
dot.render(directory='documentation', view=False)

print("\nRan code successfully. \n")
