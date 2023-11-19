from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np


# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load the dataset
print("Import Dataset ...")
df = pd.read_csv("healthcare_dataset.csv")

# Separate independent (features) and dependent (target) variables
X = df[['Age', 'Gender', 'Blood Type', 'Medical Condition']]
y = df['Billing Amount']

# Preprocess the categorical variables using one-hot encoding
categorical_features = ['Gender', 'Blood Type', 'Medical Condition']

print("Transform Dataset ...")
one_hot = OneHotEncoder(handle_unknown='ignore')
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)],
                                 remainder='passthrough')

# Apply transformation to the features
transformed_X = transformer.fit_transform(X)

print("Split Dataset ...")
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2, random_state=42)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
print("Fit Model ...")
regr = RandomForestRegressor(max_depth=2, random_state=seed)
regr.fit(X_train, y_train)

# Report training set score
print("Model Evaluation ...")
train_score = regr.score(X_train, y_train) * 100
# Report test set score
test_score = regr.score(X_test, y_test) * 100

# Write scores to a file
print("Store Metrics into txt File ...")
with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)

##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
print("Calculate Feature Importance ...")
importances = regr.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)

print("Plot Feature Importance ...")
# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()
