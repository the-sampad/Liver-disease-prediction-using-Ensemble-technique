## Step 1 : Notebook Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier


## Step 2 : Load the Data
# Next, let's load the dataset into a pandas dataframe.

df = pd.read_csv('indian_liver_patient.csv')



## Step 3 : Exploratory Data Analysis
# Before we start building our model, let's perform some exploratory data analysis to 
# understand the data better.

# Check the first few rows of the data
df.head()

# Check the shape of the data
df.shape

# Check the data types of the columns
df.dtypes

# Check the missing values in the data
df.isnull().sum()

# Check the distribution of the target variable
sns.countplot(x='Dataset', data=df)



## Step 4 : Preprocessing the Data
# We need to preprocess the data to prepare it for the model training. In this step, we will 
# perform the following operations:
# 1. Convert the gender column to binary
# 2. Replace the missing values in the Albumin_and_Globulin_Ratio column with the 
# mean value
# 3. Split the data into features and target variables
# 4. Split the data into training and testing sets

# Convert gender column to binary
df['Gender'] = df['Gender'].apply(lambda x: 1 if x=='Male' else 0)

# Replace missing values with the mean value
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())

# Split the data into features and target variables
X = df.drop(['Dataset'], axis=1)
y = df['Dataset']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



## Step 5 : Building Models
# We will build three models: Logistic Regression, Decision Tree, and XGBoost

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)



## Step 6 : Model Evaluation
# Now, let's evaluate the performance of each model using various metrics.

# Logistic Regression
print('Accuracy:', accuracy_score(y_test, lr_pred))
print('Classification Report:\n', classification_report(y_test, lr_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, lr_pred))

# Decision Tree
print('Accuracy:', accuracy_score(y_test, dt_pred))
print('Classification Report:\n', classification_report(y_test, dt_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, dt_pred))

# XGBoost
print('Accuracy:', accuracy_score(y_test, xgb_pred))
print('Classification Report:\n', classification_report(y_test, xgb_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, xgb_pred))



## Step 7: Ensemble Model
# Finally, let's combine the three models using the Ensemble technique.
# Ensemble Model
estimators = [('lr', lr), ('dt', dt), ('xgb', xgb)]
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)

# Evaluate the Ensemble Model
print('Accuracy:', accuracy_score(y_test, ensemble_pred))
print('Classification Report:\n', classification_report(y_test, ensemble_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, ensemble_pred))







