# Customer Churn Prediction Project
# Author: Aman
# Tool: Python (Pandas, Scikit-learn, Matplotlib)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\user\Downloads\Telco-Customer-Churn.csv")
print(df.head())

# Step 3: Data Cleaning
df.drop(['customerID'], axis=1, inplace=True)
df.replace(' ', np.nan, inplace=True)
df.dropna(inplace=True)

# Step 4: Convert categorical columns
for column in df.select_dtypes(['object']).columns:
    if df[column].nunique() == 2:
        df[column] = LabelEncoder().fit_transform(df[column])
    else:
        df = pd.get_dummies(df, columns=[column])

# Step 5: Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test)
print("\nModel Evaluation 📈")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Confusion Matrix Visualization
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
importances.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.show()
