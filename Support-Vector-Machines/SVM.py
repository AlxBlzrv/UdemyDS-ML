import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('D:\\Dell\\repos\\UdemyDS-ML\\Support-Vector-Machines\\data\\wine_fraud.csv')
print(df.head())

# Different values of the target variable
print(df['quality'].unique())

# Plot the distribution of each category
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.countplot(x='quality', data=df)
plt.title('Number of points for each target value')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Support-Vector-Machines\\plots\\countplot1.png")
plt.close()

# Type and category of wine
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.countplot(x='type', data=df, hue='quality')
plt.title('Number of points for each target value')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Support-Vector-Machines\\plots\\countplot2.png")
plt.close()

# Percentage of fraudulent wines
reds = df[df["type"]=='red']
whites = df[df["type"]=='white']

print("Percentage of fraud in Red Wines:")
print(100 * (len(reds[reds['quality']=='Fraud']) / len(reds)))

print("Percentage of fraud in White Wines:")
print(100 * (len(whites[whites['quality']=='Fraud']) / len(whites)))

# Features correlation
df['Fraud']= df['quality'].map({'Legit':0,'Fraud':1})
print(df.corr(numeric_only=True)['Fraud'])

plt.figure(figsize=(12, 10))
plt.rc('font', size=9)
df.corr(numeric_only=True)['Fraud'][:-1].sort_values().plot(kind='bar')
plt.title('Correlation values of wine features')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Support-Vector-Machines\\plots\\corr_plot.png")
plt.close()

# Interconnections between variables
plt.figure(figsize=(12, 10))
plt.rc('font', size=9)
sns.clustermap(df.corr(numeric_only=True), cmap='viridis')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Support-Vector-Machines\\plots\\clustermap.png")
plt.close()

# Data preprocessing
df['type'] = pd.get_dummies(df['type'], drop_first=True)
df = df.drop('Fraud', axis=1)

X = df.drop('quality', axis=1)
y = df['quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Feature scaling
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Support Vector Classifier model
svc = SVC(class_weight='balanced')

# Parameter grid for grid search
param_grid = {'C':[0.001, 0.01, 0.1, 0.5, 1],  # Regularization parameter
              'gamma':['scale', 'auto']}  # Radius of the Gaussian kernel function

# Grid search to find the best parameters
grid = GridSearchCV(svc, param_grid)
grid.fit(scaled_X_train, y_train)
print(grid.best_params_)

# Predictions using the best model
grid_pred = grid.predict(scaled_X_test)

# Confusion matrix
print(confusion_matrix(y_test, grid_pred))

# Heatmap plot of the confusion matrix
cm = confusion_matrix(y_test, grid_pred)

plt.figure(figsize=(8, 6))
plt.rc('font', size=10)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Support-Vector-Machines\\plots\\heatmap_confusion_matrix.png")
plt.close()

# Classification report
print(classification_report(y_test, grid_pred))

# Accuracy score
accuracy = accuracy_score(y_test, grid_pred)
print("Accuracy:", accuracy)
