import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('D:/work/git/udemy/Logistic-Regression/data/heart.csv')

# Display the first few rows of the dataset
print(df.head())

# Display unique values of the target column
print(df['target'].unique())

# Display information about the dataset
print(df.info())

# Display descriptive statistics of the dataset
print(df.describe().transpose())

# Count plot showing the distribution of target values
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.countplot(x='target', data=df, hue='target', palette='husl', legend=False)
plt.title('Number of points for each target value')
plt.savefig("D:/work/git/udemy/Logistic-Regression/plots/countplot.png")
plt.close()

# Pairplot showing relationships between columns
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.pairplot(df[['age','trestbps', 'chol','thalach','target']], hue='target')
plt.title('Connections between columns')
plt.savefig("D:/work/git/udemy/Logistic-Regression/plots/pairplot.png")
plt.close()

# Heatmap showing the correlation between features
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.heatmap(df.corr(),cmap='viridis',annot=True)
plt.title('Heatmap of Correlation')
plt.savefig("D:/work/git/udemy/Logistic-Regression/plots/heatmap_corr.png")
plt.close()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve

# Splitting the dataset into train and test sets
X = df.drop('target',axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Scaling the features
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Training the logistic regression model
log_model = LogisticRegressionCV()
log_model.fit(scaled_X_train,y_train)

# Displaying model parameters and coefficients
print(log_model.C_)
print(log_model.get_params())
print(log_model.coef_)

# Visualizing coefficients
coefs = pd.Series(index=X.columns,data=log_model.coef_[0])
coefs = coefs.sort_values()
plt.figure(figsize=(10, 6))
plt.rc('font', size=10)
sns.barplot(x=coefs.index,y=coefs.values, hue = coefs.index)
plt.title('Visualization of coefficients')
plt.savefig("D:/work/git/udemy/Logistic-Regression/plots/barplot.png")
plt.close()

# Predictions and evaluation
y_pred = log_model.predict(scaled_X_test)
conf_matrix_array = confusion_matrix(y_test, y_pred)
print(conf_matrix_array)

# Heatmap of confusion matrix
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.heatmap(conf_matrix_array, annot=True, fmt="d", cmap="viridis", cbar=False)
plt.title('Heatmap confusion matrix')
plt.savefig("D:/work/git/udemy/Logistic-Regression/plots/heatmap_conf_matrix.png")
plt.close()

# Classification report
class_report = classification_report(y_test, y_pred)
print(class_report)

# Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, log_model.predict_proba(scaled_X_test)[:, 1])
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig("D:/work/git/udemy/Logistic-Regression/plots/precision_recall_curve.png")
plt.close()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, log_model.predict_proba(scaled_X_test)[:, 1])
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig("D:/work/git/udemy/Logistic-Regression/plots/roc_curve.png")
plt.close()

# Example prediction for a patient
patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]

print(X_test.iloc[-1])
print(y_test.iloc[-1])

print(log_model.predict(scaler.transform(patient)))
print(log_model.predict_proba(scaler.transform(patient)))
