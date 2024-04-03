import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('D:\\Dell\\repos\\UdemyDS-ML\\K-Nearest-Neighbors\\data\\sonar.all-data.csv')
print(df.head())

# Plot the heatmap of correlation
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.heatmap(df.iloc[:, :-1].corr(), cmap='viridis', annot=False)
plt.title('Heatmap of Correlation')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\K-Nearest-Neighbors\\plots\\heatmap_corr.png")
plt.close()

# Map target labels to numerical values
df['Target'] = df['Label'].map({'R':0, 'M':1})
print(np.abs(df.corr(numeric_only=True)['Target']).sort_values().tail(6))

# Split the data into features (X) and target labels (y)
X = df.drop(['Target', 'Label'], axis=1)
y = df['Label']

# Split the data into training-validation and test sets
X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define preprocessing steps and classifier
scaler = StandardScaler()
knn = KNeighborsClassifier()
operations = [('scaler', scaler), ('knn', knn)]
pipe = Pipeline(operations)

# Define hyperparameter grid for GridSearchCV
k_values = list(range(1, 30))
param_grid = {'knn__n_neighbors': k_values}

# Perform GridSearchCV to find the best estimator
full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
full_cv_classifier.fit(X_cv, y_cv)

# Print the parameters of the best estimator
print(full_cv_classifier.best_estimator_.get_params())

# Plot mean test score vs. K values
scores = full_cv_classifier.cv_results_['mean_test_score']

plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
plt.plot(k_values, scores, 'o-')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title('Mean Test Score')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\K-Nearest-Neighbors\\plots\\plot.png")
plt.close()

# Make predictions on the test set
pred = full_cv_classifier.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, pred))

# Create a heatmap plot of the confusion matrix
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(8, 6))
plt.rc('font', size=10)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\K-Nearest-Neighbors\\plots\\heatmap_confusion_matrix.png")
plt.close()

# Print classification report
print(classification_report(y_test,pred))

# Calculate accuracy score
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)