# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('D:\\Dell\\repos\\UdemyDS-ML\\Natural-Language-Processing\\project2\\data\\moviereviews.csv')

# Display the first few rows of the dataset and information about it
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Check reviews with empty text (i.e. with spaces)
print(df['review'].str.isspace().sum())
print(df[df['review'].str.isspace()])

# Remove reviews with empty text
df = df[~df['review'].str.isspace()]
df.info()

# Count rows for different target column values
print('Number of rows for different target column values:')
print(df['label'].value_counts())

# Create a CountVectorizer object to convert text into feature matrix
cv = CountVectorizer(stop_words='english')

# Convert text of negative reviews into feature matrix
matrix = cv.fit_transform(df[df['label']=='neg']['review'])
freqs = zip(cv.get_feature_names_out(), matrix.sum(axis=0).tolist()[0])    

print("Top 20 words used for Negative reviews.")
print(sorted(freqs, key=lambda x: -x[1])[:20])

# Convert text of positive reviews into feature matrix
matrix = cv.fit_transform(df[df['label']=='pos']['review'])
freqs = zip(cv.get_feature_names_out(), matrix.sum(axis=0).tolist()[0])    

print("Top 20 words used for Positive reviews.")
print(sorted(freqs, key=lambda x: -x[1])[:20])

# Split data into train and test sets
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Create and train a model using a pipeline
pipe = Pipeline([('tfidf', TfidfVectorizer()), ('svc', LinearSVC(dual=False))])
pipe.fit(X_train, y_train)

# Predict on test set and output evaluation metrics
preds = pipe.predict(X_test)
print(classification_report(y_test, preds))

# Plot and save a heatmap of the confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6), dpi=300)
plt.rc('font', size=10)
sns.heatmap(cm, annot=True, cmap='magma', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix Heatmap for LinearSVC Model')
plt.savefig(f'D:\\Dell\\repos\\UdemyDS-ML\\Natural-Language-Processing\\project2\\plots\\heatmap_confusion_matrix.png')
plt.close()

# Calculate and output model accuracy
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)
