# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv('D:\\Dell\\repos\\UdemyDS-ML\\Natural-Language-Processing\\project1\\data\\airline_tweets.csv')

# Display the first few rows of the dataset and information about it
print(df.head())
print(df.info())

# Plot the distribution of review sentiment by airline and save the plot
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.countplot(data=df, x='airline', hue='airline_sentiment')
plt.title('Distribution of review sentiment by airline')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Natural-Language-Processing\\project1\\plots\\countplot1.png")
plt.close()

# Plot the distribution of reasons for negative review and save the plot
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.countplot(data=df, x='negativereason')
plt.xticks(rotation=90);
plt.title('Distribution of reasons for negative review')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Natural-Language-Processing\\project1\\plots\\countplot2.png",
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Plot the distribution of feedback sentiments and save the plot
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.countplot(data=df, x='airline_sentiment')
plt.title('Distribution of feedback sentiments')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Natural-Language-Processing\\project1\\plots\\countplot3.png")
plt.close()

# Display the count of different sentiment categories
print(df['airline_sentiment'].value_counts())

# Prepare data for training
data = df[['airline_sentiment', 'text']]
data.head()

X = df['text']
y = df['airline_sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Vectorize text data using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train classifiers
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

log = LogisticRegression(max_iter=1000)
log.fit(X_train_tfidf, y_train)

svc = LinearSVC(dual=False)
svc.fit(X_train_tfidf, y_train)

# Function to generate classification report and confusion matrix
def report(model, plot_number):
    preds = model.predict(X_test_tfidf)
    print(classification_report(y_test, preds))
    
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.rc('font', size=10)
    sns.heatmap(cm, annot=True, cmap='magma', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix Heatmap for Model {plot_number}')
    plt.savefig(f'D:\\Dell\\repos\\UdemyDS-ML\\Natural-Language-Processing\\project1\\plots\\heatmap_confusion_matrix{plot_number}.png')
    plt.close()

    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:", accuracy)

# Evaluate Naive Bayes model
print("NB MODEL")
report(nb, 1)

# Evaluate Logistic Regression model
print("Logistic Regression")
report(log, 2)

# Evaluate Support Vector Classifier model
print('SVC')
report(svc, 3)

# Create a pipeline to apply to new tweets
pipe = Pipeline([('tfidf', TfidfVectorizer()), ('svc', LinearSVC(dual=False))])
pipe.fit(df['text'], df['airline_sentiment'])

# Test examples
new_tweet = ['good flight']
print(pipe.predict(new_tweet))

new_tweet = ['bad flight']
print(pipe.predict(new_tweet))

new_tweet = ['ok flight']
print(pipe.predict(new_tweet))
