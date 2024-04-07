# Natural Language Processing Projects

This folder contains two projects focused on Natural Language Processing (NLP). Both projects involve text data preprocessing, sentiment analysis, and model building for text classification tasks.

## Project 1: Sentiment Analysis on Airline Tweets

This project involves sentiment analysis on tweets about various airlines. The dataset used contains tweets along with their sentiment labels (positive, negative, neutral). The project includes:

- Data preprocessing steps such as handling missing values and exploring the distribution of sentiment labels.
- Text preprocessing techniques including tokenization, removing stop words, and vectorization.
- Model building using Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).
- Evaluation of model performance using metrics like accuracy, confusion matrix, and classification report.
- Visualization of results including count plots and confusion matrix heatmaps.

## Project 2: Sentiment Analysis on Movie Reviews

This project focuses on sentiment analysis of movie reviews. The dataset consists of movie reviews labeled as positive or negative sentiment. The project includes:

- Data cleaning to handle missing values and remove empty text reviews.
- Exploratory data analysis to understand the distribution of sentiment labels.
- Text preprocessing including vectorization using CountVectorizer and TfidfVectorizer.
- Model training using a Linear Support Vector Classifier (LinearSVC) in a pipeline with TF-IDF vectorization.
- Evaluation of model performance with metrics such as accuracy and a classification report.
- Visualization of evaluation results with a confusion matrix heatmap.

## Getting Started

To get started with each project:

1. Clone this repository to your local machine.
2. Navigate to the respective project directory (project1 or project2).
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Explore the Python code files to understand the data preprocessing, model building, and evaluation steps.
5. Run the Python scripts to execute the projects and analyze the results.

## Requirements

To run the code in these projects, you will need Python installed on your system along with the required libraries such as Pandas, Seaborn, Matplotlib, and Scikit-learn. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
