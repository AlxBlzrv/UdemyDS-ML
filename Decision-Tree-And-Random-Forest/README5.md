# Project: Decision Trees and Random Forests for Telecom Customer Churn Prediction

This project focuses on implementing Decision Trees and Random Forests for predicting customer churn in a telecommunications company. Decision Trees and Random Forests are powerful supervised machine learning algorithms used for classification tasks.

## Dataset

The dataset used in this project is a telecom customer churn dataset that contains various features describing the customers' attributes and usage patterns, such as gender, seniority, partner status, contract type, internet service, monthly charges, and whether they have churned or not. Each sample in the dataset represents a customer, and the target variable "Churn" indicates whether the customer has churned (left the service) or not.

## Project Structure

The project directory includes:

- Python code files for implementing Decision Trees and Random Forests algorithms, data preprocessing, model training, and evaluation.
- Plots directory containing visualization plots:
  - `countplot`: Plot showing the distribution of churn and non-churn customers.
  - `violinplot`: Violin plot displaying the distribution of total charges across different churn categories.
  - `boxplot`: Distribution chart for different types of contracts and their relationship with churn.
  - `barplot`: Bar plot illustrating the correlation of features with churn prediction.
  - `histplot`: Distribution of values in the tenure column.
  - `scatterplot`: Distribution plot for TotalCharges and MonthlyCharges columns.
  - `plot`: Plot showing the percentage of churn for different tenure values.
- Data directory containing the telecom customer churn dataset (`Telco-Customer-Churn.csv`).
- README.md providing project information and instructions.

## Getting Started

To get started with the project:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Explore the Python code files for implementing Decision Trees and Random Forests, data preprocessing, and model evaluation.
5. Review the visualization plots in the `plots` directory to understand the data distribution and relationships.
6. Examine the telecom customer churn dataset file (`Telco-Customer-Churn.csv`) to understand the structure of the data and the features involved.
7. Run the Python scripts to train the Decision Trees and Random Forests models and evaluate their performance in predicting customer churn.

## Requirements

To run the code in this project, you will need Python installed on your system along with the required libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
