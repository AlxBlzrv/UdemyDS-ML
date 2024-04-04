# Project: Support Vector Machines (SVM) for Wine Fraud Detection

This project focuses on implementing Support Vector Machines (SVM) for detecting fraud in wine quality classification. Support Vector Machines are powerful supervised machine learning algorithms used for classification and regression tasks.

## Dataset

The dataset used in this project is a wine quality dataset that contains various features describing the properties of different wines, such as acidity, pH level, type (red or white), and quality. Each sample in the dataset represents a wine instance, and the target variable "quality" indicates whether the wine is labeled as legitimate or fraudulent.

## Project Structure

The project directory includes:

- Python code files for implementing SVM algorithm, data preprocessing, model training, and evaluation.
- Plots directory containing visualization plots:
  - `countplot1`: Plot showing the distribution of wine quality categories.
  - `countplot2`: Plot illustrating the relationship between wine type and quality categories.
  - `corr_plot`: Bar plot displaying the correlation values of wine features with fraud detection.
  - `clustermap`: Clustermap representing the correlations between different wine features.
  - `heatmap_confusion_matrix`: Heatmap representation of the confusion matrix.
- Data directory containing the wine dataset (`wine_fraud.csv`).
- README.md providing project information and instructions.

## Getting Started

To get started with the project:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Explore the Python code files for implementing SVM, data preprocessing, and model evaluation.
5. Review the visualization plots in the `plots` directory to understand the data distribution and relationships.
6. Examine the wine dataset file (`wine_fraud.csv`) to understand the structure of the data and the features involved.
7. Run the Python scripts to train the SVM model and evaluate its performance in fraud detection.

## Requirements

To run the code in this project, you will need Python installed on your system along with the required libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt```
