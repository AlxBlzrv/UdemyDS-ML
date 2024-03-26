# Project 2: Logistic Regression & Model Evaluation

This project focuses on logistic regression modeling and model evaluation. Logistic regression is a popular classification algorithm used for binary classification tasks. In this project, I train a logistic regression model on a dataset and evaluate its performance using various evaluation metrics and visualization techniques.

## Dataset

The dataset used in this project contains information about heart disease diagnosis. It includes the following features:

- **age**: Age of the patient.
- **sex**: Gender of the patient (0 = female, 1 = male).
- **cp**: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic).
- **trestbps**: Resting blood pressure (in mm Hg).
- **chol**: Serum cholesterol level (in mg/dL).
- **fbs**: Fasting blood sugar > 120 mg/dL (1 = true, 0 = false).
- **restecg**: Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy).
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise induced angina (1 = yes, 0 = no).
- **oldpeak**: ST depression induced by exercise relative to rest.
- **slope**: Slope of the peak exercise ST segment.
- **ca**: Number of major vessels (0-3) colored by fluoroscopy.
- **thal**: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect).
- **target**: Presence of heart disease (1 = disease present, 0 = disease not present).

## Project Structure

The project consists of:

- Python code for data preprocessing, model training, and evaluation.
- Visualization plots including countplots, pairplots, heatmaps, precision-recall curves, and ROC curves.
- Data file containing the heart disease diagnosis dataset.
- Intermediate result files after preprocessing and model training.
- README file providing instructions and information about the project.

## Getting Started

To get started with the project, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the necessary dependencies by running `pip install -r requirements.txt`.
4. Explore the Python code files for data preprocessing, model training, and evaluation.
5. Review the visualization plots generated during the analysis.
6. Examine the data file (`heart.csv`) to understand the structure of the dataset.
7. Check the intermediate result files to review the project's output after preprocessing and model training.

## Requirements

To run the code in this project, you will need Python installed on your system along with the necessary libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt```
