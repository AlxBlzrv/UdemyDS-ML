# DBSCAN Clustering Analysis

This folder contains a project focused on clustering analysis using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. The goal is to segment wholesale customers based on their purchasing behavior.

## Project Overview

The project involves the following steps:

1. **Data Loading:** The main dataset, "Wholesale customers Data Set", is used for analysis. This dataset contains information about the annual spending of customers on various product categories.

2. **Data Preprocessing:** Initial data exploration is conducted, including handling missing values and understanding the distribution of features. Categorical variables such as 'Channel' and 'Region' are encoded for analysis.

3. **Visualization:** Several visualization techniques such as scatter plots, histograms, and cluster maps are utilized to understand the data distribution and relationships between variables.

4. **Modeling:** DBSCAN clustering algorithm is applied to group customers based on their spending patterns. The optimal epsilon value is determined using the Elbow Method, and the model is fitted with the chosen epsilon value.

5. **Evaluation:** The percentage of outlier points is calculated for different epsilon values to assess the effectiveness of the clustering.

6. **Analysis:** After clustering, the data is analyzed to understand the characteristics of each cluster and identify any outliers.

## Getting Started

To get started with this project:

1. Clone this repository to your local machine.
2. Navigate to the project directory (`DBSCAN`).
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Explore the Python code files (`DBSCAN.py`) to understand the data preprocessing, model building, and visualization steps.
5. Run the Jupyter Notebook to execute the project and analyze the results.

## Requirements

To run the code in this project, you will need Python installed on your system along with the required libraries such as Pandas, Matplotlib, Seaborn, Scikit-learn, and NumPy. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Data Description

The main dataset contains information about wholesale customers' annual spending on various product categories, such as fresh products, milk, grocery items, frozen products, detergents_paper, and delicatessen. Additionally, it includes categorical variables indicating the channel (Horeca or Retail) and region of the customers.

## Project Structure

```
DBSCAN-clustering/
│
├── data/
│   └── wholesome_customers_data.csv   # Main dataset with customer spending information
│
├── plots/                             # Directory to store visualization plots
│
├── DBSCAN.py                          # Containing the project code
│
└── README8.md                          # Project README file
```

This project aims to provide insights into customer segmentation based on their purchasing behavior, allowing businesses to tailor marketing strategies and services accordingly.
