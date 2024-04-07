# Country Clustering Analysis

This folder contains a project focused on clustering analysis of countries based on various features. The goal is to identify similarities among countries and regions by experimenting with different numbers of clusters.

## Project Overview

The project involves the following steps:

1. **Data Loading:** Two CSV files are provided in the `data` directory. The main file contains information about countries, while the secondary file provides additional information about ISO codes.

2. **Data Preprocessing:** Initial data exploration is conducted, including handling missing values and understanding the distribution of features.

3. **Feature Engineering:** Data is prepared for clustering by encoding categorical variables and scaling numerical features.

4. **Determining Optimal Number of Clusters:** The Elbow Method is employed to find the optimal number of clusters based on the sum of squared distances.

5. **Model Building:** K-Means clustering algorithm is applied with the chosen number of clusters to group countries based on similarities in their features.

6. **Visualization:** Results are visualized using a choropleth map to display clusters on a geographical map.

## Getting Started

To get started with this project:

1. Clone this repository to your local machine.
2. Navigate to the project directory (`K-means-clustering`).
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Explore the Python code files (`Kmeans.py`) to understand the data preprocessing, model building, and visualization steps.
5. Run the Jupyter Notebook to execute the project and analyze the results.

## Requirements

To run the code in this project, you will need Python installed on your system along with the required libraries such as Pandas, Matplotlib, Seaborn, Scikit-learn, and Plotly. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Data Description

The main dataset contains information about countries, including various socio-economic indicators such as population, GDP per capita, literacy rate, and more. The secondary dataset provides information about ISO codes for countries, which is used for mapping countries to their respective geographical regions.

## Project Structure

```
K-means-clustering/
│
├── data/
│   ├── CIA_Country_Facts.csv   # Main dataset with country information
│   └── country_iso_codes.csv   # Secondary dataset with ISO codes
│
├── plots/                      # Directory to store visualization plots
│
├── Kmeans.py                   # Containing the project code
│
└── README.md                   # Project README file
```

This project aims to provide insights into the similarities among countries and regions based on various socio-economic indicators, facilitating better understanding and analysis of global trends and patterns.
