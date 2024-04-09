# Principal Component Analysis (PCA)

This folder contains solutions to verification assignments related to Principal Component Analysis (PCA).

## Objective

The goal is to utilize Principal Component Analysis to determine which handwritten digits differ most significantly from each other. 

Imagine you're working on an image recognition task for a postal company. It would be highly beneficial to automatically read digits, even if they are handwritten (currently, postal companies use automated digit recognition, which often outperforms human capabilities). The postal company's manager would like to identify which digits are recognized most difficultly to acquire more labeled data for those digits. By employing PCA, you'll ascertain which digits are more distinguishable from others.

## Project Overview

This project includes the following steps:

1. **Data Loading:** The dataset, "digits.csv", containing handwritten digits is loaded for analysis.

2. **Data Exploration:** Initial exploration of the dataset is conducted to understand its structure and attributes.

3. **Data Visualization:** Various visualization techniques are employed to visualize the handwritten digits, including heatmaps and scatter plots in 2D and 3D PCA spaces.

4. **Preprocessing:** The pixel data is standardized using StandardScaler.

5. **Principal Component Analysis (PCA):** PCA is applied to reduce the dimensionality of the data and extract principal components.

6. **Analysis:** The results are analyzed to determine which digits are better separated from others.

## Getting Started

To get started with this project:

1. Clone this repository to your local machine.
2. Navigate to the project directory (`Principal-Component-Analysis`).
3. Ensure you have Python installed on your system along with necessary libraries listed in the `requirements.txt`.
4. Run the Python script `PCA.py` to execute the project and generate visualizations.
5. Explore the results and analysis in the generated plots and outputs.

## Requirements

To run the code in this project, you will need Python installed on your system along with the required libraries such as NumPy, Pandas, Matplotlib, Seaborn, Plotly, and Scikit-learn. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Data Description

The dataset contains pixel values representing handwritten digits (0-9) in a tabular format. Each row corresponds to an image of a handwritten digit, and each column represents a pixel's grayscale intensity.

## Project Structure

```
Principal-Component-Analysis/
│
├── data/
│   └── digits.csv                      # Dataset containing pixel values of handwritten digits
│
├── plots/                              # Directory to store visualization plots
│
├── PCA.py                              # Python script containing the project code
│
└── README9.md                           # Project README file
```

This project aims to provide insights into the distinguishability of handwritten digits using PCA, aiding in the optimization of data collection efforts for improving digit recognition systems.
