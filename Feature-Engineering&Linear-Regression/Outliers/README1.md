# Project 1: Feature Engineering & Linear Regression

This project focuses on feature engineering and linear regression modeling. Feature engineering is a crucial step in the machine learning pipeline where I construct new features from existing data to improve the predictive power of my models. In this project, I explore various techniques for feature engineering and apply them to a dataset. Subsequently, I build a machine learning model, specifically linear regression, based on these engineered features.

## Dataset

The dataset used in this project is the Ames Housing Data. It contains the following columns:

- **PID**: Identification number.
- **MS SubClass**: Identifies the type of dwelling involved in the sale.
- **MS Zoning**: Identifies the general zoning classification of the sale.
- **Lot Frontage**: Linear feet of street connected to property.
- **Lot Area**: Lot size in square feet.
- **Street**: Type of road access to property.
- **Alley**: Type of alley access to property.
- **Lot Shape**: General shape of property.
- **Land Contour**: Flatness of the property.
- **Utilities**: Type of utilities available.
- **Lot Config**: Lot configuration.
- **Land Slope**: Slope of property.
- **Neighborhood**: Physical locations within Ames city limits.
- **Condition 1**: Proximity to various conditions (e.g., main road or railroad).
- **Condition 2**: Proximity to various conditions (if more than one is present).
- **Bldg Type**: Type of dwelling.
- **House Style**: Style of dwelling.


## Project Structure

The project consists of:

- Python codes for data preprocessing, feature engineering, and modeling.
- Various visualization plots including scatterplots, distplots, and heatmaps etc.
- Data files used in the project.
- A file containing detailed descriptions of each column in the dataset.
- Intermediate results file after preprocessing and modeling.


## Getting Started

To get started with the project, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the necessary dependencies by running `pip install -r requirements.txt`.
4. Explore the `requirements.txt` file located in the main directory to see the list of required Python libraries.
5. Run the Python code in your preferred environment for data preprocessing, feature engineering, and modeling.
6. Examine the various visualization plots generated during the analysis.
7. Review the data files and the `Ames_Housing_Feature_Description.txt` file for detailed information on the dataset columns.
8. Check the intermediate result files to review the project's output after preprocessing and modeling.


## Requirements

To run the code in this project, you will need Python installed on your system along with the necessary libraries such as NumPy, Pandas, Matplotlib, and Scikit-learn. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
