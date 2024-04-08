# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

import warnings

# Suppressing warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Loading the dataset
df = pd.read_csv('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\data\\wholesome_customers_data.csv')

# Displaying initial information about the dataset
print(df.head())  # Print the first few rows of the dataset
print(df.info())  # Print information about the dataset
print(df.describe())  # Print statistical information about the dataset

# Creating a scatter plot for the relationship between 'Milk' and 'Grocery' variables
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.scatterplot(data=df, x='Milk', y='Grocery', hue='Channel')
plt.legend(loc='best')
plt.title('Relationship between the variables MILK and GROCERY')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\plots\\scatterplot1.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Creating a histogram to display information about the 'Milk' column
plt.figure(figsize=(8, 6), dpi=300)
plt.rc('font', size=10)
sns.histplot(df, x='Milk', hue='Channel', multiple="stack")
plt.title('Displaying information about the MILK column')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\plots\\histplot.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Creating a cluster map to show correlation between spending categories
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.clustermap(df.drop(['Region','Channel'],axis=1).corr(), annot=True);
plt.title('Correlation Between Spending Categories')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\plots\\clustermap.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Creating pair plots for all columns with coloring by Region
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.pairplot(df, hue='Region', palette='Set1')
plt.title('PairPlot for All Columns with Coloring by Region')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\plots\\pairplot.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Scaling the data using StandardScaler
scaler = StandardScaler()
scaled_X = scaler.fit_transform(df)
print(scaled_X)

# Determining outlier percentage for various epsilon values
outlier_percent = []

for eps in np.linspace(0.001, 3, 50):
    
    # Create Model
    dbscan = DBSCAN(eps=eps, min_samples=2*scaled_X.shape[1])
    dbscan.fit(scaled_X)
    
    # Log percentage of points that are outliers
    perc_outliers = 100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
    
    outlier_percent.append(perc_outliers)

# Plotting percentage of outlier points depending on epsilon values
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.lineplot(x=np.linspace(0.001,3,50), y=outlier_percent)
plt.ylabel("Percentage of Points Classified as Outliers")
plt.xlabel("Epsilon Value")
plt.title('Percentage of outlier points depending on epsilon values')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\plots\\lineplot.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Fitting DBSCAN model with epsilon value = 2
dbscan = DBSCAN(eps=2)
dbscan.fit(scaled_X)

# Creating scatter plot for the relationship between 'Milk' and 'Grocery' variables with updated labels
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.scatterplot(data=df, x='Grocery', y='Milk', hue=dbscan.labels_)
plt.legend(loc='best')
plt.title('Relationship between the variables MILK and GROCERY (update)')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\plots\\scatterplot2.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Creating scatter plot for the relationship between 'Milk' and 'Detergents_Paper' variables
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.scatterplot(data=df, x='Detergents_Paper', y='Milk', hue=dbscan.labels_)
plt.legend(loc='best')
plt.title('Relationship between the variables Milk and Detergents_Paper')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\plots\\scatterplot3.png', 
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Adding DBSCAN labels to the dataset
df['Labels'] = dbscan.labels_
print(df.head())

# Grouping data by labels and calculating means for each category
cats = df.drop(['Channel','Region'],axis=1)
cat_means = cats.groupby('Labels').mean()
print(cat_means)

# Scaling the means using MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(cat_means)
scaled_means = pd.DataFrame(data, cat_means.index, cat_means.columns)
print(scaled_means)

# Creating a heatmap to visualize correlations between all columns of the dataframe
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.heatmap(scaled_means)
plt.title('Correlation Heatmap between all columns of the dataframe')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\plots\\heatmap1.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Creating a heatmap to visualize correlations between columns of the dataframe without outliers
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.heatmap(scaled_means.loc[[0,1]], annot=True)
plt.title('Correlation Heatmap between columns of the dataframe without outliers')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\DBSCAN\\plots\\heatmap2.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()
