# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import warnings

# Suppressing warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Loading the dataset
df = pd.read_csv('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\data\\CIA_Country_Facts.csv')

# Displaying initial information about the dataset
print(df.head())
print(df.info())
print(df.describe().transpose())

# Visualization: Histogram of Population
plt.figure(figsize=(8, 6), dpi=300)
plt.rc('font', size=10)
sns.histplot(data=df, x='Population')
plt.title('Visualization of the population column of different countries')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\histplot1.png')
plt.close()

# Visualization: Histogram of Population (< 5 billion)
plt.figure(figsize=(8, 6), dpi=300)
plt.rc('font', size=10)
sns.histplot(data=df[df['Population']<500000000], x='Population')
plt.title('Population of countries (< 5 billion)')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\histplot2.png')
plt.close()

# Visualization: Barplot of GDP per capita by Region
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.barplot(data=df, y='GDP ($ per capita)', x='Region', estimator=np.mean)
plt.xticks(rotation=90);
plt.title('Gross domestic product per capita ($)')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\barplot.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Visualization: Scatterplot of GDP vs Phones per 1000 with Region hue
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.scatterplot(data=df, x='GDP ($ per capita)', y='Phones (per 1000)', hue='Region')
plt.legend(loc=(1.05,0.5))
plt.title('Association between GDP and Phones (per 1000) columns')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\scatterplot1.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Visualization: Scatterplot of GDP vs Literacy with Region hue
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.scatterplot(data=df, x='GDP ($ per capita)', y='Literacy (%)', hue='Region')
plt.legend(loc='best')
plt.title('Association between GDP and Literacy columns')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\scatterplot2.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Visualization: Correlation Heatmap of all numeric columns
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.heatmap(df.corr(numeric_only=True))
plt.title('Correlation Heatmap between all columns of the dataframe')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\heatmap.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Visualization: Hierarchical data clustering (Clustermap)
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
sns.clustermap(df.corr(numeric_only=True))
plt.title('Hierarchical data clustering')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\clustermap.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Handling Missing Data
print('Missing data:')
print(df.isnull().sum())

# Filling missing values in 'Agriculture' column with 0
df[df['Agriculture'].isnull()] = df[df['Agriculture'].isnull()].fillna(0)
print('Missing data (update):')
print(df.isnull().sum())

# Filling missing values in 'Climate' column with mean of respective Region
df['Climate'] = df['Climate'].fillna(df.groupby('Region')['Climate'].transform('mean'))
print('Missing data (update):')
print(df.isnull().sum())

# Filling missing values in 'Literacy (%)' column with mean of respective Region
df['Literacy (%)'] = df['Literacy (%)'].fillna(df.groupby('Region')['Literacy (%)'].transform('mean'))
print('Missing data (update):')
print(df.isnull().sum())

# Dropping rows with any remaining missing values
df = df.dropna()

# Preprocessing Data for K-Means Clustering
X = df.drop("Country",axis=1)
X = pd.get_dummies(X)

# Scaling the features
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Determining optimal number of clusters (K) using Elbow Method
ssd = []
for k in range(2,30):
    model = KMeans(n_clusters=k)
    model.fit(scaled_X)
    ssd.append(model.inertia_)

# Visualization: Elbow Method Plot
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
plt.plot(range(2,30), ssd, 'o--')
plt.xlabel("K Value")
plt.ylabel("Sum of Squared Distances")
plt.title('Finding best K value')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\K-value-plot1.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Visualization: Difference in consecutive SSDs for Elbow Method
plt.figure(figsize=(10,6), dpi=200)
plt.rc('font', size=10)
pd.Series(ssd).diff().plot(kind='bar')
plt.title('Finding best K value')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\K-value-plot2.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Clustering Data with chosen number of clusters (K=3)
model = KMeans(n_clusters=3)
model.fit(scaled_X)

# Assigning cluster labels to the data
X['K=3 Clusters'] = model.labels_

# Evaluating correlation of K=3 clusters with other features
print(X.corr()['K=3 Clusters'].sort_values())

# Mapping ISO codes to countries for visualization
iso_codes = pd.read_csv("D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\data\\country_iso_codes.csv")
iso_mapping = iso_codes.set_index('Country')['ISO Code'].to_dict()
df['ISO Code'] = df['Country'].map(iso_mapping)

# Adding cluster labels to the dataframe
df['Cluster'] = model.labels_

# Visualizing Clusters on a Choropleth map
fig = px.choropleth(df, locations="ISO Code",
                    color="Cluster", # Color based on cluster labels
                    hover_name="Country", # Country name for hover information
                    color_continuous_scale='Turbo'
                    )

# Displaying the Choropleth map
fig.show()

# Saving the Choropleth map as an image
fig.write_image("D:\\Dell\\repos\\UdemyDS-ML\\K-means-clustering\\plots\\choropleth_plot.png")
