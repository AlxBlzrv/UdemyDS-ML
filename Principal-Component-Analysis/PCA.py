# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import plotly.graph_objects as go

# Loading the dataset
digits = pd.read_csv('D:\\Dell\\repos\\UdemyDS-ML\\Principal-Component-Analysis\\data\\digits.csv')

# Displaying initial information about the dataset
print(digits.head())  # Print the first few rows of the dataset
print(digits.info())  # Print information about the dataset
print(digits.describe())  # Print statistical information about the dataset

# Extracting pixel data and removing the label column
pixels = digits.drop('number_label', axis=1)
print(pixels)

# Extracting pixel data for a single image
single_image = pixels.iloc[0]
print(single_image)

# Reshaping the pixel data to visualize as an image
single_image.to_numpy()
single_image.to_numpy().shape
single_image.to_numpy().reshape(8, 8)

# Plotting heatmap of the single image
plt.figure(figsize=(10, 6), dpi=200)
plt.rc('font', size=10)
plt.imshow(single_image.to_numpy().reshape(8, 8))
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\Principal-Component-Analysis\\plots\\heatmap1.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Plotting grayscale heatmap of the single image
plt.figure(figsize=(10, 6), dpi=200)
plt.rc('font', size=10)
plt.imshow(single_image.to_numpy().reshape(8, 8), cmap='gray')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\Principal-Component-Analysis\\plots\\heatmap2.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Plotting heatmap with annotations of the single image
plt.figure(figsize=(10, 6), dpi=200)
plt.rc('font', size=10)
sns.heatmap(single_image.to_numpy().reshape(8, 8), annot=True, cmap='gray')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\Principal-Component-Analysis\\plots\\heatmap3.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Standardizing the pixel data
scaler = StandardScaler()
scaled_pixels = scaler.fit_transform(pixels)
print(scaled_pixels)

# Applying Principal Component Analysis (PCA) with 2 components
pca_model = PCA(n_components=2)
pca_pixels = pca_model.fit_transform(scaled_pixels)
print(pca_pixels)

# Calculating the total variability explained by the 2 main components
np.sum(pca_model.explained_variance_ratio_)

# Plotting scatterplot of the data in 2-dimensional PCA space
plt.figure(figsize=(10, 6), dpi=150)
plt.rc('font', size=10)
labels = digits['number_label'].values
sns.scatterplot(x=pca_pixels[:, 0], y=pca_pixels[:, 1], hue=labels, palette='Set1')
plt.legend(loc=(1.05, 0))
plt.title('Scatterplot for numbers in 2-dimensional principal component space, by column number_label')
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\Principal-Component-Analysis\\plots\\scatterplot.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Observation: Number 4 is best separated from others

# Reapplying PCA with 3 components for 3D visualization
pca_model = PCA(n_components=3)
pca_pixels = pca_model.fit_transform(scaled_pixels)

# Plotting scatterplot of the data in 3-dimensional PCA space
plt.figure(figsize=(8, 8), dpi=150)
plt.rc('font', size=12)
ax = plt.axes(projection='3d')
ax.scatter3D(pca_pixels[:, 0], pca_pixels[:, 1], pca_pixels[:, 2], c=digits['number_label'])
plt.savefig('D:\\Dell\\repos\\UdemyDS-ML\\Principal-Component-Analysis\\plots\\scatterplot3d.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Plotting 3D scatterplot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=pca_pixels[:, 0],
    y=pca_pixels[:, 1],
    z=pca_pixels[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=digits['number_label'],
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig.update_layout(
    width=800,
    height=800,
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    )
)

fig.write_html("D:\\Dell\\repos\\UdemyDS-ML\\Principal-Component-Analysis\\plots\\scatterplot3d.html")
