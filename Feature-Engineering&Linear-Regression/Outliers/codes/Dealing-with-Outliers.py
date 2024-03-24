import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading data
df = pd.read_csv("D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/data/Ames_Housing_Data.csv")

# Heatmap correlation
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.heatmap(df.corr(numeric_only=True))
plt.title('Heatmap of Correlation')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/plots/heatmap_corr.png")
plt.close()

# Output Sale Price correlation
print(df.corr(numeric_only=True)['SalePrice'].sort_values())

# Histogram with kernel density estimation
sns.displot(df["SalePrice"], kde=True)
plt.xticks(rotation=45)
plt.title('Distribution of SalePrice')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/plots/displot.png")
plt.close()

# Scatterplots with outliers
sns.scatterplot(x='Overall Qual', y='SalePrice', data=df)
plt.title('Scatterplot of Overall Qual vs SalePrice with outliers')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/plots/scatterplot1_with_outliers.png")
plt.close()

print(df[(df['Overall Qual']>8) & (df['SalePrice']<200000)])

sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df)
plt.title('Scatterplot of Gr Liv Area vs SalePrice with outliers')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/plots/scatterplot2_with_outliers.png")
plt.close()

print(df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)])
print(df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)].index)

# Removing outliers
ind_drop = df[(df['Overall Qual'] > 8) & (df['SalePrice'] < 200000)].index
df = df.drop(ind_drop, axis=0)

ind_drop = df[(df['Gr Liv Area'] > 4000) & (df['SalePrice'] < 400000)].index
df = df.drop(ind_drop, axis=0)

# Scatterplots without outliers
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df)
plt.title('Scatterplot of Gr Liv Area vs SalePrice without outliers')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/plots/scatterplot2_without_outliers.png")
plt.close()

sns.scatterplot(x='Overall Qual', y='SalePrice', data=df)
plt.title('Scatterplot of Overall Qual vs SalePrice without outliers')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/plots/scatterplot1_without_outliers.png")
plt.close()

# save DataFrame
df.to_csv("D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/data/Ames_outliers_removed.csv", index=False)