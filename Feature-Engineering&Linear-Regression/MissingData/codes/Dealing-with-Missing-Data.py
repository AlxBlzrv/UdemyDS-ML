import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/data/Ames_Housing_Feature_Description.txt','r') as f: 
    print(f.read())

df = pd.read_csv("D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/data/Ames_outliers_removed.csv")

print(df.head())
print(len(df.columns))
print(df.info())

df = df.drop('PID',axis=1)
print(len(df.columns))

print(df.isnull())
print(df.isnull().sum())
print(100* df.isnull().sum() / len(df))

def percent_missing(df):
    percent_nan = 100* df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan>0].sort_values()
    return percent_nan

percent_nan = percent_missing(df)

plt.figure(figsize=(15, 15))
plt.rc('font', size=10)
sns.barplot(x=percent_nan.index, y=percent_nan, hue=percent_nan.index, palette='husl', dodge=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Number of missing values')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/barplot1.png")
plt.close()

plt.figure(figsize=(15, 11))
plt.rc('font', size=10)
sns.barplot(x=percent_nan.index, y=percent_nan, hue=percent_nan.index, palette='husl', dodge=False)
plt.xticks(rotation=90)
# Set the threshold value to 1%
plt.ylim(0,1)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Number of missing values (zoom)')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/barplot2.png")
plt.close()

# Compare with threshold value
print(percent_nan[percent_nan < 1])
print(100/len(df))
print(df[df['Total Bsmt SF'].isnull()])
print(df[df['Bsmt Half Bath'].isnull()])

# Filling numeric columns
bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)

# Filling text columns
bsmt_str_cols =  ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')

percent_nan = percent_missing(df)

plt.figure(figsize=(15, 11))
plt.rc('font', size=10)
sns.barplot(x=percent_nan.index, y=percent_nan, hue=percent_nan.index, palette='husl', dodge=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Number of missing values after filling (1)')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/barplot3.png")
plt.close()

df = df.dropna(axis=0,subset= ['Electrical','Garage Cars'])
percent_nan = percent_missing(df)

plt.figure(figsize=(15, 8))
plt.rc('font', size=10)
sns.barplot(x=percent_nan.index, y=percent_nan, hue=percent_nan.index, palette='husl', dodge=False)
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Number of missing values after deletion (1)')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/barplot4.png")
plt.close()

df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)

percent_nan = percent_missing(df)

plt.figure(figsize=(15, 12))
plt.rc('font', size=10)
sns.barplot(x=percent_nan.index, y=percent_nan, hue=percent_nan.index, palette='husl', dodge=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Number of missing values after filling (2)')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/barplot5.png")
plt.close()

print(df[['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']])

gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[gar_str_cols] = df[gar_str_cols].fillna('None')
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

percent_nan = percent_missing(df)

plt.figure(figsize=(15, 12))
plt.rc('font', size=10)
sns.barplot(x=percent_nan.index, y=percent_nan, hue=percent_nan.index, palette='husl', dodge=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Number of missing values after filling (3)')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/barplot6.png")
plt.close()

print(percent_nan.index)
print(df[['Lot Frontage', 'Fireplace Qu', 'Fence', 'Alley', 'Misc Feature','Pool QC']])
df = df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)

percent_nan = percent_missing(df)

plt.figure(figsize=(15, 12))
plt.rc('font', size=10)
sns.barplot(x=percent_nan.index, y=percent_nan, hue=percent_nan.index, palette='husl', dodge=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Number of missing values after deletion (2)')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/barplot7.png")
plt.close()

df['Fireplace Qu'] = df['Fireplace Qu'].fillna("None")
percent_nan = percent_missing(df)

plt.figure(figsize=(15, 12))
plt.rc('font', size=10)
sns.barplot(x=percent_nan.index, y=percent_nan, hue=percent_nan.index, palette='husl', dodge=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Number of missing values after filling (4)')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/barplot8.png")
plt.close()

print(df['Neighborhood'].unique())
plt.figure(figsize=(8,12))
plt.rc('font', size=10)
sns.boxplot(x='Lot Frontage', y='Neighborhood', data=df, orient='h', hue='Neighborhood', palette='husl', dodge=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Boxplot of Lot Frontage by Neighborhood')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/boxplot.png")
plt.close()

print(df.groupby('Neighborhood')['Lot Frontage'])
print(df.groupby('Neighborhood')['Lot Frontage'].mean())

print(df.head()['Lot Frontage'])
print(df[df['Lot Frontage'].isnull()])
print(df.iloc[21:26]['Lot Frontage'])
print(df.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean())))
print(df.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean())).iloc[21:26])

df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean()))
percent_nan = percent_missing(df)

plt.figure(figsize=(15, 12))
plt.rc('font', size=10)
sns.barplot(x=percent_nan.index, y=percent_nan, hue=percent_nan.index, palette='husl', dodge=False)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Number of missing values after filling (5)')
plt.savefig("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/plots/barplot9.png")
plt.close()

df['Lot Frontage'] = df['Lot Frontage'].fillna(0)
percent_nan = percent_missing(df)
print(percent_nan)

df.to_csv("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/data/Ames_NO_Missing_Data.csv",index=False)