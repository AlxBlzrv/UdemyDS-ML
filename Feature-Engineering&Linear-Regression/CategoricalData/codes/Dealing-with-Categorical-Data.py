import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D:/work/git/udemy/Feature-Engineering&Linear-Regression/MissingData/data/Ames_NO_Missing_Data.csv")

print(df.head())

with open('D:/work/git/udemy/Feature-Engineering&Linear-Regression/Outliers/data/Ames_Housing_Feature_Description.txt','r') as f: 
    print(f.read())

# Convert to string
df['MS SubClass'] = df['MS SubClass'].apply(str)

print(df.select_dtypes(include='object'))

df_nums = df.select_dtypes(exclude='object')
df_objs = df.select_dtypes(include='object')

print(df_nums.info())
print(df_objs.info())

df_objs = pd.get_dummies(df_objs,drop_first=True)
final_df = pd.concat([df_nums,df_objs],axis=1)

print(final_df)
print(final_df.corr()['SalePrice'].sort_values())

df.to_csv("D:/work/git/udemy/Feature-Engineering&Linear-Regression/CategoricalData/data/Ames_NO_Missing_Data.csv",index=False)