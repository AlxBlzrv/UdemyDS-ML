# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Load the dataset
df = pd.read_csv('D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\data\\Telco-Customer-Churn.csv')
print(df.head())

# Display statistical information for columns
print(df.info())
print(df.describe())

# Check for missing values
print(df.isna().sum())

# Plot the distribution of churn
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.countplot(data=df, x='Churn')
plt.title('Number of customers for each churn status')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\countplot1.png")
plt.close()

# Plot the distribution of TotalCharges across churn categories
plt.figure(figsize=(10, 8))
plt.rc('font', size=10)
sns.violinplot(data=df, x='Churn', y='TotalCharges')
plt.title('Distribution of the TotalCharges column by churn status')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\violinplot.png")
plt.close()

# Plot the distribution of TotalCharges by Contract type and churn status
plt.figure(figsize=(10,4),dpi=200)
plt.rc('font', size=10)
sns.boxplot(data=df, y='TotalCharges', x='Contract', hue='Churn')
plt.legend(loc=(1.01,0.5))
plt.title('Distribution of TotalCharges by Contract and Churn')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\boxplot.png")
plt.close()

# Correlation analysis
print(df.columns)
corr_df  = pd.get_dummies(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod','Churn']]).corr()

# Print correlations excluding 'Churn_Yes' and 'Churn_No'
print(corr_df['Churn_Yes'].sort_values().iloc[1:-1])

# Plot feature correlations with churn
plt.figure(figsize=(10,5),dpi=200)
plt.rc('font', size=8)
sns.barplot(x=corr_df['Churn_Yes'].sort_values().iloc[1:-1].index,y=corr_df['Churn_Yes'].sort_values().iloc[1:-1].values)
plt.title("Feature Correlation with Churn")
plt.xticks(rotation=90)
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\barplot1.png",
            bbox_inches='tight', pad_inches=0.1)
plt.close()

# Explore contract types
print(df['Contract'].unique())

# Plot tenure distribution
plt.figure(figsize=(10,4),dpi=200)
plt.rc('font', size=8)
sns.histplot(data=df, x='tenure', bins=60)
plt.title('Distribution of tenure')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\histplot.png")
plt.close()

# Plot tenure distribution by Contract and Churn
plt.figure(figsize=(10,3),dpi=200)
plt.rc('font', size=10)
sns.displot(data=df,x='tenure',bins=70,col='Contract',row='Churn')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\displot.png")
plt.close()

# Plot MonthlyCharges vs TotalCharges with Churn differentiation
plt.figure(figsize=(10,4),dpi=200)
plt.rc('font', size=10)
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Churn', 
                linewidth=0.5, alpha=0.5, palette='Dark2')
plt.title('Distribution of TotalCharges vs MonthlyCharges by Churn')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\scatterplot1.png")
plt.close()

# Cohort analysis
no_churn = df.groupby(['Churn','tenure']).count().transpose()['No']
yes_churn = df.groupby(['Churn','tenure']).count().transpose()['Yes']
churn_rate = 100 * yes_churn / (no_churn+yes_churn)
print(churn_rate.transpose()['customerID'])

# Plot churn percentage by tenure
plt.figure(figsize=(10,4),dpi=200)
plt.rc('font', size=10)
churn_rate.iloc[0].plot()
plt.ylabel('Churn Percentage')
plt.title('Churn Percentage by Tenure')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\plot.png")
plt.close()

# Define Tenure Cohorts
def cohort(tenure):
    if tenure < 13:
        return '0-12 Months'
    elif tenure < 25:
        return '12-24 Months'
    elif tenure < 49:
        return '24-48 Months'
    else:
        return "Over 48 Months"

# Apply Tenure Cohorts to dataframe
df['Tenure Cohort'] = df['tenure'].apply(cohort)
print(df.head(10)[['tenure','Tenure Cohort']])

# Plot TotalCharges vs MonthlyCharges with Tenure Cohorts
plt.figure(figsize=(10,4),dpi=200)
plt.rc('font', size=10)
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Tenure Cohort', 
                linewidth=0.5, alpha=0.5, palette='Dark2')
plt.title('Distribution of TotalCharges vs MonthlyCharges by Tenure Cohort')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\scatterplot2.png")
plt.close()

# Plot count of churn/non-churn customers by Tenure Cohort
plt.figure(figsize=(10, 4))
plt.rc('font', size=10)
sns.countplot(data=df, x='Tenure Cohort', hue='Churn')
plt.title('Count of churn and non-churn customers by Tenure Cohort')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\countplot2.png")
plt.close()

# Plot churn/non-churn customers by Tenure Cohort and Contract
plt.figure(figsize=(10,4),dpi=200)
plt.rc('font', size= 10)
sns.catplot(data=df, x='Tenure Cohort', hue='Churn', 
            col='Contract', kind='count')
plt.title('Count of churn and non-churn customers by Tenure Cohort and Contract')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\catplot.png")
plt.close()


# Decision Tree
X = df.drop(['Churn','customerID'],axis=1)
X = pd.get_dummies(X,drop_first=True)
y = df['Churn']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Initializing and fitting Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X_train,y_train)

# Predictions and evaluation
preds = dt.predict(X_test)
print(classification_report(y_test, preds))

# Heatmap plot of the confusion matrix
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
plt.rc('font', size=10)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap for DecisionTreeClassifier')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\heatmap_confusion_matrix1.png")
plt.close()

# Accuracy score
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)

# Feature Importance
print(dt.feature_importances_)
imp_feats = pd.DataFrame(data=dt.feature_importances_, 
                         index=X.columns,
                         columns=['Feature Importance']).sort_values("Feature Importance")

print(imp_feats)

plt.figure(figsize=(14,6),dpi=200)
plt.rc('font', size=10)
sns.barplot(data=imp_feats.sort_values('Feature Importance'),
            x=imp_feats.sort_values('Feature Importance').index,
            y='Feature Importance')
plt.xticks(rotation=90)
plt.title("Feature Importance for Decision Tree")
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\barplot2.png",
            bbox_inches='tight', pad_inches=0.1)
plt.close()

plt.figure(figsize=(12,8),dpi=150)
plt.rc('font', size=10)
plot_tree(dt,filled=True,feature_names=X.columns.tolist())
plt.title("Decision Tree visualization graph")
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\plot_tree.png")
plt.close()

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)

preds = rf.predict(X_test)
print(classification_report(y_test, preds))

# Heatmap plot of the confusion matrix
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
plt.rc('font', size=10)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap for RandomForestClassifier')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\heatmap_confusion_matrix2.png")
plt.close()

# Accuracy score
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)

# Boosted Trees
ada_model = AdaBoostClassifier()
ada_model.fit(X_train,y_train)

preds = ada_model.predict(X_test)
print(classification_report(y_test,preds))

# Heatmap plot of the confusion matrix
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
plt.rc('font', size=10)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap for AdaBoostClassifier')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\heatmap_confusion_matrix3.png")
plt.close()

# Accuracy score
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)

# Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

preds = gb_model.predict(X_test)
print(classification_report(y_test, preds))

# Heatmap plot of the confusion matrix
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
plt.rc('font', size=10)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap for GradientBoostingClassifier')
plt.savefig("D:\\Dell\\repos\\UdemyDS-ML\\Decision-Tree-And-Random-Forest\\plots\\heatmap_confusion_matrix4.png")
plt.close()

# Accuracy score
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)
