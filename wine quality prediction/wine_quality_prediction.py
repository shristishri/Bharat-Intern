 np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#now we are importing the dataframe
df = pd.read_csv('C:\\Users\\akash\\Downloads\\internship\\ml\\wine quality prediction\\winequality.csv')
print('This is our training database')
print('--------------------------------------------------------------\n')

print(df.describe().T)
print('--------------------------------------------------------------')
df.nunique()
df.duplicated().sum()

sns.set(style="whitegrid")
print(df['quality'].value_counts())
fig = plt.figure(figsize = (10,6))

plt.figure(figsize = (15,15))
sns.heatmap(df.corr(),annot=True, cmap= 'PuBuGn')
color = sns.color_palette("pastel")
print('--------------------------------------------------------------')

fig, ax1 = plt.subplots(3,4, figsize=(24,30))
k = 0
columns = list(df.columns)
for i in range(3):
    for j in range(4):
            sns.distplot(df[columns[k]], ax = ax1[i][j], color = 'red')
            k += 1
plt.show()
print('--------------------------------------------------------------')

def log_transform(col):
    return np.log(col[0])

df['residual sugar'] = df[['residual sugar']].apply(log_transform, axis=1)
df['chlorides'] = df[['chlorides']].apply(log_transform, axis=1)
df['free sulfur dioxide'] = df[['free sulfur dioxide']].apply(log_transform, axis=1)
df['total sulfur dioxide'] = df[['total sulfur dioxide']].apply(log_transform, axis=1)
df['sulphates'] = df[['sulphates']].apply(log_transform, axis=1)
color = sns.color_palette("pastel")

fig, ax1 = plt.subplots(3,4, figsize=(24,30))
k = 0
columns = list(df.columns)
for i in range(3):
    for j in range(4):
            sns.distplot(df[columns[k]], ax = ax1[i][j], color = 'green')
            k += 1
plt.show()
print('--------------------------------------------------------------')

df.corr()['quality'].sort_values(ascending=False)
df_3 = df[df.quality==3]     # MINORITY          
df_4 = df[df.quality==4]     # MINORITY          
df_5 = df[df.quality==5]     # MAJORITY
df_6 = df[df.quality==6]     # MAJORITY
df_7 = df[df.quality==7]     # MINORITY
df_8 = df[df.quality==8]     # MINORITY
# Oversample MINORITY Class to make balance data :
from sklearn.utils import resample

df_3_up = resample(df_3, replace=True, n_samples=600, random_state=12) 
df_4_up = resample(df_4, replace=True, n_samples=600, random_state=12) 
df_7_up = resample(df_7, replace=True, n_samples=600, random_state=12) 
df_8_up = resample(df_8, replace=True, n_samples=600, random_state=12) 

# Decreases the rows of Majority one's to make balance data :
df_5_down = df[df.quality==5].sample(n=600).reset_index(drop=True)
df_6_down = df[df.quality==6].sample(n=600).reset_index(drop=True)
# Combine downsampled majority class with upsampled minority class
Balanced_df = pd.concat([df_3_up, df_4_up, df_7_up, 
                         df_8_up, df_5_down, df_6_down]).reset_index(drop=True)


# Display new class counts
Balanced_df.quality.value_counts()   
plt.figure(figsize=(10,6))
sns.countplot(x='quality', data=Balanced_df, order=[3, 4, 5, 6, 7, 8], palette='pastel')
plt.figure(figsize = (12,6))
sns.barplot(x='quality', y = 'alcohol', data = df, palette = 'coolwarm')
plt.figure(figsize=(15,15))
Balanced_df.corr().quality.apply(lambda x: abs(x)).sort_values(ascending=False).iloc[1:11][::-1].plot(kind='barh',color='pink') 
# calculating the top 10 highest correlated features
# with respect to the target variable i.e. "quality"
plt.title("Top 10 highly correlated features", size=20, pad=26)
plt.xlabel("Correlation coefficient")
plt.ylabel("Features")
print('--------------------------------------------------------------')

selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides',
                     'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                     'sulphates', 'alcohol']
x = Balanced_df[selected_features]
y = Balanced_df.quality
from sklearn.model_selection import train_test_split

# Splitting the data into 70% and 30% to construct Training and Testing Data respectively.
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3,random_state=13)
from sklearn.neighbors import KNeighborsClassifier  
# For weights = 'uniform'
for n_neighbors in [5,10,15,20]:
    model = KNeighborsClassifier(n_neighbors)
    model.fit(x_train, y_train) 
    scr = model.score(x_test, y_test)
    print("For n_neighbors = ", n_neighbors  ," score is ",scr)
print('--------------------------------------------------------------')

# For weights = 'distance'
for n_neighbors in [5,10,15,20]:
    model = KNeighborsClassifier(n_neighbors, weights='distance')
    model.fit(x_train, y_train) 
    scr = model.score(x_test, y_test)
    print("For n_neighbors = ", n_neighbors  ," score is ",scr)
print('--------------------------------------------------------------')

# Creating a k-nearest neighbors Classifier
KNN = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Train the model using the training set
KNN.fit(x_train, y_train) 
results = KNN.fit(x_train, y_train)
train_predictions = KNN.predict(x_train)
test_predictions = KNN.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix

print("\n Train Data: KNN_Confusion Matrix:\n ")
print(confusion_matrix(y_train, train_predictions))
print('--------------------------------------------------------------')

print("\n Train Data: KNN_Classification Report:\n ")
print(classification_report(y_train,train_predictions))
print('--------------------------------------------------------------')

print("\n \n Test Data: KNN_Confusion Matrix: \n ")
print(confusion_matrix(y_test, test_predictions))
print('--------------------------------------------------------------')

print("\n Test Data: KNN_Classification Report:\n ")
print(classification_report(y_test,test_predictions))

print('-----------------------------------------------')
print('PROGRAMMED BY SHRISTI SHRIVASTAVA')
print('-----------------------------------------------')
