import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Load the dataset from CSV
df = pd.read_csv('C:\\Users\\VINAY BHARADWAJ\\Desktop\\internship\\data.csv')


# Correlation matrix to understand feature relationships
cor_matrix = df.corr()
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm')
plt.title("------Correlation Matrix-------",color='r')
plt.show()

# Preprocessing: Selecting features and target variable
x = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
y = df['price']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression() #here we created a regression model
# Fitting the model on the training data
model.fit(x_train, y_train)
# Model Evaluation
y_pred = model.predict(x_test)
meanerror = mean_squared_error(y_test, y_pred)
r_square = r2_score(y_test, y_pred)

print("Mean Squared Error:", meanerror)
print("R-squared:", r_square)
print('-------------------------------------------------------------')
# now we will plot the predicted and actual price
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices---------->")
plt.ylabel("Predicted Prices------------>")
plt.title("Actual vs. Predicted Price",color='r')
plt.show()
print('-------------------------------------------------------------')

# We can also create a residual plot to check the model's performance
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices-------------->")
plt.ylabel("Residuals------------>")
plt.title("Residual Plot",c='r')
plt.show()
print('-------------------------------------------------------------')

#now we will train our model to predict on new data
new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]
predicted_price = model.predict(new_data)
print('Prediction on the basis of the following data:  ',new_data,'\t')
print("Predicted Price:", predicted_price[0])
print('-------------------------------------------------------------')

a=input('\nDo you want to predict your data? (y for yes and n for no): ')
if a=='y':
    while a=='y':
        n=int(input('Enter number of data: '))
        new=list(map(int,input('Enter Data(seperated by single space character): ').strip().split()))[:n]
        c=new
        new=np.array(new,ndmin=2)
        predicted_price = model.predict(new)
        print('Prediction on the basis of the following given data: ',c)
        print("Predicted Price: ", predicted_price[0])
        a=input('\nDo you want to predict your data? (y for yes and n for no): ')
    else:
        print('\n--------------------Thank you--------------------\nHouse price prediction project by Shristi Shrivastava')
elif a!='y' or a!='n':
    print('Invalid input')
    a=input('\nDo you want to predict your data? (y for yes and n for no): ')
else:
    print('\n--------------------Thank you--------------------\nHouse price prediction project by Shristi Shrivastava')
