"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')

train = train[(train['Commodities'] == 'APPLE GOLDEN DELICIOUS')]
#Filter out outliers using Interquartile Range method
Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1


#filter out the outliers
index = train[(train["avg_price_per_kg"] >= 15)].index
index2 = train[(train["avg_price_per_kg"] >7) & (train["Weight_Kg"] == 400)].index
#drop outliers
train.drop(index, inplace=True)
train.drop(index2, inplace=True)
y_train = train['avg_price_per_kg']
X_train = train[['Weight_Kg','Total_Qty_Sold','Stock_On_Hand']]

# Fit model
regr_tree = DecisionTreeRegressor(max_depth=2,random_state=42)
print ("Training Model...")
regr_tree.fit(X_train,y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/apples_simple_lm_regression_own.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(regr_tree, open(save_path,'wb'))
