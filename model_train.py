#!/usr/bin/env python
# coding: utf-8

# # Task 3 - Modeling
# 
# This notebook will get you started by helping you to load the data, but then it'll be up to you to complete the task! If you need help, refer to the `modeling_walkthrough.ipynb` notebook.
# 
# 
# ## Section 1 - Setup
# 
# First, we need to mount this notebook to our Google Drive folder, in order to access the CSV data file. If you haven't already, watch this video https://www.youtube.com/watch?v=woHxvbBLarQ to help you mount your Google Drive folder.

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# We want to use dataframes once again to store and manipulate the data.

# In[2]:


get_ipython().system('pip install pandas')


# In[3]:


import pandas as pd


# ---
# 
# ## Section 2 - Data loading
# 
# Similar to before, let's load our data from Google Drive for the 3 datasets provided. Be sure to upload the datasets into Google Drive, so that you can access them here.

# In[4]:


path = "/content/drive/MyDrive/Forage - Cognizant AI Program/Task 3/Resources/"

sales_df = pd.read_csv(f"{path}sales.csv")
sales_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
sales_df.head()


# In[5]:


stock_df = pd.read_csv(f"{path}sensor_stock_levels.csv")
stock_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
stock_df.head()


# In[6]:


temp_df = pd.read_csv(f"{path}sensor_storage_temperature.csv")
temp_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
temp_df.head()


# In[7]:


# to get information on dataset
sales_df.info()


# Now it's up to you, refer back to the steps in your strategic plan to complete this task. Good luck!

# In[9]:


stock_df.info()


# In[10]:



temp_df.info()


# ###### All data looks good in three datasets instead timestamp. So, we will convert this to work properly with each datasets

# In[11]:


def convert_to_datetime(data: pd.DataFrame = None, column: str = None):

  dummy = data.copy()
  dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
  return dummy


# In[12]:


sales_df = convert_to_datetime(sales_df, 'timestamp')
sales_df.info()


# In[13]:


stock_df = convert_to_datetime(stock_df, 'timestamp')
stock_df.info()


# In[14]:


temp_df = convert_to_datetime(temp_df, 'timestamp')
temp_df.info()


# #### Section 5 - Merging all Datasets
# ###### We can revisit the problem statement : 
# “Can we accurately predict the stock levels of products, based on sales data and sensor data, on an hourly basis in order to more intelligently procure products from our suppliers.”
# 
# So, to predict our model likewise, we shall merge the all three datasets and make a new DataFrame to work on to help us for model prediction. For that, now we shall first transform the timestamp column to predict hourly basis output.

# In[15]:



sales_df.head()


# In[17]:


# Importing Datetime method

from datetime import datetime

def convert_timestamp_to_hourly(data: pd.DataFrame = None, column: str = None):
  dummy = data.copy()
  new_ts = dummy[column].tolist()
  new_ts = [i.strftime('%Y-%m-%d %H:00:00') for i in new_ts]
  new_ts = [datetime.strptime(i, '%Y-%m-%d %H:00:00') for i in new_ts]
  dummy[column] = new_ts
  return dummy


# In[18]:


sales_df = convert_timestamp_to_hourly(sales_df, 'timestamp')
sales_df.head()


# In[19]:


stock_df = convert_timestamp_to_hourly(stock_df, 'timestamp')
stock_df.head()


# In[20]:


temp_df = convert_timestamp_to_hourly(temp_df, 'timestamp')
temp_df.head()


# In[ ]:





# ###### Now, all the timestamp column have transformed and reduced to 00. Next, we shall aggregate the datasets to combine rows that have the same value on timestamp.

# In[21]:


# Now we shall aggregate sales dataset on 'quantity' and groupby 'timestamp' and 'product_id' column

sales_agg = sales_df.groupby(['timestamp', 'product_id']).agg({'quantity': 'sum'}).reset_index()
sales_agg.head()


# In[23]:


# Now, for the stock data, we shall group it in the same way and aggregate the estimated_stock_pct.
stock_agg = stock_df.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()
stock_agg.head()



# In[24]:


# In the temparature data, the product_id does not exist. So, we shall groupby using 'timestamp' and then aggregate it.
temp_agg = temp_df.groupby(['timestamp']).agg({'temperature': 'mean'}).reset_index()
temp_agg.head()


# In[25]:


# hence, we receive the mean()/ average temparature of the storge in the warehouse by unique hours during the week. We shall use base table stock_agg and merge the others onto it. 


merged_df = stock_agg.merge(sales_agg, on=['timestamp', 'product_id'], how='left')
merged_df.head()


# In[26]:


# left merge is used
merged_df = merged_df.merge(temp_agg, on='timestamp', how='left')
merged_df.head()


# In[28]:


# Check for null values
merged_df.info()


# In[29]:


# We can see from above output that presence of null so to fill with 0
merged_df['quantity'] = merged_df['quantity'].fillna(0)
merged_df.info()


# In[ ]:





# ###### Now, some extra features are included to get better result
# 

# In[31]:


product_categories = sales_df[['product_id', 'category']]
product_categories = product_categories.drop_duplicates()

product_price = sales_df[['product_id', 'unit_price']]
product_price = product_price.drop_duplicates()


# In[32]:


merged_df = merged_df.merge(product_categories, on="product_id", how="left")
merged_df.head()


# In[33]:


merged_df = merged_df.merge(product_price, on="product_id", how="left")
merged_df.head()


# In[35]:


merged_df.info()


# ###### Section 6 - Feature Engineering
# So, we have now our clean and merged dataset. Now, we can transform this to get suitable format for our Machine learningModel.Henc, instead Categorical feature we shall use Numerical features.
# 
# We can explode 'timestamp' to break it into day of the week, day of the month, and hour.

# In[36]:


merged_df['timestamp_day_of_month'] = merged_df['timestamp'].dt.day
merged_df['timestamp_day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['timestamp_hour'] = merged_df['timestamp'].dt.hour
merged_df.drop(columns=['timestamp'], inplace=True)
merged_df.head()


# In[ ]:





# ###### Now to work with Category to convert 'category to numeric using dummy variables.
# 
# A dummy variable is a binary flag contains only 0's and 1's. This indicates whether a row fits a particular value of that column or not. 

# In[37]:


merged_df = pd.get_dummies(merged_df, columns=['category'])
merged_df.head()


# ###### We can see that product_id column is not numeric. It's an unique column that rows represents by combining with 'timestamp' hourly basis, so, as an ID we can remove this from processing the model.

# In[38]:


merged_df.drop(columns=['product_id'], inplace=True)
merged_df.head()


# ###### We will use cross-validation.
# To ensure that the trained Machine learning model is able to perform robustly, we shall use it to test it several times on random samples of data. For that, we will use a K-fold strategy to train the machine learning model on K (K is an integer to be decided) random samples of the data.

# In[39]:


#  Now, we shall create our target variable y and independent variables X
X = merged_df.drop(columns=['estimated_stock_pct'])
y = merged_df['estimated_stock_pct']
print(X.shape)
print(y.shape)


# In[ ]:





# ###### We have 28 predictor variables that we will train our machine learning model on and 10845 rows of data.
# 
# Now let's define how many folds we want to complete during training, and how much of the dataset to assign to training, leaving the rest for test.
# 

# In[40]:


K = 10
split = 0.75


# ###### From Ensamble learning we shall use RandomForestRegressor.

# In[41]:


get_ipython().system('pip install scikit-learn')


# In[42]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# ###### And now let's create a loop to train K models with a 70-30% random split of the data each time between training and test samples

# In[43]:


accuracy = []

for fold in range(0, K):

  # Instantiate algorithm
  model = RandomForestRegressor()
  scaler = StandardScaler()

  # Create training and test samples
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

  # Scale X data, we scale the data because it helps the algorithm to converge
  # and helps the algorithm to not be greedy with large values
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)

  # Train model
  trained_model = model.fit(X_train, y_train)

  # Generate predictions on test sample
  y_pred = trained_model.predict(X_test)

  # Compute accuracy, using mean absolute error
  mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
  accuracy.append(mae)
  print(f"Fold {fold + 1}: MAE = {mae:.3f}")

print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")


# ###### We can see that the Mean Absolute Error (MAE), the performance metrics, is almost exactly the same each time. This shows that the performance of the model is consistent across the different random samples of the data.
# 
# N.B. Even though the model is predicting robustly, the MAE is not as the average value of the target variable is around 0.51. This means that the accuracy percentage is around 50%. At this stage, we have small samples of the data, we can report back to the business with these findings and recommend that the the dataset needs to be further engineered, or more datasets need to be added.

# ###### We can use the trained model to intepret those features that are signficant when the model is predicting the target variable. We can use matplotlib and numpy to visualuse the results, so we should install and import this package.

# In[46]:


import matplotlib.pyplot as plt
import numpy as np 


# In[47]:


features = [i.split("__")[0] for i in X.columns]
importances = model.feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots(figsize=(10, 20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# 

# In[ ]:





# ###### This feature importance visualisation tells us:
# 1. The product categories are not so important.
# 2. The unit price, and the temperature are important in predicting the stock.
# 3. The hour of the day is also important for predicting stock.
# 4. These insights tells that now report can be generate, and provide to the business.
# 
# 
# 
# 
# 
# 

# 
