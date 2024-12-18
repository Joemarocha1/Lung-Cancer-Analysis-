#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# In[58]:


data = pd.read_csv(r'C:\Users\joema\Downloads\lung cancer 2.csv')


# In[59]:


print(data.head(3))


# In[60]:


Columns_of_interest = ['AGE', 'GENDER' , 'SMOKING', 'LUNG_CANCER' ]
print(data[Columns_of_interest])


# In[15]:


age_range = ['AGE']
max_age = data[age_range].max() 
min_age = data[age_range].min() 
print(max_age)
print(min_age)


# In[61]:


import matplotlib.pyplot as plt 

age_counts = data['AGE'].value_counts().sort_index()  # Count and sort ages

# Step 3: Create the bar graph
plt.figure(figsize=(10, 6))  # Set the figure size
plt.bar(age_counts.index, age_counts.values, color='skyblue')

plt.xlabel('Age')  # X-axis label
plt.ylabel('Count')  # Y-axis label
plt.title('Distribution of Age')  # Graph title

# Step 4: Show the graph
plt.show()


# In[62]:


# Group by age and smoking status, and count occurrences
age_smoking = data.groupby(['AGE', 'SMOKING']).size().unstack(fill_value=0)

# Create a grouped bar chart
plt.figure(figsize=(12, 6))
age_smoking.plot(kind='bar', stacked=False, color=['skyblue', 'purple'])

plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Smoking Status by Age')
plt.legend(['Non-Smoker (0)', 'Smoker (1)'], title='Smoking Status')

plt.show()



# In[64]:


lungcancer_smoking = data.groupby(['SMOKING', 'LUNG_CANCER']).size().unstack(fill_value=0)

plt.figure(figsize=(9,5))
lungcancer_smoking.plot(kind='bar', stacked = False, color=['blue', 'purple'])

plt.xlabel('smoking')
plt.ylabel('lung cancer')
plt.title('Lung Cancer/ smokers vs non- smokers')

plt.show()


# In[78]:


bins = [29, 39, 49, 59, 69, 79, 89, float('inf')]  # Bin edges
labels = [ '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

# Step 3: Create a new column with age groups
data['Age_Group'] = pd.cut(data['AGE'], bins=bins, labels=labels, right=True)

# Step 2: Define your columns for the heatmap
x_col = 'Age_Group'  # X-axis: AGE
y_col = 'SMOKING'  # Y-axis: SMOKING
z_col = 'LUNG_CANCER'  # Color intensity: LUNG_CANCER

# Step 3: Reshape the data using pivot_table
heatmap_data = data.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc='mean')

# Step 4: Plot the heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Lung Cancer Probability'})
plt.title("Heatmap of Lung Cancer Probability by Age and Smoking Status")

plt.close()


# In[84]:


bins = [29, 39, 49, 59, 69, 79, 89, float('inf')]  # Bin edges
labels = [ '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

# Step 3: Create a new column with age groups
data['Age_Group'] = pd.cut(data['AGE'], bins=bins, labels=labels, right=True)

x_col1 = 'Age_Group'
y_col1 = 'YELLOW_FINGERS'
z_col1 = 'LUNG_CANCER'

heatmap_data = data.pivot_table(index= y_col1, columns = x_col1, values = z_col1) 

plt.figure(figsize=(8,4))
sns.heatmap(heatmap_data, cmap="RdYlGn", annot=True, fmt=".2f", cbar_kws={'label': 'Lung Cancer Probability'})
plt.title("Heatmap of Lung Cancer Probability by Age and Yellow finger Status")
plt.show() 


# In[88]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

X = data['SMOKING']
Y = data['LUNG_CANCER']

# Step 1: Reshape the feature if it's in 1D form
X = X.values.reshape(-1, 1)
Y = Y.values.reshape(-1, 1)

#split data for training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#model evaluation
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[ ]:




